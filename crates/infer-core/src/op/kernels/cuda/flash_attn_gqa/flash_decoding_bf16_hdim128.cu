// ============================================================================
// Flash Decoding BF16, head_dim = 128 — split-K across KV sequence
//
// Optimised version (see cuda-optimized-skill run_main, v2):
//   - Grid = (num_q_heads, N_SPLIT) instead of (num_q_heads,) alone.  Each
//     block does online softmax over its KV chunk and writes a partial
//     (m, l, O) to a global workspace.
//   - A small pass-2 kernel merges the N_SPLIT partials per query head using
//     the standard log-sum-exp rule.
//   - Intra-block group-merge scratch (`s_acc`) is stored in bf16 (not fp32)
//     to cut per-block shared memory from 24.96 KB to 20.8 KB.  This takes
//     the Block Limit SMEM from 3 to 4 blocks per SM on Ampere (SM_8x).
//
// Measured on NVIDIA A10 (sm_86), head_dim=128, num_q_heads=32, num_kv_heads=8,
// seq_len=2048 (decode phase, q_seq_len=1):
//
//   | Version               | Kernel median | Speedup vs baseline |
//   | --------------------- | ------------- | ------------------- |
//   | baseline (pre-split)  | 0.0589 ms     | 1.00x               |
//   | v1 split-K N_SPLIT=8  | 0.0303 ms     | 1.94x               |
//   | v2 + bf16 s_acc       | 0.0255 ms     | 2.31x  (this file)  |
//
//   - DRAM throughput:  20% -> 53% of peak (NCU)
//   - Achieved occupancy: 17% -> 57%
//   - L2 hit rate kept at ~75% via the split-K chunking
//   - Full NCU report (pass-1 kernel) available under
//     cuda-optimized-skill/flash_decoding_bf16_hdim128_opt/optimize_runs/run_main/iter_v2/full.ncu-rep
//
// Correctness verified against a bf16-quantised PyTorch reference
// (atol=2e-2, rtol=2e-2, seq_len 2048): max |delta| ≈ 4.88e-4.
// ============================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>
#include <stdio.h>
#include "flash_attn_gqa.h"

// ----------------------------------------------------------------------------
// Async copy macros (PTX cp.async)
// ----------------------------------------------------------------------------
#define CP_ASYNC_CG(dst, src, bytes)                                           \
    asm volatile(                                                              \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(dst),     \
        "l"(src), "n"(bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// head_dim = 128, 16 groups of 16 lanes per block = 256 threads
#define HD128              128
#define BN128              16     // tokens per KV micro-tile
#define THREADS_PER_KEY128 16     // lanes per group (each lane owns 8 dims)
#define NUM_GROUPS128      16     // 256 / 16
#define N_SPLIT            8      // KV-axis partitions per q_head

// ============================================================================
// PASS 1 — per-chunk online softmax
//
// grid = (num_q_heads, N_SPLIT)   block = 256
// Each block computes the partial softmax over tokens
// [blockIdx.y * chunk_size, min(seq_len, (blockIdx.y+1) * chunk_size) ).
// Writes  M_g[q, s]               : running max
//         L_g[q, s]               : running denominator (post-rescale)
//         O_g[q, s, 0..127]       : un-normalised partial O (fp32)
// ============================================================================
__global__ void flash_decode_bf16_hdim128_splitk_pass1_kernel(
    const __nv_bfloat16* __restrict__ Q,      // [num_q_heads, 128]
    const __nv_bfloat16* __restrict__ K,      // [seq_len, num_kv_heads, 128]
    const __nv_bfloat16* __restrict__ V,      // [seq_len, num_kv_heads, 128]
    float*               __restrict__ M_g,    // [num_q_heads, N_SPLIT]
    float*               __restrict__ L_g,    // [num_q_heads, N_SPLIT]
    float*               __restrict__ O_g,    // [num_q_heads, N_SPLIT, 128]
    int32_t*             seq_len_ptr,
    int num_kv_heads,
    int group_size,
    float sm_scale)
{
    int32_t seq_len = *seq_len_ptr + 1;
    int q_head_idx  = blockIdx.x;
    int split_idx   = blockIdx.y;
    int kv_head_idx = q_head_idx / group_size;

    // Compute chunk_size on device so the host FFI does not need to know
    // the true seq_len.  chunk_size is ceil(seq_len / N_SPLIT) rounded up
    // to a multiple of BN128 so cp.async micro-tiles stay aligned.
    int per_split   = (seq_len + N_SPLIT - 1) / N_SPLIT;
    int chunk_size  = ((per_split + BN128 - 1) / BN128) * BN128;

    int chunk_begin = split_idx * chunk_size;
    if (chunk_begin >= seq_len) {
        // Fully-empty chunk publishes identity (m=-inf, l=0, O=0) for pass-2 safety.
        if (threadIdx.x == 0) {
            int mo = q_head_idx * N_SPLIT + split_idx;
            M_g[mo] = -1e30f;
            L_g[mo] = 0.0f;
        }
        if (threadIdx.x < HD128) {
            O_g[(q_head_idx * N_SPLIT + split_idx) * HD128 + threadIdx.x] = 0.0f;
        }
        return;
    }
    int chunk_end = chunk_begin + chunk_size;
    if (chunk_end > seq_len) chunk_end = seq_len;

    // SMEM layout (bytes):
    //   s_q        : 128 * 2                    = 256
    //   s_k (2-st) : 2 * 16 * 128 * 2           = 8192
    //   s_v (2-st) : 2 * 16 * 128 * 2           = 8192
    //   s_m        : 16 * 4                     = 64
    //   s_s        : 16 * 4                     = 64
    //   s_acc_bf16 : 16 * 128 * 2               = 4096   (bf16, not fp32)
    //   TOTAL                                    = 20864 B  (~20.4 KB)
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* s_q       = smem;
    __nv_bfloat16* s_k       = s_q + HD128;
    __nv_bfloat16* s_v       = s_k + (2 * BN128 * HD128);
    float*         s_m       = reinterpret_cast<float*>(s_v + (2 * BN128 * HD128));
    float*         s_s       = s_m + NUM_GROUPS128;
    __nv_bfloat16* s_acc_bf  = reinterpret_cast<__nv_bfloat16*>(s_s + NUM_GROUPS128);

    int tid  = threadIdx.x;
    int gid  = tid / THREADS_PER_KEY128;     // group (token-row) id
    int lane = tid % THREADS_PER_KEY128;     // dim-chunk id (each owns 8 dims)

    // Q load (16 threads do 16 * float4 = 128 bf16).
    if (tid < 16) {
        reinterpret_cast<float4*>(s_q)[tid] =
            reinterpret_cast<const float4*>(Q + q_head_idx * HD128)[tid];
    }

    // Per-thread online-softmax state.
    float row_max = -1e20f;
    float row_sum = 0.0f;
    float acc[8]  = {0.0f};

    auto get_smem_ptr = [](__nv_bfloat16* p) -> uint64_t {
        return (uint64_t)__cvta_generic_to_shared(p);
    };

    auto fetch_kv = [&](int token_base, int stage) {
        int token_idx = token_base + gid;
        if (token_idx < chunk_end) {
            const __nv_bfloat16* k_ptr = K + (token_idx * num_kv_heads + kv_head_idx) * HD128 + lane * 8;
            const __nv_bfloat16* v_ptr = V + (token_idx * num_kv_heads + kv_head_idx) * HD128 + lane * 8;
            __nv_bfloat16* sk_dst = &s_k[(stage * BN128 + gid) * HD128 + lane * 8];
            __nv_bfloat16* sv_dst = &s_v[(stage * BN128 + gid) * HD128 + lane * 8];
            CP_ASYNC_CG(get_smem_ptr(sk_dst), k_ptr, 16);
            CP_ASYNC_CG(get_smem_ptr(sv_dst), v_ptr, 16);
        }
    };

    // Prime the pipeline for the first micro-tile.
    fetch_kv(chunk_begin, 0);
    CP_ASYNC_COMMIT_GROUP();

    for (int i = chunk_begin; i < chunk_end; i += BN128) {
        int cur_stage  = ((i - chunk_begin) / BN128) % 2;
        int next_stage = 1 - cur_stage;

        if (i + BN128 < chunk_end) {
            fetch_kv(i + BN128, next_stage);
        }
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        int current_token_idx = i + gid;
        if (current_token_idx < chunk_end) {
            __nv_bfloat16* sk_local = &s_k[(cur_stage * BN128 + gid) * HD128];
            __nv_bfloat16* sv_local = &s_v[(cur_stage * BN128 + gid) * HD128];

            // Q · K (16 lanes * 8 dims = 128 bf16 dot-product).
            float4 q_vec = reinterpret_cast<float4*>(s_q)[lane];
            float4 k_vec = reinterpret_cast<float4*>(sk_local)[lane];
            __nv_bfloat162* q_p2 = reinterpret_cast<__nv_bfloat162*>(&q_vec);
            __nv_bfloat162* k_p2 = reinterpret_cast<__nv_bfloat162*>(&k_vec);

            float score = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                __nv_bfloat162 res = __hmul2(q_p2[j], k_p2[j]);
                score += __low2float(res) + __high2float(res);
            }
            // Intra-group reduction across the 16 lanes.
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                score += __shfl_xor_sync(0xffff, score, offset);
            }
            score *= sm_scale;

            // Online-softmax update.
            float old_max  = row_max;
            row_max        = fmaxf(row_max, score);
            float exp_scale = __expf(old_max - row_max);
            float p         = __expf(score - row_max);
            row_sum = row_sum * exp_scale + p;

            // Accumulate V.
            float4 v_vec = reinterpret_cast<float4*>(sv_local)[lane];
            __nv_bfloat16* v_p = reinterpret_cast<__nv_bfloat16*>(&v_vec);
            #pragma unroll
            for (int d = 0; d < 8; ++d) {
                acc[d] = acc[d] * exp_scale + p * __bfloat162float(v_p[d]);
            }
        }
    }

    // Intra-block merge: publish per-group (m, l, acc_bf16) to SMEM.
    if (lane == 0) {
        s_m[gid] = row_max;
        s_s[gid] = row_sum;
    }
    {
        float4 pack;
        __nv_bfloat16* pack_bf = reinterpret_cast<__nv_bfloat16*>(&pack);
        #pragma unroll
        for (int d = 0; d < 8; ++d) pack_bf[d] = __float2bfloat16(acc[d]);
        float4* s_acc_f4 = reinterpret_cast<float4*>(s_acc_bf + gid * HD128);
        s_acc_f4[lane] = pack;
    }
    __syncthreads();

    // First HD128 threads merge the 16 groups and publish the per-block partial.
    if (tid < HD128) {
        float block_max = -1e20f;
        #pragma unroll
        for (int g = 0; g < NUM_GROUPS128; ++g) {
            block_max = fmaxf(block_max, s_m[g]);
        }
        float block_sum = 0.0f;
        float block_acc = 0.0f;
        #pragma unroll
        for (int g = 0; g < NUM_GROUPS128; ++g) {
            float w = __expf(s_m[g] - block_max);
            block_sum += s_s[g] * w;
            block_acc += __bfloat162float(s_acc_bf[g * HD128 + tid]) * w;
        }
        // Publish UN-NORMALIZED partial O (block_acc, NOT block_acc / block_sum):
        // pass-2 needs the raw (m, l, o) tuple.
        int gmem_o = (q_head_idx * N_SPLIT + split_idx) * HD128 + tid;
        O_g[gmem_o] = block_acc;

        if (tid == 0) {
            int gmem_ml = q_head_idx * N_SPLIT + split_idx;
            M_g[gmem_ml] = block_max;
            L_g[gmem_ml] = block_sum;
        }
    }
}

// ============================================================================
// PASS 2 — merge N_SPLIT partials per q_head, normalise, write final bf16 O.
// grid = (num_q_heads,)   block = HD128
// ============================================================================
__global__ void flash_decode_bf16_hdim128_splitk_pass2_kernel(
    const float*         __restrict__ M_g,
    const float*         __restrict__ L_g,
    const float*         __restrict__ O_g,
    __nv_bfloat16*       __restrict__ O)
{
    int q_head_idx = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float s_m[N_SPLIT];
    __shared__ float s_l[N_SPLIT];
    if (tid < N_SPLIT) {
        s_m[tid] = M_g[q_head_idx * N_SPLIT + tid];
        s_l[tid] = L_g[q_head_idx * N_SPLIT + tid];
    }
    __syncthreads();

    float g_m = -1e30f;
    #pragma unroll
    for (int s = 0; s < N_SPLIT; ++s) g_m = fmaxf(g_m, s_m[s]);

    float g_l = 0.0f;
    float g_o = 0.0f;
    #pragma unroll
    for (int s = 0; s < N_SPLIT; ++s) {
        float w = __expf(s_m[s] - g_m);
        g_l += s_l[s] * w;
        g_o += O_g[(q_head_idx * N_SPLIT + s) * HD128 + tid] * w;
    }
    float out = (g_l > 0.0f) ? (g_o / g_l) : 0.0f;
    O[q_head_idx * HD128 + tid] = __float2bfloat16(out);
}

// ============================================================================
// Cached workspace for pass-1 -> pass-2 communication.
//
// Workspace size is O(num_q_heads * N_SPLIT * head_dim) and independent of
// seq_len, so we allocate once per unique (num_q_heads, head_dim) and reuse
// across calls.  Assumes all calls into this function within a process
// either share the same config or see it grow monotonically (true for
// RustInfer's per-model decode pipeline where the shape is fixed after init).
// ============================================================================
struct FlashDecodeSplitKWorkspace {
    float*  m = nullptr;
    float*  l = nullptr;
    float*  o = nullptr;
    int num_q_heads = 0;
    int head_dim    = 0;
};
static FlashDecodeSplitKWorkspace g_ws;

static void ensure_workspace(int num_q_heads, int head_dim, cudaStream_t stream) {
    if (g_ws.num_q_heads == num_q_heads && g_ws.head_dim == head_dim) return;

    if (g_ws.m) cudaFreeAsync(g_ws.m, stream);
    if (g_ws.l) cudaFreeAsync(g_ws.l, stream);
    if (g_ws.o) cudaFreeAsync(g_ws.o, stream);

    size_t ml_bytes = (size_t)num_q_heads * N_SPLIT           * sizeof(float);
    size_t o_bytes  = (size_t)num_q_heads * N_SPLIT * head_dim * sizeof(float);
    cudaMallocAsync(&g_ws.m, ml_bytes, stream);
    cudaMallocAsync(&g_ws.l, ml_bytes, stream);
    cudaMallocAsync(&g_ws.o, o_bytes,  stream);

    g_ws.num_q_heads = num_q_heads;
    g_ws.head_dim    = head_dim;
}

// ============================================================================
// Public FFI entry point — signature is preserved from the pre-optimisation
// revision of this file so that crate/op/kernels/cuda/flash_attn_gqa/mod.rs
// does NOT need to be re-linked.
// ============================================================================
extern "C"
void flash_decoding_cu_bf16_hdim128(
    const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr,
    const __nv_bfloat16* v_ptr,
    __nv_bfloat16* o_ptr,
    int32_t* kv_seq_len,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    cudaStream_t stream)
{
    ensure_workspace(num_q_heads, head_dim, stream);

    const int   group_size = num_q_heads / num_kv_heads;
    const float sm_scale   = 1.0f / sqrtf((float)head_dim);

    size_t smem_size = (HD128 * sizeof(__nv_bfloat16)) +
                       (2 * BN128 * HD128 * 2 * sizeof(__nv_bfloat16)) +
                       (NUM_GROUPS128 * 2 * sizeof(float)) +
                       (NUM_GROUPS128 * HD128 * sizeof(__nv_bfloat16));

    dim3 grid1(num_q_heads, N_SPLIT, 1);
    flash_decode_bf16_hdim128_splitk_pass1_kernel<<<grid1, 256, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr,
        g_ws.m, g_ws.l, g_ws.o,
        kv_seq_len,
        num_kv_heads, group_size, sm_scale);

    flash_decode_bf16_hdim128_splitk_pass2_kernel<<<num_q_heads, HD128, 0, stream>>>(
        g_ws.m, g_ws.l, g_ws.o, o_ptr);
}
