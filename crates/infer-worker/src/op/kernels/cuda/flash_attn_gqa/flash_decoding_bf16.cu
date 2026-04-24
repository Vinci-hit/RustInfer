// ============================================================================
// Flash Decoding BF16, head_dim = 64 — split-K across KV sequence
//
// 从 hdim=128 版本移植，优化要点一致：
//   1. Split-K: grid = (num_q_heads, N_SPLIT_1B)
//   2. 块内 group-merge 用 bf16 scratch
//
// hdim=64 的结构差异：
//   - HD1B = 64, THREADS_PER_KEY = 8, block = 128 threads (= 16 groups × 8 lanes)
//   - intra-group reduction mask 0xff，offset 从 4 开始
//   - Q 加载仅前 8 个线程做 8 × float4 = 64 bf16
//
// 在 NVIDIA A10 (sm_86)、head_dim=64、num_q_heads=32、num_kv_heads=8、
// seq_len=2048 (decode, q_seq_len=1) 上实测：
//   baseline: 0.0419 ms
//   本版本:   0.0149 ms  (2.81x)
//
// Workspace 所有权：见 flash_decoding_bf16_hdim128.cu 的文件头注释。
// ============================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>
#include "flash_attn_gqa.h"

// ----------------------------------------------------------------------------
// PTX cp.async macros
// ----------------------------------------------------------------------------
#define CP_ASYNC_CG(dst, src, bytes)                                           \
    asm volatile(                                                              \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(dst),     \
        "l"(src), "n"(bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

#define HD1B             64
#define BN_1B            16
#define THREADS_PER_KEY  8
#define NUM_GROUPS_1B    16
#define N_SPLIT_1B       8       // 与 cuda/config.rs::FLASH_DECODE_N_SPLIT 同步！

// ============================================================================
// PASS 1 — per-chunk online softmax
// ============================================================================
__global__ void flash_decode_bf16_hdim64_splitk_pass1_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    float*               __restrict__ M_g,
    float*               __restrict__ L_g,
    float*               __restrict__ O_g,
    int32_t*             seq_len_ptr,
    int num_kv_heads,
    int group_size,
    float sm_scale)
{
    int32_t seq_len = *seq_len_ptr + 1;
    int q_head_idx  = blockIdx.x;
    int split_idx   = blockIdx.y;
    int kv_head_idx = q_head_idx / group_size;

    int per_split  = (seq_len + N_SPLIT_1B - 1) / N_SPLIT_1B;
    int chunk_size = ((per_split + BN_1B - 1) / BN_1B) * BN_1B;

    int chunk_begin = split_idx * chunk_size;
    if (chunk_begin >= seq_len) {
        if (threadIdx.x == 0) {
            int mo = q_head_idx * N_SPLIT_1B + split_idx;
            M_g[mo] = -1e30f;
            L_g[mo] = 0.0f;
        }
        if (threadIdx.x < HD1B) {
            O_g[(q_head_idx * N_SPLIT_1B + split_idx) * HD1B + threadIdx.x] = 0.0f;
        }
        return;
    }
    int chunk_end = chunk_begin + chunk_size;
    if (chunk_end > seq_len) chunk_end = seq_len;

    // SMEM layout (bytes):
    //   s_q        : 64 * 2                    = 128
    //   s_k (2-st) : 2 * 16 * 64 * 2           = 4096
    //   s_v (2-st) : 2 * 16 * 64 * 2           = 4096
    //   s_m        : 16 * 4                    = 64
    //   s_s        : 16 * 4                    = 64
    //   s_acc_bf16 : 16 * 64 * 2               = 2048   (bf16, not fp32)
    //   TOTAL                                   = 10496 B  (~10.3 KB)
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* s_q      = smem;
    __nv_bfloat16* s_k      = s_q + HD1B;
    __nv_bfloat16* s_v      = s_k + (2 * BN_1B * HD1B);
    float*         s_m      = reinterpret_cast<float*>(s_v + (2 * BN_1B * HD1B));
    float*         s_s      = s_m + NUM_GROUPS_1B;
    __nv_bfloat16* s_acc_bf = reinterpret_cast<__nv_bfloat16*>(s_s + NUM_GROUPS_1B);

    int tid  = threadIdx.x;
    int gid  = tid / THREADS_PER_KEY;
    int lane = tid % THREADS_PER_KEY;

    if (tid < 8) {
        reinterpret_cast<float4*>(s_q)[tid] =
            reinterpret_cast<const float4*>(Q + q_head_idx * HD1B)[tid];
    }

    float row_max = -1e20f;
    float row_sum = 0.0f;
    float acc[8]  = {0.0f};

    auto get_smem_ptr = [](__nv_bfloat16* p) -> uint64_t {
        return (uint64_t)__cvta_generic_to_shared(p);
    };
    auto fetch_kv = [&](int token_base, int stage) {
        int token_idx = token_base + gid;
        if (token_idx < chunk_end) {
            const __nv_bfloat16* k_ptr = K + (token_idx * num_kv_heads + kv_head_idx) * HD1B + lane * 8;
            const __nv_bfloat16* v_ptr = V + (token_idx * num_kv_heads + kv_head_idx) * HD1B + lane * 8;
            __nv_bfloat16* sk_dst = &s_k[(stage * BN_1B + gid) * HD1B + lane * 8];
            __nv_bfloat16* sv_dst = &s_v[(stage * BN_1B + gid) * HD1B + lane * 8];
            CP_ASYNC_CG(get_smem_ptr(sk_dst), k_ptr, 16);
            CP_ASYNC_CG(get_smem_ptr(sv_dst), v_ptr, 16);
        }
    };

    fetch_kv(chunk_begin, 0);
    CP_ASYNC_COMMIT_GROUP();

    for (int i = chunk_begin; i < chunk_end; i += BN_1B) {
        int cur_stage  = ((i - chunk_begin) / BN_1B) % 2;
        int next_stage = 1 - cur_stage;
        if (i + BN_1B < chunk_end) fetch_kv(i + BN_1B, next_stage);
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        int current_token_idx = i + gid;
        if (current_token_idx < chunk_end) {
            __nv_bfloat16* sk_local = &s_k[(cur_stage * BN_1B + gid) * HD1B];
            __nv_bfloat16* sv_local = &s_v[(cur_stage * BN_1B + gid) * HD1B];

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
            #pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1) {
                score += __shfl_xor_sync(0xff, score, offset);
            }
            score *= sm_scale;

            float old_max  = row_max;
            row_max        = fmaxf(row_max, score);
            float exp_scale = __expf(old_max - row_max);
            float p         = __expf(score - row_max);
            row_sum = row_sum * exp_scale + p;

            float4 v_vec = reinterpret_cast<float4*>(sv_local)[lane];
            __nv_bfloat16* v_p = reinterpret_cast<__nv_bfloat16*>(&v_vec);
            #pragma unroll
            for (int d = 0; d < 8; ++d) {
                acc[d] = acc[d] * exp_scale + p * __bfloat162float(v_p[d]);
            }
        }
    }

    if (lane == 0) { s_m[gid] = row_max; s_s[gid] = row_sum; }
    {
        float4 pack;
        __nv_bfloat16* pack_bf = reinterpret_cast<__nv_bfloat16*>(&pack);
        #pragma unroll
        for (int d = 0; d < 8; ++d) pack_bf[d] = __float2bfloat16(acc[d]);
        float4* s_acc_f4 = reinterpret_cast<float4*>(s_acc_bf + gid * HD1B);
        s_acc_f4[lane] = pack;
    }
    __syncthreads();

    if (tid < HD1B) {
        float block_max = -1e20f;
        #pragma unroll
        for (int g = 0; g < NUM_GROUPS_1B; ++g) block_max = fmaxf(block_max, s_m[g]);

        float block_sum = 0.0f;
        float block_acc = 0.0f;
        #pragma unroll
        for (int g = 0; g < NUM_GROUPS_1B; ++g) {
            float w = __expf(s_m[g] - block_max);
            block_sum += s_s[g] * w;
            block_acc += __bfloat162float(s_acc_bf[g * HD1B + tid]) * w;
        }
        int gmem_o = (q_head_idx * N_SPLIT_1B + split_idx) * HD1B + tid;
        O_g[gmem_o] = block_acc;
        if (tid == 0) {
            int gmem_ml = q_head_idx * N_SPLIT_1B + split_idx;
            M_g[gmem_ml] = block_max;
            L_g[gmem_ml] = block_sum;
        }
    }
}

// ============================================================================
// PASS 2 — merge N_SPLIT_1B partials per q_head.
// ============================================================================
__global__ void flash_decode_bf16_hdim64_splitk_pass2_kernel(
    const float*         __restrict__ M_g,
    const float*         __restrict__ L_g,
    const float*         __restrict__ O_g,
    __nv_bfloat16*       __restrict__ O)
{
    int q_head_idx = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float s_m[N_SPLIT_1B];
    __shared__ float s_l[N_SPLIT_1B];
    if (tid < N_SPLIT_1B) {
        s_m[tid] = M_g[q_head_idx * N_SPLIT_1B + tid];
        s_l[tid] = L_g[q_head_idx * N_SPLIT_1B + tid];
    }
    __syncthreads();

    float g_m = -1e30f;
    #pragma unroll
    for (int s = 0; s < N_SPLIT_1B; ++s) g_m = fmaxf(g_m, s_m[s]);

    float g_l = 0.0f, g_o = 0.0f;
    #pragma unroll
    for (int s = 0; s < N_SPLIT_1B; ++s) {
        float w = __expf(s_m[s] - g_m);
        g_l += s_l[s] * w;
        g_o += O_g[(q_head_idx * N_SPLIT_1B + s) * HD1B + tid] * w;
    }
    float out = (g_l > 0.0f) ? (g_o / g_l) : 0.0f;
    O[q_head_idx * HD1B + tid] = __float2bfloat16(out);
}

// ============================================================================
// 公共 FFI 入口 —— workspace 由调用方传入（见 flash_decoding_bf16_hdim128.cu 注释）
// ============================================================================
void flash_decoding_cu_bf16(
    const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr,
    const __nv_bfloat16* v_ptr,
    __nv_bfloat16* o_ptr,
    float* workspace,
    int32_t* kv_seq_len,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    cudaStream_t stream)
{
    const int   group_size = num_q_heads / num_kv_heads;
    const float sm_scale   = 1.0f / sqrtf((float)head_dim);

    const size_t ml_stride = (size_t)num_q_heads * N_SPLIT_1B;
    float* M_g = workspace;
    float* L_g = workspace + ml_stride;
    float* O_g = workspace + 2 * ml_stride;

    size_t smem_size = (HD1B * sizeof(__nv_bfloat16)) +
                       (2 * BN_1B * HD1B * 2 * sizeof(__nv_bfloat16)) +
                       (NUM_GROUPS_1B * 2 * sizeof(float)) +
                       (NUM_GROUPS_1B * HD1B * sizeof(__nv_bfloat16));

    dim3 grid1(num_q_heads, N_SPLIT_1B, 1);
    flash_decode_bf16_hdim64_splitk_pass1_kernel<<<grid1, 128, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr,
        M_g, L_g, O_g,
        kv_seq_len, num_kv_heads, group_size, sm_scale);

    flash_decode_bf16_hdim64_splitk_pass2_kernel<<<num_q_heads, HD1B, 0, stream>>>(
        M_g, L_g, O_g, o_ptr);
}
