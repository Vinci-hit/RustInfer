// ============================================================================
// Flash Decoding, head_dim = 128 — split-K across KV sequence
//
// 泛型化实现：一份模板代码同时实例化 BF16 / FP16 两个 extern "C" 入口。
// FP16 版本按 BF16 的 split-K 算法走，废弃老的非 split-K 实现。
//
// 优化要点（来自 cuda-optimized-skill run_main v2 winner）：
//   1. Split-K: grid = (num_q_heads, N_SPLIT)。每 block 仅对 KV chunk 做
//      online softmax，写部分 (m, l, O) 到调用方提供的 workspace；再由一个
//      很小的 pass-2 kernel 做 log-sum-exp 合并。
//   2. 块内 group-merge scratch 用低精度 (bf16/half) 存储，降 per-block SMEM
//      从 24.96 KB 到 20.8 KB，Block Limit SMEM 在 Ampere 上 3 → 4，
//      theoretical occupancy 50% → 67%。
//
// 在 NVIDIA A10 (sm_86)、head_dim=128、num_q_heads=32、num_kv_heads=8、
// seq_len=2048 (decode, q_seq_len=1) 上实测（BF16）：
//   baseline (pre-split):   0.0589 ms
//   v1 split-K N_SPLIT=8:   0.0303 ms  (1.94x)
//   v2 + bf16 s_acc:        0.0255 ms  (2.31x)  ← 本文件
//
// Workspace 所有权：
//   pass-1 → pass-2 的 (m, l, O) scratch 由 Rust 侧 `CudaConfig` 预分配并通过
//   `workspace` 入参传入（详见 `cuda/config.rs::ensure_flash_decode_workspace`）。
//   Kernel 内部无任何 static / cudaMalloc，保证完整 CUDA Graph 兼容。
//
//   Workspace 内存布局（单个 fp32 平板）：
//     [0,                              num_q_heads * N_SPLIT)                 = M
//     [num_q_heads * N_SPLIT,        2*num_q_heads * N_SPLIT)                 = L
//     [2*num_q_heads * N_SPLIT,      (2 + head_dim) * num_q_heads * N_SPLIT)  = O
//   总 fp32 数 = num_q_heads * N_SPLIT * (2 + head_dim)
//   调用方保证 `workspace` 至少这么大。
// ============================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>
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

#define HD128              128
#define BN128              16     // tokens per KV micro-tile
#define THREADS_PER_KEY128 16     // lanes per group (each lane owns 8 dims)
#define NUM_GROUPS128      16     // 256 / 16
#define N_SPLIT            8      // KV-axis partitions per q_head
                                  // 与 cuda/config.rs::FLASH_DECODE_N_SPLIT 同步！

// ----------------------------------------------------------------------------
// Element traits: 把元素类型 → half2 类型 + 标量 float 互转 封装为 trait。
// 匿名命名空间避免与 flash_decoding_hdim64.cu 的同名 trait 发生 ODR 冲突。
// ----------------------------------------------------------------------------
namespace {
template <class Elem> struct DecodeTraits128;

template <> struct DecodeTraits128<__nv_bfloat16> {
    using Elem   = __nv_bfloat16;
    using Elem2  = __nv_bfloat162;
    static __device__ __forceinline__ Elem2 mul2(Elem2 a, Elem2 b) { return __hmul2(a, b); }
    static __device__ __forceinline__ float low(Elem2 x)  { return __low2float(x);  }
    static __device__ __forceinline__ float high(Elem2 x) { return __high2float(x); }
    static __device__ __forceinline__ float to_float(Elem x) { return __bfloat162float(x); }
    static __device__ __forceinline__ Elem  from_float(float x) { return __float2bfloat16(x); }
};

template <> struct DecodeTraits128<__half> {
    using Elem   = __half;
    using Elem2  = __half2;
    static __device__ __forceinline__ Elem2 mul2(Elem2 a, Elem2 b) { return __hmul2(a, b); }
    static __device__ __forceinline__ float low(Elem2 x)  { return __low2float(x);  }
    static __device__ __forceinline__ float high(Elem2 x) { return __high2float(x); }
    static __device__ __forceinline__ float to_float(Elem x) { return __half2float(x); }
    static __device__ __forceinline__ Elem  from_float(float x) { return __float2half(x); }
};
}

// ============================================================================
// PASS 1 — per-chunk online softmax
// ============================================================================
template <class Elem>
__global__ void flash_decode_hdim128_splitk_pass1_kernel(
    const Elem* __restrict__ Q,
    const Elem* __restrict__ K,
    const Elem* __restrict__ V,
    float*      __restrict__ M_g,
    float*      __restrict__ L_g,
    float*      __restrict__ O_g,
    int32_t*    seq_len_ptr,
    int num_kv_heads,
    int group_size,
    float sm_scale)
{
    using Traits = DecodeTraits128<Elem>;
    using Elem2  = typename Traits::Elem2;

    int32_t seq_len = *seq_len_ptr + 1;
    int q_head_idx  = blockIdx.x;
    int split_idx   = blockIdx.y;
    int kv_head_idx = q_head_idx / group_size;

    // chunk_size 设备端动态计算，向上对齐到 BN128 保证 cp.async 对齐
    int per_split   = (seq_len + N_SPLIT - 1) / N_SPLIT;
    int chunk_size  = ((per_split + BN128 - 1) / BN128) * BN128;

    int chunk_begin = split_idx * chunk_size;
    if (chunk_begin >= seq_len) {
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
    //   s_acc_lp   : 16 * 128 * 2               = 4096   (低精度, not fp32)
    //   TOTAL                                    = 20864 B  (~20.4 KB)
    extern __shared__ __align__(16) unsigned char smem_bytes_128[];
    Elem*  s_q       = reinterpret_cast<Elem*>(smem_bytes_128);
    Elem*  s_k       = s_q + HD128;
    Elem*  s_v       = s_k + (2 * BN128 * HD128);
    float* s_m       = reinterpret_cast<float*>(s_v + (2 * BN128 * HD128));
    float* s_s       = s_m + NUM_GROUPS128;
    Elem*  s_acc_lp  = reinterpret_cast<Elem*>(s_s + NUM_GROUPS128);

    int tid  = threadIdx.x;
    int gid  = tid / THREADS_PER_KEY128;
    int lane = tid % THREADS_PER_KEY128;

    if (tid < 16) {
        reinterpret_cast<float4*>(s_q)[tid] =
            reinterpret_cast<const float4*>(Q + q_head_idx * HD128)[tid];
    }

    float row_max = -1e20f;
    float row_sum = 0.0f;
    float acc[8]  = {0.0f};

    auto get_smem_ptr = [](Elem* p) -> uint64_t {
        return (uint64_t)__cvta_generic_to_shared(p);
    };

    auto fetch_kv = [&](int token_base, int stage) {
        int token_idx = token_base + gid;
        if (token_idx < chunk_end) {
            const Elem* k_ptr = K + (token_idx * num_kv_heads + kv_head_idx) * HD128 + lane * 8;
            const Elem* v_ptr = V + (token_idx * num_kv_heads + kv_head_idx) * HD128 + lane * 8;
            Elem* sk_dst = &s_k[(stage * BN128 + gid) * HD128 + lane * 8];
            Elem* sv_dst = &s_v[(stage * BN128 + gid) * HD128 + lane * 8];
            CP_ASYNC_CG(get_smem_ptr(sk_dst), k_ptr, 16);
            CP_ASYNC_CG(get_smem_ptr(sv_dst), v_ptr, 16);
        }
    };

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
            Elem* sk_local = &s_k[(cur_stage * BN128 + gid) * HD128];
            Elem* sv_local = &s_v[(cur_stage * BN128 + gid) * HD128];

            float4 q_vec = reinterpret_cast<float4*>(s_q)[lane];
            float4 k_vec = reinterpret_cast<float4*>(sk_local)[lane];
            Elem2* q_p2 = reinterpret_cast<Elem2*>(&q_vec);
            Elem2* k_p2 = reinterpret_cast<Elem2*>(&k_vec);

            float score = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                Elem2 res = Traits::mul2(q_p2[j], k_p2[j]);
                score += Traits::low(res) + Traits::high(res);
            }
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                score += __shfl_xor_sync(0xffff, score, offset);
            }
            score *= sm_scale;

            float old_max  = row_max;
            row_max        = fmaxf(row_max, score);
            float exp_scale = __expf(old_max - row_max);
            float p         = __expf(score - row_max);
            row_sum = row_sum * exp_scale + p;

            float4 v_vec = reinterpret_cast<float4*>(sv_local)[lane];
            Elem* v_p = reinterpret_cast<Elem*>(&v_vec);
            #pragma unroll
            for (int d = 0; d < 8; ++d) {
                acc[d] = acc[d] * exp_scale + p * Traits::to_float(v_p[d]);
            }
        }
    }

    if (lane == 0) {
        s_m[gid] = row_max;
        s_s[gid] = row_sum;
    }
    {
        float4 pack;
        Elem* pack_lp = reinterpret_cast<Elem*>(&pack);
        #pragma unroll
        for (int d = 0; d < 8; ++d) pack_lp[d] = Traits::from_float(acc[d]);
        float4* s_acc_f4 = reinterpret_cast<float4*>(s_acc_lp + gid * HD128);
        s_acc_f4[lane] = pack;
    }
    __syncthreads();

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
            block_acc += Traits::to_float(s_acc_lp[g * HD128 + tid]) * w;
        }
        // UN-normalised partial O (pass-2 does the final divide).
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
// PASS 2 — merge N_SPLIT partials per q_head, normalise, write low-precision O.
// ============================================================================
template <class Elem>
__global__ void flash_decode_hdim128_splitk_pass2_kernel(
    const float* __restrict__ M_g,
    const float* __restrict__ L_g,
    const float* __restrict__ O_g,
    Elem*        __restrict__ O)
{
    using Traits = DecodeTraits128<Elem>;

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
    O[q_head_idx * HD128 + tid] = Traits::from_float(out);
}

// ============================================================================
// 模板化 launch 函数
// ============================================================================
template <class Elem>
static void launch_flash_decoding_hdim128_impl(
    const Elem* q_ptr,
    const Elem* k_ptr,
    const Elem* v_ptr,
    Elem*       o_ptr,
    float*      workspace,
    int32_t*    kv_seq_len,
    int32_t     num_q_heads,
    int32_t     num_kv_heads,
    int32_t     head_dim,
    cudaStream_t stream)
{
    const int   group_size = num_q_heads / num_kv_heads;
    const float sm_scale   = 1.0f / sqrtf((float)head_dim);

    // Workspace slicing: [M | L | O]
    //   M 长度 = num_q_heads * N_SPLIT
    //   L 长度 = num_q_heads * N_SPLIT
    //   O 长度 = num_q_heads * N_SPLIT * head_dim
    const size_t ml_stride = (size_t)num_q_heads * N_SPLIT;
    float* M_g = workspace;
    float* L_g = workspace + ml_stride;
    float* O_g = workspace + 2 * ml_stride;

    size_t smem_size = (HD128 * sizeof(Elem)) +
                       (2 * BN128 * HD128 * 2 * sizeof(Elem)) +
                       (NUM_GROUPS128 * 2 * sizeof(float)) +
                       (NUM_GROUPS128 * HD128 * sizeof(Elem));

    dim3 grid1(num_q_heads, N_SPLIT, 1);
    flash_decode_hdim128_splitk_pass1_kernel<Elem><<<grid1, 256, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr,
        M_g, L_g, O_g,
        kv_seq_len,
        num_kv_heads, group_size, sm_scale);

    flash_decode_hdim128_splitk_pass2_kernel<Elem><<<num_q_heads, HD128, 0, stream>>>(
        M_g, L_g, O_g, o_ptr);
}

// ============================================================================
// 公共 FFI 入口
// ============================================================================
extern "C"
void flash_decoding_cu_bf16_hdim128(
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
    launch_flash_decoding_hdim128_impl<__nv_bfloat16>(
        q_ptr, k_ptr, v_ptr, o_ptr, workspace, kv_seq_len,
        num_q_heads, num_kv_heads, head_dim, stream);
}

extern "C"
void flash_decoding_cu_fp16_hdim128(
    const __half* q_ptr,
    const __half* k_ptr,
    const __half* v_ptr,
    __half* o_ptr,
    float* workspace,
    int32_t* kv_seq_len,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    cudaStream_t stream)
{
    launch_flash_decoding_hdim128_impl<__half>(
        q_ptr, k_ptr, v_ptr, o_ptr, workspace, kv_seq_len,
        num_q_heads, num_kv_heads, head_dim, stream);
}
