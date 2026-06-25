// flash_attn_paged_decode.cu
// -----------------------------------------------------------------------------
// Paged Flash-Decoding over a global KV pool.
//
// This follows the split-KV + LSE-combine structure used by FlashAttention
// decoding: pass1 computes a locally-normalized partial O plus LSE per KV split;
// pass2 combines split partials with exp(lse_split - global_lse).  K/V rows are
// gathered from the paged pool through block_tables.  The public ABI is still one
// host entry point per dtype; the caller supplies the existing decode workspace.
// -----------------------------------------------------------------------------

#include "flash_attn_gqa.h"
#include "flash_attn_paged_common.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cute/tensor.hpp>

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <type_traits>

namespace flash_paged_decode {

using namespace cute;

static constexpr int kMaxSplits    = 8;
static constexpr int kMinChunkSize = 32;
static constexpr int kBN           = 16;
static constexpr int kNumGroups    = 16;
static constexpr int kElemPerLane  = 8;

template <class E> struct ET;
template <> struct ET<__nv_bfloat16> {
    __device__ __forceinline__ static float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
    __device__ __forceinline__ static __nv_bfloat16 from_f(float x) { return __float2bfloat16_rn(x); }
};
template <> struct ET<__half> {
    __device__ __forceinline__ static float to_f(__half x) { return __half2float(x); }
    __device__ __forceinline__ static __half from_f(float x) { return __float2half_rn(x); }
};

#define PD_CP_ASYNC_CG(dst_smem, src_gmem, bytes)                              \
    asm volatile(                                                              \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" :: "l"(dst_smem), \
        "l"(src_gmem), "n"(bytes))
#define PD_CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define PD_CP_ASYNC_WAIT_GROUP(n)  asm volatile("cp.async.wait_group %0;\n" :: "n"(n))

__device__ __forceinline__ unsigned tpk_mask(int kTPK) {
    return (kTPK == 32) ? 0xffffffffu : ((1u << kTPK) - 1u);
}

__device__ __forceinline__ void token_to_page(
    int token,
    int block_size,
    int& block_idx,
    int& block_off)
{
    if (block_size == 16) {
        block_idx = token >> 4;
        block_off = token & 15;
    } else if (block_size == 32) {
        block_idx = token >> 5;
        block_off = token & 31;
    } else {
        block_idx = token / block_size;
        block_off = token - block_idx * block_size;
    }
}

template <class Elem, int HeadDim>
__global__ void paged_decode_pass1_kernel(
    const Elem* __restrict__ q_ptr,
    int64_t q_stride_b, int64_t q_stride_h,
    const Elem* __restrict__ k_pool,
    const Elem* __restrict__ v_pool,
    Elem* __restrict__ o_ptr,
    int64_t o_stride_b, int64_t o_stride_h,
    const uint32_t* __restrict__ block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* __restrict__ kv_lens,
    float* __restrict__ workspace_partial_o,
    float* __restrict__ workspace_partial_lse,
    int32_t* __restrict__ workspace_num_splits,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    int splits_cap)
{
    static_assert(HeadDim % kElemPerLane == 0, "HeadDim must be a multiple of 8");
    constexpr int kThreadsPerKey = HeadDim / kElemPerLane;
    constexpr int kBlockThreads = kNumGroups * kThreadsPerKey;

    const int qh = blockIdx.x;
    const int split = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;

    const int kv_len = kv_lens[b];
    if (kv_len <= 0) {
        if (split == 0 && tid < HeadDim) {
            Elem* o_row = o_ptr + static_cast<int64_t>(b) * o_stride_b + static_cast<int64_t>(qh) * o_stride_h;
            o_row[tid] = ET<Elem>::from_f(0.f);
            if (tid == 0) workspace_num_splits[b] = 0;
        }
        return;
    }

    // Adaptive split count: the host caps the number of KV splits by GPU
    // occupancy (batch*q_heads*splits ~ a few waves). At high batch the grid is
    // already full with splits_cap==1, so we take the single-split fast path and
    // avoid the FP32 partial-O global round-trip entirely. kMaxSplits still bounds
    // the workspace layout, so splits_cap is clamped into [1, kMaxSplits].
    int eff_splits = splits_cap;
    if (eff_splits < 1) eff_splits = 1;
    if (eff_splits > kMaxSplits) eff_splits = kMaxSplits;
    int chunk_size = (kv_len + eff_splits - 1) / eff_splits;
    if (chunk_size < kMinChunkSize) chunk_size = kMinChunkSize;
    chunk_size = (chunk_size + kBN - 1) & ~(kBN - 1);
    const int num_splits = (kv_len + chunk_size - 1) / chunk_size;
    if (split >= num_splits) return;
    if (split == 0 && tid == 0) workspace_num_splits[b] = num_splits;

    const int chunk_begin = split * chunk_size;
    const int chunk_end = min(chunk_begin + chunk_size, kv_len);
    const int group_size = num_q_heads / num_kv_heads;
    const int kv_head = qh / group_size;

    const Elem* q_bh = q_ptr + static_cast<int64_t>(b) * q_stride_b + static_cast<int64_t>(qh) * q_stride_h;
    Elem* o_bh = o_ptr + static_cast<int64_t>(b) * o_stride_b + static_cast<int64_t>(qh) * o_stride_h;
    const uint32_t* block_table = block_tables + static_cast<int64_t>(b) * max_blocks_per_seq;

    extern __shared__ unsigned char smem_raw[];
    Elem* s_q = reinterpret_cast<Elem*>(smem_raw);
    Elem* s_k = s_q + HeadDim;
    Elem* s_v = s_k + 2 * kBN * HeadDim;
    float* s_m = reinterpret_cast<float*>(s_v + 2 * kBN * HeadDim);
    float* s_s = s_m + kNumGroups;
    Elem* s_acc = reinterpret_cast<Elem*>(s_s + kNumGroups);

    const int gid = tid / kThreadsPerKey;
    const int lane = tid % kThreadsPerKey;

    if (tid < HeadDim / kElemPerLane) {
        reinterpret_cast<float4*>(s_q)[tid] = reinterpret_cast<const float4*>(q_bh)[tid];
    }

    float row_max = -INFINITY;
    float row_sum = 0.f;
    float acc[kElemPerLane];
    #pragma unroll
    for (int d = 0; d < kElemPerLane; ++d) acc[d] = 0.f;

    auto get_smem_ptr = [](void* p) -> uint64_t {
        return static_cast<uint64_t>(__cvta_generic_to_shared(p));
    };
    auto fetch_kv = [&](int token_base, int stage) {
        const int token_idx = token_base + gid;
        Elem* sk_dst = s_k + (stage * kBN + gid) * HeadDim + lane * kElemPerLane;
        Elem* sv_dst = s_v + (stage * kBN + gid) * HeadDim + lane * kElemPerLane;
        if (token_idx < chunk_end) {
            int logical_block, block_off;
            token_to_page(token_idx, block_size, logical_block, block_off);
            const uint32_t physical_block = block_table[logical_block];
            const Elem* k_row = paged_kv_row(k_pool, physical_block, block_off, block_size, num_kv_heads, kv_head, HeadDim);
            const Elem* v_row = paged_kv_row(v_pool, physical_block, block_off, block_size, num_kv_heads, kv_head, HeadDim);
            PD_CP_ASYNC_CG(get_smem_ptr(sk_dst), k_row + lane * kElemPerLane, 16);
            PD_CP_ASYNC_CG(get_smem_ptr(sv_dst), v_row + lane * kElemPerLane, 16);
        } else {
            reinterpret_cast<float4*>(sk_dst)[0] = make_float4(0.f, 0.f, 0.f, 0.f);
            reinterpret_cast<float4*>(sv_dst)[0] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };

    fetch_kv(chunk_begin, 0);
    PD_CP_ASYNC_COMMIT_GROUP();
    const unsigned shfl_mask = tpk_mask(kThreadsPerKey);

    for (int i = chunk_begin; i < chunk_end; i += kBN) {
        const int cur_stage = ((i - chunk_begin) / kBN) & 1;
        const int next_stage = cur_stage ^ 1;
        if (i + kBN < chunk_end) fetch_kv(i + kBN, next_stage);
        PD_CP_ASYNC_COMMIT_GROUP();
        PD_CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        const int t = i + gid;
        if (t < chunk_end) {
            float4 q_vec = reinterpret_cast<float4*>(s_q)[lane];
            float4 k_vec = reinterpret_cast<float4*>(s_k + (cur_stage * kBN + gid) * HeadDim)[lane];
            float s_qk = 0.f;
            if constexpr (std::is_same_v<Elem, __nv_bfloat16>) {
                __nv_bfloat162* q2 = reinterpret_cast<__nv_bfloat162*>(&q_vec);
                __nv_bfloat162* k2 = reinterpret_cast<__nv_bfloat162*>(&k_vec);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    __nv_bfloat162 r = __hmul2(q2[j], k2[j]);
                    s_qk += __low2float(r) + __high2float(r);
                }
            } else {
                __half2* q2 = reinterpret_cast<__half2*>(&q_vec);
                __half2* k2 = reinterpret_cast<__half2*>(&k_vec);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    __half2 r = __hmul2(q2[j], k2[j]);
                    s_qk += __low2float(r) + __high2float(r);
                }
            }
            #pragma unroll
            for (int off = kThreadsPerKey / 2; off > 0; off >>= 1) {
                s_qk += __shfl_xor_sync(shfl_mask, s_qk, off);
            }
            s_qk *= softmax_scale;

            const float old_max = row_max;
            row_max = fmaxf(row_max, s_qk);
            const float exp_scale = __expf(old_max - row_max);
            const float p = __expf(s_qk - row_max);
            row_sum = row_sum * exp_scale + p;

            float4 v_vec = reinterpret_cast<float4*>(s_v + (cur_stage * kBN + gid) * HeadDim)[lane];
            Elem* v_p = reinterpret_cast<Elem*>(&v_vec);
            #pragma unroll
            for (int d = 0; d < kElemPerLane; ++d) {
                acc[d] = acc[d] * exp_scale + p * ET<Elem>::to_f(v_p[d]);
            }
        }
    }

    if (lane == 0) {
        s_m[gid] = row_max;
        s_s[gid] = row_sum;
    }
    {
        float4 pack;
        Elem* pack_e = reinterpret_cast<Elem*>(&pack);
        #pragma unroll
        for (int d = 0; d < kElemPerLane; ++d) pack_e[d] = ET<Elem>::from_f(acc[d]);
        reinterpret_cast<float4*>(s_acc + gid * HeadDim)[lane] = pack;
    }
    __syncthreads();

    if (tid < HeadDim) {
        float block_max = -INFINITY;
        #pragma unroll
        for (int g = 0; g < kNumGroups; ++g) block_max = fmaxf(block_max, s_m[g]);

        float block_sum = 0.f;
        float block_acc = 0.f;
        #pragma unroll
        for (int g = 0; g < kNumGroups; ++g) {
            const float w = __expf(s_m[g] - block_max);
            block_sum += s_s[g] * w;
            block_acc += ET<Elem>::to_f(s_acc[g * HeadDim + tid]) * w;
        }
        const float inv_sum = block_sum > 0.f ? 1.f / block_sum : 0.f;

        if (num_splits == 1) {
            o_bh[tid] = ET<Elem>::from_f(block_acc * inv_sum);
        } else {
            auto partial_shape = make_shape(num_q_heads, Int<kMaxSplits>{}, Int<HeadDim>{});
            auto partial_stride = make_stride(kMaxSplits * HeadDim, HeadDim, _1{});
            Tensor partial_o = make_tensor(make_gmem_ptr(workspace_partial_o + static_cast<int64_t>(b) * num_q_heads * kMaxSplits * HeadDim),
                                           partial_shape, partial_stride);
            partial_o(qh, split, tid) = block_acc * inv_sum;
            if (tid == 0) {
                auto lse_shape = make_shape(num_q_heads, Int<kMaxSplits>{});
                auto lse_stride = make_stride(kMaxSplits, _1{});
                Tensor partial_lse = make_tensor(make_gmem_ptr(workspace_partial_lse + static_cast<int64_t>(b) * num_q_heads * kMaxSplits),
                                                 lse_shape, lse_stride);
                partial_lse(qh, split) = block_sum > 0.f ? (block_max + __logf(block_sum)) : -INFINITY;
            }
        }
    }
}

template <class Elem, int HeadDim>
__global__ void paged_decode_combine_kernel(
    Elem* __restrict__ o_ptr,
    int64_t o_stride_b, int64_t o_stride_h,
    const float* __restrict__ workspace_partial_o,
    const float* __restrict__ workspace_partial_lse,
    const int32_t* __restrict__ workspace_num_splits,
    int num_q_heads)
{
    const int qh = blockIdx.x;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int num_splits = workspace_num_splits[b];
    if (num_splits <= 1) return;

    auto lse_shape = make_shape(num_q_heads, Int<kMaxSplits>{});
    auto lse_stride = make_stride(kMaxSplits, _1{});
    Tensor partial_lse = make_tensor(make_gmem_ptr(const_cast<float*>(workspace_partial_lse + static_cast<int64_t>(b) * num_q_heads * kMaxSplits)),
                                     lse_shape, lse_stride);

    __shared__ float s_max;
    __shared__ float s_sum;
    if (tid == 0) {
        float m = -INFINITY;
        #pragma unroll
        for (int s = 0; s < kMaxSplits; ++s) {
            if (s >= num_splits) break;
            m = fmaxf(m, partial_lse(qh, s));
        }
        float l = 0.f;
        #pragma unroll
        for (int s = 0; s < kMaxSplits; ++s) {
            if (s >= num_splits) break;
            l += __expf(partial_lse(qh, s) - m);
        }
        s_max = m;
        s_sum = l;
    }
    __syncthreads();

    if (tid < HeadDim) {
        auto partial_shape = make_shape(num_q_heads, Int<kMaxSplits>{}, Int<HeadDim>{});
        auto partial_stride = make_stride(kMaxSplits * HeadDim, HeadDim, _1{});
        Tensor partial_o = make_tensor(make_gmem_ptr(const_cast<float*>(workspace_partial_o + static_cast<int64_t>(b) * num_q_heads * kMaxSplits * HeadDim)),
                                       partial_shape, partial_stride);
        float acc = 0.f;
        #pragma unroll
        for (int s = 0; s < kMaxSplits; ++s) {
            if (s >= num_splits) break;
            acc += __expf(partial_lse(qh, s) - s_max) * partial_o(qh, s, tid);
        }
        const float inv = s_sum > 0.f ? 1.f / s_sum : 0.f;
        Elem* o_row = o_ptr + static_cast<int64_t>(b) * o_stride_b + static_cast<int64_t>(qh) * o_stride_h;
        o_row[tid] = ET<Elem>::from_f(acc * inv);
    }
}

// Number of SMs on the active device, queried once. Used to pick the KV-split
// count by occupancy so we only split far enough to fill the GPU.
static int device_sm_count() {
    static int sm = 0;
    static std::once_flag once;
    std::call_once(once, [&]() {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) { sm = 132; return; }
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) { sm = 132; return; }
        sm = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 132;
    });
    return sm;
}

// Pick the KV-split cap: the minimum number of splits whose grid
// (batch*q_heads*splits) still covers ~2 waves of the GPU, clamped to
// [1, kMaxSplits]. At high batch this returns 1 (grid already full → no
// FP32 partial-O round-trip); at batch 1 it returns kMaxSplits.
static int compute_splits_cap(int batch, int num_q_heads) {
    const long target = 2L * device_sm_count();
    const long base = static_cast<long>(batch) * num_q_heads;
    if (base <= 0) return 1;
    long cap = (target + base - 1) / base;  // ceil(target / base)
    if (cap < 1) cap = 1;
    if (cap > kMaxSplits) cap = kMaxSplits;
    return static_cast<int>(cap);
}

template <class Elem, int HeadDim>
static cudaError_t launch_impl(
    const Elem* q, int64_t qsb, int64_t qsh,
    const Elem* k_pool,
    const Elem* v_pool,
    Elem* o, int64_t osb, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    float* workspace,
    int batch, int num_q_heads, int num_kv_heads,
    float softmax_scale,
    cudaStream_t stream)
{
    if (batch <= 0) return cudaSuccess;
    if (workspace == nullptr) return cudaErrorInvalidValue;

    float* partial_o = workspace;
    float* partial_lse = partial_o + static_cast<int64_t>(batch) * num_q_heads * kMaxSplits * HeadDim;
    int32_t* num_splits = reinterpret_cast<int32_t*>(partial_lse + static_cast<int64_t>(batch) * num_q_heads * kMaxSplits);

    constexpr int kThreadsPerKey = HeadDim / kElemPerLane;
    constexpr int kBlockThreads = kNumGroups * kThreadsPerKey;
    const int splits_cap = compute_splits_cap(batch, num_q_heads);
    dim3 grid1(num_q_heads, splits_cap, batch);
    dim3 block1(kBlockThreads);

    const size_t smem_size =
        static_cast<size_t>(HeadDim) * sizeof(Elem) +
        static_cast<size_t>(2) * kBN * HeadDim * sizeof(Elem) * 2 +
        static_cast<size_t>(kNumGroups) * 2 * sizeof(float) +
        static_cast<size_t>(kNumGroups) * HeadDim * sizeof(Elem);

    auto pass1 = paged_decode_pass1_kernel<Elem, HeadDim>;
    static std::once_flag pass1_attr_once;
    static cudaError_t pass1_attr_err = cudaSuccess;
    std::call_once(pass1_attr_once, [&]() {
        pass1_attr_err = cudaFuncSetAttribute(pass1, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size));
    });
    if (pass1_attr_err != cudaSuccess) return pass1_attr_err;

    pass1<<<grid1, block1, smem_size, stream>>>(
        q, qsb, qsh, k_pool, v_pool, o, osb, osh,
        block_tables, max_blocks_per_seq, block_size, kv_lens,
        partial_o, partial_lse, num_splits,
        num_q_heads, num_kv_heads, softmax_scale, splits_cap);

    // splits_cap==1 forces num_splits==1 for every sequence, so pass1 writes the
    // final O directly and the combine pass is a no-op — skip its launch.
    if (splits_cap > 1) {
        dim3 grid2(num_q_heads, 1, batch);
        dim3 block2(HeadDim);
        paged_decode_combine_kernel<Elem, HeadDim><<<grid2, block2, 0, stream>>>(
            o, osb, osh, partial_o, partial_lse, num_splits, num_q_heads);
    }

    return cudaGetLastError();
}

template <class Elem>
static cudaError_t launch_dispatch(
    const Elem* q, int64_t qsb, int64_t qsh,
    const Elem* k_pool,
    const Elem* v_pool,
    Elem* o, int64_t osb, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    float* workspace,
    int batch,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale,
    cudaStream_t stream)
{
    switch (head_dim) {
    case 64:  return launch_impl<Elem,  64>(q,qsb,qsh,k_pool,v_pool,o,osb,osh,block_tables,max_blocks_per_seq,block_size,kv_lens,workspace,batch,num_q_heads,num_kv_heads,softmax_scale,stream);
    case 128: return launch_impl<Elem, 128>(q,qsb,qsh,k_pool,v_pool,o,osb,osh,block_tables,max_blocks_per_seq,block_size,kv_lens,workspace,batch,num_q_heads,num_kv_heads,softmax_scale,stream);
    case 192: return launch_impl<Elem, 192>(q,qsb,qsh,k_pool,v_pool,o,osb,osh,block_tables,max_blocks_per_seq,block_size,kv_lens,workspace,batch,num_q_heads,num_kv_heads,softmax_scale,stream);
    case 256: return launch_impl<Elem, 256>(q,qsb,qsh,k_pool,v_pool,o,osb,osh,block_tables,max_blocks_per_seq,block_size,kv_lens,workspace,batch,num_q_heads,num_kv_heads,softmax_scale,stream);
    default:
        fprintf(stderr, "[flash_attn_paged_decode] unsupported head_dim=%d (supported: 64, 128, 192, 256)\n", head_dim);
        return cudaErrorInvalidValue;
    }
}

}  // namespace flash_paged_decode

extern "C" void launch_flash_attn_paged_decode_bf16(
    const __nv_bfloat16* q, int64_t qsb, int64_t qsh,
    const __nv_bfloat16* k_pool,
    const __nv_bfloat16* v_pool,
          __nv_bfloat16* o, int64_t osb, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    float* workspace,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream)
{
    cudaError_t err = flash_paged_decode::launch_dispatch<__nv_bfloat16>(
        q, qsb, qsh, k_pool, v_pool, o, osb, osh,
        block_tables, max_blocks_per_seq, block_size, kv_lens, workspace,
        batch, num_q_heads, num_kv_heads, head_dim, softmax_scale, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[flash_attn_paged_decode_bf16] launch error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_flash_attn_paged_decode_fp16(
    const __half* q, int64_t qsb, int64_t qsh,
    const __half* k_pool,
    const __half* v_pool,
          __half* o, int64_t osb, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    float* workspace,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream)
{
    cudaError_t err = flash_paged_decode::launch_dispatch<__half>(
        q, qsb, qsh, k_pool, v_pool, o, osb, osh,
        block_tables, max_blocks_per_seq, block_size, kv_lens, workspace,
        batch, num_q_heads, num_kv_heads, head_dim, softmax_scale, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[flash_attn_paged_decode_fp16] launch error: %s\n", cudaGetErrorString(err));
    }
}
