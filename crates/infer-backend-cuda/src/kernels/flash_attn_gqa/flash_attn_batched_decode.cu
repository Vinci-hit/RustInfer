// flash_attn_batched_decode.cu
// -----------------------------------------------------------------------------
// Batched Flash-Decoding: q_len = 1, arbitrary batch with independent KV caches.
//
// Design goals
//   * One kernel launch handles B requests, each with its own KV cache buffer.
//   * Split-KV along the sequence dim so long contexts can saturate SMs.
//   * Per-request dynamic `num_splits` inside the kernel — grid shape stays
//     fixed at (num_q_heads, MaxSplits, B) so CUDA Graph replay is safe.
//   * Single-split requests write directly to the final output (no reduction).
//   * Graph-friendly: caller provides fixed device pointer arrays for K/V and
//     a `req_to_slot` remap; nothing is allocated/freed per step.
//
// Pipeline
//   Pass1 (attention per chunk):
//       grid = (num_q_heads, MaxSplits, batch),  block = 128 threads
//       each block computes attention for one (req, q_head, kv_chunk)
//       if it's the only chunk for that request: write directly to O
//       otherwise: write to workspace[req, q_head, split, :] + lse
//   Pass2 (LSE reduction):
//       grid = (num_q_heads, batch),  block = HeadDim threads
//       each block merges `num_splits[req]` partials into final O
//       if only one split existed: bypass (Pass1 already wrote O)
//
// Workspace layout (float):
//       partial_o  : [B, Hq, MaxSplits, HeadDim]
//       partial_lse: [B, Hq, MaxSplits]
//       num_splits : [B]      (int32 packed as f32)
//   Total size = B * Hq * MaxSplits * (HeadDim + 1) * 4 bytes + B * 4 bytes
// -----------------------------------------------------------------------------

#include "flash_attn_gqa.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <mutex>

namespace flash_batched_decode {

// MaxSplits governs the maximum number of KV chunks a single request can
// be split into.  The actual `num_splits` is computed from kv_len at runtime.
// 16 splits × 128-tokens-per-chunk covers kv up to 2048 with chunk=128; longer
// contexts automatically use a larger chunk size (chunk = ceil(kv/MaxSplits)).
static constexpr int kMaxSplits       = 8;    // 历史 e1d242c 版用 8，512 → 256 减少空 block
static constexpr int kMinChunkSize    = 16;   // 旧版 BN_1B=16，让短 kv (~128) 也能 8-split 并行

// Number of threads per CTA in Pass 1; also used for HeadDim-wise work split.
// Picked so that HeadDim / Threads is an integer vector size ≥ 4 bytes.
static constexpr int kPass1Threads = 128;

// Pass 2 uses exactly HeadDim threads (each thread handles one output element).
// Since HeadDim ∈ {64, 128, 192, 256}, this fits in 1-2 warps cleanly.

// ---- element traits ---------------------------------------------------------
template <class E> struct ET;
template <> struct ET<__nv_bfloat16> {
    using h = __nv_bfloat16;
    __device__ __forceinline__ static float to_f(h x) { return __bfloat162float(x); }
    __device__ __forceinline__ static h from_f(float x) { return __float2bfloat16_rn(x); }
};
template <> struct ET<__half> {
    using h = __half;
    __device__ __forceinline__ static float to_f(h x) { return __half2float(x); }
    __device__ __forceinline__ static h from_f(float x) { return __float2half_rn(x); }
};

// ============================================================================
// Pass 1 kernel  ——  cp.async double-buffered, BF16 hmul2 score, bf16 scratch
//                   accumulator (移植自 e1d242c hdim64/128 split-K 实现，扩展为
//                   batched: 通过 req_to_slot[b] 解 KV slot，per-request 动态
//                   num_splits 与 chunk_size 与原 batched 接口保持一致)
// ============================================================================
//
// 设计要点：
//   * block 形状 = (kNumGroups=16) × (kThreadsPerKey = HeadDim/8) 个线程
//                  hd64→128, hd128→256, hd192→384, hd256→512
//   * 每 lane 持有 8 个 bf16 = 4 个 bf16x2 = 16B（匹配 cp.async / float4 一次一行）
//   * 每 group 处理一个 KV token：lane-mask (1<<kThreadsPerKey)-1 内 shfl 求 score
//   * online softmax 与 V 累加 fuse 在同一循环；acc[8] 全模板长 = HeadDim/kThreadsPerKey
//   * cp.async 双 stage 同时拉 K 与 V；每 16 token 一个 commit_group
//   * group-merge：tid<HeadDim 的线程 reduce 16 个 group 的 (m, l, V_acc)，写出
//     locally-normalised partial_o + lse（与现有 pass2 兼容）；num_splits==1 时
//     直通最终 O。
//
// PTX cp.async helpers (sm_80+)
// -----------------------------------------------------------------------------
#define BD_CP_ASYNC_CG(dst_smem, src_gmem, bytes)                              \
    asm volatile(                                                              \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(dst_smem),\
        "l"(src_gmem), "n"(bytes))
#define BD_CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define BD_CP_ASYNC_WAIT_GROUP(n)  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

static constexpr int kBN          = 16;   // tokens per micro-tile
static constexpr int kNumGroups   = 16;   // groups per CTA (also per micro-tile)
static constexpr int kElemPerLane = 8;    // 8 bf16/fp16 per lane (16 bytes)

// Mask helper —— 单 PTX shfl_xor 限制：mask 所有 1-bit 必须落在同一 warp 内 (32 个 lane)
// kThreadsPerKey ≤ 32，所以 (1ull << kThreadsPerKey) - 1 始终是合法 32-bit lane mask。
// 注：当 kThreadsPerKey == 32 时 shift 32 是 UB，要用 (kTPK==32 ? ~0u : ((1u<<kTPK)-1u))
__device__ __forceinline__ unsigned tpk_mask(int kTPK) {
    return (kTPK == 32) ? 0xffffffffu : ((1u << kTPK) - 1u);
}

template <class Elem, int HeadDim>
__global__ void pass1_kernel(
    const Elem* __restrict__ q_ptr,
    int64_t q_stride_b, int64_t q_stride_h,
    const Elem* const* __restrict__ k_cache_ptrs,
    const Elem* const* __restrict__ v_cache_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
    Elem* __restrict__ o_ptr,
    int64_t o_stride_b, int64_t o_stride_h,
    const int32_t* __restrict__ req_to_slot,
    const int32_t* __restrict__ kv_lens,
    float*  __restrict__ workspace_partial_o,   // [B, Hq, MaxSplits, HeadDim]
    float*  __restrict__ workspace_partial_lse, // [B, Hq, MaxSplits]
    int32_t* __restrict__ workspace_num_splits, // [B]
    int num_q_heads, int num_kv_heads,
    float softmax_scale)
{
    static_assert(HeadDim % kElemPerLane == 0,
                  "HeadDim must be a multiple of 8");
    constexpr int kThreadsPerKey = HeadDim / kElemPerLane;     // 8/16/24/32
    constexpr int kBlockThreads  = kNumGroups * kThreadsPerKey;
    constexpr int kQLoadIters    = HeadDim / 32;               // 8/16/24/32 / 8

    const int qh    = blockIdx.x;
    const int split = blockIdx.y;
    const int b     = blockIdx.z;
    const int tid   = threadIdx.x;

    const int slot   = req_to_slot[b];
    const int kv_len = kv_lens[b];
    if (kv_len <= 0) {
        if (split == 0 && tid < HeadDim) {
            Elem* o_row = o_ptr + (int64_t)b * o_stride_b + (int64_t)qh * o_stride_h;
            o_row[tid] = ET<Elem>::from_f(0.f);
            if (tid == 0) workspace_num_splits[b] = 0;
        }
        return;
    }

    // chunk_size 与原 batched 保持一致（保证 workspace 布局不变）
    int chunk_size = (kv_len + kMaxSplits - 1) / kMaxSplits;
    if (chunk_size < kMinChunkSize) chunk_size = kMinChunkSize;
    chunk_size = (chunk_size + kBN - 1) & ~(kBN - 1);     // align to BN
    const int num_splits = (kv_len + chunk_size - 1) / chunk_size;
    if (split >= num_splits) return;
    if (split == 0 && tid == 0) workspace_num_splits[b] = num_splits;

    const int chunk_begin = split * chunk_size;
    const int chunk_end   = min(chunk_begin + chunk_size, kv_len);

    const int group_size = num_q_heads / num_kv_heads;
    const int kv_head    = qh / group_size;

    const Elem* q_bh   = q_ptr + (int64_t)b * q_stride_b + (int64_t)qh * q_stride_h;
    const Elem* k_slot = k_cache_ptrs[slot] + (int64_t)kv_head * kv_stride_h;
    const Elem* v_slot = v_cache_ptrs[slot] + (int64_t)kv_head * kv_stride_h;

    // --------------------------------------------------------------------- smem
    // s_q   : HeadDim Elems
    // s_k   : 2 * kBN * HeadDim Elems
    // s_v   : 2 * kBN * HeadDim Elems
    // s_m   : kNumGroups float
    // s_s   : kNumGroups float
    // s_acc : kNumGroups * HeadDim Elems   (bf16 scratch)
    extern __shared__ unsigned char smem_raw[];
    Elem*  s_q    = reinterpret_cast<Elem*>(smem_raw);
    Elem*  s_k    = s_q + HeadDim;
    Elem*  s_v    = s_k + 2 * kBN * HeadDim;
    float* s_m    = reinterpret_cast<float*>(s_v + 2 * kBN * HeadDim);
    float* s_s    = s_m + kNumGroups;
    Elem*  s_acc  = reinterpret_cast<Elem*>(s_s + kNumGroups);

    const int gid  = tid / kThreadsPerKey;
    const int lane = tid % kThreadsPerKey;

    // ---- Load Q（tid < HeadDim/8 的线程，每个搬 16B = 8 Elem 一次）-----------
    if (tid < HeadDim / kElemPerLane) {
        reinterpret_cast<float4*>(s_q)[tid] =
            reinterpret_cast<const float4*>(q_bh)[tid];
    }
    (void)kQLoadIters;  // 用 (HeadDim/8) 个线程一次 16B 载完

    float row_max = -INFINITY;
    float row_sum = 0.f;
    float acc[kElemPerLane];
    #pragma unroll
    for (int d = 0; d < kElemPerLane; ++d) acc[d] = 0.f;

    auto get_smem_ptr = [](void* p) -> uint64_t {
        return (uint64_t)__cvta_generic_to_shared(p);
    };
    auto fetch_kv = [&](int token_base, int stage) {
        int token_idx = token_base + gid;
        if (token_idx < chunk_end) {
            const Elem* k_ptr = k_slot + (int64_t)token_idx * kv_stride_s + lane * kElemPerLane;
            const Elem* v_ptr = v_slot + (int64_t)token_idx * kv_stride_s + lane * kElemPerLane;
            Elem* sk_dst = s_k + (stage * kBN + gid) * HeadDim + lane * kElemPerLane;
            Elem* sv_dst = s_v + (stage * kBN + gid) * HeadDim + lane * kElemPerLane;
            BD_CP_ASYNC_CG(get_smem_ptr(sk_dst), k_ptr, 16);
            BD_CP_ASYNC_CG(get_smem_ptr(sv_dst), v_ptr, 16);
        }
    };

    // 预取第一个 micro-tile
    fetch_kv(chunk_begin, 0);
    BD_CP_ASYNC_COMMIT_GROUP();

    const unsigned shfl_mask = tpk_mask(kThreadsPerKey);

    for (int i = chunk_begin; i < chunk_end; i += kBN) {
        const int cur_stage  = ((i - chunk_begin) / kBN) & 1;
        const int next_stage = cur_stage ^ 1;
        if (i + kBN < chunk_end) fetch_kv(i + kBN, next_stage);
        BD_CP_ASYNC_COMMIT_GROUP();
        BD_CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        const int t = i + gid;
        if (t < chunk_end) {
            float4 q_vec = reinterpret_cast<float4*>(s_q)[lane];
            float4 k_vec = reinterpret_cast<float4*>(
                s_k + (cur_stage * kBN + gid) * HeadDim)[lane];

            // --- score = <q, k> via 4 个 bf16x2 hmul2 + warp-mask reduce ----
            float s_qk = 0.f;
            if constexpr (sizeof(Elem) == 2) {
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
            }
            // intra-group reduce
            #pragma unroll
            for (int off = kThreadsPerKey / 2; off > 0; off >>= 1) {
                s_qk += __shfl_xor_sync(shfl_mask, s_qk, off);
            }
            s_qk *= softmax_scale;

            // online softmax + V acc fuse
            const float old_max  = row_max;
            row_max              = fmaxf(row_max, s_qk);
            const float exp_scale = __expf(old_max - row_max);
            const float p         = __expf(s_qk - row_max);
            row_sum               = row_sum * exp_scale + p;

            float4 v_vec = reinterpret_cast<float4*>(
                s_v + (cur_stage * kBN + gid) * HeadDim)[lane];
            Elem* v_p = reinterpret_cast<Elem*>(&v_vec);
            #pragma unroll
            for (int d = 0; d < kElemPerLane; ++d) {
                acc[d] = acc[d] * exp_scale + p * ET<Elem>::to_f(v_p[d]);
            }
        }
    }

    // group-level: lane==0 写 m/l 到 smem
    if (lane == 0) {
        s_m[gid] = row_max;
        s_s[gid] = row_sum;
    }
    // 把 acc[8] 打包成 16B 写到 s_acc[gid * HeadDim + lane*8]
    {
        float4 pack;
        Elem* pack_bf = reinterpret_cast<Elem*>(&pack);
        #pragma unroll
        for (int d = 0; d < kElemPerLane; ++d) pack_bf[d] = ET<Elem>::from_f(acc[d]);
        reinterpret_cast<float4*>(s_acc + gid * HeadDim)[lane] = pack;
    }
    __syncthreads();

    // --- group merge：每个 tid (< HeadDim) 一个 head_dim slot 计算最终 partial ---
    if (tid < HeadDim) {
        float bm = -INFINITY;
        #pragma unroll
        for (int g = 0; g < kNumGroups; ++g) bm = fmaxf(bm, s_m[g]);

        float bs = 0.f;
        float ba = 0.f;
        #pragma unroll
        for (int g = 0; g < kNumGroups; ++g) {
            float w = __expf(s_m[g] - bm);
            bs += s_s[g] * w;
            ba += ET<Elem>::to_f(s_acc[g * HeadDim + tid]) * w;
        }
        const float inv_l = (bs > 0.f) ? (1.f / bs) : 0.f;

        if (num_splits == 1) {
            // 直通：归一后写 final O
            Elem* o_row = o_ptr + (int64_t)b * o_stride_b + (int64_t)qh * o_stride_h;
            o_row[tid] = ET<Elem>::from_f(ba * inv_l);
        } else {
            // 兼容 pass2：写 locally-normalised partial_o + lse
            const int64_t po_base = ((int64_t)b * num_q_heads + qh) * kMaxSplits + split;
            float* po = workspace_partial_o + po_base * HeadDim;
            po[tid] = ba * inv_l;
            if (tid == 0) {
                workspace_partial_lse[po_base] =
                    (bs > 0.f) ? (bm + __logf(bs)) : -INFINITY;
            }
        }
    }
}

// ============================================================================
// Pass 2 kernel — LSE-weighted merge across splits
// ============================================================================
template <class Elem, int HeadDim>
__global__ void pass2_kernel(
    Elem* __restrict__ o_ptr,
    int64_t o_stride_b, int64_t o_stride_h,
    const float* __restrict__ workspace_partial_o,
    const float* __restrict__ workspace_partial_lse,
    const int32_t* __restrict__ workspace_num_splits,
    int num_q_heads)
{
    const int qh = blockIdx.x;
    const int b  = blockIdx.z;  // grid is (Hq, 1, B)
    const int tid = threadIdx.x;

    const int num_splits = workspace_num_splits[b];
    if (num_splits <= 1) return;  // Pass 1 already wrote O directly

    // All threads compute m* from lse table (cheap, only num_splits entries).
    const int64_t lse_base = ((int64_t)b * num_q_heads + qh) * kMaxSplits;
    const float* lse_ptr = workspace_partial_lse + lse_base;

    // Broadcast m_star via smem (or just compute in every thread — small trip).
    __shared__ float m_star;
    __shared__ float l_star;
    if (tid == 0) {
        float m = -INFINITY;
        #pragma unroll
        for (int s = 0; s < kMaxSplits; ++s) {
            if (s >= num_splits) break;
            float v = lse_ptr[s];
            if (v > m) m = v;
        }
        float l = 0.f;
        #pragma unroll
        for (int s = 0; s < kMaxSplits; ++s) {
            if (s >= num_splits) break;
            l += __expf(lse_ptr[s] - m);
        }
        m_star = m;
        l_star = l;
    }
    __syncthreads();

    const float inv_l = (l_star > 0.f) ? (1.f / l_star) : 0.f;

    // Merge this thread's head-dim element.
    //   partial_o_s is locally-normalised (= softmax(chunk_s) @ V_s).
    //   Final:  o = Σ exp(lse_s - m*) * partial_o_s  /  Σ exp(lse_s - m*)
    if (tid < HeadDim) {
        const int64_t po_base = ((int64_t)b * num_q_heads + qh) * kMaxSplits;
        const float* po = workspace_partial_o + po_base * HeadDim;

        float acc = 0.f;
        #pragma unroll
        for (int s = 0; s < kMaxSplits; ++s) {
            if (s >= num_splits) break;
            float w = __expf(lse_ptr[s] - m_star);
            acc += w * po[s * HeadDim + tid];
        }

        Elem* o_row = o_ptr + (int64_t)b * o_stride_b + (int64_t)qh * o_stride_h;
        o_row[tid] = ET<Elem>::from_f(acc * inv_l);
    }
}

// ============================================================================
// Launcher
// ============================================================================
template <class Elem, int HeadDim>
static cudaError_t launch_impl(
    const Elem*  q_ptr,      int64_t qsb, int64_t qsh,
    const Elem* const* k_ptrs,
    const Elem* const* v_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          Elem*  o_ptr,      int64_t osb, int64_t osh,
    const int32_t* req_to_slot,
    const int32_t* kv_lens,
    float*   workspace,
    int batch, int num_q_heads, int num_kv_heads,
    float softmax_scale,
    cudaStream_t stream)
{
    // Workspace partition.
    float* partial_o  = workspace;
    float* partial_lse = partial_o + (int64_t)batch * num_q_heads * kMaxSplits * HeadDim;
    int32_t* num_splits = reinterpret_cast<int32_t*>(partial_lse + (int64_t)batch * num_q_heads * kMaxSplits);

    // Pass 1
    {
        constexpr int kThreadsPerKey = HeadDim / kElemPerLane;       // 8/16/24/32
        constexpr int kBlockThreads  = kNumGroups * kThreadsPerKey;  // 128/256/384/512
        dim3 grid(num_q_heads, kMaxSplits, batch);
        dim3 block(kBlockThreads);
        // smem (bytes):
        //   s_q   : HeadDim * sizeof(Elem)
        //   s_k   : 2 * kBN * HeadDim * sizeof(Elem)
        //   s_v   : 2 * kBN * HeadDim * sizeof(Elem)
        //   s_m   : kNumGroups * 4
        //   s_s   : kNumGroups * 4
        //   s_acc : kNumGroups * HeadDim * sizeof(Elem)
        const size_t smem_size =
            (size_t)HeadDim * sizeof(Elem) +
            (size_t)2 * kBN * HeadDim * sizeof(Elem) * 2 +     // s_k + s_v
            (size_t)kNumGroups * 2 * sizeof(float) +
            (size_t)kNumGroups * HeadDim * sizeof(Elem);
        auto kernel = pass1_kernel<Elem, HeadDim>;
        // `cudaFuncSetAttribute` 是 host-同步 API，CUDA Graph stream capture
        // 不允许在 capture 中调用。本属性是 per-kernel 的全局状态，整个 process
        // 只需调一次；用 `std::once_flag` 守卫。template instantiation 会让
        // 每个 (Elem, HeadDim) 组合各自有独立的 once_flag，刚好对应每个 kernel
        // 函数地址。
        static std::once_flag pass1_attr_once;
        static cudaError_t pass1_attr_err = cudaSuccess;
        std::call_once(pass1_attr_once, [&]() {
            pass1_attr_err = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);
        });
        if (pass1_attr_err != cudaSuccess) return pass1_attr_err;
        kernel<<<grid, block, smem_size, stream>>>(
            q_ptr, qsb, qsh,
            k_ptrs, v_ptrs, kv_stride_s, kv_stride_h,
            o_ptr, osb, osh,
            req_to_slot, kv_lens,
            partial_o, partial_lse, num_splits,
            num_q_heads, num_kv_heads, softmax_scale);
    }

    // Pass 2
    {
        dim3 grid(num_q_heads, 1, batch);
        dim3 block(HeadDim);
        auto kernel = pass2_kernel<Elem, HeadDim>;
        kernel<<<grid, block, 0, stream>>>(
            o_ptr, osb, osh,
            partial_o, partial_lse, num_splits,
            num_q_heads);
    }

    return cudaGetLastError();
}

template <class Elem>
static cudaError_t launch_dispatch(
    const Elem*  q,  int64_t qsb, int64_t qsh,
    const Elem* const* k_ptrs,
    const Elem* const* v_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          Elem*  o,  int64_t osb, int64_t osh,
    const int32_t* req_to_slot,
    const int32_t* kv_lens,
    float*   workspace,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream)
{
    switch (head_dim) {
    case 64:  return launch_impl<Elem,  64>(q,qsb,qsh, k_ptrs,v_ptrs,kv_stride_s,kv_stride_h,
                                            o,osb,osh, req_to_slot,kv_lens,workspace,
                                            batch,num_q_heads,num_kv_heads,
                                            softmax_scale,stream);
    case 128: return launch_impl<Elem, 128>(q,qsb,qsh, k_ptrs,v_ptrs,kv_stride_s,kv_stride_h,
                                            o,osb,osh, req_to_slot,kv_lens,workspace,
                                            batch,num_q_heads,num_kv_heads,
                                            softmax_scale,stream);
    case 192: return launch_impl<Elem, 192>(q,qsb,qsh, k_ptrs,v_ptrs,kv_stride_s,kv_stride_h,
                                            o,osb,osh, req_to_slot,kv_lens,workspace,
                                            batch,num_q_heads,num_kv_heads,
                                            softmax_scale,stream);
    case 256: return launch_impl<Elem, 256>(q,qsb,qsh, k_ptrs,v_ptrs,kv_stride_s,kv_stride_h,
                                            o,osb,osh, req_to_slot,kv_lens,workspace,
                                            batch,num_q_heads,num_kv_heads,
                                            softmax_scale,stream);
    default:
        fprintf(stderr, "[flash_batched_decode] unsupported head_dim=%d "
                        "(supported: 64, 128, 192, 256)\n", head_dim);
        return cudaErrorInvalidValue;
    }
}

}  // namespace flash_batched_decode

// ============================================================================
// Public C ABI
// ============================================================================
extern "C" {

// Required workspace size (in bytes) for a given (batch, num_q_heads, head_dim).
int64_t flash_attn_batched_decode_workspace_bytes(
    int batch, int num_q_heads, int head_dim)
{
    constexpr int kMaxSplits = flash_batched_decode::kMaxSplits;
    int64_t partial_o   = (int64_t)batch * num_q_heads * kMaxSplits * head_dim * (int64_t)sizeof(float);
    int64_t partial_lse = (int64_t)batch * num_q_heads * kMaxSplits * (int64_t)sizeof(float);
    int64_t num_splits  = (int64_t)batch * (int64_t)sizeof(int32_t);
    return partial_o + partial_lse + num_splits;
}

void launch_flash_attn_batched_decode_bf16(
    const __nv_bfloat16* q, int64_t qsb, int64_t qsh,
    const __nv_bfloat16* const* k_ptrs,
    const __nv_bfloat16* const* v_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          __nv_bfloat16* o, int64_t osb, int64_t osh,
    const int32_t* req_to_slot,
    const int32_t* kv_lens,
    float* workspace,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream)
{
    cudaError_t err = flash_batched_decode::launch_dispatch<__nv_bfloat16>(
        q, qsb, qsh, k_ptrs, v_ptrs, kv_stride_s, kv_stride_h,
        o, osb, osh, req_to_slot, kv_lens, workspace,
        batch, num_q_heads, num_kv_heads, head_dim, softmax_scale, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[flash_attn_batched_decode_bf16] launch error: %s\n",
                cudaGetErrorString(err));
    }
}

void launch_flash_attn_batched_decode_fp16(
    const __half* q, int64_t qsb, int64_t qsh,
    const __half* const* k_ptrs,
    const __half* const* v_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          __half* o, int64_t osb, int64_t osh,
    const int32_t* req_to_slot,
    const int32_t* kv_lens,
    float* workspace,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream)
{
    cudaError_t err = flash_batched_decode::launch_dispatch<__half>(
        q, qsb, qsh, k_ptrs, v_ptrs, kv_stride_s, kv_stride_h,
        o, osb, osh, req_to_slot, kv_lens, workspace,
        batch, num_q_heads, num_kv_heads, head_dim, softmax_scale, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[flash_attn_batched_decode_fp16] launch error: %s\n",
                cudaGetErrorString(err));
    }
}

} // extern "C"


// =============================================================================
// Stand-alone test main (same pattern as flash_attn_gqa_prefill.cu)
//   nvcc -std=c++17 -arch=sm_80 -O3 -I<cutlass-include> \
//        -DFLASH_ATTN_BATCHED_DECODE_STANDALONE_TEST \
//        flash_attn_batched_decode.cu -o bdecode_test
// =============================================================================
#ifdef FLASH_ATTN_BATCHED_DECODE_STANDALONE_TEST
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_MUST(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){ \
    std::cerr<<cudaGetErrorString(_e)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);}} while(0)

template <class E> struct HT;
template <> struct HT<__nv_bfloat16> {
    static __nv_bfloat16 f(float x) { return __float2bfloat16(x); }
    static float to(__nv_bfloat16 x) { return __bfloat162float(x); }
    static const char* n() { return "bf16"; }
};
template <> struct HT<__half> {
    static __half f(float x) { return __float2half(x); }
    static float to(__half x) { return __half2float(x); }
    static const char* n() { return "fp16"; }
};

__global__ void ref_decode(
    const float* Q, int64_t qsb, int64_t qsh,
    const float* K, const float* V,
    int64_t kvs_s, int64_t kvs_h,
    float* O, int64_t osb, int64_t osh,
    int Hq, int Hkv, int HD, int kv_len, float scale)
{
    int b  = blockIdx.z;
    int qh = blockIdx.y;
    int kvh = qh / (Hq / Hkv);

    const float* q = Q + b*qsb + qh*qsh;
    const float* k_base = K + (int64_t)b * (kv_len * kvs_s) + kvh*kvs_h;
    const float* v_base = V + (int64_t)b * (kv_len * kvs_s) + kvh*kvs_h;
    float* o = O + b*osb + qh*osh;

    float m = -INFINITY;
    for (int t = 0; t < kv_len; ++t) {
        const float* kp = k_base + (int64_t)t * kvs_s;
        float s = 0.f;
        for (int d = 0; d < HD; ++d) s += q[d] * kp[d];
        s *= scale;
        if (s > m) m = s;
    }
    float denom = 0.f;
    for (int d = 0; d < HD; ++d) o[d] = 0.f;
    for (int t = 0; t < kv_len; ++t) {
        const float* kp = k_base + (int64_t)t * kvs_s;
        const float* vp = v_base + (int64_t)t * kvs_s;
        float s = 0.f;
        for (int d = 0; d < HD; ++d) s += q[d] * kp[d];
        float e = __expf(s * scale - m);
        denom += e;
        for (int d = 0; d < HD; ++d) o[d] += e * vp[d];
    }
    float inv = (denom==0.f)?1.f:1.f/denom;
    for (int d = 0; d < HD; ++d) o[d] *= inv;
}

extern "C" int64_t flash_attn_batched_decode_workspace_bytes(int,int,int);
extern "C" void launch_flash_attn_batched_decode_bf16(
    const __nv_bfloat16*, int64_t, int64_t,
    const __nv_bfloat16* const*, const __nv_bfloat16* const*,
    int64_t, int64_t,
    __nv_bfloat16*, int64_t, int64_t,
    const int32_t*, const int32_t*,
    float*, int, int, int, int, float, cudaStream_t);
extern "C" void launch_flash_attn_batched_decode_fp16(
    const __half*, int64_t, int64_t,
    const __half* const*, const __half* const*,
    int64_t, int64_t,
    __half*, int64_t, int64_t,
    const int32_t*, const int32_t*,
    float*, int, int, int, int, float, cudaStream_t);

template <class Elem>
bool run_case(int B, int Hq, int Hkv, int HD, std::vector<int> kv_lens_host, std::mt19937& rng) {
    const int max_kv = *std::max_element(kv_lens_host.begin(), kv_lens_host.end());
    const float scale = 1.f / std::sqrt((float)HD);

    // Build independent KV caches: B buffers each [max_kv, Hkv, HD]
    std::vector<std::vector<Elem>>  h_k(B), h_v(B);
    std::vector<std::vector<float>> fk(B),  fv(B);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int b = 0; b < B; ++b) {
        h_k[b].resize((size_t)max_kv * Hkv * HD);
        h_v[b].resize((size_t)max_kv * Hkv * HD);
        fk[b].resize(h_k[b].size()); fv[b].resize(h_v[b].size());
        for (size_t i = 0; i < h_k[b].size(); ++i) {
            float a = dist(rng), c = dist(rng);
            fk[b][i] = a; fv[b][i] = c;
            h_k[b][i] = HT<Elem>::f(a); h_v[b][i] = HT<Elem>::f(c);
        }
    }
    // Q: [B, Hq, HD]
    std::vector<Elem>  h_q((size_t)B * Hq * HD);
    std::vector<float> fq((size_t)B * Hq * HD);
    for (size_t i = 0; i < fq.size(); ++i) { float x = dist(rng); fq[i] = x; h_q[i] = HT<Elem>::f(x); }

    // Allocate device buffers
    Elem* d_q=nullptr; CUDA_MUST(cudaMalloc(&d_q, h_q.size()*sizeof(Elem)));
    CUDA_MUST(cudaMemcpy(d_q, h_q.data(), h_q.size()*sizeof(Elem), cudaMemcpyHostToDevice));

    std::vector<Elem*> d_k_slots(B), d_v_slots(B);
    for (int b = 0; b < B; ++b) {
        CUDA_MUST(cudaMalloc(&d_k_slots[b], h_k[b].size()*sizeof(Elem)));
        CUDA_MUST(cudaMalloc(&d_v_slots[b], h_v[b].size()*sizeof(Elem)));
        CUDA_MUST(cudaMemcpy(d_k_slots[b], h_k[b].data(), h_k[b].size()*sizeof(Elem), cudaMemcpyHostToDevice));
        CUDA_MUST(cudaMemcpy(d_v_slots[b], h_v[b].data(), h_v[b].size()*sizeof(Elem), cudaMemcpyHostToDevice));
    }
    Elem** d_k_ptrs=nullptr; CUDA_MUST(cudaMalloc(&d_k_ptrs, B*sizeof(Elem*)));
    Elem** d_v_ptrs=nullptr; CUDA_MUST(cudaMalloc(&d_v_ptrs, B*sizeof(Elem*)));
    CUDA_MUST(cudaMemcpy(d_k_ptrs, d_k_slots.data(), B*sizeof(Elem*), cudaMemcpyHostToDevice));
    CUDA_MUST(cudaMemcpy(d_v_ptrs, d_v_slots.data(), B*sizeof(Elem*), cudaMemcpyHostToDevice));

    int32_t* d_kvlen=nullptr; CUDA_MUST(cudaMalloc(&d_kvlen, B*sizeof(int32_t)));
    CUDA_MUST(cudaMemcpy(d_kvlen, kv_lens_host.data(), B*sizeof(int32_t), cudaMemcpyHostToDevice));

    std::vector<int32_t> slot_map(B);
    for (int b = 0; b < B; ++b) slot_map[b] = b;
    int32_t* d_slot=nullptr; CUDA_MUST(cudaMalloc(&d_slot, B*sizeof(int32_t)));
    CUDA_MUST(cudaMemcpy(d_slot, slot_map.data(), B*sizeof(int32_t), cudaMemcpyHostToDevice));

    Elem* d_o=nullptr; CUDA_MUST(cudaMalloc(&d_o, (size_t)B*Hq*HD*sizeof(Elem)));
    CUDA_MUST(cudaMemset(d_o, 0, (size_t)B*Hq*HD*sizeof(Elem)));

    int64_t ws_bytes = flash_attn_batched_decode_workspace_bytes(B, Hq, HD);
    float* d_ws=nullptr; CUDA_MUST(cudaMalloc(&d_ws, ws_bytes));

    const int64_t qsb = (int64_t)Hq * HD, qsh = HD;
    const int64_t osb = qsb, osh = qsh;
    const int64_t kvs_s = (int64_t)Hkv * HD, kvs_h = HD;

    if constexpr (std::is_same_v<Elem, __nv_bfloat16>) {
        launch_flash_attn_batched_decode_bf16(
            d_q, qsb, qsh,
            (const __nv_bfloat16* const*)d_k_ptrs,
            (const __nv_bfloat16* const*)d_v_ptrs,
            kvs_s, kvs_h,
            d_o, osb, osh,
            d_slot, d_kvlen,
            d_ws,
            B, Hq, Hkv, HD, scale, 0);
    } else {
        launch_flash_attn_batched_decode_fp16(
            d_q, qsb, qsh,
            (const __half* const*)d_k_ptrs,
            (const __half* const*)d_v_ptrs,
            kvs_s, kvs_h,
            d_o, osb, osh,
            d_slot, d_kvlen,
            d_ws,
            B, Hq, Hkv, HD, scale, 0);
    }
    CUDA_MUST(cudaDeviceSynchronize());

    // Reference on f32 (per batch, because kv_lens differ we need per-b launch)
    std::vector<float> ref_o((size_t)B*Hq*HD, 0.f);
    for (int b = 0; b < B; ++b) {
        const int kv_len = kv_lens_host[b];
        // copy per-b slices
        float* d_Q=nullptr; cudaMalloc(&d_Q, (size_t)Hq*HD*sizeof(float));
        float* d_K=nullptr; cudaMalloc(&d_K, (size_t)kv_len*Hkv*HD*sizeof(float));
        float* d_V=nullptr; cudaMalloc(&d_V, (size_t)kv_len*Hkv*HD*sizeof(float));
        float* d_O=nullptr; cudaMalloc(&d_O, (size_t)Hq*HD*sizeof(float));
        cudaMemcpy(d_Q, fq.data() + (size_t)b*Hq*HD, Hq*HD*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, fk[b].data(), (size_t)kv_len*Hkv*HD*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, fv[b].data(), (size_t)kv_len*Hkv*HD*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_O, 0, (size_t)Hq*HD*sizeof(float));

        ref_decode<<<dim3(1, Hq, 1)>>>(
            d_Q, (int64_t)Hq*HD, HD,
            d_K, d_V, (int64_t)Hkv*HD, HD,
            d_O, (int64_t)Hq*HD, HD,
            Hq, Hkv, HD, kv_len, scale);
        cudaDeviceSynchronize();

        std::vector<float> tmp(Hq*HD);
        cudaMemcpy(tmp.data(), d_O, Hq*HD*sizeof(float), cudaMemcpyDeviceToHost);
        std::copy(tmp.begin(), tmp.end(), ref_o.begin() + (size_t)b*Hq*HD);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    }

    std::vector<Elem> got((size_t)B*Hq*HD);
    CUDA_MUST(cudaMemcpy(got.data(), d_o, got.size()*sizeof(Elem), cudaMemcpyDeviceToHost));

    const float atol = std::is_same_v<Elem, __nv_bfloat16> ? 7e-2f : 1e-2f;
    const float rtol = std::is_same_v<Elem, __nv_bfloat16> ? 1e-2f : 5e-3f;
    double max_err = 0; long bad = 0;
    for (size_t i = 0; i < ref_o.size(); ++i) {
        float a = HT<Elem>::to(got[i]);
        float r = ref_o[i];
        float e = std::fabs(a - r);
        float tol = atol + rtol * std::fabs(r);
        max_err = std::max<double>(max_err, e);
        if (e > tol) ++bad;
    }
    std::printf("  [%s] B=%d Hq=%d Hkv=%d HD=%d kv_lens=", HT<Elem>::n(), B, Hq, Hkv, HD);
    for (int x : kv_lens_host) std::printf("%d,", x);
    std::printf("  max_err=%.4e bad=%ld/%zu  %s\n", max_err, bad, ref_o.size(), bad==0?"OK":"FAIL");

    cudaFree(d_q); cudaFree(d_o); cudaFree(d_kvlen); cudaFree(d_slot); cudaFree(d_ws);
    for (int b = 0; b < B; ++b) { cudaFree(d_k_slots[b]); cudaFree(d_v_slots[b]); }
    cudaFree(d_k_ptrs); cudaFree(d_v_ptrs);
    return bad == 0;
}

int main() {
    std::mt19937 rng(0xBEEF);
    bool ok = true;

    std::cout << "=== BF16 ===\n";
    ok &= run_case<__nv_bfloat16>(1,  8, 2,  64, {100},                  rng);
    ok &= run_case<__nv_bfloat16>(3,  4, 2,  64, {10, 20, 30},           rng);
    ok &= run_case<__nv_bfloat16>(3,  8, 2,  64, {99, 199, 299},         rng);
    ok &= run_case<__nv_bfloat16>(4,  8, 2, 128, {50, 50, 50, 50},       rng);   // 50+50
    ok &= run_case<__nv_bfloat16>(4, 16, 4, 128, {100, 2048, 500, 99},   rng);   // mixed
    ok &= run_case<__nv_bfloat16>(2,  8, 2, 128, {4096, 8192},           rng);   // long ctx
    ok &= run_case<__nv_bfloat16>(1,  8, 2, 192, {1234},                 rng);
    ok &= run_case<__nv_bfloat16>(1,  8, 2, 256, {1024},                 rng);

    std::cout << "=== FP16 ===\n";
    ok &= run_case<__half>(1,  8, 2,  64, {100},                  rng);
    ok &= run_case<__half>(3,  4, 2,  64, {10, 20, 30},           rng);
    ok &= run_case<__half>(4,  8, 2, 128, {50, 50, 50, 50},       rng);
    ok &= run_case<__half>(4, 16, 4, 128, {100, 2048, 500, 99},   rng);
    ok &= run_case<__half>(2,  8, 2, 128, {4096, 8192},           rng);

    std::cout << (ok?"ALL TESTS PASSED\n":"SOME TESTS FAILED\n");
    return ok ? 0 : 1;
}
#endif // FLASH_ATTN_BATCHED_DECODE_STANDALONE_TEST
