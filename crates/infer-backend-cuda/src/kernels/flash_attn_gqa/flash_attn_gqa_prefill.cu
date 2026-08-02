// flash_attn_gqa_prefill.cu
// -----------------------------------------------------------------------------
// A modern, batched, stride-aware Flash-Attention GQA prefill kernel (CuTe/SM80).
//
// Highlights:
//   * Arbitrary batch size      (gridDim.z = B).
//   * Arbitrary stride layout   (Q/K/V/O: stride_b, stride_s, stride_h).
//     Caller passes real tensor strides; NHD / HND / sliced-view all work.
//   * Arbitrary head_dim        (dispatched statically: 64, 128, 192, 256).
//     Adding more sizes = adding one line in the dispatcher below.
//   * 2-stage cp.async pipeline on K/V; single-shot load of Q.
//   * Predicate-masked epilogue for non-multiple-of-BlockM seq lengths.
//
// Public C ABI (used by Rust FFI and stand-alone tests):
//
//   void launch_flash_attn_prefill_bf16(
//       const __nv_bfloat16* Q,  int64_t qsb, int64_t qss, int64_t qsh,
//       const __nv_bfloat16* K,  int64_t ksb, int64_t kss, int64_t ksh,
//       const __nv_bfloat16* V,  int64_t vsb, int64_t vss, int64_t vsh,
//             __nv_bfloat16* O,  int64_t osb, int64_t oss, int64_t osh,
//       int batch, int q_len, int kv_len,
//       int num_q_heads, int num_kv_heads, int head_dim,
//       float softmax_scale, int is_causal,
//       cudaStream_t stream);
//
//   void launch_flash_attn_prefill_fp16(... same shape with __half ...);
//
// Legacy ABI (kept for backwards compatibility):
//   launch_flash_attn_cute_128x64x64_tile         (head_dim=64, bf16, B=1, NHD)
//   launch_flash_attn_cute_128x64x64_tile_fp16    (head_dim=64, fp16, B=1, NHD)
//   launch_flash_attn_cute_bf16_hdim128           (head_dim=128, bf16, B=1, NHD)
//   launch_flash_attn_cute_fp16_hdim128           (head_dim=128, fp16, B=1, NHD)
// They are implemented as thin wrappers on top of the new ABI.
//
// Note: head_dim=128 version lives in flash_attn_gqa_prefill_hdim128.cu for the
// legacy hdim128 symbol; we provide our own templated path here that covers
// 64/128/192/256 uniformly.  The legacy hdim128 TU still owns its own
// launch_flash_attn_cute_bf16_hdim128 / _fp16_hdim128 symbols; we *do not*
// redefine them here to avoid duplicate symbols.
// -----------------------------------------------------------------------------

#include "flash_attn_gqa.h"
#include "cuda_kernel_attr.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

#include <cstdint>
#include <cstdio>

namespace flash_attn_prefill {

using namespace cute;

// =============================================================================
// Element -> MMA atom mapping
// =============================================================================
template <class Elem> struct MmaAtomFor;
template <> struct MmaAtomFor<__nv_bfloat16> {
    using type = SM80_16x8x16_F32BF16BF16F32_TN;
};
template <> struct MmaAtomFor<__half> {
    using type = SM80_16x8x16_F32F16F16F32_TN;
};

// =============================================================================
// Static kernel traits (BlockM / BlockN fixed; HeadDim is a template parameter)
// =============================================================================
static constexpr int kNWarps   = 4;
static constexpr int kBlockM   = 128;
static constexpr int kBlockN   = 64;

template <class Elem_, int HeadDim_>
struct KTraits {
    using Elem = Elem_;
    static constexpr int HeadDim = HeadDim_;
    static_assert(HeadDim % 64 == 0, "HeadDim must be a multiple of 64");

    // Swizzled smem atom: <3,3,3> over a 8x64 tile -> 128B lines, bank-conflict free
    using SmemAtom = decltype(composition(
        Swizzle<3,3,3>{},
        Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

    using SmemLayoutQ  = decltype(tile_to_shape(SmemAtom{},
                                      Shape<Int<kBlockM>, Int<HeadDim>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(SmemAtom{},
                                      Shape<Int<kBlockN>, Int<HeadDim>>{}));
    using SmemLayoutO  = SmemLayoutQ;

    // V transposed (for P @ V, where V reads as B-operand of MMA)
    using SmemLayoutVt =
        decltype(composition(SmemLayoutKV{},
                             make_layout(Shape<Int<HeadDim>, Int<kBlockN>>{},
                                         GenRowMajor{})));
    using SmemLayoutVtNoSwi =
        decltype(get_nonswizzle_portion(SmemLayoutVt{}));

    // Async gmem -> smem copy atom (128-bit cp.async)
    using GmemCopyAtom =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Elem>;

    // A 16x8 thread layout, each thread copies 1x8 elements (16B = 128b).
    // That covers 16 rows x 64 cols per instruction; we tile it over (BlockM, HD)
    // / (BlockN, HD) via partition_*.
    using GmemTiledCopy = decltype(make_tiled_copy(
        GmemCopyAtom{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}));

    // Smem -> reg ldmatrix atoms
    using S2RAtomAB   = Copy_Atom<SM75_U32x4_LDSM_N, Elem>;
    using S2RAtomVT   = Copy_Atom<SM75_U16x8_LDSM_T, Elem>;

    // Output copy: smem -> gmem (universal 128-bit copy, predicated in M)
    using GmemCopyOut =
        Copy_Atom<UniversalCopy<uint128_t>, Elem>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
        GmemCopyOut{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}));

    // MMA tile: (16*NWarps, 64, 64). K-direction iterations are handled by
    // partition_fragment_* automatically when HeadDim > 64.
    using WarpLayout = Layout<Shape<Int<kNWarps>, _1, _1>>;
    using MmaTile    = Tile<Int<16 * kNWarps>, _64, _64>;
    using Mma        = decltype(make_tiled_mma(
        typename MmaAtomFor<Elem>::type{},
        WarpLayout{},
        MmaTile{}));

    static constexpr int NumThreads = 32 * kNWarps;
};

template <class Elem, class LayoutQ, class LayoutKV>
struct SharedStorage {
    array_aligned<Elem, cosize_v<LayoutQ>>  smem_q;
    array_aligned<Elem, cosize_v<LayoutKV>> smem_k;
    array_aligned<Elem, cosize_v<LayoutKV>> smem_v;
};

// =============================================================================
// Device helpers: reductions, layout conversions, mask, softmax scaling.
// =============================================================================

template <bool ZeroInit=true, class T0, class L0, class T1, class L1, class Op>
__device__ __forceinline__
void thread_reduce(Tensor<T0, L0> const& src, Tensor<T1, L1>& dst, Op op) {
    static_assert(L0::rank == 2, "src must be 2D");
    static_assert(L1::rank == 1, "dst must be 1D");
    CUTE_STATIC_ASSERT_V(size<0>(dst) == size<0>(src));
    #pragma unroll
    for (int mi = 0; mi < size<0>(src); ++mi) {
        dst(mi) = ZeroInit ? src(mi, 0) : op(dst(mi), src(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(src); ++ni) {
            dst(mi) = op(dst(mi), src(mi, ni));
        }
    }
}

template <int Threads>
struct Allreduce {
    static_assert(Threads == 32 || Threads == 16 || Threads == 8 || Threads == 4);
    template <class T, class Op>
    static __device__ __forceinline__ T run(T x, Op op) {
        constexpr int Off = Threads / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, Off));
        return Allreduce<Off>::run(x, op);
    }
};
template <>
struct Allreduce<2> {
    template <class T, class Op>
    static __device__ __forceinline__ T run(T x, Op op) {
        return op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    }
};

template <class T0, class L0, class T1, class L1, class Op>
__device__ __forceinline__
void quad_allreduce(Tensor<T0, L0>& dst, Tensor<T1, L1>& src, Op op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); ++i) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

struct MaxOp { __device__ __forceinline__ float operator()(float a, float b) const { return max(a, b); } };
struct SumOp { __device__ __forceinline__ float operator()(float a, float b) const { return a + b; } };

template <bool ZeroInit, class T0, class L0, class T1, class L1>
__device__ __forceinline__
void reduce_max_rows(Tensor<T0, L0> const& src, Tensor<T1, L1>& dst) {
    MaxOp op;
    thread_reduce<ZeroInit>(src, dst, op);
    quad_allreduce(dst, dst, op);
}
template <bool ZeroInit, class T0, class L0, class T1, class L1>
__device__ __forceinline__
void reduce_sum_rows(Tensor<T0, L0> const& src, Tensor<T1, L1>& dst) {
    SumOp op;
    thread_reduce<ZeroInit>(src, dst, op);
}

// Reshape MMA-C accumulator layout (MMA=4, M, N) -> (row, col) 2D view.
template <class Layout>
__forceinline__ __device__ auto to_rowcol(Layout l) {
    static_assert(decltype(size<0>(l))::value == 4);
    static_assert(decltype(rank(l))::value == 3);
    auto x = logical_divide(l, Shape<_2>{});   // ((2,2), M, N)
    return make_layout(make_layout(get<0,1>(x), get<1>(x)),
                       make_layout(get<0,0>(x), get<2>(x)));
}

// Convert MMA-C accumulator (float) to MMA-A input layout (bf16/fp16) for the 2nd GEMM.
template <class MmaT, class Layout>
__forceinline__ __device__ auto to_A_regs(Layout l) {
    using X = Underscore;
    static_assert(decltype(size<0>(l))::value == 4);
    static_assert(decltype(rank(l))::value == 3);
    constexpr int K = get<2>(typename MmaT::Shape_MNK{});
    static_assert(K == 8 || K == 16);
    if constexpr (K == 8) { return l; }
    else {
        auto x = logical_divide(l, Shape<X, X, _2>{});
        return make_layout(make_layout(get<0>(x), get<2,0>(x)), get<1>(x), get<2,1>(x));
    }
}

template <class To, class Eng, class Lay>
__forceinline__ __device__ auto convert_type(Tensor<Eng, Lay> const& src) {
    using From = typename Eng::value_type;
    constexpr int N = decltype(size(src))::value;
    cutlass::NumericArrayConverter<To, From, N> conv;
    auto frag = conv(*reinterpret_cast<const cutlass::Array<From, N>*>(src.data()));
    return make_tensor(make_rmem_ptr<To>(&frag), src.layout());
}

// Causal mask: set scores to -inf where col > row + causal_shift.
// `tensor_` is MMA-C fragment (rank-3, size<0>=4).
template <class Eng, class Lay>
__forceinline__ __device__
void apply_causal_mask(Tensor<Eng, Lay>& tensor_,
                       int col_idx_offset_,
                       int row_idx_offset,
                       int warp_row_stride,
                       int causal_shift,
                       int kv_upper_bound) {
    static_assert(Lay::rank == 3);
    static_assert(decltype(size<0>(tensor_))::value == 4);
    auto t = make_tensor(tensor_.data(), to_rowcol(tensor_.layout()));
    const int lane_id = threadIdx.x & 31;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

    #pragma unroll
    for (int mi = 0; mi < size<0,1>(t); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0,0>(t); ++i) {
            const int row_idx = row_idx_base + i * 8;
            const int col_limit = row_idx + 1 + causal_shift;
            #pragma unroll
            for (int nj = 0; nj < size<1,1>(t); ++nj) {
                const int col_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1,0>(t); ++j) {
                    const int col_idx = col_base + j;
                    if (col_idx >= col_limit || col_idx >= kv_upper_bound) {
                        t(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
        }
    }
}

// Non-causal: only mask columns >= kv_len (for the ragged last tile).
template <class Eng, class Lay>
__forceinline__ __device__
void apply_col_bound_mask(Tensor<Eng, Lay>& tensor_,
                          int col_idx_offset_,
                          int kv_upper_bound) {
    static_assert(Lay::rank == 3);
    auto t = make_tensor(tensor_.data(), to_rowcol(tensor_.layout()));
    const int lane_id = threadIdx.x & 31;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0,1>(t); ++mi) {
        #pragma unroll
        for (int i = 0; i < size<0,0>(t); ++i) {
            #pragma unroll
            for (int nj = 0; nj < size<1,1>(t); ++nj) {
                const int col_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1,0>(t); ++j) {
                    if (col_base + j >= kv_upper_bound) {
                        t(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
        }
    }
}

template <bool ScaleMax=true, class T0, class L0, class T1, class L1>
__forceinline__ __device__
void scale_apply_exp2(Tensor<T0, L0>& s, Tensor<T1, L1> const& m, float scale) {
    static_assert(L0::rank == 2);
    static_assert(L1::rank == 1);
    CUTE_STATIC_ASSERT_V(size<0>(m) == size<0>(s));
    #pragma unroll
    for (int mi = 0; mi < size<0>(s); ++mi) {
        const float mm = (m(mi) == -INFINITY) ? 0.f
                          : m(mi) * (ScaleMax ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(s); ++ni) {
            s(mi, ni) = exp2f(s(mi, ni) * scale - mm);
        }
    }
}

// Second GEMM (acc += P @ V^T) with overlapped smem->reg loads on B operand.
template <class T0, class T1, class T2, class T3, class M, class TC, class ThrC>
__forceinline__ __device__
void gemm_rs(T0& acc, T1& tA, T2& tB, T3 const& tsB, M mma, TC copyB, ThrC thrB) {
    CUTE_STATIC_ASSERT_V(size<1>(tA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tA) == size<2>(tB));
    auto tB_view = thrB.retile_D(tB);
    CUTE_STATIC_ASSERT_V(size<1>(tsB) == size<1>(tB_view));
    cute::copy(copyB, tsB(_, _, _0{}), tB_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tA); ++i) {
        if (i < size<2>(tA) - 1) {
            cute::copy(copyB, tsB(_, _, i + 1), tB_view(_, _, i + 1));
        }
        cute::gemm(mma, tA(_, _, i), tB(_, _, i), acc);
    }
}

// =============================================================================
// Main kernel
//   grid = (m_blocks, num_q_heads, batch)
//   block threads = 32 * kNWarps
// =============================================================================
template <class Traits>
__global__ void flash_attn_prefill_kernel(
    const typename Traits::Elem* __restrict__ q_ptr,
    int64_t q_stride_b, int64_t q_stride_s, int64_t q_stride_h,
    const typename Traits::Elem* __restrict__ k_ptr,
    int64_t k_stride_b, int64_t k_stride_s, int64_t k_stride_h,
    const typename Traits::Elem* __restrict__ v_ptr,
    int64_t v_stride_b, int64_t v_stride_s, int64_t v_stride_h,
          typename Traits::Elem* __restrict__ o_ptr,
    int64_t o_stride_b, int64_t o_stride_s, int64_t o_stride_h,
    int q_len, int kv_len,
    int num_q_heads, int num_kv_heads,
    float softmax_scale,
    int is_causal)
{
    using Elem = typename Traits::Elem;
    constexpr int HD = Traits::HeadDim;

    const int block_m    = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    const int batch_idx  = blockIdx.z;
    const int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);

    // Per-row base pointers (batch and head offsets folded in).
    const Elem* q_bh = q_ptr + batch_idx * q_stride_b + q_head_idx  * q_stride_h;
    const Elem* k_bh = k_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h;
    const Elem* v_bh = v_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h;
          Elem* o_bh = o_ptr + batch_idx * o_stride_b + q_head_idx  * o_stride_h;

    // Build global tensors shaped (seq, head_dim) with arbitrary row stride.
    auto Q = make_tensor(make_gmem_ptr(q_bh),
                         make_shape(q_len, Int<HD>{}),
                         make_stride(q_stride_s, _1{}));
    auto K = make_tensor(make_gmem_ptr(k_bh),
                         make_shape(kv_len, Int<HD>{}),
                         make_stride(k_stride_s, _1{}));
    auto V = make_tensor(make_gmem_ptr(v_bh),
                         make_shape(kv_len, Int<HD>{}),
                         make_stride(v_stride_s, _1{}));
    auto O = make_tensor(make_gmem_ptr(o_bh),
                         make_shape(q_len, Int<HD>{}),
                         make_stride(o_stride_s, _1{}));

    // Block-level tiles (N tile = _ for K/V, iterated at runtime).
    auto gQ = local_tile(Q, Shape<Int<kBlockM>, Int<HD>>{}, make_coord(block_m, 0));
    auto gK = local_tile(K, Shape<Int<kBlockN>, Int<HD>>{}, make_coord(_,       0));
    auto gV = local_tile(V, Shape<Int<kBlockN>, Int<HD>>{}, make_coord(_,       0));
    auto gO = local_tile(O, Shape<Int<kBlockM>, Int<HD>>{}, make_coord(block_m, 0));

    // Shared memory.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    using SharedT = SharedStorage<Elem,
                                  typename Traits::SmemLayoutQ,
                                  typename Traits::SmemLayoutKV>;
    SharedT& smem = *reinterpret_cast<SharedT*>(smem_raw);

    auto sQ  = make_tensor(make_smem_ptr(smem.smem_q.begin()), typename Traits::SmemLayoutQ{});
    auto sK  = make_tensor(make_smem_ptr(smem.smem_k.begin()), typename Traits::SmemLayoutKV{});
    auto sV  = make_tensor(make_smem_ptr(smem.smem_v.begin()), typename Traits::SmemLayoutKV{});
    auto sVt = make_tensor(sV.data(), typename Traits::SmemLayoutVt{});
    auto sVtNS = make_tensor(sV.data(), typename Traits::SmemLayoutVtNoSwi{});

    // Copy engines.
    typename Traits::GmemTiledCopy  copy_q{};
    typename Traits::GmemTiledCopy  copy_kv{};
    typename Traits::GmemTiledCopyO copy_o_gmem{};
    typename Traits::Mma            mma{};

    auto thr_copy_q  = copy_q .get_slice(threadIdx.x);
    auto thr_copy_kv = copy_kv.get_slice(threadIdx.x);

    auto tQgQ = thr_copy_q .partition_S(gQ);       // (CPY, M, K)
    auto tQsQ = thr_copy_q .partition_D(sQ);
    auto tKgK = thr_copy_kv.partition_S(gK);       // (CPY, N, K, n_tiles)
    auto tKsK = thr_copy_kv.partition_D(sK);
    auto tVgV = thr_copy_kv.partition_S(gV);
    auto tVsV = thr_copy_kv.partition_D(sV);

    // --- Predicate tensors for ragged Q / KV in seq dim ---
    auto cQ = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<HD>{}));
    auto tQcQ = thr_copy_q.partition_S(cQ);
    auto cK = make_identity_tensor(make_shape(Int<kBlockN>{}, Int<HD>{}));
    auto tKcK = thr_copy_kv.partition_S(cK);

    // -- Load Q (predicated in M) --
    #pragma unroll
    for (int m = 0; m < size<1>(tQgQ); ++m) {
        const int row = block_m * kBlockM + get<0>(tQcQ(0, m, 0));
        if (row < q_len) {
            cute::copy(copy_q, tQgQ(_, m, _), tQsQ(_, m, _));
        } else {
            cute::clear(tQsQ(_, m, _));
        }
    }
    cp_async_fence();

    // -- Launch first K tile (predicated in N) --
    auto load_kv_tile = [&](auto tXgX, auto tXsX, auto tXcX,
                            int n_tile, int kv_upper) {
        #pragma unroll
        for (int n = 0; n < size<1>(tXsX); ++n) {
            const int row = n_tile * kBlockN + get<0>(tXcX(0, n, 0));
            if (row < kv_upper) {
                cute::copy(copy_kv, tXgX(_, n, _, n_tile), tXsX(_, n, _));
            } else {
                cute::clear(tXsX(_, n, _));
            }
        }
    };
    load_kv_tile(tKgK, tKsK, tKcK, 0, kv_len);
    cp_async_fence();

    // -- Register fragments --
    auto thr_mma = mma.get_slice(threadIdx.x);
    auto rQ = thr_mma.partition_fragment_A(sQ);                 // Q fragments
    auto rK = thr_mma.partition_fragment_B(sK);                 // K fragments
    auto rS = partition_fragment_C(mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    auto rO = partition_fragment_C(mma, Shape<Int<kBlockM>, Int<HD>>{});
    auto rVt = thr_mma.partition_fragment_B(sVtNS);
    clear(rS); clear(rO);

    // -- smem -> reg copy atoms --
    auto s2r_q  = make_tiled_copy_A(typename Traits::S2RAtomAB{}, mma);
    auto s2r_k  = make_tiled_copy_B(typename Traits::S2RAtomAB{}, mma);
    auto s2r_v  = make_tiled_copy_B(typename Traits::S2RAtomVT{}, mma);
    auto thr_s2r_q = s2r_q.get_slice(threadIdx.x);
    auto thr_s2r_k = s2r_k.get_slice(threadIdx.x);
    auto thr_s2r_v = s2r_v.get_slice(threadIdx.x);
    auto tXsQ = thr_s2r_q.partition_S(sQ);
    auto tXrQ = thr_s2r_q.retile_D(rQ);
    auto tXsK = thr_s2r_k.partition_S(sK);
    auto tXrK = thr_s2r_k.retile_D(rK);
    auto tOsVt = thr_s2r_v.partition_S(sVt);

    // -- Softmax state --
    auto rS_rc_layout = to_rowcol(rS.layout());
    Tensor row_max = make_tensor<float>(Shape<Int<size<0>(rS_rc_layout)>>{});
    Tensor row_sum = make_tensor<float>(Shape<Int<size<0>(rS_rc_layout)>>{});
    CUTE_UNROLL
    for (int i = 0; i < size(row_max); ++i) { row_max(i) = -5e4f; row_sum(i) = 0.f; }

    const float scale_log2 = softmax_scale * float(M_LOG2E);

    // -- Wait for Q in smem, pull Q into registers (once) --
    cp_async_wait<1>();   // Q done; K tile 0 may still be in flight
    __syncthreads();
    cute::copy(s2r_q, tXsQ, tXrQ);

    // -- KV tile range (causal prune) --
    int n_block_max;
    if (is_causal) {
        const int max_k = (block_m + 1) * kBlockM + (kv_len - q_len);
        n_block_max = (max_k + kBlockN - 1) / kBlockN;
        const int total = (kv_len + kBlockN - 1) / kBlockN;
        n_block_max = min(n_block_max, total);
    } else {
        n_block_max = (kv_len + kBlockN - 1) / kBlockN;
    }
    if (n_block_max <= 0) {
        // Nothing to attend to (can happen for causal + empty KV). Write zeros.
        // Fallthrough: rO is already zero and row_sum=0 → epilogue writes zeros.
        n_block_max = 0;
    }

    const int warp_id     = threadIdx.x / 32;
    const int lane_id     = threadIdx.x & 31;
    const int row_idx_ofs = block_m * kBlockM + warp_id * 16 + lane_id / 4;

    // =========================== Main KV loop ===========================
    for (int nt = 0; nt < n_block_max; ++nt) {
        // Issue V load for this tile (overlap with QK^T).
        load_kv_tile(tVgV, tVsV, tKcK, nt, kv_len);
        cp_async_fence();

        // Wait for this tile's K to be ready.
        cp_async_wait<1>();   // V may still be in flight
        __syncthreads();

        // --- S = Q @ K^T ---
        clear(rS);
        cute::copy(s2r_k, tXsK, tXrK);
        cute::gemm(mma, rQ, rK, rS);

        // Issue next K prefetch (if any).
        if (nt + 1 < n_block_max) {
            load_kv_tile(tKgK, tKsK, tKcK, nt + 1, kv_len);
        }
        cp_async_fence();

        // --- Mask ---
        if (is_causal) {
            apply_causal_mask(rS, kBlockN * nt, row_idx_ofs,
                              16 * kNWarps, kv_len - q_len, kv_len);
        } else {
            // Only the last tile can be ragged in N.
            if ((nt + 1) * kBlockN > kv_len) {
                apply_col_bound_mask(rS, kBlockN * nt, kv_len);
            }
        }

        // --- Online softmax ---
        auto scores = make_tensor(rS.data(), to_rowcol(rS.layout()));
        if (nt == 0) {
            reduce_max_rows<true>(scores, row_max);
            scale_apply_exp2(scores, row_max, scale_log2);
            reduce_sum_rows<true>(scores, row_sum);
        } else {
            Tensor m_prev = make_fragment_like(row_max);
            cute::copy(row_max, m_prev);
            reduce_max_rows<false>(scores, row_max);
            auto rO_rc = make_tensor(rO.data(), to_rowcol(rO.layout()));
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                const float m_cur = (row_max(mi) == -INFINITY) ? 0.f : row_max(mi);
                const float sc    = exp2f((m_prev(mi) - m_cur) * scale_log2);
                row_sum(mi) *= sc;
                #pragma unroll
                for (int ni = 0; ni < size<1>(rO_rc); ++ni) rO_rc(mi, ni) *= sc;
            }
            scale_apply_exp2(scores, row_max, scale_log2);
            reduce_sum_rows<false>(scores, row_sum);
        }

        // --- Wait for V to be ready, then acc += P @ V ---
        cp_async_wait<1>();   // keep the next K prefetch in flight
        __syncthreads();

        auto rP = convert_type<Elem>(rS);
        auto tOrP = make_tensor(
            rP.data(),
            to_A_regs<typename Traits::Mma>(rP.layout()));
        gemm_rs(rO, tOrP, rVt, tOsVt, mma, s2r_v, thr_s2r_v);
    }

    // Drain remaining cp.async groups (next-K prefetch from the last iteration).
    cp_async_wait<0>();
    __syncthreads();

    // --- Final softmax normalization: row_sum needs warp-group reduce ---
    SumOp sum_op;
    quad_allreduce(row_sum, row_sum, sum_op);
    auto rO_rc = make_tensor(rO.data(), to_rowcol(rO.layout()));
    #pragma unroll
    for (int mi = 0; mi < size<0>(rO_rc); ++mi) {
        const float s = row_sum(mi);
        const float inv = (s == 0.f || s != s) ? 1.f : (1.f / s);
        #pragma unroll
        for (int ni = 0; ni < size<1>(rO_rc); ++ni) rO_rc(mi, ni) *= inv;
    }

    // --- Write back: reg -> smem -> gmem (reusing Q's smem buffer) ---
    auto rO_out = convert_type<Elem>(rO);
    auto sO = make_tensor(sQ.data(), typename Traits::SmemLayoutO{});

    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Elem>;
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(threadIdx.x);
    auto taccOrO = smem_thr_copy_O.retile_S(rO_out);
    auto taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    __syncthreads();

    auto thr_copy_o = copy_o_gmem.get_slice(threadIdx.x);
    auto tOsO = thr_copy_o.partition_S(sO);
    auto tOgO = thr_copy_o.partition_D(gO);
    auto rO_tmp = make_tensor<Elem>(shape(tOgO));
    cute::copy(copy_o_gmem, tOsO, rO_tmp);

    auto cO   = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<HD>{}));
    auto tOcO = thr_copy_o.partition_D(cO);
    #pragma unroll
    for (int i = 0; i < size(rO_tmp); ++i) {
        const int m = get<0>(tOcO(i));
        if (block_m * kBlockM + m < q_len) {
            tOgO(i) = rO_tmp(i);
        }
    }
}

// =============================================================================
// Launcher (templated)
// =============================================================================
template <class Elem, int HD>
static cudaError_t launch_impl(
    const Elem* q, int64_t qsb, int64_t qss, int64_t qsh,
    const Elem* k, int64_t ksb, int64_t kss, int64_t ksh,
    const Elem* v, int64_t vsb, int64_t vss, int64_t vsh,
          Elem* o, int64_t osb, int64_t oss, int64_t osh,
    int batch, int q_len, int kv_len,
    int num_q_heads, int num_kv_heads,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    using Traits = KTraits<Elem, HD>;
    const int m_blocks = (q_len + kBlockM - 1) / kBlockM;
    dim3 grid(m_blocks, num_q_heads, batch);
    dim3 block(Traits::NumThreads);

    const int smem_size = int(sizeof(SharedStorage<Elem,
                                 typename Traits::SmemLayoutQ,
                                 typename Traits::SmemLayoutKV>));

    auto kernel = flash_attn_prefill_kernel<Traits>;
    static rustinfer::cuda::PerDeviceKernelAttribute prefill_attr;
    const cudaError_t prefill_attr_err = prefill_attr.set_max_dynamic_shared_memory(
        reinterpret_cast<const void*>(kernel), smem_size);
    if (prefill_attr_err != cudaSuccess) return prefill_attr_err;

    kernel<<<grid, block, smem_size, stream>>>(
        q, qsb, qss, qsh,
        k, ksb, kss, ksh,
        v, vsb, vss, vsh,
        o, osb, oss, osh,
        q_len, kv_len, num_q_heads, num_kv_heads,
        softmax_scale, is_causal);
    return cudaGetLastError();
}

template <class Elem>
static cudaError_t launch_dispatch(
    const Elem* q, int64_t qsb, int64_t qss, int64_t qsh,
    const Elem* k, int64_t ksb, int64_t kss, int64_t ksh,
    const Elem* v, int64_t vsb, int64_t vss, int64_t vsh,
          Elem* o, int64_t osb, int64_t oss, int64_t osh,
    int batch, int q_len, int kv_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    switch (head_dim) {
    case 64:
        return launch_impl<Elem, 64>(q,qsb,qss,qsh, k,ksb,kss,ksh, v,vsb,vss,vsh,
                                     o,osb,oss,osh, batch,q_len,kv_len,
                                     num_q_heads,num_kv_heads, softmax_scale,is_causal,stream);
    case 128:
        return launch_impl<Elem,128>(q,qsb,qss,qsh, k,ksb,kss,ksh, v,vsb,vss,vsh,
                                     o,osb,oss,osh, batch,q_len,kv_len,
                                     num_q_heads,num_kv_heads, softmax_scale,is_causal,stream);
    case 192:
        return launch_impl<Elem,192>(q,qsb,qss,qsh, k,ksb,kss,ksh, v,vsb,vss,vsh,
                                     o,osb,oss,osh, batch,q_len,kv_len,
                                     num_q_heads,num_kv_heads, softmax_scale,is_causal,stream);
    case 256:
        return launch_impl<Elem,256>(q,qsb,qss,qsh, k,ksb,kss,ksh, v,vsb,vss,vsh,
                                     o,osb,oss,osh, batch,q_len,kv_len,
                                     num_q_heads,num_kv_heads, softmax_scale,is_causal,stream);
    default:
        fprintf(stderr, "[flash_attn_prefill] unsupported head_dim=%d; "
                        "supported: 64, 128, 192, 256\n", head_dim);
        return cudaErrorInvalidValue;
    }
}

// =============================================================================
// Ragged kernel
//   grid = (total_q_tiles, num_q_heads)  — each block handles (req, q_tile, qh)
//   Q / O are packed over the batch: [total_q_tokens, num_q_heads, head_dim]
//   K / V live in independent per-slot cache buffers, addressed via a device
//   pointer array.  All control arrays (req_to_slot, kv_lens, cu_q_lens,
//   block2req, block2tile) have stable addresses so CUDA Graphs can capture
//   the launch.
// =============================================================================
template <class Traits>
__global__ void flash_attn_ragged_kernel(
    const typename Traits::Elem* __restrict__ q_ptr,    // [total_q, Hq, HD] packed
    int64_t q_stride_s, int64_t q_stride_h,
    const typename Traits::Elem* const* __restrict__ k_cache_ptrs,
    const typename Traits::Elem* const* __restrict__ v_cache_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          typename Traits::Elem* __restrict__ o_ptr,    // [total_q, Hq, HD] packed
    int64_t o_stride_s, int64_t o_stride_h,
    const int32_t* __restrict__ req_to_slot,            // [B]
    const int32_t* __restrict__ kv_lens,                // [B]
    const int32_t* __restrict__ cu_q_lens,              // [B+1]
    const int32_t* __restrict__ block2req,              // [total_q_tiles]
    const int32_t* __restrict__ block2tile,             // [total_q_tiles]
    int num_q_heads, int num_kv_heads,
    float softmax_scale,
    int is_causal)
{
    using Elem = typename Traits::Elem;
    constexpr int HD = Traits::HeadDim;

    const int flat_tile  = blockIdx.x;
    const int q_head_idx = blockIdx.y;

    // Lookup which request / q-tile-within-request this CTA owns.
    const int req     = block2req[flat_tile];
    const int block_m = block2tile[flat_tile];

    const int q_start = cu_q_lens[req];
    const int q_end   = cu_q_lens[req + 1];
    const int q_len   = q_end - q_start;
    const int kv_len  = kv_lens[req];
    if (q_len <= 0) return;

    const int slot = req_to_slot[req];
    const int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);

    // Per-request base pointers.
    //   Q / O are packed [total_q_tokens, Hq, HD], so we jump by
    //   q_start * q_stride_s (token offset) and then by q_head_idx * q_stride_h.
    const Elem* q_bh = q_ptr + (int64_t)q_start * q_stride_s + (int64_t)q_head_idx  * q_stride_h;
          Elem* o_bh = o_ptr + (int64_t)q_start * o_stride_s + (int64_t)q_head_idx  * o_stride_h;
    //   K / V point to a per-slot cache buffer; jump only over KV-head.
    const Elem* k_bh = k_cache_ptrs[slot] + (int64_t)kv_head_idx * kv_stride_h;
    const Elem* v_bh = v_cache_ptrs[slot] + (int64_t)kv_head_idx * kv_stride_h;

    // Global tensors.
    auto Q = make_tensor(make_gmem_ptr(q_bh),
                         make_shape(q_len, Int<HD>{}),
                         make_stride(q_stride_s, _1{}));
    auto K = make_tensor(make_gmem_ptr(k_bh),
                         make_shape(kv_len, Int<HD>{}),
                         make_stride(kv_stride_s, _1{}));
    auto V = make_tensor(make_gmem_ptr(v_bh),
                         make_shape(kv_len, Int<HD>{}),
                         make_stride(kv_stride_s, _1{}));
    auto O = make_tensor(make_gmem_ptr(o_bh),
                         make_shape(q_len, Int<HD>{}),
                         make_stride(o_stride_s, _1{}));

    auto gQ = local_tile(Q, Shape<Int<kBlockM>, Int<HD>>{}, make_coord(block_m, 0));
    auto gK = local_tile(K, Shape<Int<kBlockN>, Int<HD>>{}, make_coord(_,       0));
    auto gV = local_tile(V, Shape<Int<kBlockN>, Int<HD>>{}, make_coord(_,       0));
    auto gO = local_tile(O, Shape<Int<kBlockM>, Int<HD>>{}, make_coord(block_m, 0));

    // ------------------------------------------------------------------
    // From here on: identical structure to `flash_attn_prefill_kernel`.
    // ------------------------------------------------------------------

    extern __shared__ __align__(16) unsigned char smem_raw[];
    using SharedT = SharedStorage<Elem,
                                  typename Traits::SmemLayoutQ,
                                  typename Traits::SmemLayoutKV>;
    SharedT& smem = *reinterpret_cast<SharedT*>(smem_raw);

    auto sQ    = make_tensor(make_smem_ptr(smem.smem_q.begin()), typename Traits::SmemLayoutQ{});
    auto sK    = make_tensor(make_smem_ptr(smem.smem_k.begin()), typename Traits::SmemLayoutKV{});
    auto sV    = make_tensor(make_smem_ptr(smem.smem_v.begin()), typename Traits::SmemLayoutKV{});
    auto sVt   = make_tensor(sV.data(), typename Traits::SmemLayoutVt{});
    auto sVtNS = make_tensor(sV.data(), typename Traits::SmemLayoutVtNoSwi{});

    typename Traits::GmemTiledCopy  copy_q{};
    typename Traits::GmemTiledCopy  copy_kv{};
    typename Traits::GmemTiledCopyO copy_o_gmem{};
    typename Traits::Mma            mma{};

    auto thr_copy_q  = copy_q .get_slice(threadIdx.x);
    auto thr_copy_kv = copy_kv.get_slice(threadIdx.x);

    auto tQgQ = thr_copy_q .partition_S(gQ);
    auto tQsQ = thr_copy_q .partition_D(sQ);
    auto tKgK = thr_copy_kv.partition_S(gK);
    auto tKsK = thr_copy_kv.partition_D(sK);
    auto tVgV = thr_copy_kv.partition_S(gV);
    auto tVsV = thr_copy_kv.partition_D(sV);

    auto cQ = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<HD>{}));
    auto tQcQ = thr_copy_q.partition_S(cQ);
    auto cK = make_identity_tensor(make_shape(Int<kBlockN>{}, Int<HD>{}));
    auto tKcK = thr_copy_kv.partition_S(cK);

    // Load Q (predicated in M).
    #pragma unroll
    for (int m = 0; m < size<1>(tQgQ); ++m) {
        const int row = block_m * kBlockM + get<0>(tQcQ(0, m, 0));
        if (row < q_len) {
            cute::copy(copy_q, tQgQ(_, m, _), tQsQ(_, m, _));
        } else {
            cute::clear(tQsQ(_, m, _));
        }
    }
    cp_async_fence();

    auto load_kv_tile = [&](auto tXgX, auto tXsX, auto tXcX,
                            int n_tile, int kv_upper) {
        #pragma unroll
        for (int n = 0; n < size<1>(tXsX); ++n) {
            const int row = n_tile * kBlockN + get<0>(tXcX(0, n, 0));
            if (row < kv_upper) {
                cute::copy(copy_kv, tXgX(_, n, _, n_tile), tXsX(_, n, _));
            } else {
                cute::clear(tXsX(_, n, _));
            }
        }
    };
    load_kv_tile(tKgK, tKsK, tKcK, 0, kv_len);
    cp_async_fence();

    auto thr_mma = mma.get_slice(threadIdx.x);
    auto rQ  = thr_mma.partition_fragment_A(sQ);
    auto rK  = thr_mma.partition_fragment_B(sK);
    auto rS  = partition_fragment_C(mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    auto rO  = partition_fragment_C(mma, Shape<Int<kBlockM>, Int<HD>>{});
    auto rVt = thr_mma.partition_fragment_B(sVtNS);
    clear(rS); clear(rO);

    auto s2r_q  = make_tiled_copy_A(typename Traits::S2RAtomAB{}, mma);
    auto s2r_k  = make_tiled_copy_B(typename Traits::S2RAtomAB{}, mma);
    auto s2r_v  = make_tiled_copy_B(typename Traits::S2RAtomVT{}, mma);
    auto thr_s2r_q = s2r_q.get_slice(threadIdx.x);
    auto thr_s2r_k = s2r_k.get_slice(threadIdx.x);
    auto thr_s2r_v = s2r_v.get_slice(threadIdx.x);
    auto tXsQ = thr_s2r_q.partition_S(sQ);
    auto tXrQ = thr_s2r_q.retile_D(rQ);
    auto tXsK = thr_s2r_k.partition_S(sK);
    auto tXrK = thr_s2r_k.retile_D(rK);
    auto tOsVt = thr_s2r_v.partition_S(sVt);

    auto rS_rc_layout = to_rowcol(rS.layout());
    Tensor row_max = make_tensor<float>(Shape<Int<size<0>(rS_rc_layout)>>{});
    Tensor row_sum = make_tensor<float>(Shape<Int<size<0>(rS_rc_layout)>>{});
    CUTE_UNROLL
    for (int i = 0; i < size(row_max); ++i) { row_max(i) = -5e4f; row_sum(i) = 0.f; }

    const float scale_log2 = softmax_scale * float(M_LOG2E);

    cp_async_wait<1>();
    __syncthreads();
    cute::copy(s2r_q, tXsQ, tXrQ);

    // KV tile range (causal pruning is per-request).
    int n_block_max;
    if (is_causal) {
        const int max_k = (block_m + 1) * kBlockM + (kv_len - q_len);
        n_block_max = (max_k + kBlockN - 1) / kBlockN;
        const int total = (kv_len + kBlockN - 1) / kBlockN;
        n_block_max = min(n_block_max, total);
    } else {
        n_block_max = (kv_len + kBlockN - 1) / kBlockN;
    }
    if (n_block_max <= 0) n_block_max = 0;

    const int warp_id     = threadIdx.x / 32;
    const int lane_id     = threadIdx.x & 31;
    const int row_idx_ofs = block_m * kBlockM + warp_id * 16 + lane_id / 4;

    for (int nt = 0; nt < n_block_max; ++nt) {
        load_kv_tile(tVgV, tVsV, tKcK, nt, kv_len);
        cp_async_fence();
        cp_async_wait<1>();
        __syncthreads();

        clear(rS);
        cute::copy(s2r_k, tXsK, tXrK);
        cute::gemm(mma, rQ, rK, rS);

        if (nt + 1 < n_block_max) {
            load_kv_tile(tKgK, tKsK, tKcK, nt + 1, kv_len);
        }
        cp_async_fence();

        if (is_causal) {
            apply_causal_mask(rS, kBlockN * nt, row_idx_ofs,
                              16 * kNWarps, kv_len - q_len, kv_len);
        } else {
            if ((nt + 1) * kBlockN > kv_len) {
                apply_col_bound_mask(rS, kBlockN * nt, kv_len);
            }
        }

        auto scores = make_tensor(rS.data(), to_rowcol(rS.layout()));
        if (nt == 0) {
            reduce_max_rows<true>(scores, row_max);
            scale_apply_exp2(scores, row_max, scale_log2);
            reduce_sum_rows<true>(scores, row_sum);
        } else {
            Tensor m_prev = make_fragment_like(row_max);
            cute::copy(row_max, m_prev);
            reduce_max_rows<false>(scores, row_max);
            auto rO_rc = make_tensor(rO.data(), to_rowcol(rO.layout()));
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                const float m_cur = (row_max(mi) == -INFINITY) ? 0.f : row_max(mi);
                const float sc    = exp2f((m_prev(mi) - m_cur) * scale_log2);
                row_sum(mi) *= sc;
                #pragma unroll
                for (int ni = 0; ni < size<1>(rO_rc); ++ni) rO_rc(mi, ni) *= sc;
            }
            scale_apply_exp2(scores, row_max, scale_log2);
            reduce_sum_rows<false>(scores, row_sum);
        }

        cp_async_wait<1>();
        __syncthreads();

        auto rP = convert_type<Elem>(rS);
        auto tOrP = make_tensor(
            rP.data(),
            to_A_regs<typename Traits::Mma>(rP.layout()));
        gemm_rs(rO, tOrP, rVt, tOsVt, mma, s2r_v, thr_s2r_v);
    }

    cp_async_wait<0>();
    __syncthreads();

    SumOp sum_op;
    quad_allreduce(row_sum, row_sum, sum_op);
    auto rO_rc = make_tensor(rO.data(), to_rowcol(rO.layout()));
    #pragma unroll
    for (int mi = 0; mi < size<0>(rO_rc); ++mi) {
        const float s = row_sum(mi);
        const float inv = (s == 0.f || s != s) ? 1.f : (1.f / s);
        #pragma unroll
        for (int ni = 0; ni < size<1>(rO_rc); ++ni) rO_rc(mi, ni) *= inv;
    }

    auto rO_out = convert_type<Elem>(rO);
    auto sO = make_tensor(sQ.data(), typename Traits::SmemLayoutO{});

    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Elem>;
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(threadIdx.x);
    auto taccOrO = smem_thr_copy_O.retile_S(rO_out);
    auto taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    __syncthreads();

    auto thr_copy_o = copy_o_gmem.get_slice(threadIdx.x);
    auto tOsO = thr_copy_o.partition_S(sO);
    auto tOgO = thr_copy_o.partition_D(gO);
    auto rO_tmp = make_tensor<Elem>(shape(tOgO));
    cute::copy(copy_o_gmem, tOsO, rO_tmp);

    auto cO   = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<HD>{}));
    auto tOcO = thr_copy_o.partition_D(cO);
    #pragma unroll
    for (int i = 0; i < size(rO_tmp); ++i) {
        const int m = get<0>(tOcO(i));
        if (block_m * kBlockM + m < q_len) {
            tOgO(i) = rO_tmp(i);
        }
    }
}

// =============================================================================
// Ragged launcher
// =============================================================================
template <class Elem, int HD>
static cudaError_t launch_ragged_impl(
    const Elem* q, int64_t qss, int64_t qsh,
    const Elem* const* k_cache_ptrs,
    const Elem* const* v_cache_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          Elem* o, int64_t oss, int64_t osh,
    const int32_t* req_to_slot,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    const int32_t* block2req,
    const int32_t* block2tile,
    int total_q_tiles,
    int num_q_heads, int num_kv_heads,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    using Traits = KTraits<Elem, HD>;
    if (total_q_tiles <= 0) return cudaSuccess;
    dim3 grid(total_q_tiles, num_q_heads, 1);
    dim3 block(Traits::NumThreads);

    const int smem_size = int(sizeof(SharedStorage<Elem,
                                 typename Traits::SmemLayoutQ,
                                 typename Traits::SmemLayoutKV>));
    auto kernel = flash_attn_ragged_kernel<Traits>;
    static rustinfer::cuda::PerDeviceKernelAttribute ragged_attr;
    const cudaError_t ragged_attr_err = ragged_attr.set_max_dynamic_shared_memory(
        reinterpret_cast<const void*>(kernel), smem_size);
    if (ragged_attr_err != cudaSuccess) return ragged_attr_err;

    kernel<<<grid, block, smem_size, stream>>>(
        q, qss, qsh,
        k_cache_ptrs, v_cache_ptrs, kv_stride_s, kv_stride_h,
        o, oss, osh,
        req_to_slot, kv_lens, cu_q_lens, block2req, block2tile,
        num_q_heads, num_kv_heads,
        softmax_scale, is_causal);
    return cudaGetLastError();
}

template <class Elem>
static cudaError_t launch_ragged_dispatch(
    const Elem* q, int64_t qss, int64_t qsh,
    const Elem* const* k_cache_ptrs,
    const Elem* const* v_cache_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          Elem* o, int64_t oss, int64_t osh,
    const int32_t* req_to_slot,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    const int32_t* block2req,
    const int32_t* block2tile,
    int total_q_tiles,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    switch (head_dim) {
    case 64:  return launch_ragged_impl<Elem, 64>(q,qss,qsh, k_cache_ptrs,v_cache_ptrs,
                                                  kv_stride_s,kv_stride_h,
                                                  o,oss,osh,
                                                  req_to_slot,kv_lens,cu_q_lens,
                                                  block2req,block2tile,total_q_tiles,
                                                  num_q_heads,num_kv_heads,
                                                  softmax_scale,is_causal,stream);
    case 128: return launch_ragged_impl<Elem,128>(q,qss,qsh, k_cache_ptrs,v_cache_ptrs,
                                                  kv_stride_s,kv_stride_h,
                                                  o,oss,osh,
                                                  req_to_slot,kv_lens,cu_q_lens,
                                                  block2req,block2tile,total_q_tiles,
                                                  num_q_heads,num_kv_heads,
                                                  softmax_scale,is_causal,stream);
    case 192: return launch_ragged_impl<Elem,192>(q,qss,qsh, k_cache_ptrs,v_cache_ptrs,
                                                  kv_stride_s,kv_stride_h,
                                                  o,oss,osh,
                                                  req_to_slot,kv_lens,cu_q_lens,
                                                  block2req,block2tile,total_q_tiles,
                                                  num_q_heads,num_kv_heads,
                                                  softmax_scale,is_causal,stream);
    case 256: return launch_ragged_impl<Elem,256>(q,qss,qsh, k_cache_ptrs,v_cache_ptrs,
                                                  kv_stride_s,kv_stride_h,
                                                  o,oss,osh,
                                                  req_to_slot,kv_lens,cu_q_lens,
                                                  block2req,block2tile,total_q_tiles,
                                                  num_q_heads,num_kv_heads,
                                                  softmax_scale,is_causal,stream);
    default:
        fprintf(stderr, "[flash_attn_ragged] unsupported head_dim=%d; "
                        "supported: 64, 128, 192, 256\n", head_dim);
        return cudaErrorInvalidValue;
    }
}

} // namespace flash_attn_prefill

// =============================================================================
// Public C ABI
// =============================================================================
extern "C" {

void launch_flash_attn_prefill_bf16(
    const __nv_bfloat16* q, int64_t qsb, int64_t qss, int64_t qsh,
    const __nv_bfloat16* k, int64_t ksb, int64_t kss, int64_t ksh,
    const __nv_bfloat16* v, int64_t vsb, int64_t vss, int64_t vsh,
          __nv_bfloat16* o, int64_t osb, int64_t oss, int64_t osh,
    int batch, int q_len, int kv_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    cudaError_t err = flash_attn_prefill::launch_dispatch<__nv_bfloat16>(
        q,qsb,qss,qsh, k,ksb,kss,ksh, v,vsb,vss,vsh, o,osb,oss,osh,
        batch,q_len,kv_len,num_q_heads,num_kv_heads,head_dim,
        softmax_scale,is_causal,stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[flash_attn_prefill_bf16] launch error: %s\n",
                cudaGetErrorString(err));
    }
}

void launch_flash_attn_prefill_fp16(
    const __half* q, int64_t qsb, int64_t qss, int64_t qsh,
    const __half* k, int64_t ksb, int64_t kss, int64_t ksh,
    const __half* v, int64_t vsb, int64_t vss, int64_t vsh,
          __half* o, int64_t osb, int64_t oss, int64_t osh,
    int batch, int q_len, int kv_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    cudaError_t err = flash_attn_prefill::launch_dispatch<__half>(
        q,qsb,qss,qsh, k,ksb,kss,ksh, v,vsb,vss,vsh, o,osb,oss,osh,
        batch,q_len,kv_len,num_q_heads,num_kv_heads,head_dim,
        softmax_scale,is_causal,stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[flash_attn_prefill_fp16] launch error: %s\n",
                cudaGetErrorString(err));
    }
}

// ---- Legacy shims removed. Use launch_flash_attn_prefill_{bf16,fp16} only. ----

void launch_flash_attn_ragged_bf16(
    const __nv_bfloat16* q, int64_t qss, int64_t qsh,
    const __nv_bfloat16* const* k_cache_ptrs,
    const __nv_bfloat16* const* v_cache_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          __nv_bfloat16* o, int64_t oss, int64_t osh,
    const int32_t* req_to_slot,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    const int32_t* block2req,
    const int32_t* block2tile,
    int total_q_tiles,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    cudaError_t err = flash_attn_prefill::launch_ragged_dispatch<__nv_bfloat16>(
        q, qss, qsh,
        k_cache_ptrs, v_cache_ptrs, kv_stride_s, kv_stride_h,
        o, oss, osh,
        req_to_slot, kv_lens, cu_q_lens, block2req, block2tile, total_q_tiles,
        num_q_heads, num_kv_heads, head_dim,
        softmax_scale, is_causal, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[flash_attn_ragged_bf16] launch error: %s\n",
                cudaGetErrorString(err));
    }
}

void launch_flash_attn_ragged_fp16(
    const __half* q, int64_t qss, int64_t qsh,
    const __half* const* k_cache_ptrs,
    const __half* const* v_cache_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          __half* o, int64_t oss, int64_t osh,
    const int32_t* req_to_slot,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    const int32_t* block2req,
    const int32_t* block2tile,
    int total_q_tiles,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    cudaError_t err = flash_attn_prefill::launch_ragged_dispatch<__half>(
        q, qss, qsh,
        k_cache_ptrs, v_cache_ptrs, kv_stride_s, kv_stride_h,
        o, oss, osh,
        req_to_slot, kv_lens, cu_q_lens, block2req, block2tile, total_q_tiles,
        num_q_heads, num_kv_heads, head_dim,
        softmax_scale, is_causal, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[flash_attn_ragged_fp16] launch error: %s\n",
                cudaGetErrorString(err));
    }
}

} // extern "C"

// =============================================================================
// Stand-alone C++ test main:
//   nvcc -std=c++17 -arch=sm_80 -O3 -I<cutlass-include> \
//        -DFLASH_ATTN_PREFILL_STANDALONE_TEST flash_attn_gqa_prefill.cu \
//        -o flash_attn_prefill_test
//   ./flash_attn_prefill_test
// =============================================================================
#ifdef FLASH_ATTN_PREFILL_STANDALONE_TEST

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace test {

#define CUDA_MUST(expr) do {                                                    \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
        std::cerr << "CUDA error " << cudaGetErrorString(_e)                    \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

// Naive f32 reference Attention on device (one block per (batch, q_head),
// one thread per q_row). Slow but correct.
__global__ void naive_attn_ref_kernel(
    const float* __restrict__ Q, int64_t qsb, int64_t qss, int64_t qsh,
    const float* __restrict__ K, int64_t ksb, int64_t kss, int64_t ksh,
    const float* __restrict__ V, int64_t vsb, int64_t vss, int64_t vsh,
          float* __restrict__ O, int64_t osb, int64_t oss, int64_t osh,
    int q_len, int kv_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    float scale, int is_causal)
{
    const int bid = blockIdx.z;
    const int qh  = blockIdx.y;
    const int kvh = qh / (num_q_heads / num_kv_heads);
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= q_len) return;

    const float* q = Q + bid*qsb + qh *qsh + row * qss;
          float* o = O + bid*osb + qh *osh + row * oss;

    const int causal_shift = kv_len - q_len;
    const int k_upper = is_causal ? min(kv_len, row + 1 + causal_shift) : kv_len;

    // Pass 1: logits max
    float m = -INFINITY;
    for (int t = 0; t < k_upper; ++t) {
        const float* k = K + bid*ksb + kvh*ksh + t * kss;
        float s = 0.f;
        for (int d = 0; d < head_dim; ++d) s += q[d] * k[d];
        s *= scale;
        if (s > m) m = s;
    }
    if (m == -INFINITY) {
        for (int d = 0; d < head_dim; ++d) o[d] = 0.f;
        return;
    }
    // Pass 2: weighted sum
    float denom = 0.f;
    for (int d = 0; d < head_dim; ++d) o[d] = 0.f;
    for (int t = 0; t < k_upper; ++t) {
        const float* k = K + bid*ksb + kvh*ksh + t * kss;
        const float* v = V + bid*vsb + kvh*vsh + t * vss;
        float s = 0.f;
        for (int d = 0; d < head_dim; ++d) s += q[d] * k[d];
        s = __expf(s * scale - m);
        denom += s;
        for (int d = 0; d < head_dim; ++d) o[d] += s * v[d];
    }
    const float inv = (denom == 0.f) ? 1.f : 1.f / denom;
    for (int d = 0; d < head_dim; ++d) o[d] *= inv;
}

template <class HT /* bf16 or fp16 host-visible proxy: use uint16_t bit-cast */>
struct HalfType;
template<> struct HalfType<__nv_bfloat16> {
    static __nv_bfloat16 from_f32(float x) { return __float2bfloat16(x); }
    static float         to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
    static const char*   name() { return "bf16"; }
};
template<> struct HalfType<__half> {
    static __half from_f32(float x) { return __float2half(x); }
    static float  to_f32(__half x)  { return __half2float(x); }
    static const char* name() { return "fp16"; }
};

struct Case {
    int B, Hq, Hkv, HD, Qn, Kn;
    bool causal;
    // If true, Q/O carry an extra padding stride (non-contiguous view test).
    bool q_padded;
    bool kv_padded;
    const char* tag;
};

template <class Elem>
bool run_case(const Case& c, std::mt19937& rng) {
    auto scale = 1.f / std::sqrt((float)c.HD);

    // Host storage in f32 for reference and for value generation.
    const size_t q_n  = (size_t)c.B * c.Qn * c.Hq  * c.HD;
    const size_t kv_n = (size_t)c.B * c.Kn * c.Hkv * c.HD;

    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> q_f(q_n), k_f(kv_n), v_f(kv_n);
    for (auto& x : q_f) x = dist(rng);
    for (auto& x : k_f) x = dist(rng);
    for (auto& x : v_f) x = dist(rng);

    // Build low-precision device buffers, potentially with "padding" in the
    // head dimension to make strides non-contiguous.
    const int q_pad_heads  = c.q_padded  ? (c.Hq  + 3) : c.Hq;
    const int kv_pad_heads = c.kv_padded ? (c.Hkv + 3) : c.Hkv;

    const int64_t qsh = c.HD;
    const int64_t qss = (int64_t)q_pad_heads * c.HD;
    const int64_t qsb = (int64_t)c.Qn * qss;
    const int64_t ksh = c.HD;
    const int64_t kss = (int64_t)kv_pad_heads * c.HD;
    const int64_t ksb = (int64_t)c.Kn * kss;
    // O uses the same stride as Q (same padding) so that kernel's stride path
    // is exercised on the output side as well.
    const int64_t osh = c.HD;
    const int64_t oss = qss;
    const int64_t osb = qsb;

    const size_t q_buf_elems = (size_t)c.B * qsb;
    const size_t k_buf_elems = (size_t)c.B * ksb;

    std::vector<Elem>  h_q(q_buf_elems, HalfType<Elem>::from_f32(0.f));
    std::vector<Elem>  h_k(k_buf_elems, HalfType<Elem>::from_f32(0.f));
    std::vector<Elem>  h_v(k_buf_elems, HalfType<Elem>::from_f32(0.f));
    std::vector<Elem>  h_o(q_buf_elems, HalfType<Elem>::from_f32(0.f));

    // Also build full f32 mirrors (with the same padding!) for the reference.
    std::vector<float> qf_full(q_buf_elems, 0.f);
    std::vector<float> kf_full(k_buf_elems, 0.f);
    std::vector<float> vf_full(k_buf_elems, 0.f);
    std::vector<float> of_full(q_buf_elems, 0.f);

    auto idx_q = [&](int b, int s, int h, int d) -> size_t {
        return (size_t)b * qsb + (size_t)s * qss + (size_t)h * qsh + d;
    };
    auto idx_k = [&](int b, int s, int h, int d) -> size_t {
        return (size_t)b * ksb + (size_t)s * kss + (size_t)h * ksh + d;
    };

    for (int b = 0; b < c.B; ++b) {
        for (int s = 0; s < c.Qn; ++s)
            for (int h = 0; h < c.Hq; ++h)
                for (int d = 0; d < c.HD; ++d) {
                    const size_t src = ((size_t)b*c.Qn*c.Hq + s*c.Hq + h) * c.HD + d;
                    const size_t dst = idx_q(b,s,h,d);
                    h_q[dst]    = HalfType<Elem>::from_f32(q_f[src]);
                    qf_full[dst] = q_f[src];
                }
        for (int s = 0; s < c.Kn; ++s)
            for (int h = 0; h < c.Hkv; ++h)
                for (int d = 0; d < c.HD; ++d) {
                    const size_t src = ((size_t)b*c.Kn*c.Hkv + s*c.Hkv + h) * c.HD + d;
                    const size_t dst = idx_k(b,s,h,d);
                    h_k[dst] = HalfType<Elem>::from_f32(k_f[src]);
                    h_v[dst] = HalfType<Elem>::from_f32(v_f[src]);
                    kf_full[dst] = k_f[src];
                    vf_full[dst] = v_f[src];
                }
    }

    // Device allocations
    Elem  *d_q=nullptr, *d_k=nullptr, *d_v=nullptr, *d_o=nullptr;
    float *d_qf=nullptr,*d_kf=nullptr,*d_vf=nullptr,*d_of=nullptr;
    CUDA_MUST(cudaMalloc(&d_q, h_q.size()*sizeof(Elem)));
    CUDA_MUST(cudaMalloc(&d_k, h_k.size()*sizeof(Elem)));
    CUDA_MUST(cudaMalloc(&d_v, h_v.size()*sizeof(Elem)));
    CUDA_MUST(cudaMalloc(&d_o, h_o.size()*sizeof(Elem)));
    CUDA_MUST(cudaMalloc(&d_qf, qf_full.size()*sizeof(float)));
    CUDA_MUST(cudaMalloc(&d_kf, kf_full.size()*sizeof(float)));
    CUDA_MUST(cudaMalloc(&d_vf, vf_full.size()*sizeof(float)));
    CUDA_MUST(cudaMalloc(&d_of, of_full.size()*sizeof(float)));

    CUDA_MUST(cudaMemcpy(d_q, h_q.data(), h_q.size()*sizeof(Elem), cudaMemcpyHostToDevice));
    CUDA_MUST(cudaMemcpy(d_k, h_k.data(), h_k.size()*sizeof(Elem), cudaMemcpyHostToDevice));
    CUDA_MUST(cudaMemcpy(d_v, h_v.data(), h_v.size()*sizeof(Elem), cudaMemcpyHostToDevice));
    CUDA_MUST(cudaMemset(d_o, 0,         h_o.size()*sizeof(Elem)));
    CUDA_MUST(cudaMemcpy(d_qf, qf_full.data(), qf_full.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_MUST(cudaMemcpy(d_kf, kf_full.data(), kf_full.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_MUST(cudaMemcpy(d_vf, vf_full.data(), vf_full.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_MUST(cudaMemset(d_of, 0,              of_full.size()*sizeof(float)));

    // --- Launch kernel under test ---
    if constexpr (std::is_same_v<Elem, __nv_bfloat16>) {
        launch_flash_attn_prefill_bf16(
            d_q,qsb,qss,qsh, d_k,ksb,kss,ksh, d_v,ksb,kss,ksh, d_o,osb,oss,osh,
            c.B, c.Qn, c.Kn, c.Hq, c.Hkv, c.HD,
            scale, c.causal ? 1 : 0, 0);
    } else {
        launch_flash_attn_prefill_fp16(
            d_q,qsb,qss,qsh, d_k,ksb,kss,ksh, d_v,ksb,kss,ksh, d_o,osb,oss,osh,
            c.B, c.Qn, c.Kn, c.Hq, c.Hkv, c.HD,
            scale, c.causal ? 1 : 0, 0);
    }
    CUDA_MUST(cudaDeviceSynchronize());

    // --- Reference on f32 ---
    const int TPB = 128;
    dim3 grid((c.Qn + TPB - 1) / TPB, c.Hq, c.B);
    naive_attn_ref_kernel<<<grid, TPB>>>(
        d_qf,qsb,qss,qsh, d_kf,ksb,kss,ksh, d_vf,ksb,kss,ksh, d_of,osb,oss,osh,
        c.Qn, c.Kn, c.Hq, c.Hkv, c.HD, scale, c.causal ? 1 : 0);
    CUDA_MUST(cudaDeviceSynchronize());

    // --- Compare ---
    CUDA_MUST(cudaMemcpy(h_o.data(),   d_o,  h_o.size()*sizeof(Elem),   cudaMemcpyDeviceToHost));
    CUDA_MUST(cudaMemcpy(of_full.data(), d_of, of_full.size()*sizeof(float), cudaMemcpyDeviceToHost));

    // tolerances (Q is bf16/fp16, accumulation in f32).
    const float atol = std::is_same_v<Elem, __nv_bfloat16> ? 5e-2f : 1e-2f;
    const float rtol = std::is_same_v<Elem, __nv_bfloat16> ? 1e-2f : 5e-3f;

    double max_abs = 0, sum_abs = 0;
    long   cnt = 0, bad = 0;
    for (int b = 0; b < c.B; ++b) {
        for (int s = 0; s < c.Qn; ++s) {
            for (int h = 0; h < c.Hq; ++h) {
                for (int d = 0; d < c.HD; ++d) {
                    const size_t p = idx_q(b,s,h,d);
                    const float got = HalfType<Elem>::to_f32(h_o[p]);
                    const float ref = of_full[p];
                    const float err = std::fabs(got - ref);
                    const float tol = atol + rtol * std::fabs(ref);
                    max_abs = std::max<double>(max_abs, err);
                    sum_abs += err;
                    ++cnt;
                    if (err > tol) ++bad;
                }
            }
        }
    }

    const bool ok = (bad == 0);
    std::printf("  [%s|%-6s] B=%d Hq=%d Hkv=%d HD=%d Qn=%d Kn=%d causal=%d "
                "q_pad=%d kv_pad=%d  max_err=%.4f mean_err=%.4f bad=%ld/%ld  %s\n",
                c.tag, HalfType<Elem>::name(), c.B, c.Hq, c.Hkv, c.HD, c.Qn, c.Kn,
                c.causal, c.q_padded, c.kv_padded,
                max_abs, sum_abs / std::max(1L, cnt), bad, cnt,
                ok ? "OK" : "FAIL");

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
    cudaFree(d_qf); cudaFree(d_kf); cudaFree(d_vf); cudaFree(d_of);
    return ok;
}

} // namespace test

int main() {
    std::mt19937 rng(0xC0DEBEEFu);

    // Matrix of cases: batch, heads, head_dim, seq lengths, causal, stride padding.
    std::vector<test::Case> cases = {
        // --- head_dim=64 ---
        {1, 4,  2,  64,  128, 128, true,  false, false, "hd64-B1-causal"},
        {1, 8,  2,  64,  160, 160, true,  false, false, "hd64-B1-ragged-causal"},
        {2, 4,  4,  64,  128, 256, false, false, false, "hd64-B2-cross"},
        {3, 8,  4,  64,  256, 256, true,  true,  true,  "hd64-B3-strided"},
        // --- head_dim=128 ---
        {1, 4,  4, 128,  128, 128, true,  false, false, "hd128-B1-causal"},
        {2, 8,  2, 128,  256, 384, false, true,  false, "hd128-B2-crossatt-strided"},
        {2, 8,  2, 128,  130, 300, true,  false, true,  "hd128-B2-ragged"},
        // --- head_dim=192 ---
        {1, 4,  4, 192,  128, 128, true,  false, false, "hd192-B1-causal"},
        {2, 8,  4, 192,  128, 200, false, true,  true,  "hd192-B2-strided"},
        // --- head_dim=256 ---
        {1, 4,  4, 256,  128, 128, true,  false, false, "hd256-B1-causal"},
    };

    bool all_ok = true;
    std::cout << "=== BF16 ===" << std::endl;
    for (const auto& c : cases) all_ok &= test::run_case<__nv_bfloat16>(c, rng);
    std::cout << "=== FP16 ===" << std::endl;
    for (const auto& c : cases) all_ok &= test::run_case<__half>(c, rng);

    std::cout << (all_ok ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    return all_ok ? 0 : 1;
}

#endif // FLASH_ATTN_PREFILL_STANDALONE_TEST
