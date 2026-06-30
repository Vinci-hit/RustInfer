// flash_attn_paged_prefill.cu
// -----------------------------------------------------------------------------
// Paged ragged/prefill attention over a global KV pool.
//
// Optimized wrappers use CuTe/SM80 MMA and a ragged Q-tile schedule: one CTA
// per (request, q_tile, q_head).  Q/O are packed affine tensors, while K/V are
// gathered from a paged pool through block_tables.
// -----------------------------------------------------------------------------

#include "flash_attn_gqa.h"
#include "flash_attn_paged_common.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <mutex>

namespace flash_attn_paged_prefill {

using namespace cute;

// =============================================================================
// Element -> MMA atom mapping and static kernel traits.
// =============================================================================
template <class Elem> struct MmaAtomFor;
template <> struct MmaAtomFor<__nv_bfloat16> {
    using type = SM80_16x8x16_F32BF16BF16F32_TN;
};
template <> struct MmaAtomFor<__half> {
    using type = SM80_16x8x16_F32F16F16F32_TN;
};

static constexpr int kNWarps = 4;
static constexpr int kBlockM = 128;
static constexpr int kBlockN = 64;

template <class Elem_, int HeadDim_>
struct KTraits {
    using Elem = Elem_;
    static constexpr int HeadDim = HeadDim_;
    static_assert(HeadDim % 64 == 0, "HeadDim must be a multiple of 64");

    using SmemAtom = decltype(composition(
        Swizzle<3,3,3>{},
        Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

    using SmemLayoutQ  = decltype(tile_to_shape(SmemAtom{},
                                      Shape<Int<kBlockM>, Int<HeadDim>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(SmemAtom{},
                                      Shape<Int<kBlockN>, Int<HeadDim>>{}));
    using SmemLayoutO  = SmemLayoutQ;

    using SmemLayoutVt =
        decltype(composition(SmemLayoutKV{},
                             make_layout(Shape<Int<HeadDim>, Int<kBlockN>>{},
                                         GenRowMajor{})));
    using SmemLayoutVtNoSwi = decltype(get_nonswizzle_portion(SmemLayoutVt{}));

    using GmemCopyOut = Copy_Atom<UniversalCopy<uint128_t>, Elem>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
        GmemCopyOut{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}));

    using S2RAtomAB = Copy_Atom<SM75_U32x4_LDSM_N, Elem>;
    using S2RAtomVT = Copy_Atom<SM75_U16x8_LDSM_T, Elem>;

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
// Device helpers copied from the CuTe prefill kernel.
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

template <class Layout>
__forceinline__ __device__ auto to_rowcol(Layout l) {
    static_assert(decltype(size<0>(l))::value == 4);
    static_assert(decltype(rank(l))::value == 3);
    auto x = logical_divide(l, Shape<_2>{});
    return make_layout(make_layout(get<0,1>(x), get<1>(x)),
                       make_layout(get<0,0>(x), get<2>(x)));
}

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

// =============================================================================
// Paged ragged CuTe kernel.
// =============================================================================
template <class Traits>
__launch_bounds__(128, 2)
__global__ void flash_attn_paged_ragged_kernel(
    const typename Traits::Elem* __restrict__ q_ptr,
    int64_t q_stride_s, int64_t q_stride_h,
    const typename Traits::Elem* __restrict__ k_pool,
    const typename Traits::Elem* __restrict__ v_pool,
          typename Traits::Elem* __restrict__ o_ptr,
    int64_t o_stride_s, int64_t o_stride_h,
    const uint32_t* __restrict__ block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* __restrict__ kv_lens,
    const int32_t* __restrict__ cu_q_lens,
    const int32_t* __restrict__ block2req,
    const int32_t* __restrict__ block2tile,
    const int32_t* __restrict__ valid_q_tiles,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    int is_causal)
{
    using Elem = typename Traits::Elem;
    constexpr int HD = Traits::HeadDim;

    const int flat_tile  = blockIdx.x;
    const int q_head_idx = blockIdx.y;
    if (valid_q_tiles != nullptr && flat_tile >= valid_q_tiles[0]) return;
    const int req     = block2req[flat_tile];
    const int block_m = block2tile[flat_tile];

    const int q_start = cu_q_lens[req];
    const int q_end   = cu_q_lens[req + 1];
    const int q_len   = q_end - q_start;
    const int kv_len  = kv_lens[req];
    if (q_len <= 0) return;

    const int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);
    const uint32_t* req_block_table = block_tables + static_cast<int64_t>(req) * max_blocks_per_seq;

    const Elem* q_bh = q_ptr + static_cast<int64_t>(q_start) * q_stride_s
                             + static_cast<int64_t>(q_head_idx) * q_stride_h;
          Elem* o_bh = o_ptr + static_cast<int64_t>(q_start) * o_stride_s
                             + static_cast<int64_t>(q_head_idx) * o_stride_h;

    auto O = make_tensor(make_gmem_ptr(o_bh),
                         make_shape(q_len, Int<HD>{}),
                         make_stride(o_stride_s, _1{}));
    auto gO = local_tile(O, Shape<Int<kBlockM>, Int<HD>>{}, make_coord(block_m, 0));

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

    // Synchronous scalar/vector-friendly global -> swizzled smem loads.  This
    // keeps the paged gather simple; once in smem, the MMA/softmax path is the
    // same CuTe tensor-core path as the regular ragged prefill kernel.
    for (int idx = threadIdx.x; idx < kBlockM * HD; idx += blockDim.x) {
        const int m = idx / HD;
        const int d = idx - m * HD;
        const int row = block_m * kBlockM + m;
        sQ(m, d) = (row < q_len) ? q_bh[static_cast<int64_t>(row) * q_stride_s + d]
                                 : Elem(0);
    }
    __syncthreads();

    auto load_paged_tile = [&](auto& sX, const Elem* pool, int n_tile) {
        for (int idx = threadIdx.x; idx < kBlockN * HD; idx += blockDim.x) {
            const int n = idx / HD;
            const int d = idx - n * HD;
            const int token = n_tile * kBlockN + n;
            Elem val = Elem(0);
            if (token < kv_len) {
                int logical_block = 0;
                int block_off = 0;
                token_to_page(token, block_size, logical_block, block_off);
                const uint32_t physical_block = req_block_table[logical_block];
                const Elem* row = paged_kv_row(
                    pool, physical_block, block_off, block_size,
                    num_kv_heads, kv_head_idx, HD);
                val = row[d];
            }
            sX(n, d) = val;
        }
    };

    typename Traits::Mma mma{};
    typename Traits::GmemTiledCopyO copy_o_gmem{};

    auto thr_mma = mma.get_slice(threadIdx.x);
    auto rQ  = thr_mma.partition_fragment_A(sQ);
    auto rK  = thr_mma.partition_fragment_B(sK);
    auto rS  = partition_fragment_C(mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    auto rO  = partition_fragment_C(mma, Shape<Int<kBlockM>, Int<HD>>{});
    auto rVt = thr_mma.partition_fragment_B(sVtNS);
    clear(rS);
    clear(rO);

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

    cute::copy(s2r_q, tXsQ, tXrQ);

    auto rS_rc_layout = to_rowcol(rS.layout());
    Tensor row_max = make_tensor<float>(Shape<Int<size<0>(rS_rc_layout)>>{});
    Tensor row_sum = make_tensor<float>(Shape<Int<size<0>(rS_rc_layout)>>{});
    CUTE_UNROLL
    for (int i = 0; i < size(row_max); ++i) { row_max(i) = -5e4f; row_sum(i) = 0.f; }

    const float scale_log2 = softmax_scale * float(M_LOG2E);

    int n_block_max;
    if (is_causal) {
        const int max_k = (block_m + 1) * kBlockM + (kv_len - q_len);
        n_block_max = (max_k + kBlockN - 1) / kBlockN;
        const int total = (kv_len + kBlockN - 1) / kBlockN;
        n_block_max = min(n_block_max, total);
    } else {
        n_block_max = (kv_len + kBlockN - 1) / kBlockN;
    }
    if (n_block_max < 0) n_block_max = 0;

    const int warp_id     = threadIdx.x / 32;
    const int lane_id     = threadIdx.x & 31;
    const int row_idx_ofs = block_m * kBlockM + warp_id * 16 + lane_id / 4;

    for (int nt = 0; nt < n_block_max; ++nt) {
        load_paged_tile(sK, k_pool, nt);
        load_paged_tile(sV, v_pool, nt);
        __syncthreads();

        clear(rS);
        cute::copy(s2r_k, tXsK, tXrK);
        cute::gemm(mma, rQ, rK, rS);

        if (is_causal) {
            apply_causal_mask(rS, kBlockN * nt, row_idx_ofs,
                              16 * kNWarps, kv_len - q_len, kv_len);
        } else if ((nt + 1) * kBlockN > kv_len) {
            apply_col_bound_mask(rS, kBlockN * nt, kv_len);
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
            auto rO_rc_prev = make_tensor(rO.data(), to_rowcol(rO.layout()));
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                const float m_cur = (row_max(mi) == -INFINITY) ? 0.f : row_max(mi);
                const float sc = exp2f((m_prev(mi) - m_cur) * scale_log2);
                row_sum(mi) *= sc;
                #pragma unroll
                for (int ni = 0; ni < size<1>(rO_rc_prev); ++ni) rO_rc_prev(mi, ni) *= sc;
            }
            scale_apply_exp2(scores, row_max, scale_log2);
            reduce_sum_rows<false>(scores, row_sum);
        }

        auto rP = convert_type<Elem>(rS);
        auto tOrP = make_tensor(rP.data(), to_A_regs<typename Traits::Mma>(rP.layout()));
        gemm_rs(rO, tOrP, rVt, tOsVt, mma, s2r_v, thr_s2r_v);
        __syncthreads();
    }

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

    auto cO = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<HD>{}));
    auto tOcO = thr_copy_o.partition_D(cO);
    #pragma unroll
    for (int i = 0; i < size(rO_tmp); ++i) {
        const int m = get<0>(tOcO(i));
        if (block_m * kBlockM + m < q_len) {
            tOgO(i) = rO_tmp(i);
        }
    }
}

template <class Elem, int HD>
static cudaError_t launch_impl(
    const Elem* q, int64_t qss, int64_t qsh,
    const Elem* k_pool,
    const Elem* v_pool,
          Elem* o, int64_t oss, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    const int32_t* block2req,
    const int32_t* block2tile,
    const int32_t* valid_q_tiles,
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
    auto kernel = flash_attn_paged_ragged_kernel<Traits>;
    static std::once_flag attr_once;
    static cudaError_t attr_err = cudaSuccess;
    std::call_once(attr_once, [&]() {
        attr_err = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    });
    if (attr_err != cudaSuccess) return attr_err;

    kernel<<<grid, block, smem_size, stream>>>(
        q, qss, qsh, k_pool, v_pool, o, oss, osh,
        block_tables, max_blocks_per_seq, block_size,
        kv_lens, cu_q_lens, block2req, block2tile, valid_q_tiles,
        num_q_heads, num_kv_heads, softmax_scale, is_causal);
    return cudaGetLastError();
}

template <class Elem>
static cudaError_t launch_dispatch(
    const Elem* q, int64_t qss, int64_t qsh,
    const Elem* k_pool,
    const Elem* v_pool,
          Elem* o, int64_t oss, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    const int32_t* block2req,
    const int32_t* block2tile,
    const int32_t* valid_q_tiles,
    int total_q_tiles,
    int batch,
    int total_q_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    if (batch <= 0 || total_q_tokens <= 0 || total_q_tiles <= 0) return cudaSuccess;
    switch (head_dim) {
    case 64:
        return launch_impl<Elem, 64>(q,qss,qsh,k_pool,v_pool,o,oss,osh,
                                     block_tables,max_blocks_per_seq,block_size,
                                     kv_lens,cu_q_lens,block2req,block2tile,valid_q_tiles,total_q_tiles,
                                     num_q_heads,num_kv_heads,softmax_scale,is_causal,stream);
    case 128:
        return launch_impl<Elem,128>(q,qss,qsh,k_pool,v_pool,o,oss,osh,
                                     block_tables,max_blocks_per_seq,block_size,
                                     kv_lens,cu_q_lens,block2req,block2tile,valid_q_tiles,total_q_tiles,
                                     num_q_heads,num_kv_heads,softmax_scale,is_causal,stream);
    case 192:
        return launch_impl<Elem,192>(q,qss,qsh,k_pool,v_pool,o,oss,osh,
                                     block_tables,max_blocks_per_seq,block_size,
                                     kv_lens,cu_q_lens,block2req,block2tile,valid_q_tiles,total_q_tiles,
                                     num_q_heads,num_kv_heads,softmax_scale,is_causal,stream);
    case 256:
        return launch_impl<Elem,256>(q,qss,qsh,k_pool,v_pool,o,oss,osh,
                                     block_tables,max_blocks_per_seq,block_size,
                                     kv_lens,cu_q_lens,block2req,block2tile,valid_q_tiles,total_q_tiles,
                                     num_q_heads,num_kv_heads,softmax_scale,is_causal,stream);
    default:
        fprintf(stderr, "[flash_attn_paged_ragged_cute] unsupported head_dim=%d (supported: 64, 128, 192, 256)\n", head_dim);
        return cudaErrorInvalidValue;
    }
}

} // namespace flash_attn_paged_prefill

extern "C" void launch_flash_attn_paged_ragged_bf16(
    const __nv_bfloat16* q, int64_t qss, int64_t qsh,
    const __nv_bfloat16* k_pool,
    const __nv_bfloat16* v_pool,
          __nv_bfloat16* o, int64_t oss, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    int batch, int total_q_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    (void)q; (void)qss; (void)qsh; (void)k_pool; (void)v_pool; (void)o; (void)oss; (void)osh;
    (void)block_tables; (void)max_blocks_per_seq; (void)block_size; (void)kv_lens; (void)cu_q_lens;
    (void)batch; (void)total_q_tokens; (void)num_q_heads; (void)num_kv_heads; (void)head_dim;
    (void)softmax_scale; (void)is_causal; (void)stream;
    fprintf(stderr, "[flash_attn_paged_ragged_bf16] legacy ABI removed; use launch_flash_attn_paged_ragged_cute_bf16 with block2req/block2tile\n");
}

extern "C" void launch_flash_attn_paged_ragged_fp16(
    const __half* q, int64_t qss, int64_t qsh,
    const __half* k_pool,
    const __half* v_pool,
          __half* o, int64_t oss, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    int batch, int total_q_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    (void)q; (void)qss; (void)qsh; (void)k_pool; (void)v_pool; (void)o; (void)oss; (void)osh;
    (void)block_tables; (void)max_blocks_per_seq; (void)block_size; (void)kv_lens; (void)cu_q_lens;
    (void)batch; (void)total_q_tokens; (void)num_q_heads; (void)num_kv_heads; (void)head_dim;
    (void)softmax_scale; (void)is_causal; (void)stream;
    fprintf(stderr, "[flash_attn_paged_ragged_fp16] legacy ABI removed; use launch_flash_attn_paged_ragged_cute_fp16 with block2req/block2tile\n");
}

extern "C" void launch_flash_attn_paged_ragged_cute_bf16(
    const __nv_bfloat16* q, int64_t qss, int64_t qsh,
    const __nv_bfloat16* k_pool,
    const __nv_bfloat16* v_pool,
          __nv_bfloat16* o, int64_t oss, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    const int32_t* block2req,
    const int32_t* block2tile,
    const int32_t* valid_q_tiles,
    int total_q_tiles,
    int batch, int total_q_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    cudaError_t err = flash_attn_paged_prefill::launch_dispatch<__nv_bfloat16>(
        q, qss, qsh, k_pool, v_pool, o, oss, osh,
        block_tables, max_blocks_per_seq, block_size, kv_lens, cu_q_lens,
        block2req, block2tile, valid_q_tiles, total_q_tiles,
        batch, total_q_tokens, num_q_heads, num_kv_heads, head_dim,
        softmax_scale, is_causal, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[flash_attn_paged_ragged_cute_bf16] launch error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_flash_attn_paged_ragged_cute_fp16(
    const __half* q, int64_t qss, int64_t qsh,
    const __half* k_pool,
    const __half* v_pool,
          __half* o, int64_t oss, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    const int32_t* block2req,
    const int32_t* block2tile,
    const int32_t* valid_q_tiles,
    int total_q_tiles,
    int batch, int total_q_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream)
{
    cudaError_t err = flash_attn_paged_prefill::launch_dispatch<__half>(
        q, qss, qsh, k_pool, v_pool, o, oss, osh,
        block_tables, max_blocks_per_seq, block_size, kv_lens, cu_q_lens,
        block2req, block2tile, valid_q_tiles, total_q_tiles,
        batch, total_q_tokens, num_q_heads, num_kv_heads, head_dim,
        softmax_scale, is_causal, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[flash_attn_paged_ragged_cute_fp16] launch error: %s\n", cudaGetErrorString(err));
    }
}
