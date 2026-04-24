// flash_attn_gqa_bf16_hdim128.cu
// Flash Attention GQA Prefill kernel for head_dim=128 (BF16)
// Adapted from flash_attn_gqa_bf16.cu (head_dim=64)
//
// Key differences from hdim64:
//   - kHeadDim = 128 (was 64)
//   - Q smem: [128, 128], KV smem: [64, 128]  (doubled in head_dim)
//   - acc_o: Shape<128, 128> (output covers full head_dim)
//   - V^T: [128, 64] (transposed from [64, 128])
//   - softmax_scale = 1/sqrt(128) instead of 1/sqrt(64)
//   - Requires >48KB shared memory → needs cudaFuncSetAttribute

#include "flash_attn_gqa.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

// --- Shared Storage ---
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage128 {
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};

using namespace cute;

// --- Helper functions (identical to hdim64 version) ---

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_128_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
#pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
#pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<int THREADS>
struct Allreduce128 {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce128<OFFSET>::run(x, op);
    }
};
template<>
struct Allreduce128<2> {
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        return x;
    }
};

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_128_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce128<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_128_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_128_<zero_init>(tensor, summary, op);
    quad_allreduce_128_(summary, summary, op);
}

template<typename T>
struct MaxOp128 {
    __device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};
template <>
struct MaxOp128<float> {
    __device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

template<typename T>
struct SumOp128 {
    __device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max_128(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp128<float> max_op;
    reduce_128_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum_128(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp128<float> sum_op;
    thread_reduce_128_<zero_init>(tensor, sum, sum_op);
}

template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol_128(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_128(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_128(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset,
                                               const int warp_row_stride, const int causal_shift) {
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
    Tensor tensor = make_tensor(tensor_.data(), convert_layout_acc_rowcol_128(tensor_.layout()));
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

#pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
#pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            const int col_idx_limit = row_idx + 1 + causal_shift;
#pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    if (col_idx >= col_idx_limit) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
        }
    }
}

template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2_128(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
#pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs_128(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs_128(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// =====================================================================
// Main kernel template (same structure as hdim64, parameterized shapes)
// =====================================================================
static constexpr int kNWarps_128 = 4;

template <class QShape, class KVShape,
          class QStride, class QSmemLayout, class TiledCopyQ, class S2RAtom,
          class KStride, class KVSmemLayout, class TiledCopyK, class S2RAtomTrans, class SmemVTransNoSwi, class SmemVTrans,
          class VStride,
          class OStride, class OSmemLayout, class TiledCopyO, class TiledMma>
__global__ void flash_attn_gqa_kernel_bf16_hdim128(
    QShape q_shape, KVShape kv_shape,
    const __nv_bfloat16* __restrict__ q_ptr, QStride dQ, QSmemLayout sQ_layout, TiledCopyQ copy_q, S2RAtom s2r_atom,
    const __nv_bfloat16* __restrict__ k_ptr, KStride dK, KVSmemLayout sKV_layout, TiledCopyK copy_kv, S2RAtomTrans s2r_atom_trans, SmemVTransNoSwi V_layout_trans_no_swi, SmemVTrans V_layout_trans,
    const __nv_bfloat16* __restrict__ v_ptr, VStride dV,
    __nv_bfloat16* __restrict__ o_ptr, OStride dO, OSmemLayout sO_layout, TiledCopyO gmem_tiled_copy_O, TiledMma mma)
{
    unsigned int q_head_idx = blockIdx.y;
    unsigned int block_m = blockIdx.x;

    // softmax scale = 1/sqrt(128), then * log2(e) for exp2-based softmax
    // sqrt(128) = 8*sqrt(2) ≈ 11.3137085
    constexpr float softmax_scale_log2 = float(M_LOG2E) / 11.3137085f;

    unsigned int kv_head_idx = q_head_idx / (size<1>(q_shape) / size<1>(kv_shape));

    Tensor Q = make_tensor(make_gmem_ptr(q_ptr), q_shape, dQ);
    Tensor K = make_tensor(make_gmem_ptr(k_ptr), kv_shape, dK);
    Tensor V = make_tensor(make_gmem_ptr(v_ptr), kv_shape, dV);
    Tensor O = make_tensor(make_gmem_ptr(o_ptr), q_shape, dO);

    // --- head_dim=128: tile Q/O as [128, 128], K/V as [64, 128] ---
    Tensor gQ = local_tile(Q(_, q_head_idx, _), Shape<Int<128>, Int<128>>{}, make_coord(block_m, 0));
    Tensor gK = local_tile(K(_, kv_head_idx, _), Shape<Int<64>, Int<128>>{}, make_coord(_, 0));
    Tensor gV = local_tile(V(_, kv_head_idx, _), Shape<Int<64>, Int<128>>{}, make_coord(_, 0));
    Tensor gO = local_tile(O(_, q_head_idx, _), Shape<Int<128>, Int<128>>{}, make_coord(block_m, 0));

    extern __shared__ __nv_bfloat16 shared_memory_128[];
    using SharedStorageType = SharedStorage128<__nv_bfloat16, QSmemLayout, KVSmemLayout, KVSmemLayout>;
    SharedStorageType& smem = *reinterpret_cast<SharedStorageType*>(shared_memory_128);

    Tensor sQ = make_tensor(make_smem_ptr(smem.smem_q.begin()), sQ_layout);
    Tensor sK = make_tensor(make_smem_ptr(smem.smem_k.begin()), sKV_layout);
    Tensor sV = make_tensor(make_smem_ptr(smem.smem_v.begin()), sKV_layout);
    Tensor sVt = make_tensor(sV.data(), V_layout_trans);
    Tensor sVtNoSwizzle = make_tensor(sV.data(), V_layout_trans_no_swi);

    int q_len = size<0>(q_shape);
    int kv_len = size<0>(kv_shape);

    // --- Async copy: Q from gmem to smem ---
    ThrCopy thr_copy_q  = copy_q.get_slice(threadIdx.x);
    ThrCopy thr_copy_kv = copy_kv.get_slice(threadIdx.x);

    Tensor tQgQ = thr_copy_q.partition_S(gQ);
    Tensor tQsQ = thr_copy_q.partition_D(sQ);
    copy(copy_q, tQgQ, tQsQ);

    Tensor tKgK = thr_copy_kv.partition_S(gK);
    Tensor tKsK = thr_copy_kv.partition_D(sK);
    Tensor tVgV = thr_copy_kv.partition_S(gV);
    Tensor tVsV = thr_copy_kv.partition_D(sV);

    copy(copy_kv, tKgK(_, _, _, 0), tKsK);  // Load first K tile

    auto thr_mma = mma.get_slice(threadIdx.x);

    // --- Registers for Q @ K^T ---
    Tensor rQ = thr_mma.partition_fragment_A(sQ);   // (MMA, MMA_M, MMA_K)
    TiledCopy s2r_copy_q = make_tiled_copy_A(s2r_atom, mma);
    ThrCopy   s2r_thr_copy_q = s2r_copy_q.get_slice(threadIdx.x);
    Tensor tXsQ = s2r_thr_copy_q.partition_S(sQ);
    Tensor tXrQ = s2r_thr_copy_q.retile_D(rQ);

    Tensor rK = thr_mma.partition_fragment_B(sK);   // (MMA, MMA_N, MMA_K)

    // rS: scores [blockM=128, blockN=64] — same as hdim64
    Tensor rS = partition_fragment_C(mma, Shape<Int<128>, Int<64>>{});
    // acc_o: output [blockM=128, head_dim=128] — doubled from hdim64
    Tensor acc_o = partition_fragment_C(mma, Shape<Int<128>, Int<128>>{});
    clear(rS);
    clear(acc_o);

    TiledCopy s2r_copy_k = make_tiled_copy_B(s2r_atom, mma);
    ThrCopy   s2r_thr_copy_k = s2r_copy_k.get_slice(threadIdx.x);
    Tensor tXsK = s2r_thr_copy_k.partition_S(sK);
    Tensor tXrK = s2r_thr_copy_k.retile_D(rK);

    // --- V transpose registers for P @ V ---
    Tensor rVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);
    TiledCopy s2r_copy_v = make_tiled_copy_B(s2r_atom_trans, mma);
    ThrCopy   s2r_thr_copy_v = s2r_copy_v.get_slice(threadIdx.x);
    Tensor tOsVt = s2r_thr_copy_v.partition_S(sVt);

    cp_async_fence();

    // --- Row-wise softmax state ---
    // Derive row count from scores fragment [128, 64] (same M dimension as acc_o [128, 128])
    // But use the score layout for row_max/row_sum since softmax operates on score rows
    Tensor scores_for_layout = make_tensor(rS.data(), convert_layout_acc_rowcol_128(rS.layout()));
    Tensor row_max = make_tensor<float>(Shape<Int<size<0>(scores_for_layout)>>{});
    Tensor row_sum = make_tensor<float>(Shape<Int<size<0>(scores_for_layout)>>{});
    CUTE_UNROLL
    for (int ii = 0; ii < size(row_max); ii++) {
        row_max(ii) = float(-5e4);
        row_sum(ii) = 0;
    }

    cp_async_wait<0>();
    __syncthreads();

    copy(s2r_copy_q, tXsQ, tXrQ);  // Q: smem → registers

    int max_k_col = (block_m + 1) * 128 + (size<0>(kv_shape) - size<0>(q_shape));
    int n_block_max = (max_k_col + 64 - 1) / 64;
    n_block_max = min(n_block_max, (size<0>(kv_shape) + 64 - 1) / 64);

    // ===== Main loop over KV tiles =====
    for (int n_tile = 0; n_tile < n_block_max; ++n_tile)
    {
        // Prefetch V for this tile
        copy(copy_kv, tVgV(_, _, _, n_tile), tVsV);

        cp_async_wait<1>();
        __syncthreads();

        // --- Compute S = Q @ K^T ---
        clear(rS);
        copy(s2r_copy_k, tXsK, tXrK);
        gemm(mma, rQ, rK, rS);

        // Prefetch next K tile
        copy(copy_kv, tKgK(_, _, _, (n_tile + 1) % n_block_max), tKsK);
        cp_async_fence();

        // --- Causal mask ---
        apply_mask_128(
            rS, 64 * n_tile, block_m * 128 + (threadIdx.x / 32) * 16 + (threadIdx.x % 32) / 4,
            16 * kNWarps_128, kv_len - q_len
        );

        // --- Online softmax ---
        Tensor scores = make_tensor(rS.data(), convert_layout_acc_rowcol_128(rS.layout()));

        if (n_tile == 0)
        {
            reduce_max_128</*zero_init=*/true>(scores, row_max);
            scale_apply_exp2_128(scores, row_max, softmax_scale_log2);
            reduce_sum_128</*zero_init=*/true>(scores, row_sum);
        }
        else
        {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            reduce_max_128</*zero_init=*/false>(scores, row_max);

            // Rescale previous acc_o
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol_128(acc_o.layout()));
#pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = row_max(mi) == -INFINITY ? 0.0f : row_max(mi);
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
#pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            scale_apply_exp2_128(scores, row_max, softmax_scale_log2);
            reduce_sum_128</*zero_init=*/false>(scores, row_sum);
        }

        // --- acc_o += P @ V ---
        cp_async_wait<0>();
        __syncthreads();

        Tensor rP = convert_type_128<__nv_bfloat16>(rS);
        Tensor tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs_128<TiledMma>(rP.layout()));
        gemm_rs_128(acc_o, tOrP, rVt, tOsVt, mma, s2r_copy_v, s2r_thr_copy_v);
    }

    // ===== Final softmax normalization =====
    SumOp128<float> sum_op;
    quad_allreduce_128_(row_sum, row_sum, sum_op);
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol_128(acc_o.layout()));
#pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = row_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
#pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= inv_sum;
        }
    }

    // ===== Write output: registers → smem → gmem =====
    Tensor rO = convert_type_128<__nv_bfloat16>(acc_o);
    // Reuse sQ shared memory for output (sO has same shape [128, 128])
    Tensor sO = make_tensor(sQ.data(), sO_layout);

    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, __nv_bfloat16>;
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(threadIdx.x);

    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    __syncthreads();

    // Smem → Gmem
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(threadIdx.x);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO_reg = make_tensor<__nv_bfloat16>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO_reg);

    // Boundary check (M dimension)
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);

#pragma unroll
    for (int i = 0; i < size(tOrO_reg); ++i) {
        int m_coord = get<0>(tOcO(i));
        if (block_m * 128 + m_coord < size<0>(q_shape)) {
            tOgO(i) = tOrO_reg(i);
        }
    }
}

// =====================================================================
// Launch function
// =====================================================================
extern "C" void launch_flash_attn_cute_bf16_hdim128(
    const __nv_bfloat16* d_Q, const __nv_bfloat16* d_K, const __nv_bfloat16* d_V, __nv_bfloat16* d_O,
    int seq_len, int* kv_len_ptr, int q_heads, int kv_heads,
    cudaStream_t stream)
{
    using namespace cute;
    int kv_len = *kv_len_ptr + seq_len;  // prefill: total kv length

    constexpr int kBlockM = 128;
    constexpr int kBlockN = 64;
    constexpr int kHeadDim = 128;

    auto q_shape  = make_shape(seq_len, q_heads, Int<kHeadDim>{});
    auto kv_shape = make_shape(kv_len, kv_heads, Int<kHeadDim>{});

    auto stride_Q = make_stride(q_heads * kHeadDim, kHeadDim, _1{});
    auto stride_K = make_stride(kv_heads * kHeadDim, kHeadDim, _1{});
    auto stride_V = make_stride(kv_heads * kHeadDim, kHeadDim, _1{});
    auto stride_O = make_stride(q_heads * kHeadDim, kHeadDim, _1{});

    // Shared memory layout atom: Swizzle<3,3,3> with [8, 64] tile
    // Tiled to cover head_dim=128 (2 tiles in the column direction)
    using SmemLayoutAtomQ = decltype(composition(Swizzle<3, 3, 3>{},
                                    Layout<Shape<_8, _64>,
                                           Stride<_64, _1>>{}));

    // Q smem: [kBlockM=128, kHeadDim=128]
    // KV smem: [kBlockN=64, kHeadDim=128]
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    auto sQ  = tile_to_shape(SmemLayoutAtomQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));
    auto sKV = tile_to_shape(SmemLayoutAtomQ{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));
    auto sO  = tile_to_shape(SmemLayoutAtomQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));

    // Async global → shared memory copy atoms
    using AtomGMEM = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, __nv_bfloat16>;

    auto copy_q = make_tiled_copy(
        AtomGMEM{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{});

    auto copy_kv = make_tiled_copy(
        AtomGMEM{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{});

    // Output copy: shared → global
    using AtomGMEM_Out = Copy_Atom<UniversalCopy<cute::uint128_t>, __nv_bfloat16>;
    auto copy_o = make_tiled_copy(
        AtomGMEM_Out{},
        make_layout(Shape<Int<128>, Int<1>>{}),
        make_layout(Shape<Int<1>, Int<8>>{}));

    // MMA configuration (same atom and warp layout as hdim64)
    auto warp_layout = make_layout(make_shape(Int<kNWarps_128>{}, Int<1>{}, Int<1>{}));
    auto tile_shape = make_tile(Int<16 * kNWarps_128>{}, Int<64>{}, Int<64>{});

    auto mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        warp_layout,
        tile_shape
    );

    using S2RAtom = Copy_Atom<SM75_U32x4_LDSM_N, __nv_bfloat16>;
    using S2RAtom_trans = Copy_Atom<SM75_U16x8_LDSM_T, __nv_bfloat16>;

    // Grid and block
    dim3 dimGrid(size(ceil_div(seq_len, Int<kBlockM>{})), size(q_heads));
    dim3 block(size(mma));

    int smem_size = int(sizeof(SharedStorage128<__nv_bfloat16, decltype(sQ), decltype(sKV), decltype(sKV)>));

    // Request extended shared memory (needed for 128 head_dim: ~64KB > default 48KB)
    auto kernel_ptr = flash_attn_gqa_kernel_bf16_hdim128<
        decltype(q_shape), decltype(kv_shape),
        decltype(stride_Q), decltype(sQ), decltype(copy_q), S2RAtom,
        decltype(stride_K), decltype(sKV), decltype(copy_kv), S2RAtom_trans, SmemLayoutVtransposedNoSwizzle, SmemLayoutVtransposed,
        decltype(stride_V),
        decltype(stride_O), decltype(sO), decltype(copy_o), decltype(mma)>;

    cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    kernel_ptr<<<dimGrid, block, smem_size, stream>>>(
        q_shape, kv_shape,
        d_Q, stride_Q, sQ, copy_q, S2RAtom{},
        d_K, stride_K, sKV, copy_kv, S2RAtom_trans{}, SmemLayoutVtransposedNoSwizzle{}, SmemLayoutVtransposed{},
        d_V, stride_V,
        d_O, stride_O, sO, copy_o, mma
    );
}
