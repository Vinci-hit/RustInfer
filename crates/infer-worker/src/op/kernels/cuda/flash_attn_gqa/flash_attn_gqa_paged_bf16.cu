#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "flash_attn_gqa.h"
#include "batch_decode_params.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>
#define CP_ASYNC_CG(dst, src, bytes)                                           \
asm volatile(                                                                \
"cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(dst),       \
"l"(src), "n"(bytes))

#define CP_ASYNC_CG_ca(dst, src, bytes)                                           \
asm volatile(                                                                \
"cp.async.ca.shared.global [%0], [%1], %2;\n" ::"l"(dst),       \
"l"(src), "n"(bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage {
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};

using namespace cute;
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
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
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};
template<>
struct Allreduce<2> {
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        return x;
    }
};
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
    // This is slightly faster
    __device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
}
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}
template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                           const int col_idx_offset_,
                                           const int row_idx_offset,
                                           const int warp_row_stride,const int causal_shift) {
    // 检查 Tensor 格式是否符合 MMA 累加器的布局要求
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
    Tensor tensor = make_tensor(tensor_.data(), convert_layout_acc_rowcol(tensor_.layout()));
    const int lane_id = threadIdx.x % 32;
    // 计算当前线程处理的列索引基础偏移
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

    // 遍历 MMA 布局的行维度 (外层)
#pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
#pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;

            // [核心修改]
            // 允许查看的最大列号 = 当前行号 + 历史长度偏移
            // 比如 row=0, shift=99 -> limit=100 (可以看 col 0~99)
            const int col_idx_limit = row_idx + 1 + causal_shift;

#pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;

                    // Causal Mask 判断
                    if (col_idx >= col_idx_limit) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
        }
    }
}
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
#pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            // The following macro will disable the use of fma.
            // See: https://github.com/pytorch/pytorch/issues/121558 for more details
            // This macro is set in PyTorch and not FlashAttention
#ifdef UNFUSE_FMA
            tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
#else
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
#endif
        }
    }
}
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};
template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}
const int kNWarps = 4;

// ============================================================================
// Kernel Traits - compile-time parameters for kernel configuration
// ============================================================================
template <int kBlockM_, int kBlockN_, int kHeadDim_, int kNWarps_>
struct FlashAttnKernelTraits {
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kNWarps = kNWarps_;

    // Shared memory layout with swizzle for bank conflict reduction
    using SmemLayoutAtom = decltype(composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8, Int<kHeadDim>>,
                                         Stride<Int<kHeadDim>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutO = SmemLayoutQ;

    using SmemLayoutVtransposed = decltype(
            composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    // Copy atoms
    using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, __nv_bfloat16>;
    using GmemCopyAtomO = Copy_Atom<UniversalCopy<cute::uint128_t>, __nv_bfloat16>;
    using S2RAtom = Copy_Atom<SM75_U32x4_LDSM_N, __nv_bfloat16>;
    using S2RAtomTrans = Copy_Atom<SM75_U16x8_LDSM_T, __nv_bfloat16>;

    // MMA
    using MMA = decltype(make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        make_layout(make_shape(Int<kNWarps>{}, Int<1>{}, Int<1>{})),
        make_tile(Int<16 * kNWarps>{}, Int<kBlockN>{}, Int<kHeadDim>{})
    ));
};

template <typename Traits, typename Params>
__global__ void flash_attn_gqa_kernel_bf16(Params params)
{
    using DType = __nv_bfloat16;
    using IdType = int32_t;

    const unsigned int batch_idx = blockIdx.z;
    const unsigned int q_head_idx = blockIdx.y;
    const unsigned int block_m = blockIdx.x;

    // GQA: map Q head to KV head
    const unsigned int kv_head_idx = q_head_idx / params.get_gqa_group_size();

    // Get sequence lengths
    const int q_seq_len = params.q_seq_len;
    const int kv_seq_len = params.get_kv_len(batch_idx);

    // Softmax scale (precomputed log2 version for efficiency)
    const float sm_scale_log2 = params.sm_scale_log2;

    // ========== Setup Shared Memory ==========
    extern __shared__ DType shared_memory[];
    using SharedStorage = SharedStorage<DType,
        typename Traits::SmemLayoutQ,
        typename Traits::SmemLayoutKV,
        typename Traits::SmemLayoutKV>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

    auto sQ_layout = typename Traits::SmemLayoutQ{};
    auto sKV_layout = typename Traits::SmemLayoutKV{};
    auto sO_layout = typename Traits::SmemLayoutO{};
    auto sVt_layout = typename Traits::SmemLayoutVtransposed{};
    auto sVt_no_swi_layout = typename Traits::SmemLayoutVtransposedNoSwizzle{};

    Tensor sQ = make_tensor(make_smem_ptr(smem.smem_q.begin()), sQ_layout);
    Tensor sK = make_tensor(make_smem_ptr(smem.smem_k.begin()), sKV_layout);
    Tensor sV = make_tensor(make_smem_ptr(smem.smem_v.begin()), sKV_layout);
    Tensor sVt = make_tensor(sV.data(), sVt_layout);
    Tensor sVtNoSwizzle = make_tensor(sV.data(), sVt_no_swi_layout);

    // ========== Setup Global Memory Tensors ==========
    // Q: [batch, q_seq, num_qo_heads, head_dim]
    auto q_shape = make_shape(params.batch_size, q_seq_len, params.num_qo_heads, Int<Traits::kHeadDim>{});
    auto q_stride = make_stride(params.q_stride_batch, params.q_stride_seq, params.q_stride_head, _1{});
    Tensor Q = make_tensor(make_gmem_ptr(params.q), q_shape, q_stride);
    Tensor gQ = local_tile(Q(batch_idx, _, q_head_idx, _),
                           Shape<Int<Traits::kBlockM>, Int<Traits::kHeadDim>>{},
                           make_coord(block_m, 0));

    // O: [batch, q_seq, num_qo_heads, head_dim]
    auto o_stride = make_stride(params.o_stride_batch, params.o_stride_seq, params.o_stride_head, _1{});
    Tensor O = make_tensor(make_gmem_ptr(params.o), q_shape, o_stride);
    Tensor gO = local_tile(O(batch_idx, _, q_head_idx, _),
                           Shape<Int<Traits::kBlockM>, Int<Traits::kHeadDim>>{},
                           make_coord(block_m, 0));

    // ========== Setup Copy Operations ==========
    auto copy_q = make_tiled_copy(
        typename Traits::GmemCopyAtom{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{});

    auto copy_kv = make_tiled_copy(
        typename Traits::GmemCopyAtom{},
        Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{});

    auto copy_o = make_tiled_copy(
        typename Traits::GmemCopyAtomO{},
        make_layout(Shape<Int<128>, Int<1>>{}),
        make_layout(Shape<Int<1>, Int<8>>{}));

    typename Traits::MMA mma;

    // Load Q to shared memory
    ThrCopy thr_copy_q = copy_q.get_slice(threadIdx.x);
    Tensor tQgQ = thr_copy_q.partition_S(gQ);
    Tensor tQsQ = thr_copy_q.partition_D(sQ);
    copy(copy_q, tQgQ, tQsQ);

    // ========== Setup MMA ==========
    auto thr_mma = mma.get_slice(threadIdx.x);
    Tensor rQ = thr_mma.partition_fragment_A(sQ);
    Tensor rK = thr_mma.partition_fragment_B(sK);
    Tensor rS = partition_fragment_C(mma, Shape<Int<Traits::kBlockM>, Int<Traits::kBlockN>>{});
    Tensor acc_o = partition_fragment_C(mma, Shape<Int<Traits::kBlockM>, Int<Traits::kHeadDim>>{});
    clear(acc_o);

    // S2R copy for Q
    TiledCopy s2r_copy_q = make_tiled_copy_A(typename Traits::S2RAtom{}, mma);
    ThrCopy s2r_thr_copy_q = s2r_copy_q.get_slice(threadIdx.x);
    Tensor tXsQ = s2r_thr_copy_q.partition_S(sQ);
    Tensor tXrQ = s2r_thr_copy_q.retile_D(rQ);

    // S2R copy for K
    TiledCopy s2r_copy_k = make_tiled_copy_B(typename Traits::S2RAtom{}, mma);
    ThrCopy s2r_thr_copy_k = s2r_copy_k.get_slice(threadIdx.x);
    Tensor tXsK = s2r_thr_copy_k.partition_S(sK);
    Tensor tXrK = s2r_thr_copy_k.retile_D(rK);

    // S2R copy for V (transposed)
    Tensor rVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    TiledCopy s2r_copy_v = make_tiled_copy_B(typename Traits::S2RAtomTrans{}, mma);
    ThrCopy s2r_thr_copy_v = s2r_copy_v.get_slice(threadIdx.x);
    Tensor tOsVt = s2r_thr_copy_v.partition_S(sVt);

    // ========== Setup row_max and row_sum for online softmax ==========
    auto rAccOut = partition_fragment_C(mma, Shape<Int<Traits::kBlockM>, Int<Traits::kHeadDim>>{});
    clear(rAccOut);
    auto ol = logical_divide(rAccOut.layout(), Shape<Int<2>>{});
    auto rAccOut_new_layout = make_layout(
        make_layout(get<1>(get<0>(ol)), get<1>(ol)),
        make_layout(get<0>(get<0>(ol)), get<2>(ol)));
    auto rAccOut_new = make_tensor(rAccOut.data(), rAccOut_new_layout);
    Tensor row_max = make_tensor<float>(Shape<Int<size<0>(rAccOut_new)>>{});
    Tensor row_sum = make_tensor<float>(Shape<Int<size<0>(rAccOut_new)>>{});

    CUTE_UNROLL
    for (int ii = 0; ii < size(row_max); ii++) {
        row_max(ii) = -INFINITY;
        row_sum(ii) = 0;
    }

    // ========== Paged KV cache access via slot_mapping ==========
    const auto& paged_kv = params.paged_kv;
    const IdType kv_start = paged_kv.kv_indptr[batch_idx];

    // Load first K tile
    DType* k_smem_ptr = reinterpret_cast<DType*>(cute::raw_pointer_cast(sK.data()));

    // Vectorized async copy: 8 bf16 = 16 bytes per cp.async (minimum valid size is 4)
    // Each token has kHeadDim contiguous elements in global memory
    constexpr int kCopyVec = 8;  // bf16 elements per async copy
    constexpr int kVecsPerToken = Traits::kHeadDim / kCopyVec;  // 64/8 = 8
    constexpr int kTotalVecs = Traits::kBlockN * kVecsPerToken;  // 64*8 = 512

    // First tile K load using slot_mapping
    for (int idx = threadIdx.x; idx < kTotalVecs; idx += blockDim.x) {
        int token_in_tile = idx / kVecsPerToken;
        int vec_idx = idx % kVecsPerToken;
        int dim_offset = vec_idx * kCopyVec;
        int smem_offset = token_in_tile * Traits::kHeadDim + dim_offset;

        if (token_in_tile < kv_seq_len) {
            IdType slot = paged_kv.slot_mapping[kv_start + token_in_tile];
            const DType* k_src = paged_kv.k_ptr
                               + slot * paged_kv.num_kv_heads * paged_kv.head_dim
                               + kv_head_idx * paged_kv.head_dim + dim_offset;
            CP_ASYNC_CG_ca(k_smem_ptr + smem_offset, k_src, 16);
        } else {
            #pragma unroll
            for (int j = 0; j < kCopyVec; j++) {
                k_smem_ptr[smem_offset + j] = DType(0);
            }
        }
    }
    CP_ASYNC_COMMIT_GROUP();
    cp_async_fence();

    cp_async_wait<0>();
    __syncthreads();

    // Load Q from smem to registers
    copy(s2r_copy_q, tXsQ, tXrQ);

    // ========== Main Loop over KV tiles ==========
    const int causal_shift = kv_seq_len - q_seq_len;
    const int max_k_col = (block_m + 1) * Traits::kBlockM + causal_shift;
    int n_block_max = (max_k_col + Traits::kBlockN - 1) / Traits::kBlockN;
    n_block_max = min(n_block_max, (kv_seq_len + Traits::kBlockN - 1) / Traits::kBlockN);

    for (int n_tile = 0; n_tile < n_block_max; ++n_tile) {
        const int kv_tile_start = n_tile * Traits::kBlockN;

        // Load V tile asynchronously (vectorized: 8 bf16 = 16 bytes)
        DType* v_smem_ptr = reinterpret_cast<DType*>(cute::raw_pointer_cast(sV.data()));
        for (int idx = threadIdx.x; idx < kTotalVecs; idx += blockDim.x) {
            int token_in_tile = idx / kVecsPerToken;
            int vec_idx = idx % kVecsPerToken;
            int dim_offset = vec_idx * kCopyVec;
            int smem_offset = token_in_tile * Traits::kHeadDim + dim_offset;
            int global_token = kv_tile_start + token_in_tile;

            if (global_token < kv_seq_len) {
                IdType slot = paged_kv.slot_mapping[kv_start + global_token];
                const DType* v_src = paged_kv.v_ptr
                                   + slot * paged_kv.num_kv_heads * paged_kv.head_dim
                                   + kv_head_idx * paged_kv.head_dim + dim_offset;
                CP_ASYNC_CG_ca(v_smem_ptr + smem_offset, v_src, 16);
            } else {
                #pragma unroll
                for (int j = 0; j < kCopyVec; j++) {
                    v_smem_ptr[smem_offset + j] = DType(0);
                }
            }
        }
        CP_ASYNC_COMMIT_GROUP();

        // Wait for K and compute Q*K^T
        cp_async_wait<1>();
        __syncthreads();

        clear(rS);
        gemm(mma, rQ, rK, rS);

        // Prefetch next K tile (vectorized)
        int next_tile = n_tile + 1;
        if (next_tile < n_block_max) {
            int next_kv_start = next_tile * Traits::kBlockN;
            for (int idx = threadIdx.x; idx < kTotalVecs; idx += blockDim.x) {
                int token_in_tile = idx / kVecsPerToken;
                int vec_idx = idx % kVecsPerToken;
                int dim_offset = vec_idx * kCopyVec;
                int smem_offset = token_in_tile * Traits::kHeadDim + dim_offset;
                int global_token = next_kv_start + token_in_tile;

                if (global_token < kv_seq_len) {
                    IdType slot = paged_kv.slot_mapping[kv_start + global_token];
                    const DType* k_src = paged_kv.k_ptr
                                       + slot * paged_kv.num_kv_heads * paged_kv.head_dim
                                       + kv_head_idx * paged_kv.head_dim + dim_offset;
                    CP_ASYNC_CG_ca(k_smem_ptr + smem_offset, k_src, 16);
                } else {
                    #pragma unroll
                    for (int j = 0; j < kCopyVec; j++) {
                        k_smem_ptr[smem_offset + j] = DType(0);
                    }
                }
            }
        }
        CP_ASYNC_COMMIT_GROUP();
        cp_async_fence();

        // Apply causal mask
        apply_mask(rS,
                   Traits::kBlockN * n_tile,
                   block_m * Traits::kBlockM + (threadIdx.x / 32) * 16 + (threadIdx.x % 32) / 4,
                   16 * Traits::kNWarps,
                   causal_shift);

        // Online softmax
        Tensor scores = make_tensor(rS.data(), convert_layout_acc_rowcol(rS.layout()));

        if (n_tile == 0) {
            reduce_max<true>(scores, row_max);
            scale_apply_exp2(scores, row_max, sm_scale_log2);
            reduce_sum<true>(scores, row_sum);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            reduce_max<false>(scores, row_max);

            Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = row_max(mi) == -INFINITY ? 0.0f : row_max(mi);
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * sm_scale_log2);
                row_sum(mi) *= scores_scale;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                    acc_o_rowcol(mi, ni) *= scores_scale;
                }
            }
            scale_apply_exp2(scores, row_max, sm_scale_log2);
            reduce_sum<false>(scores, row_sum);
        }

        // Wait for V and compute P*V
        cp_async_wait<1>();
        __syncthreads();

        Tensor rP = convert_type<DType>(rS);
        Tensor tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs<typename Traits::MMA>(rP.layout()));
        gemm_rs(acc_o, tOrP, rVt, tOsVt, mma, s2r_copy_v, s2r_thr_copy_v);
    }

    // ========== Finalize: normalize by row_sum ==========
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);

    Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = row_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= inv_sum;
        }
    }

    // ========== Write output ==========
    Tensor rO = convert_type<DType>(acc_o);
    Tensor sO = make_tensor(sQ.data(), sO_layout);

    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, DType>;
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(threadIdx.x);

    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    __syncthreads();

    auto gmem_thr_copy_O = copy_o.get_thread_slice(threadIdx.x);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_tensor<DType>(shape(tOgO));
    cute::copy(copy_o, tOsO, tOrO);

    // Boundary check and write
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);

    #pragma unroll
    for (int i = 0; i < size(tOrO); ++i) {
        int m_coord = get<0>(tOcO(i));
        if (block_m * Traits::kBlockM + m_coord < q_seq_len) {
            tOgO(i) = tOrO(i);
        }
    }
}
// ============================================================================
// PagedAttention Kernel Wrapper Function (using slot_mapping)
// ============================================================================

void flash_attn_gqa_paged_cu(
    const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr,
    const __nv_bfloat16* v_ptr,
    __nv_bfloat16* o_ptr,
    const int32_t* slot_mapping,   // [total_tokens] -> physical slot
    const int32_t* kv_indptr,      // [batch_size + 1] CSR-style
    int32_t batch_size,
    int32_t q_seq_len,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    float sm_scale,
    cudaStream_t stream
) {
    using namespace cute;
    using DType = __nv_bfloat16;
    using IdType = int32_t;

    // Kernel configuration
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 64;
    constexpr int kHeadDim = 64;
    constexpr int kNWarpsKernel = 4;

    using Traits = FlashAttnKernelTraits<kBlockM, kBlockN, kHeadDim, kNWarpsKernel>;

    // Build PagedKV
    PagedKV<DType, IdType> paged_kv(
        const_cast<DType*>(k_ptr),
        const_cast<DType*>(v_ptr),
        slot_mapping,
        kv_indptr,
        num_kv_heads,
        head_dim
    );

    // Build BatchDecodeParams
    // Q/O strides: [batch, seq, head, dim] -> batch-major
    IdType q_stride_batch = q_seq_len * num_q_heads * head_dim;
    IdType q_stride_seq = num_q_heads * head_dim;
    IdType q_stride_head = head_dim;

    BatchDecodeParams<DType, DType, DType, IdType> params(
        const_cast<DType*>(q_ptr), q_stride_batch, q_stride_seq, q_stride_head,
        paged_kv,
        o_ptr, q_stride_batch, q_stride_seq, q_stride_head,  // O has same layout as Q
        batch_size, q_seq_len,
        num_q_heads, num_kv_heads, head_dim,
        sm_scale
    );

    // Launch configuration
    typename Traits::MMA mma;
    dim3 grid(
        (q_seq_len + kBlockM - 1) / kBlockM,
        num_q_heads,
        batch_size
    );
    dim3 block(size(mma));

    int smem_size = sizeof(SharedStorage<DType,
        typename Traits::SmemLayoutQ,
        typename Traits::SmemLayoutKV,
        typename Traits::SmemLayoutKV>);

    // Launch kernel
    flash_attn_gqa_kernel_bf16<Traits><<<grid, block, smem_size, stream>>>(params);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Legacy wrapper for backward compatibility (converts block_table to slot_mapping)
// ============================================================================
void flash_attn_gqa_paged_cu_legacy(
    const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr,
    const __nv_bfloat16* v_ptr,
    __nv_bfloat16* o_ptr,
    const int32_t* block_table,
    int32_t q_seq_len,
    int32_t kv_seq_len,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t block_size,
    int32_t num_total_blocks,
    int max_num_blocks_per_seq,
    int block_table_batch_stride,
    int kv_block_stride,
    cudaStream_t stream
) {
    // This is a placeholder - in production, you would:
    // 1. Allocate device memory for slot_mapping and kv_indptr
    // 2. Launch a kernel to convert block_table to slot_mapping
    // 3. Call flash_attn_gqa_paged_cu with the converted data

    // For now, this shows the expected conversion pattern on host:
    // std::vector<int32_t> slot_mapping;
    // std::vector<int32_t> kv_indptr = {0};
    // for (int b = 0; b < batch_size; b++) {
    //     for (int s = 0; s < kv_seq_len; s++) {
    //         int logical_block = s / block_size;
    //         int token_in_block = s % block_size;
    //         int physical_block = block_table[b * block_table_batch_stride + logical_block];
    //         slot_mapping.push_back(physical_block * block_size + token_in_block);
    //     }
    //     kv_indptr.push_back(slot_mapping.size());
    // }

    fprintf(stderr, "flash_attn_gqa_paged_cu_legacy: Please use flash_attn_gqa_paged_cu with slot_mapping instead\n");
}
