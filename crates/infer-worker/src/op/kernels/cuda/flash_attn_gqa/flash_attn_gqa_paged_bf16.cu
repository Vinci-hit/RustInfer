#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "flash_attn_gqa.h"
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
template <class QShape, class KVShape,
          class QStride, class QSmemLayout, class TiledCopyQ, class S2RAtom,
          class KStride, class KVSmemLayout, class TiledCopyK, class S2RAtomTrans, class SmemVTransNoSwi, class SmemVTrans,
          class VStride,
          class OStride, class OSmemLayout, class TiledCopyO, class TiledMma>
__global__ void flash_attn_gqa_kernel_bf16_test(
    QShape q_shape, KVShape kv_shape,
    const __nv_bfloat16* __restrict__ q_ptr, QStride dQ, QSmemLayout sQ_layout, TiledCopyQ copy_q, S2RAtom s2r_atom,
    const __nv_bfloat16* __restrict__ k_ptr, KStride dK, KVSmemLayout sKV_layout, TiledCopyK copy_kv, S2RAtomTrans s2r_atom_trans, SmemVTransNoSwi V_layout_trans_no_swi,SmemVTrans V_layout_trans,
    const __nv_bfloat16* __restrict__ v_ptr, VStride dV,
    __nv_bfloat16* __restrict__ o_ptr,OStride dO, OSmemLayout sO_layout, TiledCopyO gmem_tiled_copy_O, TiledMma mma,
    const int32_t* __restrict__ block_table,
    int32_t block_size,
    int32_t num_total_blocks,
    int max_num_blocks_per_seq,
    int block_table_batch_stride,
    int kv_block_stride)
{
    unsigned int batch_idx = blockIdx.z;
    unsigned int q_head_idx = blockIdx.y;
    unsigned int block_m = blockIdx.x;
    constexpr float softmax_scale_log2 = M_LOG2E / 8.0f;
    unsigned int kv_head_idx = q_head_idx / (size<2>(q_shape) / size<2>(kv_shape)); // 简单的映射
    Tensor Q = make_tensor(make_gmem_ptr(q_ptr), q_shape, dQ);// (batch,seq,head_num,head_dim)
    Tensor K = make_tensor(make_gmem_ptr(k_ptr), kv_shape, dK);
    Tensor V = make_tensor(make_gmem_ptr(v_ptr), kv_shape, dV);
    Tensor O = make_tensor(make_gmem_ptr(o_ptr), q_shape, dO);// (batch,seq,head_num,head_dim)
    Tensor gQ = local_tile(Q(batch_idx, _,q_head_idx,_),Shape<Int<128>, Int<64>>{}, make_coord(block_m, 0)); //每个blockx处理一块q

    // 获取当前batch对应的block_table起始位置
    const int32_t* batch_block_table = block_table + batch_idx * block_table_batch_stride;
    // 创建基于block_table的KV tensor视图
    // KV数据存储在离散的block中，需要通过block_table访问
    Tensor KV_blocks = make_tensor(make_gmem_ptr(k_ptr), kv_shape, dK);
    Tensor V_blocks = make_tensor(make_gmem_ptr(v_ptr), kv_shape, dV);

    // 在循环中，我们需要根据block_table动态计算KV数据的地址
    // 这里暂时保持原来的结构，后续会在循环中修改
    Tensor gK = local_tile(KV_blocks(batch_idx, _,kv_head_idx,_),Shape<Int<64>, Int<64>>{}, make_coord(_, 0)); // 把第0个维度堆叠起来，变成（64,64，N/64）
    Tensor gV = local_tile(V_blocks(batch_idx, _,kv_head_idx,_),Shape<Int<64>, Int<64>>{}, make_coord(_, 0));
    Tensor gO = local_tile(O(batch_idx, _,q_head_idx,_),Shape<Int<128>, Int<64>>{}, make_coord(block_m, 0));
    extern __shared__ __nv_bfloat16 shared_memory[];
    using SharedStorage = SharedStorage<__nv_bfloat16, QSmemLayout, KVSmemLayout,KVSmemLayout>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sQ = make_tensor(make_smem_ptr(smem.smem_q.begin()), sQ_layout);
    Tensor sK = make_tensor(make_smem_ptr(smem.smem_k.begin()), sKV_layout);
    Tensor sV = make_tensor(make_smem_ptr(smem.smem_v.begin()), sKV_layout);
    Tensor sVt = make_tensor(sV.data(), V_layout_trans);
    Tensor sVtNoSwizzle = make_tensor(sV.data(), V_layout_trans_no_swi);
    int q_len = size<0>(q_shape);
    int kv_len = size<0>(kv_shape);
    // 创建一个假的 Tensor 用于分配寄存器
    ThrCopy thr_copy_q  = copy_q.get_slice(threadIdx.x);
    ThrCopy thr_copy_kv = copy_kv.get_slice(threadIdx.x);
    Tensor tQgQ = thr_copy_q.partition_S(gQ);
    Tensor tQsQ = thr_copy_q.partition_D(sQ);
    copy(copy_q, tQgQ, tQsQ);
    Tensor tKgK = thr_copy_kv.partition_S(gK);
    Tensor tKsK = thr_copy_kv.partition_D(sK);
    Tensor tVgV = thr_copy_kv.partition_S(gV);
    Tensor tVsV = thr_copy_kv.partition_D(sV);

    // 第一轮K数据加载（n_tile=0），使用block_table和异步cp
    int first_block_idx_in_seq = 0;
    int32_t first_physical_block_idx = batch_block_table[first_block_idx_in_seq];
    int32_t first_block_offset = first_physical_block_idx * block_size * size<2>(kv_shape) * size<3>(kv_shape);
    int32_t first_head_offset = kv_head_idx * size<3>(kv_shape);

    const __nv_bfloat16* k_global_ptr = k_ptr + first_block_offset + first_head_offset;
    __nv_bfloat16* k_smem_ptr = reinterpret_cast<__nv_bfloat16*>(cute::raw_pointer_cast(sK.data()));

    // 使用异步cp加载K数据（第一个tile，offset=0）
    for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
         int k_token = i / 64;
         int k_dim = i % 64;
         if (k_token < block_size) {
             int global_idx = k_token * size<2>(kv_shape) * size<3>(kv_shape) + k_dim;
             CP_ASYNC_CG_ca(k_smem_ptr + i, k_global_ptr + global_idx, 16);
         } else {
             k_smem_ptr[i] = __nv_bfloat16(0);
         }
     }
    CP_ASYNC_COMMIT_GROUP();

    auto thr_mma = mma.get_slice(threadIdx.x);
    // 用于计算 S = Q * K^T
    Tensor rQ = thr_mma.partition_fragment_A(sQ); // (MMA, MMA_M, MMA_K)
    // 想办法把sQ里面的值塞进rQ，但是要重排列
    TiledCopy s2r_copy_q = make_tiled_copy_A(s2r_atom, mma);
    ThrCopy   s2r_thr_copy_q = s2r_copy_q.get_slice(threadIdx.x);
    Tensor tXsQ = s2r_thr_copy_q.partition_S(sQ);
    Tensor tXrQ = s2r_thr_copy_q.retile_D(rQ);
    CUTE_STATIC_ASSERT_V(size<1>(tXsQ) == size<1>(tXrQ));

    Tensor rK = thr_mma.partition_fragment_B(sK); // (MMA, MMA_N, MMA_K)

    Tensor rS = partition_fragment_C(mma, Shape<Int<128>, Int<64>>{});  // (MMA=4, MMA_M, MMA_N)
    Tensor acc_o = partition_fragment_C(mma, Shape<Int<128>, Int<64>>{});  // MMA, MMA_M, MMA_K
    clear(rS);
    clear(acc_o);
    TiledCopy s2r_copy_k = make_tiled_copy_B(s2r_atom, mma);
    ThrCopy   s2r_thr_copy_k = s2r_copy_k.get_slice(threadIdx.x);
    Tensor tXsK = s2r_thr_copy_k.partition_S(sK); // (CPY, MMA_N, MMA_K)
    Tensor tXrK = s2r_thr_copy_k.retile_D(rK);    // (CPY, MMA_N, MMA_K)

    Tensor rVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);
    TiledCopy s2r_copy_v = make_tiled_copy_B(s2r_atom_trans, mma);
    ThrCopy   s2r_thr_copy_v = s2r_copy_v.get_slice(threadIdx.x);
    Tensor tOsVt = s2r_thr_copy_v.partition_S(sVt); // (CPY, MMA_N, MMA_K, PIPE)

    cp_async_fence(); // 提交

    //用于存储每个线程持有的最终结果
    auto rAccOut = partition_fragment_C(mma, Shape<Int<128>, Int<64>>{});
    clear(rAccOut);
    auto ol = logical_divide(rAccOut.layout(), Shape<Int<2>>{});
    auto rAccOut_new_layout =
        make_layout(make_layout(get<1>(get<0>(ol)), get<1>(ol)),
                    make_layout(get<0>(get<0>(ol)), get<2>(ol)));
    auto rAccOut_new = make_tensor(rAccOut.data(), rAccOut_new_layout);
    Tensor row_max = make_tensor<float>(Shape<Int<size<0>(rAccOut_new)>>{});
    Tensor row_sum = make_tensor<float>(Shape<Int<size<0>(rAccOut_new)>>{});
    CUTE_UNROLL
    for (int ii = 0; ii < size(row_max); ii++) {
        row_max(ii) = float(-5e4);
        row_sum(ii) = 0;
    }
    cp_async_wait<0>();
    __syncthreads();//进入循环前，等Q和KV读入共享内存

    copy(s2r_copy_q, tXsQ, tXrQ); // 把Q从共享内存读入寄存器
    int max_k_col = (block_m + 1) * 128 + (size<0>(kv_shape) - size<0>(q_shape));
    int n_block_max = (max_k_col + 64 - 1) / 64;
    n_block_max = min(n_block_max, (size<0>(kv_shape) + 64 - 1) / 64);
     for (int n_tile = 0; n_tile < n_block_max; ++n_tile)
     {
         // 计算当前KV tile的物理block信息
         int kv_token_start = n_tile * 64;
         int block_idx_in_seq = kv_token_start / block_size;
         int token_offset_in_block = kv_token_start % block_size;

         // 从block_table获取实际的物理block索引_idx_in_se
         int32_t physical_block_idx = batch_block_table[block_idx_in_seq];

         // 计算物理block的数据指针
         // KV数据布局：[num_blocks, block_size, num_kv_heads, head_dim]
         int32_t block_offset = physical_block_idx * block_size * size<2>(kv_shape) * size<3>(kv_shape);

         // 计算token在物理block内的偏移量
         int32_t token_in_block_offset = token_offset_in_block * size<2>(kv_shape) * size<3>(kv_shape);
         int32_t head_offset = kv_head_idx * size<3>(kv_shape);

         // 从全局内存异步加载V数据到共享内存
         const __nv_bfloat16* v_global_ptr = v_ptr + block_offset + token_in_block_offset + head_offset;
         __nv_bfloat16* v_smem_ptr = reinterpret_cast<__nv_bfloat16*>(cute::raw_pointer_cast(sV.data()));

         // V数据：[64个token, head_dim=64] - 使用异步cp
         for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
             int v_token = i / 64;
             int v_dim = i % 64;
             // 只加载有效的token数量
             if (v_token + token_offset_in_block < block_size) {
                 int global_idx = v_token * size<2>(kv_shape) * size<3>(kv_shape) + v_dim;
                 CP_ASYNC_CG_ca(v_smem_ptr + i, v_global_ptr + global_idx, 16); // 每个元素2字节
             } else {
                 // 填充0
                 v_smem_ptr[i] = __nv_bfloat16(0);
             }
         }
         CP_ASYNC_COMMIT_GROUP();

         cp_async_wait<1>();
         __syncthreads();//进入循环前，等上一轮的K存入共享内存完毕
         clear(rS);
         gemm(mma, rQ, rK, rS); //用完了rK，可以读下一轮的了

         // 下一轮K的预加载，需要使用block_table
         int next_tile = n_tile + 1;
         if (next_tile < n_block_max) {
             int next_kv_token_start = next_tile * 64;
             int next_block_idx_in_seq = next_kv_token_start / block_size;
             int next_token_offset_in_block = next_kv_token_start % block_size;
             int32_t next_physical_block_idx = batch_block_table[next_block_idx_in_seq];
             int32_t next_block_offset = next_physical_block_idx * block_size * size<2>(kv_shape) * size<3>(kv_shape);
             int32_t next_token_in_block_offset = next_token_offset_in_block * size<2>(kv_shape) * size<3>(kv_shape);
             int32_t next_head_offset = kv_head_idx * size<3>(kv_shape);

             const __nv_bfloat16* k_global_ptr = k_ptr + next_block_offset + next_token_in_block_offset + next_head_offset;
             __nv_bfloat16* k_smem_ptr = reinterpret_cast<__nv_bfloat16*>(cute::raw_pointer_cast(sK.data()));

             // 使用异步cp加载K数据
             for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
                 int k_token = i / 64;
                 int k_dim = i % 64;
                 if (k_token + next_token_offset_in_block < block_size) {
                     int global_idx = k_token * size<2>(kv_shape) * size<3>(kv_shape) + k_dim;
                     CP_ASYNC_CG_ca(k_smem_ptr + i, k_global_ptr + global_idx, 16);
                 } else {
                     k_smem_ptr[i] = __nv_bfloat16(0);
                 }
             }
         }
         CP_ASYNC_COMMIT_GROUP();
         cp_async_fence();
         apply_mask(
             rS, 64 *n_tile, block_m * 128 + (threadIdx.x / 32) * 16 + (threadIdx.x % 32) / 4,16*kNWarps, kv_len - q_len
         );

        //softmax
         Tensor scores = make_tensor(rS.data(), convert_layout_acc_rowcol(rS.layout()));//把tensorcore计算完后的乱序结果视为有序

        if (n_tile == 0)
        {
            reduce_max</*zero_init=*/true>(scores, row_max);
            scale_apply_exp2(scores, row_max, softmax_scale_log2);
            reduce_sum</*zero_init=*/true>(scores, row_sum);
        }else
        {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            reduce_max</*zero_init=*/false>(scores, row_max);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = row_max(mi) == -INFINITY ? 0.0f : row_max(mi);
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
#pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            reduce_sum</*zero_init=*/false>(scores, row_sum);
        }

         Tensor rP = convert_type<__nv_bfloat16>(rS);
         Tensor tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMma>(rP.layout()));
         gemm_rs(acc_o, tOrP, rVt, tOsVt, mma, s2r_copy_v, s2r_thr_copy_v);
     }

    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = row_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;

        float scale = inv_sum; // 推理通常没有 dropout

#pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= scale; // 只做归一化
        }
    }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = convert_type<__nv_bfloat16>(acc_o);
    // sO 复用 sQ 的 Shared Memory 空间 (节省显存)
    Tensor sO = make_tensor(sQ.data(), sO_layout);

    // 定义 Smem -> Reg 的拷贝策略 (用于 Output)
    // 这里的 Atom 必须支持向量化写入，且要配合 MMA 的布局
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, __nv_bfloat16>;
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(threadIdx.x);

    // retile_S: 将 MMA 布局的 rO 重排为 Copy 布局
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    // 执行拷贝: Reg -> Smem
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    __syncthreads(); // 必须同步，等待所有线程写完 Smem

    // ----------------------------------------------------------------
    // 2. Global Memory 写回 (Smem -> Gmem)
    // ----------------------------------------------------------------

    // 获取 Gmem Copy 的线程切片
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(threadIdx.x);

    // 切分 Source (Smem) 和 Dest (Gmem)
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    // 创建寄存器中转站 (用于谓词 Masking 拷贝)
    Tensor tOrO = make_tensor<__nv_bfloat16>(shape(tOgO));

    // 从 Smem 读到寄存器
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    // ----------------------------------------------------------------
    // 3. 边界检查与最终写入
    // ----------------------------------------------------------------

    // 构造 Identity Tensor 用于计算逻辑坐标 (Row, Col)
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO))); // (BLK_M, BLK_K)
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO); // 切分坐标

    // 构造谓词 Mask (Predicate Tensor)
    // 用于处理 HeadDim (K维度) 不对齐的情况
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));

    #pragma unroll
    for (int i = 0; i < size(tOrO); ++i) {
        // 获取当前元素的逻辑坐标 (m, n)
        // 注意：tOcO 的布局和 tOrO 是一一对应的
        // get<0> 是 M 坐标 (Row)，get<1> 是 N 坐标 (Col/HeadDim)
        int m_coord = get<0>(tOcO(i));

        // 判断是否越界 (M 维度)
        // size<0>(q_shape) 是 seq_len
        // block_m * 128 是当前 Block 的起始行
        if (block_m * 128 + m_coord < size<0>(q_shape)) {
            tOgO(i) = tOrO(i);
        }
    }
}
// ============================================================================
// PagedAttention Kernel Wrapper Function
// ============================================================================

void flash_attn_gqa_paged_cu(
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
    using namespace cute;

    int batch_size = 1; // paged kernel当前只支持单batch
    int kv_len = kv_seq_len;

    // 配置 Block 大小
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 64;
    constexpr int kHeadDim = 64;

    // 定义 Shapes
    auto q_shape = make_shape(batch_size, q_seq_len, num_q_heads, Int<kHeadDim>{});
    auto kv_shape = make_shape(num_total_blocks, block_size, num_kv_heads, Int<kHeadDim>{});

    // 定义 Strides
    auto stride_Q = make_stride(q_seq_len * num_q_heads * kHeadDim, num_q_heads * kHeadDim, kHeadDim, _1{});
    auto stride_K = make_stride(block_size * num_kv_heads * kHeadDim, num_kv_heads * kHeadDim, kHeadDim, _1{});
    auto stride_V = make_stride(block_size * num_kv_heads * kHeadDim, num_kv_heads * kHeadDim, kHeadDim, _1{});
    auto stride_O = make_stride(q_seq_len * num_q_heads * kHeadDim, num_q_heads * kHeadDim, kHeadDim, _1{});

    // Shared Memory Layouts
    using SmemLayoutAtomQ = decltype(composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8,_64>,
                                         Stride<_64,_1>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVtransposed = decltype(
            composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    auto sQ = tile_to_shape(SmemLayoutAtomQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));
    auto sKV = tile_to_shape(SmemLayoutAtomQ{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));
    auto sO = tile_to_shape(SmemLayoutAtomQ{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));

    // Copy Atoms
    using AtomGMEM = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, __nv_bfloat16>;

    auto copy_q = make_tiled_copy(
        AtomGMEM{},
        Layout<Shape<_16,_8>,Stride<_8,_1>>{},
        Layout<Shape< _1,_8>>{});

    auto copy_kv = make_tiled_copy(
        AtomGMEM{},
        Layout<Shape<_16,_8>,Stride<_8,_1>>{},
        Layout<Shape< _1,_8>>{});

    using AtomGMEM_Out = Copy_Atom<UniversalCopy<cute::uint128_t>, __nv_bfloat16>;
    auto copy_o = make_tiled_copy(
        AtomGMEM_Out{},
        make_layout(Shape<Int<128>, Int<1>>{}),
        make_layout(Shape<Int<1>, Int<8>>{}));

    auto warp_layout = make_layout(make_shape(Int<kNWarps>{}, Int<1>{}, Int<1>{}));

    // 定义 Tile 大小
    auto tile_shape = make_tile(Int<16 * kNWarps>{}, Int<64>{}, Int<64>{});

    // 生成 MMA
    auto mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        warp_layout,
        tile_shape
    );

    using S2RAtom = Copy_Atom<SM75_U32x4_LDSM_N, __nv_bfloat16>;
    using S2RAtom_trans = Copy_Atom<SM75_U16x8_LDSM_T, __nv_bfloat16>;

    // Launch Config
    dim3 dimGrid(size(ceil_div(q_seq_len, Int<kBlockM>{})),
               size(num_q_heads),
               batch_size);
    dim3 block(size(mma));

    int smem_size = int(sizeof(SharedStorage<__nv_bfloat16, decltype(sQ), decltype(sKV), decltype(sKV)>));

    // 启动 Paged Kernel
    flash_attn_gqa_kernel_bf16_test<<<dimGrid, block, smem_size, stream>>>(
        q_shape, kv_shape,
        q_ptr, stride_Q, sQ, copy_q, S2RAtom{},
        k_ptr, stride_K, sKV, copy_kv, S2RAtom_trans{},SmemLayoutVtransposedNoSwizzle{},SmemLayoutVtransposed{},
        v_ptr, stride_V,
        o_ptr, stride_O, sO, copy_o, mma,
        block_table,
        block_size,
        num_total_blocks,
        max_num_blocks_per_seq,
        block_table_batch_stride,
        kv_block_stride
    );

    CUDA_CHECK(cudaGetLastError());
}
