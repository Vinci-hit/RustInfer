#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "scatter_kv_batch.h"

// -----------------------------------------------------------------------------
// Batched K/V scatter.
//
// 调用者不再传 "已加 pos 偏移的 dst 指针数组"：kernel 自己用
//   cache_ptrs[layer_idx * max_slots + slot_indices[seq]]
// 拿到该 seq 在本层的 cache base，再用 `seq_positions[seq] * dst_row_stride`
// 算到写入起点。
//
// 这样 kernel 只依赖两类 device scratch：
//   1. `*_cache_ptrs`            —— `[layer_num, max_slots]` u64 表；
//                                   runner 在扩容 / 成员变化时填一次，
//                                   scatter 和 attention 共用。
//   2. `slot_indices / seq_positions / seq_starts / seq_lens`
//                                —— `[B]` 小数组；runner 在 step 入口
//                                   一次性 H2D，所有层共用。
//
// 因此 op 内部 **零 malloc / 零 sync / 零 per-step H2D**，并且指针地址全部
// step 作用域稳定，天然 CUDA-Graph-capturable。K/V 合一次 launch。
// -----------------------------------------------------------------------------

template <typename T>
__global__ void scatter_kv_batch_kernel(
    const T*          k_src,
    const T*          v_src,
    void* const* __restrict__ k_cache_ptrs,
    void* const* __restrict__ v_cache_ptrs,
    int layer_idx,
    int max_slots,
    const int* __restrict__ slot_indices,
    const int* __restrict__ seq_positions,
    const int* __restrict__ seq_starts,
    const int* __restrict__ seq_lens,
    int kv_dim,
    int k_src_row_stride,
    int v_src_row_stride,
    int dst_row_stride)
{
    const int seq = blockIdx.x;
    const int len = seq_lens[seq];
    if (len <= 0) return;

    const int slot  = slot_indices[seq];
    const int pos   = seq_positions[seq];
    const int start = seq_starts[seq];

    const int table_off = layer_idx * max_slots + slot;
    T* k_dst = reinterpret_cast<T*>(k_cache_ptrs[table_off]) + pos * dst_row_stride;
    T* v_dst = reinterpret_cast<T*>(v_cache_ptrs[table_off]) + pos * dst_row_stride;

    const int total = len * kv_dim;
    const int tid   = threadIdx.x;
    const int step  = blockDim.x;

    for (int idx = tid; idx < total; idx += step) {
        const int token = idx / kv_dim;
        const int dim   = idx - token * kv_dim;

        const int src_row = start + token;
        const T k_val = k_src[src_row * k_src_row_stride + dim];
        const T v_val = v_src[src_row * v_src_row_stride + dim];

        k_dst[token * dst_row_stride + dim] = k_val;
        v_dst[token * dst_row_stride + dim] = v_val;
    }
}

template <typename T>
static inline void launch_scatter_kv_batch(
    const T*     k_src,
    const T*     v_src,
    void* const* k_cache_ptrs,
    void* const* v_cache_ptrs,
    int layer_idx,
    int max_slots,
    const int*   slot_indices,
    const int*   seq_positions,
    const int*   seq_starts,
    const int*   seq_lens,
    int batch,
    int kv_dim,
    int k_src_row_stride,
    int v_src_row_stride,
    int dst_row_stride,
    cudaStream_t stream)
{
    if (batch <= 0) return;
    dim3 grid(batch);
    dim3 block(256);
    scatter_kv_batch_kernel<T><<<grid, block, 0, stream>>>(
        k_src, v_src,
        k_cache_ptrs, v_cache_ptrs,
        layer_idx, max_slots,
        slot_indices, seq_positions, seq_starts, seq_lens,
        kv_dim,
        k_src_row_stride, v_src_row_stride,
        dst_row_stride);
}

// --- BF16 -------------------------------------------------------------------
extern "C" void scatter_kv_batch_bf16(
    const void*  k_src,
    const void*  v_src,
    void* const* k_cache_ptrs,
    void* const* v_cache_ptrs,
    int          layer_idx,
    int          max_slots,
    const int*   slot_indices,
    const int*   seq_positions,
    const int*   seq_starts,
    const int*   seq_lens,
    int          batch,
    int          kv_dim,
    int          k_src_row_stride_elems,
    int          v_src_row_stride_elems,
    int          dst_row_stride_elems,
    cudaStream_t stream)
{
    launch_scatter_kv_batch<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(k_src),
        reinterpret_cast<const __nv_bfloat16*>(v_src),
        k_cache_ptrs, v_cache_ptrs,
        layer_idx, max_slots,
        slot_indices, seq_positions, seq_starts, seq_lens,
        batch, kv_dim,
        k_src_row_stride_elems, v_src_row_stride_elems, dst_row_stride_elems,
        stream);
}

// --- FP16 -------------------------------------------------------------------
extern "C" void scatter_kv_batch_fp16(
    const void*  k_src,
    const void*  v_src,
    void* const* k_cache_ptrs,
    void* const* v_cache_ptrs,
    int          layer_idx,
    int          max_slots,
    const int*   slot_indices,
    const int*   seq_positions,
    const int*   seq_starts,
    const int*   seq_lens,
    int          batch,
    int          kv_dim,
    int          k_src_row_stride_elems,
    int          v_src_row_stride_elems,
    int          dst_row_stride_elems,
    cudaStream_t stream)
{
    launch_scatter_kv_batch<__half>(
        reinterpret_cast<const __half*>(k_src),
        reinterpret_cast<const __half*>(v_src),
        k_cache_ptrs, v_cache_ptrs,
        layer_idx, max_slots,
        slot_indices, seq_positions, seq_starts, seq_lens,
        batch, kv_dim,
        k_src_row_stride_elems, v_src_row_stride_elems, dst_row_stride_elems,
        stream);
}

// --- F32 --------------------------------------------------------------------
extern "C" void scatter_kv_batch_f32(
    const void*  k_src,
    const void*  v_src,
    void* const* k_cache_ptrs,
    void* const* v_cache_ptrs,
    int          layer_idx,
    int          max_slots,
    const int*   slot_indices,
    const int*   seq_positions,
    const int*   seq_starts,
    const int*   seq_lens,
    int          batch,
    int          kv_dim,
    int          k_src_row_stride_elems,
    int          v_src_row_stride_elems,
    int          dst_row_stride_elems,
    cudaStream_t stream)
{
    launch_scatter_kv_batch<float>(
        reinterpret_cast<const float*>(k_src),
        reinterpret_cast<const float*>(v_src),
        k_cache_ptrs, v_cache_ptrs,
        layer_idx, max_slots,
        slot_indices, seq_positions, seq_starts, seq_lens,
        batch, kv_dim,
        k_src_row_stride_elems, v_src_row_stride_elems, dst_row_stride_elems,
        stream);
}

// -----------------------------------------------------------------------------
// Paged K/V scatter.
//
// Writes K/V rows into a global paged pool laid out as
//   [num_blocks, block_size, kv_dim]
// using per-sequence block tables [batch, max_blocks_per_seq].
// -----------------------------------------------------------------------------

template <typename T>
__global__ void scatter_kv_paged_kernel(
    const T* __restrict__ k_src,
    const T* __restrict__ v_src,
    T* __restrict__ k_pool,
    T* __restrict__ v_pool,
    const unsigned int* __restrict__ block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int* __restrict__ seq_positions,
    const int* __restrict__ seq_starts,
    const int* __restrict__ seq_lens,
    int kv_dim,
    int k_src_row_stride,
    int v_src_row_stride)
{
    const int seq = blockIdx.x;
    const int len = seq_lens[seq];
    if (len <= 0) return;

    const int base_pos = seq_positions[seq];
    const int start = seq_starts[seq];
    const int total = len * kv_dim;
    const int tid = threadIdx.x;
    const int step = blockDim.x;

    for (int idx = tid; idx < total; idx += step) {
        const int token = idx / kv_dim;
        const int dim = idx - token * kv_dim;
        const int logical_pos = base_pos + token;
        const int block_idx = logical_pos / block_size;
        const int block_off = logical_pos - block_idx * block_size;
        const unsigned int physical_block = block_tables[seq * max_blocks_per_seq + block_idx];

        const int src_row = start + token;
        const T k_val = k_src[src_row * k_src_row_stride + dim];
        const T v_val = v_src[src_row * v_src_row_stride + dim];
        const size_t dst = (static_cast<size_t>(physical_block) * block_size + block_off) * kv_dim + dim;
        k_pool[dst] = k_val;
        v_pool[dst] = v_val;
    }
}

template <typename T>
static inline void launch_scatter_kv_paged(
    const T* k_src,
    const T* v_src,
    T* k_pool,
    T* v_pool,
    const unsigned int* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int* seq_positions,
    const int* seq_starts,
    const int* seq_lens,
    int batch,
    int kv_dim,
    int k_src_row_stride,
    int v_src_row_stride,
    cudaStream_t stream)
{
    if (batch <= 0) return;
    dim3 grid(batch);
    dim3 block(256);
    scatter_kv_paged_kernel<T><<<grid, block, 0, stream>>>(
        k_src, v_src, k_pool, v_pool, block_tables, max_blocks_per_seq, block_size,
        seq_positions, seq_starts, seq_lens, kv_dim,
        k_src_row_stride, v_src_row_stride);
}

extern "C" void scatter_kv_paged_bf16(
    const void* k_src,
    const void* v_src,
    void* k_pool,
    void* v_pool,
    const unsigned int* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int* seq_positions,
    const int* seq_starts,
    const int* seq_lens,
    int batch,
    int kv_dim,
    int k_src_row_stride_elems,
    int v_src_row_stride_elems,
    cudaStream_t stream)
{
    launch_scatter_kv_paged<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(k_src),
        reinterpret_cast<const __nv_bfloat16*>(v_src),
        reinterpret_cast<__nv_bfloat16*>(k_pool),
        reinterpret_cast<__nv_bfloat16*>(v_pool),
        block_tables, max_blocks_per_seq, block_size,
        seq_positions, seq_starts, seq_lens,
        batch, kv_dim, k_src_row_stride_elems, v_src_row_stride_elems,
        stream);
}

extern "C" void scatter_kv_paged_fp16(
    const void* k_src,
    const void* v_src,
    void* k_pool,
    void* v_pool,
    const unsigned int* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int* seq_positions,
    const int* seq_starts,
    const int* seq_lens,
    int batch,
    int kv_dim,
    int k_src_row_stride_elems,
    int v_src_row_stride_elems,
    cudaStream_t stream)
{
    launch_scatter_kv_paged<__half>(
        reinterpret_cast<const __half*>(k_src),
        reinterpret_cast<const __half*>(v_src),
        reinterpret_cast<__half*>(k_pool),
        reinterpret_cast<__half*>(v_pool),
        block_tables, max_blocks_per_seq, block_size,
        seq_positions, seq_starts, seq_lens,
        batch, kv_dim, k_src_row_stride_elems, v_src_row_stride_elems,
        stream);
}

extern "C" void scatter_kv_paged_f32(
    const void* k_src,
    const void* v_src,
    void* k_pool,
    void* v_pool,
    const unsigned int* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int* seq_positions,
    const int* seq_starts,
    const int* seq_lens,
    int batch,
    int kv_dim,
    int k_src_row_stride_elems,
    int v_src_row_stride_elems,
    cudaStream_t stream)
{
    launch_scatter_kv_paged<float>(
        reinterpret_cast<const float*>(k_src),
        reinterpret_cast<const float*>(v_src),
        reinterpret_cast<float*>(k_pool),
        reinterpret_cast<float*>(v_pool),
        block_tables, max_blocks_per_seq, block_size,
        seq_positions, seq_starts, seq_lens,
        batch, kv_dim, k_src_row_stride_elems, v_src_row_stride_elems,
        stream);
}
