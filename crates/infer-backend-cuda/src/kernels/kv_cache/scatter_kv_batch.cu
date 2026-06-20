#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "scatter_kv_batch.h"

// -----------------------------------------------------------------------------
// Batched K/V scatter — vectorized per-token copy (inspired by vLLM
// reshape_and_cache_flash).
//
// 设计核心改进（相比旧 per-seq scalar kernel）：
//   - 每个 token 行用 float4 (16B = 8 bf16) 向量化拷贝，带宽利用率 ~8x 提升。
//   - 去掉了内层的 idx/kv_dim 整数除法。
//   - Grid = (batch, token_blocks): blockIdx.x = seq, blockIdx.y 分摊行数。
//     当 seq_len 长时，多个 y-blocks 协作，所有 SM 都能参与。
//
// 保持 API 完全兼容，仍然 **零 malloc / 零 sync / 零 per-step H2D**。
// CUDA-Graph-capturable（grid 大小和指针地址 step 间稳定）。
// -----------------------------------------------------------------------------

template <typename T>
__global__ __launch_bounds__(256)
void scatter_kv_batch_kernel(
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
    T* k_dst_base = reinterpret_cast<T*>(k_cache_ptrs[table_off]) + pos * dst_row_stride;
    T* v_dst_base = reinterpret_cast<T*>(v_cache_ptrs[table_off]) + pos * dst_row_stride;

    constexpr int ELEMS_PER_VEC = 16 / sizeof(T); // 8 for bf16/fp16, 4 for f32
    const int vecs_per_row = kv_dim / ELEMS_PER_VEC;

    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;

    // y-dimension distributes tokens across multiple blocks for long sequences
    const int y_blocks = gridDim.y;
    const int token_start = blockIdx.y;

    const bool k_contig = (k_src_row_stride == kv_dim);
    const bool v_contig = (v_src_row_stride == kv_dim);
    const bool dst_contig = (dst_row_stride == kv_dim);
    const bool all_contig = k_contig && v_contig && dst_contig;

    for (int token = token_start; token < len; token += y_blocks) {
        const int src_row = start + token;
        const T* k_row = k_src + src_row * k_src_row_stride;
        const T* v_row = v_src + src_row * v_src_row_stride;
        T* k_dst = k_dst_base + token * dst_row_stride;
        T* v_dst = v_dst_base + token * dst_row_stride;

        if (all_contig) {
            // Fast path: float4 vectorized copy
            const float4* k_src_v = reinterpret_cast<const float4*>(k_row);
            const float4* v_src_v = reinterpret_cast<const float4*>(v_row);
            float4* k_dst_v = reinterpret_cast<float4*>(k_dst);
            float4* v_dst_v = reinterpret_cast<float4*>(v_dst);

            for (int i = tid; i < vecs_per_row; i += block_threads) {
                k_dst_v[i] = k_src_v[i];
                v_dst_v[i] = v_src_v[i];
            }
            // Tail (kv_dim not divisible by ELEMS_PER_VEC — rare for real models)
            for (int i = vecs_per_row * ELEMS_PER_VEC + tid; i < kv_dim; i += block_threads) {
                k_dst[i] = k_row[i];
                v_dst[i] = v_row[i];
            }
        } else {
            // Strided path (qkv narrow view): element-wise
            for (int i = tid; i < kv_dim; i += block_threads) {
                k_dst[i] = k_row[i];
                v_dst[i] = v_row[i];
            }
        }
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
    // y-blocks: let multiple blocks share long sequences to fill SMs.
    // For H20 (80 SMs), use enough y-blocks so that batch * y_blocks >= 80.
    // Cap at 32 to avoid over-subscription for short sequences.
    int y_blocks = 1;
    if (batch < 32) {
        y_blocks = min(32, max(1, 80 / batch));
    }
    dim3 grid(batch, y_blocks);
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
// Vectorized per-token with y-block parallelism (same pattern as batched).
// -----------------------------------------------------------------------------

template <typename T>
__global__ __launch_bounds__(256)
void scatter_kv_paged_kernel(
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
    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;
    const int y_blocks = gridDim.y;

    constexpr int ELEMS_PER_VEC = 16 / sizeof(T);
    const int vecs_per_row = kv_dim / ELEMS_PER_VEC;
    // Paged dst stride is always kv_dim (contiguous within a page slot)
    const bool src_contig = (k_src_row_stride == kv_dim) && (v_src_row_stride == kv_dim);

    for (int token = static_cast<int>(blockIdx.y); token < len; token += y_blocks) {
        const int logical_pos = base_pos + token;
        const int blk_idx = logical_pos / block_size;
        const int blk_off = logical_pos - blk_idx * block_size;
        const unsigned int physical_block = block_tables[seq * max_blocks_per_seq + blk_idx];

        const int src_row = start + token;
        const T* k_row = k_src + src_row * k_src_row_stride;
        const T* v_row = v_src + src_row * v_src_row_stride;

        const size_t dst_offset = (static_cast<size_t>(physical_block) * block_size + blk_off) * kv_dim;
        T* k_dst = k_pool + dst_offset;
        T* v_dst = v_pool + dst_offset;

        if (src_contig) {
            // Vectorized path
            const float4* k_src_v = reinterpret_cast<const float4*>(k_row);
            const float4* v_src_v = reinterpret_cast<const float4*>(v_row);
            float4* k_dst_v = reinterpret_cast<float4*>(k_dst);
            float4* v_dst_v = reinterpret_cast<float4*>(v_dst);

            for (int i = tid; i < vecs_per_row; i += block_threads) {
                k_dst_v[i] = k_src_v[i];
                v_dst_v[i] = v_src_v[i];
            }
            for (int i = vecs_per_row * ELEMS_PER_VEC + tid; i < kv_dim; i += block_threads) {
                k_dst[i] = k_row[i];
                v_dst[i] = v_row[i];
            }
        } else {
            for (int i = tid; i < kv_dim; i += block_threads) {
                k_dst[i] = k_row[i];
                v_dst[i] = v_row[i];
            }
        }
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
    int y_blocks = 1;
    if (batch < 32) {
        y_blocks = min(32, max(1, 80 / batch));
    }
    dim3 grid(batch, y_blocks);
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
