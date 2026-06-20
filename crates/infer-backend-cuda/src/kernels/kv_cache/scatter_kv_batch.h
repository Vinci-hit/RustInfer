#pragma once
#include <cuda_runtime.h>

extern "C" {

// Batched K/V scatter —— 一次 launch 完成整 batch 的 K/V 写入，K/V 合一份。
//
// See scatter_kv_batch.cu 注释了解 plan 的职责分层。
void scatter_kv_batch_bf16(
    const void*  k_src,
    const void*  v_src,
    void* const* k_cache_ptrs,       // [layer_num, max_slots]
    void* const* v_cache_ptrs,       // [layer_num, max_slots]
    int          layer_idx,
    int          max_slots,
    const int*   slot_indices,       // [B]
    const int*   seq_positions,      // [B]
    const int*   seq_starts,         // [B]
    const int*   seq_lens,           // [B]
    int          batch,
    int          kv_dim,
    int          k_src_row_stride_elems,
    int          v_src_row_stride_elems,
    int          dst_row_stride_elems,
    cudaStream_t stream);

void scatter_kv_batch_fp16(
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
    cudaStream_t stream);

void scatter_kv_batch_f32(
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
    cudaStream_t stream);

void scatter_kv_paged_bf16(
    const void*  k_src,
    const void*  v_src,
    void*        k_pool,
    void*        v_pool,
    const unsigned int* block_tables,
    int          max_blocks_per_seq,
    int          block_size,
    const int*   seq_positions,
    const int*   seq_starts,
    const int*   seq_lens,
    int          batch,
    int          kv_dim,
    int          k_src_row_stride_elems,
    int          v_src_row_stride_elems,
    cudaStream_t stream);

void scatter_kv_paged_fp16(
    const void*  k_src,
    const void*  v_src,
    void*        k_pool,
    void*        v_pool,
    const unsigned int* block_tables,
    int          max_blocks_per_seq,
    int          block_size,
    const int*   seq_positions,
    const int*   seq_starts,
    const int*   seq_lens,
    int          batch,
    int          kv_dim,
    int          k_src_row_stride_elems,
    int          v_src_row_stride_elems,
    cudaStream_t stream);

void scatter_kv_paged_f32(
    const void*  k_src,
    const void*  v_src,
    void*        k_pool,
    void*        v_pool,
    const unsigned int* block_tables,
    int          max_blocks_per_seq,
    int          block_size,
    const int*   seq_positions,
    const int*   seq_starts,
    const int*   seq_lens,
    int          batch,
    int          kv_dim,
    int          k_src_row_stride_elems,
    int          v_src_row_stride_elems,
    cudaStream_t stream);

} // extern "C"
