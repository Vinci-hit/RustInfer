#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

int rustinfer_fa3_init_kernel_attributes(int max_dynamic_smem);

int rustinfer_fa3_varlen_paged_bf16_hd128(
    const void* q,
    const void* k_pool,
    const void* v_pool,
    void* o,
    float* softmax_lse,
    const int32_t* cu_seqlens_q,
    const int32_t* seqused_k,
    const int32_t* page_table,
    int64_t page_table_batch_stride,
    int32_t* tile_count_semaphore,
    int b,
    int max_seqlen_q,
    int max_pages_per_seq,
    int page_size,
    int num_pages,
    int q_extent,
    int h,
    int h_k,
    int device_id,
    int num_sm,
    int64_t q_row_stride,
    int64_t q_head_stride,
    int64_t k_row_stride,
    int64_t k_head_stride,
    int64_t k_page_stride,
    int64_t v_row_stride,
    int64_t v_head_stride,
    int64_t v_page_stride,
    int64_t o_row_stride,
    int64_t o_head_stride,
    float softmax_scale,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
