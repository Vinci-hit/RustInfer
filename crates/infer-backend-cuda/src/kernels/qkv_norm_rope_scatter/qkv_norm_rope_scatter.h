#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

void qkv_norm_rope_scatter_bf16(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* sin_cache,
    const __nv_bfloat16* cos_cache,
    const int* positions,
    __nv_bfloat16* k_pool,
    __nv_bfloat16* v_pool,
    const unsigned int* block_tables,
    const int* seq_positions,
    const int* seq_starts,
    const int* seq_lens,
    int num_tokens,
    int batch,
    int head_num,
    int kv_head_num,
    int head_dim,
    int kv_dim,
    long long q_row_stride,
    long long k_row_stride,
    long long v_row_stride,
    int max_blocks_per_seq,
    int block_size,
    float q_eps,
    float k_eps,
    cudaStream_t stream);

} // extern "C"
