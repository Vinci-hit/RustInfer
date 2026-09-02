#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void gated_delta_rule_f32_forward(
    const float* query,
    const float* key,
    const float* value,
    const float* a,
    const float* b,
    const float* a_log,
    const float* dt_bias,
    float* recurrent_state,
    const int* state_slots,
    const int* cu_seqlens,
    float* output,
    int num_tokens,
    int batch,
    int num_key_heads,
    int num_value_heads,
    int key_head_dim,
    int value_head_dim,
    int num_slots,
    int64_t query_row_stride,
    int64_t key_row_stride,
    int64_t value_row_stride,
    int64_t a_row_stride,
    int64_t b_row_stride,
    cudaStream_t stream);

void gated_delta_rule_bf16_forward(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const float* a_log,
    const __nv_bfloat16* dt_bias,
    float* recurrent_state,
    const int* state_slots,
    const int* cu_seqlens,
    __nv_bfloat16* output,
    int num_tokens,
    int batch,
    int num_key_heads,
    int num_value_heads,
    int key_head_dim,
    int value_head_dim,
    int num_slots,
    int64_t query_row_stride,
    int64_t key_row_stride,
    int64_t value_row_stride,
    int64_t a_row_stride,
    int64_t b_row_stride,
    cudaStream_t stream);

void gated_delta_rule_f16_forward(
    const __half* query,
    const __half* key,
    const __half* value,
    const __half* a,
    const __half* b,
    const float* a_log,
    const __half* dt_bias,
    float* recurrent_state,
    const int* state_slots,
    const int* cu_seqlens,
    __half* output,
    int num_tokens,
    int batch,
    int num_key_heads,
    int num_value_heads,
    int key_head_dim,
    int value_head_dim,
    int num_slots,
    int64_t query_row_stride,
    int64_t key_row_stride,
    int64_t value_row_stride,
    int64_t a_row_stride,
    int64_t b_row_stride,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
