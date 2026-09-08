#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void causal_conv1d_silu_f32_forward(
    const float* input,
    const float* weight,
    float* conv_state,
    const int* state_slots,
    const int* cu_seqlens,
    float* output,
    int num_tokens,
    int batch,
    int channels,
    int kernel_size,
    int num_slots,
    cudaStream_t stream);

void causal_conv1d_silu_bf16_forward(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* conv_state,
    const int* state_slots,
    const int* cu_seqlens,
    __nv_bfloat16* output,
    int num_tokens,
    int batch,
    int channels,
    int kernel_size,
    int num_slots,
    cudaStream_t stream);

void causal_conv1d_silu_f16_forward(
    const __half* input,
    const __half* weight,
    __half* conv_state,
    const int* state_slots,
    const int* cu_seqlens,
    __half* output,
    int num_tokens,
    int batch,
    int channels,
    int kernel_size,
    int num_slots,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
