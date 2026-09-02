#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void gated_rmsnorm_f32_forward(
    const float* input,
    const float* gate,
    const float* weight,
    float* output,
    int outer0,
    int outer1,
    int head_dim,
    int64_t input_stride0,
    int64_t input_stride1,
    int64_t gate_stride0,
    int64_t gate_stride1,
    int64_t output_stride0,
    int64_t output_stride1,
    float eps,
    cudaStream_t stream);

void gated_rmsnorm_bf16_forward(
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate,
    const float* weight,
    __nv_bfloat16* output,
    int outer0,
    int outer1,
    int head_dim,
    int64_t input_stride0,
    int64_t input_stride1,
    int64_t gate_stride0,
    int64_t gate_stride1,
    int64_t output_stride0,
    int64_t output_stride1,
    float eps,
    cudaStream_t stream);

void gated_rmsnorm_f16_forward(
    const __half* input,
    const __half* gate,
    const float* weight,
    __half* output,
    int outer0,
    int outer1,
    int head_dim,
    int64_t input_stride0,
    int64_t input_stride1,
    int64_t gate_stride0,
    int64_t gate_stride1,
    int64_t output_stride0,
    int64_t output_stride1,
    float eps,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
