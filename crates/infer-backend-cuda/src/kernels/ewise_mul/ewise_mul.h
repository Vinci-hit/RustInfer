#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// Element-wise multiply: dst[i] = a[i] * b[i]
void ewise_mul_f32_forward(float* dst, const float* a, const float* b, int n, cudaStream_t stream);
void ewise_mul_bf16_forward(__nv_bfloat16* dst, const __nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream);
void ewise_mul_f16_forward(__half* dst, const __half* a, const __half* b, int n, cudaStream_t stream);

// In-place element-wise mul: a[i] *= b[i]
void ewise_mul_inplace_f32_forward(float* a, const float* b, int n, cudaStream_t stream);
void ewise_mul_inplace_bf16_forward(__nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream);
void ewise_mul_inplace_f16_forward(__half* a, const __half* b, int n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
