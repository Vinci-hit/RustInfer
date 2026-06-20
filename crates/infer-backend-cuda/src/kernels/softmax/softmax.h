#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

void softmax_f32_forward(float* output, const float* input, int rows, int cols, cudaStream_t stream);
void softmax_bf16_forward(__nv_bfloat16* output, const __nv_bfloat16* input, int rows, int cols, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
