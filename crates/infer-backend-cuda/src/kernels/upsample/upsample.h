#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

void upsample_nearest_2x_f32_forward(float* output, const float* input,
    int batch, int channels, int h_in, int w_in, cudaStream_t stream);
void upsample_nearest_2x_bf16_forward(__nv_bfloat16* output, const __nv_bfloat16* input,
    int batch, int channels, int h_in, int w_in, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
