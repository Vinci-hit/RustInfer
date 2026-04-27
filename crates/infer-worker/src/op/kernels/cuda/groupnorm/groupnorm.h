#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

void groupnorm_f32_forward(float* output, const float* input, const float* weight, const float* bias,
    int batch, int channels, int spatial, int num_groups, float eps, cudaStream_t stream);
void groupnorm_bf16_forward(__nv_bfloat16* output, const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, int batch, int channels, int spatial, int num_groups, float eps, cudaStream_t stream);

// Fused GroupNorm + SiLU variants — produce `silu(groupnorm(x))` in a single
// kernel launch, saving the extra full-tensor read/write that a separate SiLU
// would cost. Used by the VAE decoder's ResnetBlock where every GN is followed
// by SiLU.
void groupnorm_silu_f32_forward(float* output, const float* input, const float* weight, const float* bias,
    int batch, int channels, int spatial, int num_groups, float eps, cudaStream_t stream);
void groupnorm_silu_bf16_forward(__nv_bfloat16* output, const __nv_bfloat16* input,
    const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    int batch, int channels, int spatial, int num_groups, float eps, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
