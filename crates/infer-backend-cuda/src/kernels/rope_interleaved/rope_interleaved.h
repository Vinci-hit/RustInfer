#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

// Apply interleaved RoPE on a tensor of shape [seq, n_heads*head_dim],
// where pairs (x[2k], x[2k+1]) are rotated by (cos[k], sin[k]).
// cos, sin: [seq, head_dim/2] (F32).
//
// Kernel processes n_heads "copies" of the same rotation per token.
void rope_interleaved_f32_forward(
    float* x,
    const float* cos, const float* sin,
    int seq, int n_heads, int head_dim,
    cudaStream_t stream);

void rope_interleaved_bf16_forward(
    __nv_bfloat16* x,
    const float* cos, const float* sin,
    int seq, int n_heads, int head_dim,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
