#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// Scaled Dot-Product Attention (BF16).
// Inputs:
//   q: [B, H, S_q, D]
//   k: [B, H, S_kv, D]
//   v: [B, H, S_kv, D]
//   output: [B, H, S_q, D]
//
// Workspace must be at least (B*H*S_q*S_kv) BF16 elements (for scores matrix).
void sdpa_bf16_forward(
    __nv_bfloat16* output,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* scores_workspace,
    int batch, int num_heads, int s_q, int s_kv, int head_dim,
    float scale,
    cudaStream_t stream);

void sdpa_f32_forward(
    float* output,
    const float* q,
    const float* k,
    const float* v,
    float* scores_workspace,
    int batch, int num_heads, int s_q, int s_kv, int head_dim,
    float scale,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
