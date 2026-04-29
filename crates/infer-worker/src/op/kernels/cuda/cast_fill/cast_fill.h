#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// dtype cast kernels
void cast_f32_to_bf16_forward(__nv_bfloat16* dst, const float* src, int n, cudaStream_t stream);
void cast_bf16_to_f32_forward(float* dst, const __nv_bfloat16* src, int n, cudaStream_t stream);
void cast_f32_to_f16_forward(__half* dst, const float* src, int n, cudaStream_t stream);
void cast_f16_to_f32_forward(float* dst, const __half* src, int n, cudaStream_t stream);

// broadcast a single row [D] into multiple rows [num_rows, D]
void broadcast_row_bf16_forward(__nv_bfloat16* dst, const __nv_bfloat16* row, int num_rows, int D, cudaStream_t stream);
void broadcast_row_f32_forward(float* dst, const float* row, int num_rows, int D, cudaStream_t stream);

// repeat last row: src [n_src, D], fill dst rows [n_src..target_len] with copy of src[n_src-1]
void fill_repeat_last_row_bf16_forward(__nv_bfloat16* dst, int n_src, int target_len, int D, cudaStream_t stream);
void fill_repeat_last_row_f32_forward(float* dst, int n_src, int target_len, int D, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
