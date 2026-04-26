#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 通用 ND permute: 按 flat 索引从 old_strides / new_strides 映射。
// 传入最多 8 维 (MAX_DIMS=8)。
//
// new_shape[j] = old_shape[perm[j]]
// new_strides[j] = product of new_shape[j+1..]
// old_strides 由调用方提供（源 tensor 的 strides）
// perm 提供新→旧维度映射

void permute_f32_forward(
    float* dst,
    const float* src,
    int ndim,
    const int64_t* new_shape,
    const int64_t* new_strides,
    const int64_t* old_strides,
    const int* perm,
    int64_t num_elements,
    cudaStream_t stream);

void permute_bf16_forward(
    __nv_bfloat16* dst,
    const __nv_bfloat16* src,
    int ndim,
    const int64_t* new_shape,
    const int64_t* new_strides,
    const int64_t* old_strides,
    const int* perm,
    int64_t num_elements,
    cudaStream_t stream);

void permute_f16_forward(
    __half* dst,
    const __half* src,
    int ndim,
    const int64_t* new_shape,
    const int64_t* new_strides,
    const int64_t* old_strides,
    const int* perm,
    int64_t num_elements,
    cudaStream_t stream);

void permute_i32_forward(
    int32_t* dst,
    const int32_t* src,
    int ndim,
    const int64_t* new_shape,
    const int64_t* new_strides,
    const int64_t* old_strides,
    const int* perm,
    int64_t num_elements,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
