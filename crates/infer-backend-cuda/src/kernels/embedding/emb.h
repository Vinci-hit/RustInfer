#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void embedding_kernel_cu_bf16(
    __nv_bfloat16* output,
    const int* input_token_ids,
    const __nv_bfloat16* weight,
    int token_len,
    int dim,
    int vocab_start,
    int local_vocab_size,
    cudaStream_t stream
);

void embedding_kernel_cu_fp16(
    __half* output,
    const int* input_token_ids,
    const __half* weight,
    int token_len,
    int dim,
    int vocab_start,
    int local_vocab_size,
    cudaStream_t stream
);

void embedding_kernel_cu_fp32(
    float* output,
    const int* input_token_ids,
    const float* weight,
    int token_len,
    int dim,
    int vocab_start,
    int local_vocab_size,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
