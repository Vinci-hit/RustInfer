#include "emb.h"

#include <cuda_bf16.h>
// --- CUDA Kernel (BF16版本) ---
__global__ void embedding_kernel_bf16x8(
    float4* output,
    const int* input_token_ids,
    const float4* weight,
    int dim, int vocab_size
) {
    const int token_idx = blockIdx.x;
    int32_t token = input_token_ids[token_idx];
    if (token >= vocab_size) {
        return;
    }
    output[token_idx * dim] = weight[token * dim];
}
void embedding_kernel_cu_bf16x8(
    __nv_bfloat16* output,
    const int* input_token_ids,
    const __nv_bfloat16* weight,
    int token_len,
    int dim,
    int vocab_size,
    cudaStream_t stream
) {
    dim3 block(dim / 8);
    auto out_f4 = reinterpret_cast<float4*>(output);
    auto weight_f4 = reinterpret_cast<const float4*>(weight);
    dim /= 8;
    // --- 启动内核 ---
    embedding_kernel_bf16x8<<<token_len, block, 0, stream>>>(
        out_f4, input_token_ids, weight_f4, dim, vocab_size
    );
}

__global__ void embedding_kernel(
    float4* output,
    const int* input_token_ids,
    const float4* weight,
    int dim, int vocab_size
) {
    const int token_idx = blockIdx.x;
    int32_t token = input_token_ids[token_idx];
    if (token >= vocab_size) {
        return;
    }

    float4* output_ptr_start = output + token_idx * dim;
    const float4* weight_ptr_start = weight + token * dim;

    for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        output_ptr_start[i] = weight_ptr_start[i];
    }
}

#include <cstdio>
void embedding_kernel_cu_fp32x4(
    float* output,
    const int* input_token_ids,
    const float* weight,
    int token_len,
    int dim,
    int vocab_size,
    cudaStream_t stream
) {
    
    constexpr int32_t thread_num = 128;
    
    // --- 类型转换 ---
    float4* out_f4 = reinterpret_cast<float4*>(output);
    const float4* weight_f4 = reinterpret_cast<const float4*>(weight);
    dim /= 4;
    // --- 启动内核 ---
    embedding_kernel<<<token_len, thread_num, 0, stream>>>(
        out_f4, input_token_ids, weight_f4, dim, vocab_size
    );
}