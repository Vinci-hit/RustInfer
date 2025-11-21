#include "emb.h"

// ============================================================================
//  在这里填写您的 CUDA C++ 内核实现
// ============================================================================

/*
 * 优化思路：
 * 
 * 1.  Grid 划分：
 *     - 使用 2D Grid: `dim3 grid(num_vecs_per_row, num_tokens)`
 *       - x 维度对应 embedding_dim 内部的向量化 chunk。
 *       - y 维度对应不同的 token。
 *     - 每个线程负责拷贝一个 `float4` (或 `float2`)。
 * 
 * 2.  向量化访存 (float4):
 *     - 将 `weight` 和 `output` 指针强制转换为 `float4*`。
 *     - 每个线程计算其在目标 `output` 行中的 `float4` 索引 `vec_idx`。
 *     - 从 `weight` 矩阵的 `token_id` 对应行的 `vec_idx` 位置读取一个 `float4`。
 *     - 将这个 `float4` 直接写入 `output` 矩阵的对应位置。
 *     - 这是一个纯粹的、并行的内存拷贝操作。
 */

__global__ void embedding_kernel(
    float4* output,
    const int* input_token_ids,
    const float4* weight,
    int token_len, int dim, int vocab_size
) {
    const int token_idx = blockIdx.x;

    if (token_idx >= token_len) {
        return;
    }
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
    
    constexpr int32_t max_seq_len = 512;
    constexpr int32_t thread_num = 128;
    
    // --- 类型转换 ---
    float4* out_f4 = reinterpret_cast<float4*>(output);
    const float4* weight_f4 = reinterpret_cast<const float4*>(weight);
    dim /= 4;
    // --- 启动内核 ---
    embedding_kernel<<<max_seq_len, thread_num, 0, stream>>>(
        out_f4, input_token_ids, weight_f4, token_len, dim, vocab_size
    );
}