#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

void embedding_kernel_cu_bf16x8(
    __nv_bfloat16* output,
    const int* input_token_ids,
    const __nv_bfloat16* weight,
    int token_len,
    int dim,
    int vocab_size,
    cudaStream_t stream
);

/// @brief 执行 Embedding 查找，采用 float4 向量化访存优化。
///        要求 embedding_dim 必须是 4 的倍数。
///
/// @param output           输出张量的设备指针, shape: [token_len, dim]。
/// @param input_token_ids  输入的 token ID 数组的设备指针, shape: [token_len]。
/// @param weight           权重矩阵（查询表）的设备指针, shape: [vocab_size, dim]。
/// @param token_len        输入的 token 数量。
/// @param dim              嵌入维度。
/// @param vocab_size       词表范围
/// @param stream           CUDA stream。
void embedding_kernel_cu_fp32x4(
    float* output,
    const int* input_token_ids,
    const float* weight,
    int token_len,
    int dim,
    int vocab_size,
    cudaStream_t stream
);


#ifdef __cplusplus
}
#endif