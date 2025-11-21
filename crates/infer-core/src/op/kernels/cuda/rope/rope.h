#include <stdint.h>
#include <cuda_runtime.h> // 包含 cudaStream_t 定义

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 在 CUDA 设备上执行 Rotary Positional Embedding (RoPE) 旋转核。
 * 
 * 这是一个就地 (in-place) 操作，会修改 input_q 和 input_k 的内容。
 * 
 * @param dim Q 和 K 向量的总旋转维度 (例如: 128)。
 * @param kv_dim K 向量旋转的维度 (例如: 64，如果 kv_dim < dim)。
 * @param head_size Attention Head 的大小 (用于 RoPE 缓存的索引)。
 * @param input_q Q 张量的设备指针 (可读写，将就地修改)。
 * @param input_k K 张量的设备指针 (可读写，将就地修改)。
 * @param input_pos 当前位置索引张量的设备指针 (通常是单个 i32 元素)。
 * @param sin_cache 正弦缓存张量的设备指针 (只读)。
 * @param cos_cache 余弦缓存张量的设备指针 (只读)。
 * @param stream CUDA stream (可为 NULL/0)。
 */
void rope_kernel_cu(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    float* input_q,
    float* input_k,
    int32_t input_pos,
    int32_t seq_len,
    const float* sin_cache,
    const float* cos_cache,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

