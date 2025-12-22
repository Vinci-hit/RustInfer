#include <stdint.h>
#include <cuda_runtime.h> // 包含 cudaStream_t 定义
#include <cuda_bf16.h>    // 包含 __nv_bfloat16 定义

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

void sin_cos_cache_calc_cu(
    int32_t head_size,
    int32_t max_seq_len,
    float* sin_cache,
    float* cos_cache,
    cudaStream_t stream);

/**
 * @brief 在 CUDA 设备上执行 Rotary Positional Embedding (RoPE) 旋转核 (BF16版本)。
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
void rope_kernel_cu_bf16(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    __nv_bfloat16* input_q,
    __nv_bfloat16* input_k,
    int32_t input_pos,
    int32_t seq_len,
    __nv_bfloat16* sin_cache,
    __nv_bfloat16* cos_cache,
    cudaStream_t stream);

/**
 * @brief 计算并填充正弦和余弦旋转嵌入 (RoPE) 的缓存的 CUDA 内核 (BF16版本)。
 *
 * @param head_size 旋转维度的大小 (K)。
 * @param max_seq_len 序列的最大长度 (M)。
 * @param sin_cache 正弦值输出张量的设备指针, 形状 [max_seq_len, head_size]。
 * @param cos_cache 余弦值输出张量的设备指针, 形状 [max_seq_len, head_size]。
 * @param stream CUDA stream (可为 NULL/0)。
 */
void sin_cos_cache_calc_cu_bf16(
    int32_t head_size,
    int32_t max_seq_len,
    __nv_bfloat16* sin_cache,
    __nv_bfloat16* cos_cache,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
