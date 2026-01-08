#include <stdint.h>
#include <cuda_runtime.h> // 包含 cudaStream_t 定义
#include <cuda_bf16.h>
#include <cublasLt.h>
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            /* 在生产代码中可能需要更复杂的错误处理机制 */                           \
        }                                                                         \
    } while (0)
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 在 CUDA 设备上执行 Flash Attention GQA (Prefill/Decode 模式)。
 * 
 * 该函数是 Rust FFI 的入口点，调用底层的 CUDA Kernel 来计算 O = Softmax(Q K^T) V。
 * 
 * @param q_ptr Query 张量的设备指针 (只读), 形状 [Q_S, Q_D]。
 * @param k_ptr K Cache 的设备指针 (只读), 形状 [Max_S, KV_D]。
 * @param v_ptr V Cache 的设备指针 (只读), 形状 [Max_S, KV_D]。
 * @param o_ptr 输出张量的设备指针 (可写), 形状 [Q_S, Q_D]。
 * @param q_seq_len Q 的实际序列长度 (S_Q)。
 * @param kv_seq_len K/V Cache 的有效历史长度 (S_KV_history)。不包含最新的长度！
 * @param num_q_heads Query 头数量 (N_Q)。
 * @param num_kv_heads K/V 头数量 (N_KV)。
 * @param head_dim 单个 Attention Head 的维度 (D_H)。
 * @param stream CUDA stream (可为 NULL/0)。
 */
void flash_attn_gqa_cu(
    const float* q_ptr,
    const float* k_ptr,
    const float* v_ptr,
    float* o_ptr,
    int32_t q_seq_len,
    int32_t kv_seq_len,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    cudaStream_t stream
);

void flash_decoding_cu(
    const float* q_ptr,
    const float* k_ptr,
    const float* v_ptr,
    float* o_ptr,
    int32_t q_seq_len,
    int32_t kv_seq_len,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    cudaStream_t stream
);

void flash_decoding_cu_bf16(
    const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr,
    const __nv_bfloat16* v_ptr,
    __nv_bfloat16* o_ptr,
    int32_t kv_seq_len,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    cudaStream_t stream);

void launch_flash_attn_cute_128x64x64_tile(
    const __nv_bfloat16* d_Q, const __nv_bfloat16* d_K, const __nv_bfloat16* d_V, __nv_bfloat16* d_O,
    int seq_len, int kv_len, int q_heads, int kv_heads,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif