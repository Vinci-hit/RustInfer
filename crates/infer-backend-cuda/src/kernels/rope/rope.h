#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- F32 RoPE (z-image 等用，保留旧 pos_offset + seq_idx 语义) -----
void rope_kernel_cu(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    float* input_q,
    float* input_k,
    int32_t* input_pos,
    int32_t seq_len,
    const float* sin_cache,
    const float* cos_cache,
    cudaStream_t stream);

void sin_cos_cache_calc_cu(
    int32_t head_size,
    int32_t max_seq_len,
    float rope_theta,
    float* sin_cache,
    float* cos_cache,
    cudaStream_t stream);

// ---- BF16 / FP16 RoPE：per-row pos，唯一版本 ----
//
// positions[i] 给第 i 行的绝对位置。prefill 时 caller 把 [p, p+1, ..., p+seq_len-1]
// 写入；decode / batch decode 时 caller 按每 seq 的实际 pos 填。
// q_row_stride / k_row_stride 允许 input 非连续（比如从 fused qkv slice 出来的
// q/k 段），单位都是元素数（bf16/fp16）。
void rope_kernel_cu_bf16(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    __nv_bfloat16* input_q,
    __nv_bfloat16* input_k,
    const int32_t* positions,      // [seq_len]
    int32_t seq_len,
    int32_t q_row_stride,
    int32_t k_row_stride,
    __nv_bfloat16* sin_cache,
    __nv_bfloat16* cos_cache,
    cudaStream_t stream);

void rope_kernel_cu_fp16(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    __half* input_q,
    __half* input_k,
    const int32_t* positions,
    int32_t seq_len,
    int32_t q_row_stride,
    int32_t k_row_stride,
    __half* sin_cache,
    __half* cos_cache,
    cudaStream_t stream);

void sin_cos_cache_calc_cu_bf16(
    int32_t head_size,
    int32_t max_seq_len,
    float rope_theta,
    __nv_bfloat16* sin_cache,
    __nv_bfloat16* cos_cache,
    float factor,
    float low_freq_factor,
    float high_freq_factor,
    float original_max_pos_emb,
    cudaStream_t stream);

void sin_cos_cache_calc_cu_fp16(
    int32_t head_size,
    int32_t max_seq_len,
    float rope_theta,
    __half* sin_cache,
    __half* cos_cache,
    float factor,
    float low_freq_factor,
    float high_freq_factor,
    float original_max_pos_emb,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
