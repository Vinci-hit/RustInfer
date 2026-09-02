#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- F32/BF16/FP16 partial-or-full RoPE, one absolute position per row ----
void rope_kernel_cu(
    float* q,
    float* k,
    const float* sin_cache,
    const float* cos_cache,
    const int32_t* positions,
    int32_t num_tokens,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_dim,
    int32_t rotary_dim,
    int64_t q_row_stride,
    int64_t k_row_stride,
    cudaStream_t stream);

void sin_cos_cache_calc_cu(
    int32_t head_size,
    int32_t max_seq_len,
    float rope_theta,
    float* sin_cache,
    float* cos_cache,
    cudaStream_t stream);

void rope_kernel_cu_bf16(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const __nv_bfloat16* sin_cache,
    const __nv_bfloat16* cos_cache,
    const int32_t* positions,
    int32_t num_tokens,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_dim,
    int32_t rotary_dim,
    int64_t q_row_stride,
    int64_t k_row_stride,
    cudaStream_t stream);

void rope_kernel_cu_fp16(
    __half* q,
    __half* k,
    const __half* sin_cache,
    const __half* cos_cache,
    const int32_t* positions,
    int32_t num_tokens,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_dim,
    int32_t rotary_dim,
    int64_t q_row_stride,
    int64_t k_row_stride,
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
