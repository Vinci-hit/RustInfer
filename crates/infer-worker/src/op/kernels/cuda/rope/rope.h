#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

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
    cudaStream_t stream
);

void sin_cos_cache_calc_cu(
    int32_t head_size,
    int32_t max_seq_len,
    float rope_theta,
    float* sin_cache,
    float* cos_cache,
    cudaStream_t stream);

void rope_kernel_cu_bf16(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    __nv_bfloat16* input_q,
    __nv_bfloat16* input_k,
    int32_t* input_pos,
    int32_t seq_len,
    __nv_bfloat16* sin_cache,
    __nv_bfloat16* cos_cache,
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

void rope_kernel_cu_fp16(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    __half* input_q,
    __half* input_k,
    int32_t* input_pos,
    int32_t seq_len,
    __half* sin_cache,
    __half* cos_cache,
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
