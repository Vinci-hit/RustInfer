#include "rope.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(err));                                    \
        }                                                                       \
    } while (0)

namespace {

template <typename T>
__device__ __forceinline__ float load_float(T value);

template <>
__device__ __forceinline__ float load_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float load_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <>
__device__ __forceinline__ float load_float<__half>(__half value) {
    return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T store_float(float value);

template <>
__device__ __forceinline__ float store_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ __half store_float<__half>(float value) {
    return __float2half_rn(value);
}

template <typename T>
__global__ void rope_partial_kernel(
    T* q,
    T* k,
    const T* __restrict__ sin_cache,
    const T* __restrict__ cos_cache,
    const int* __restrict__ positions,
    int max_heads,
    int head_num,
    int kv_head_num,
    int head_dim,
    int rotary_dim,
    long long q_row_stride,
    long long k_row_stride) {
    const int flat_block = static_cast<int>(blockIdx.x);
    const int token = flat_block / max_heads;
    const int head = flat_block - token * max_heads;
    const int pair = static_cast<int>(threadIdx.x);
    const int half_rotary = rotary_dim / 2;
    if (pair >= half_rotary) {
        return;
    }

    const int position = positions[token];
    const long long cache_offset = static_cast<long long>(position) * half_rotary + pair;
    const float sin_value = load_float(sin_cache[cache_offset]);
    const float cos_value = load_float(cos_cache[cache_offset]);

    if (head < head_num) {
        T* q_head = q + static_cast<long long>(token) * q_row_stride
                      + static_cast<long long>(head) * head_dim;
        const float left = load_float(q_head[pair]);
        const float right = load_float(q_head[pair + half_rotary]);
        q_head[pair] = store_float<T>(left * cos_value - right * sin_value);
        q_head[pair + half_rotary] = store_float<T>(left * sin_value + right * cos_value);
    }
    if (head < kv_head_num) {
        T* k_head = k + static_cast<long long>(token) * k_row_stride
                      + static_cast<long long>(head) * head_dim;
        const float left = load_float(k_head[pair]);
        const float right = load_float(k_head[pair + half_rotary]);
        k_head[pair] = store_float<T>(left * cos_value - right * sin_value);
        k_head[pair + half_rotary] = store_float<T>(left * sin_value + right * cos_value);
    }
}

template <typename T>
void launch_rope(
    T* q,
    T* k,
    const T* sin_cache,
    const T* cos_cache,
    const int* positions,
    int num_tokens,
    int head_num,
    int kv_head_num,
    int head_dim,
    int rotary_dim,
    long long q_row_stride,
    long long k_row_stride,
    cudaStream_t stream) {
    if (num_tokens <= 0) {
        return;
    }
    const int max_heads = head_num > kv_head_num ? head_num : kv_head_num;
    const int blocks = num_tokens * max_heads;
    const int threads = rotary_dim / 2;
    rope_partial_kernel<T><<<blocks, threads, 0, stream>>>(
        q,
        k,
        sin_cache,
        cos_cache,
        positions,
        max_heads,
        head_num,
        kv_head_num,
        head_dim,
        rotary_dim,
        q_row_stride,
        k_row_stride);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void sin_cos_calc_kernel(
    int rotary_dim,
    int max_seq_len,
    float rope_theta,
    T* sin_cache,
    T* cos_cache,
    float factor,
    float low_freq_factor,
    float high_freq_factor,
    float original_max_pos_emb) {
    const int frequency_index = static_cast<int>(threadIdx.x + blockDim.x * blockIdx.x);
    const int half_rotary = rotary_dim / 2;
    if (frequency_index >= half_rotary) {
        return;
    }

    const int dim = 2 * frequency_index;
    float frequency = 1.0f / powf(rope_theta, static_cast<float>(dim) / rotary_dim);
    if (factor > 1.0f) {
        const float low_freq_wavelen = original_max_pos_emb / low_freq_factor;
        const float high_freq_wavelen = original_max_pos_emb / high_freq_factor;
        const float wavelen = 2.0f * 3.14159265358979323846f / frequency;
        if (wavelen > low_freq_wavelen) {
            frequency /= factor;
        } else if (wavelen >= high_freq_wavelen) {
            const float smooth =
                (original_max_pos_emb / wavelen - low_freq_factor)
                / (high_freq_factor - low_freq_factor);
            frequency = (1.0f - smooth) * frequency / factor + smooth * frequency;
        }
    }

    for (int position = 0; position < max_seq_len; ++position) {
        const float angle = static_cast<float>(position) * frequency;
        const int offset = position * half_rotary + frequency_index;
        sin_cache[offset] = store_float<T>(sinf(angle));
        cos_cache[offset] = store_float<T>(cosf(angle));
    }
}

template <typename T>
void launch_sin_cos(
    int rotary_dim,
    int max_seq_len,
    float rope_theta,
    T* sin_cache,
    T* cos_cache,
    float factor,
    float low_freq_factor,
    float high_freq_factor,
    float original_max_pos_emb,
    cudaStream_t stream) {
    constexpr int threads = 256;
    const int half_rotary = rotary_dim / 2;
    const int blocks = (half_rotary + threads - 1) / threads;
    sin_cos_calc_kernel<T><<<blocks, threads, 0, stream>>>(
        rotary_dim,
        max_seq_len,
        rope_theta,
        sin_cache,
        cos_cache,
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_pos_emb);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

extern "C" void rope_kernel_cu(
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
    cudaStream_t stream) {
    launch_rope(
        q, k, sin_cache, cos_cache, positions, num_tokens, head_num,
        kv_head_num, head_dim, rotary_dim, q_row_stride, k_row_stride, stream);
}

extern "C" void rope_kernel_cu_bf16(
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
    cudaStream_t stream) {
    launch_rope(
        q, k, sin_cache, cos_cache, positions, num_tokens, head_num,
        kv_head_num, head_dim, rotary_dim, q_row_stride, k_row_stride, stream);
}

extern "C" void rope_kernel_cu_fp16(
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
    cudaStream_t stream) {
    launch_rope(
        q, k, sin_cache, cos_cache, positions, num_tokens, head_num,
        kv_head_num, head_dim, rotary_dim, q_row_stride, k_row_stride, stream);
}

extern "C" void sin_cos_cache_calc_cu(
    int32_t rotary_dim,
    int32_t max_seq_len,
    float rope_theta,
    float* sin_cache,
    float* cos_cache,
    cudaStream_t stream) {
    launch_sin_cos(
        rotary_dim, max_seq_len, rope_theta, sin_cache, cos_cache,
        1.0f, 1.0f, 1.0f, 1.0f, stream);
}

extern "C" void sin_cos_cache_calc_cu_bf16(
    int32_t rotary_dim,
    int32_t max_seq_len,
    float rope_theta,
    __nv_bfloat16* sin_cache,
    __nv_bfloat16* cos_cache,
    float factor,
    float low_freq_factor,
    float high_freq_factor,
    float original_max_pos_emb,
    cudaStream_t stream) {
    launch_sin_cos(
        rotary_dim, max_seq_len, rope_theta, sin_cache, cos_cache,
        factor, low_freq_factor, high_freq_factor, original_max_pos_emb, stream);
}

extern "C" void sin_cos_cache_calc_cu_fp16(
    int32_t rotary_dim,
    int32_t max_seq_len,
    float rope_theta,
    __half* sin_cache,
    __half* cos_cache,
    float factor,
    float low_freq_factor,
    float high_freq_factor,
    float original_max_pos_emb,
    cudaStream_t stream) {
    launch_sin_cos(
        rotary_dim, max_seq_len, rope_theta, sin_cache, cos_cache,
        factor, low_freq_factor, high_freq_factor, original_max_pos_emb, stream);
}
