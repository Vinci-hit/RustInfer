#include "rope_interleaved.h"

// x shape: [seq, n_heads, head_dim]
// For each token s, head h, pair k (0..head_dim/2):
//   a = x[s, h, 2k]; b = x[s, h, 2k+1]
//   x[s, h, 2k]   = a*cos[s,k] - b*sin[s,k]
//   x[s, h, 2k+1] = a*sin[s,k] + b*cos[s,k]

template<typename T>
__device__ __forceinline__ float to_f(T);
template<> __device__ __forceinline__ float to_f(__nv_bfloat16 x){ return __bfloat162float(x); }
template<> __device__ __forceinline__ float to_f(float x){ return x; }

template<typename T>
__device__ __forceinline__ T from_f(float);
template<> __device__ __forceinline__ __nv_bfloat16 from_f(float x){ return __float2bfloat16(x); }
template<> __device__ __forceinline__ float from_f(float x){ return x; }

template<typename T>
__global__ void rope_interleaved_kernel(
    T* x,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int seq, int n_heads, int head_dim)
{
    int s = blockIdx.x;
    int h = blockIdx.y;
    int half = head_dim / 2;
    int tid = threadIdx.x;

    const float* cos_row = cos_cache + s * half;
    const float* sin_row = sin_cache + s * half;

    T* x_row = x + (s * n_heads + h) * head_dim;

    for (int k = tid; k < half; k += blockDim.x) {
        float c = cos_row[k];
        float si = sin_row[k];
        float a = to_f<T>(x_row[2*k]);
        float b = to_f<T>(x_row[2*k + 1]);
        x_row[2*k]     = from_f<T>(a * c - b * si);
        x_row[2*k + 1] = from_f<T>(a * si + b * c);
    }
}

extern "C" void rope_interleaved_f32_forward(
    float* x,
    const float* cos_cache, const float* sin_cache,
    int seq, int n_heads, int head_dim,
    cudaStream_t stream)
{
    int half = head_dim / 2;
    int threads = half < 256 ? ((half + 31) / 32 * 32) : 256;
    if (threads < 32) threads = 32;
    dim3 blocks(seq, n_heads);
    rope_interleaved_kernel<float><<<blocks, threads, 0, stream>>>(
        x, cos_cache, sin_cache, seq, n_heads, head_dim);
}

extern "C" void rope_interleaved_bf16_forward(
    __nv_bfloat16* x,
    const float* cos_cache, const float* sin_cache,
    int seq, int n_heads, int head_dim,
    cudaStream_t stream)
{
    int half = head_dim / 2;
    int threads = half < 256 ? ((half + 31) / 32 * 32) : 256;
    if (threads < 32) threads = 32;
    dim3 blocks(seq, n_heads);
    rope_interleaved_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        x, cos_cache, sin_cache, seq, n_heads, head_dim);
}
