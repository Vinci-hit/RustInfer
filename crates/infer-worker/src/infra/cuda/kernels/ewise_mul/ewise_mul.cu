#include "ewise_mul.h"

template<typename T> __device__ __forceinline__ float to_float(T x);
template<> __device__ __forceinline__ float to_float(__nv_bfloat16 x){ return __bfloat162float(x); }
template<> __device__ __forceinline__ float to_float(__half x){ return __half2float(x); }
template<> __device__ __forceinline__ float to_float(float x){ return x; }

template<typename T> __device__ __forceinline__ T from_float(float x);
template<> __device__ __forceinline__ __nv_bfloat16 from_float(float x){ return __float2bfloat16(x); }
template<> __device__ __forceinline__ __half from_float(float x){ return __float2half(x); }
template<> __device__ __forceinline__ float from_float(float x){ return x; }

template<typename T>
__global__ void ewise_mul_kernel(T* dst, const T* a, const T* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        dst[i] = from_float<T>(to_float<T>(a[i]) * to_float<T>(b[i]));
    }
}

template<typename T>
__global__ void ewise_mul_inplace_kernel(T* a, const T* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        a[i] = from_float<T>(to_float<T>(a[i]) * to_float<T>(b[i]));
    }
}

#define DEFINE_LAUNCHERS(TYPE, SUFFIX) \
extern "C" void ewise_mul_##SUFFIX##_forward(TYPE* dst, const TYPE* a, const TYPE* b, int n, cudaStream_t stream) { \
    int threads = 256; int blocks = (n + threads - 1) / threads; if (blocks < 1) blocks = 1; \
    ewise_mul_kernel<TYPE><<<blocks, threads, 0, stream>>>(dst, a, b, n); \
} \
extern "C" void ewise_mul_inplace_##SUFFIX##_forward(TYPE* a, const TYPE* b, int n, cudaStream_t stream) { \
    int threads = 256; int blocks = (n + threads - 1) / threads; if (blocks < 1) blocks = 1; \
    ewise_mul_inplace_kernel<TYPE><<<blocks, threads, 0, stream>>>(a, b, n); \
}

DEFINE_LAUNCHERS(float, f32)
DEFINE_LAUNCHERS(__nv_bfloat16, bf16)
DEFINE_LAUNCHERS(__half, f16)
