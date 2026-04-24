#include "scalar.h"

// ===================== scalar_mul kernels =====================

__global__ void scalar_mul_f32_kernel(float* dst, const float* src, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // float4 vectorized path
    int n4 = n / 4;
    for (int i = idx; i < n4; i += stride) {
        float4 v = reinterpret_cast<const float4*>(src)[i];
        v.x *= val; v.y *= val; v.z *= val; v.w *= val;
        reinterpret_cast<float4*>(dst)[i] = v;
    }
    // tail elements
    int base = n4 * 4;
    for (int i = base + idx; i < n; i += stride) {
        dst[i] = src[i] * val;
    }
}

void scalar_mul_f32_forward(float* dst, const float* src, float val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_mul_f32_kernel<<<blocks, threads, 0, stream>>>(dst, src, val, n);
}

__global__ void scalar_mul_bf16_kernel(__nv_bfloat16* dst, const __nv_bfloat16* src, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    __nv_bfloat162 val2 = __float2bfloat162_rn(val);
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __nv_bfloat162 v = reinterpret_cast<const __nv_bfloat162*>(src)[i];
        reinterpret_cast<__nv_bfloat162*>(dst)[i] = __hmul2(v, val2);
    }
    if (n % 2 != 0 && idx == 0) {
        dst[n - 1] = __float2bfloat16(__bfloat162float(src[n - 1]) * val);
    }
}

void scalar_mul_bf16_forward(__nv_bfloat16* dst, const __nv_bfloat16* src, float val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_mul_bf16_kernel<<<blocks, threads, 0, stream>>>(dst, src, val, n);
}

__global__ void scalar_mul_f16_kernel(__half* dst, const __half* src, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    __half2 val2 = __float2half2_rn(val);
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __half2 v = reinterpret_cast<const __half2*>(src)[i];
        reinterpret_cast<__half2*>(dst)[i] = __hmul2(v, val2);
    }
    if (n % 2 != 0 && idx == 0) {
        dst[n - 1] = __float2half(__half2float(src[n - 1]) * val);
    }
}

void scalar_mul_f16_forward(__half* dst, const __half* src, float val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_mul_f16_kernel<<<blocks, threads, 0, stream>>>(dst, src, val, n);
}

// ===================== scalar_add kernels =====================

__global__ void scalar_add_f32_kernel(float* dst, const float* src, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n4 = n / 4;
    for (int i = idx; i < n4; i += stride) {
        float4 v = reinterpret_cast<const float4*>(src)[i];
        v.x += val; v.y += val; v.z += val; v.w += val;
        reinterpret_cast<float4*>(dst)[i] = v;
    }
    int base = n4 * 4;
    for (int i = base + idx; i < n; i += stride) {
        dst[i] = src[i] + val;
    }
}

void scalar_add_f32_forward(float* dst, const float* src, float val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_add_f32_kernel<<<blocks, threads, 0, stream>>>(dst, src, val, n);
}

__global__ void scalar_add_bf16_kernel(__nv_bfloat16* dst, const __nv_bfloat16* src, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    __nv_bfloat162 val2 = __float2bfloat162_rn(val);
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __nv_bfloat162 v = reinterpret_cast<const __nv_bfloat162*>(src)[i];
        reinterpret_cast<__nv_bfloat162*>(dst)[i] = __hadd2(v, val2);
    }
    if (n % 2 != 0 && idx == 0) {
        dst[n - 1] = __float2bfloat16(__bfloat162float(src[n - 1]) + val);
    }
}

void scalar_add_bf16_forward(__nv_bfloat16* dst, const __nv_bfloat16* src, float val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_add_bf16_kernel<<<blocks, threads, 0, stream>>>(dst, src, val, n);
}

__global__ void scalar_add_f16_kernel(__half* dst, const __half* src, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    __half2 val2 = __float2half2_rn(val);
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __half2 v = reinterpret_cast<const __half2*>(src)[i];
        reinterpret_cast<__half2*>(dst)[i] = __hadd2(v, val2);
    }
    if (n % 2 != 0 && idx == 0) {
        dst[n - 1] = __float2half(__half2float(src[n - 1]) + val);
    }
}

void scalar_add_f16_forward(__half* dst, const __half* src, float val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_add_f16_kernel<<<blocks, threads, 0, stream>>>(dst, src, val, n);
}
