#include "broadcast_mul.h"

// dst[row * D + col] = a[row * D + col] * b[col]
// 1 thread per element, vectorized with float4 on the D dimension

__global__ void broadcast_mul_f32_kernel(float* dst, const float* a, const float* b, int rows, int D) {
    int total = rows * D;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // float4 vectorized path (4 consecutive cols)
    int total4 = total / 4;
    for (int i = idx; i < total4; i += stride) {
        int flat = i * 4;
        float4 va = reinterpret_cast<const float4*>(a)[i];
        int col0 = flat % D;
        // only vectorize if all 4 elements are in the same row (col0 + 3 < D)
        if (col0 + 3 < D) {
            float b0 = b[col0], b1 = b[col0+1], b2 = b[col0+2], b3 = b[col0+3];
            va.x *= b0; va.y *= b1; va.z *= b2; va.w *= b3;
            reinterpret_cast<float4*>(dst)[i] = va;
        } else {
            // fallback: element-wise
            float vals[4] = {va.x, va.y, va.z, va.w};
            for (int k = 0; k < 4; k++) {
                int f = flat + k;
                dst[f] = a[f] * b[f % D];
            }
        }
    }
    int base = total4 * 4;
    for (int i = base + idx; i < total; i += stride) {
        dst[i] = a[i] * b[i % D];
    }
}

void broadcast_mul_f32_forward(float* dst, const float* a, const float* b, int rows, int D, cudaStream_t stream) {
    int total = rows * D;
    constexpr int threads = 256;
    int blocks = (total / 4 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    broadcast_mul_f32_kernel<<<blocks, threads, 0, stream>>>(dst, a, b, rows, D);
}

__global__ void broadcast_mul_bf16_kernel(__nv_bfloat16* dst, const __nv_bfloat16* a, const __nv_bfloat16* b, int rows, int D) {
    int total = rows * D;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total2 = total / 2;
    for (int i = idx; i < total2; i += stride) {
        int flat = i * 2;
        __nv_bfloat162 va = reinterpret_cast<const __nv_bfloat162*>(a)[i];
        int col0 = flat % D;
        if (col0 + 1 < D) {
            __nv_bfloat162 vb = *reinterpret_cast<const __nv_bfloat162*>(&b[col0]);
            reinterpret_cast<__nv_bfloat162*>(dst)[i] = __hmul2(va, vb);
        } else {
            dst[flat] = __float2bfloat16(__bfloat162float(a[flat]) * __bfloat162float(b[flat % D]));
            if (flat + 1 < total)
                dst[flat+1] = __float2bfloat16(__bfloat162float(a[flat+1]) * __bfloat162float(b[(flat+1) % D]));
        }
    }
    if (total % 2 != 0 && idx == 0) {
        int f = total - 1;
        dst[f] = __float2bfloat16(__bfloat162float(a[f]) * __bfloat162float(b[f % D]));
    }
}

void broadcast_mul_bf16_forward(__nv_bfloat16* dst, const __nv_bfloat16* a, const __nv_bfloat16* b, int rows, int D, cudaStream_t stream) {
    int total = rows * D;
    constexpr int threads = 256;
    int blocks = (total / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    broadcast_mul_bf16_kernel<<<blocks, threads, 0, stream>>>(dst, a, b, rows, D);
}

__global__ void broadcast_mul_f16_kernel(__half* dst, const __half* a, const __half* b, int rows, int D) {
    int total = rows * D;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total2 = total / 2;
    for (int i = idx; i < total2; i += stride) {
        int flat = i * 2;
        __half2 va = reinterpret_cast<const __half2*>(a)[i];
        int col0 = flat % D;
        if (col0 + 1 < D) {
            __half2 vb = *reinterpret_cast<const __half2*>(&b[col0]);
            reinterpret_cast<__half2*>(dst)[i] = __hmul2(va, vb);
        } else {
            dst[flat] = __float2half(__half2float(a[flat]) * __half2float(b[flat % D]));
            if (flat + 1 < total)
                dst[flat+1] = __float2half(__half2float(a[flat+1]) * __half2float(b[(flat+1) % D]));
        }
    }
    if (total % 2 != 0 && idx == 0) {
        int f = total - 1;
        dst[f] = __float2half(__half2float(a[f]) * __half2float(b[f % D]));
    }
}

void broadcast_mul_f16_forward(__half* dst, const __half* a, const __half* b, int rows, int D, cudaStream_t stream) {
    int total = rows * D;
    constexpr int threads = 256;
    int blocks = (total / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    broadcast_mul_f16_kernel<<<blocks, threads, 0, stream>>>(dst, a, b, rows, D);
}
