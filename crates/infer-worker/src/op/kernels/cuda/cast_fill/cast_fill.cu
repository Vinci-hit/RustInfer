#include "cast_fill.h"

// ===== dtype cast =====

__global__ void cast_f32_to_bf16_k(__nv_bfloat16* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) dst[i] = __float2bfloat16(src[i]);
}
__global__ void cast_bf16_to_f32_k(float* dst, const __nv_bfloat16* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) dst[i] = __bfloat162float(src[i]);
}
__global__ void cast_f32_to_f16_k(__half* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) dst[i] = __float2half(src[i]);
}
__global__ void cast_f16_to_f32_k(float* dst, const __half* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) dst[i] = __half2float(src[i]);
}

#define LAUNCH_CAST(FN, DST_T, SRC_T) \
extern "C" void FN##_forward(DST_T* dst, const SRC_T* src, int n, cudaStream_t stream) { \
    int t = 256; int b = (n + t - 1)/t; if (b<1) b=1; FN##_k<<<b, t, 0, stream>>>(dst, src, n); \
}

LAUNCH_CAST(cast_f32_to_bf16, __nv_bfloat16, float)
LAUNCH_CAST(cast_bf16_to_f32, float, __nv_bfloat16)
LAUNCH_CAST(cast_f32_to_f16, __half, float)
LAUNCH_CAST(cast_f16_to_f32, float, __half)

// ===== broadcast row =====

template<typename T>
__global__ void broadcast_row_k(T* dst, const T* row, int num_rows, int D) {
    int total = num_rows * D;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < total; i += stride) {
        dst[i] = row[i % D];
    }
}

extern "C" void broadcast_row_bf16_forward(__nv_bfloat16* dst, const __nv_bfloat16* row, int num_rows, int D, cudaStream_t stream) {
    int total = num_rows * D;
    int t = 256; int b = (total + t - 1)/t; if (b<1) b=1;
    broadcast_row_k<__nv_bfloat16><<<b, t, 0, stream>>>(dst, row, num_rows, D);
}
extern "C" void broadcast_row_f32_forward(float* dst, const float* row, int num_rows, int D, cudaStream_t stream) {
    int total = num_rows * D;
    int t = 256; int b = (total + t - 1)/t; if (b<1) b=1;
    broadcast_row_k<float><<<b, t, 0, stream>>>(dst, row, num_rows, D);
}

// ===== fill rows [n_src..target_len] with copy of row[n_src-1] =====

template<typename T>
__global__ void fill_repeat_last_k(T* dst, int n_src, int target_len, int D) {
    int pad = target_len - n_src;
    int total = pad * D;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < total; i += stride) {
        int row = i / D;
        int col = i % D;
        dst[(n_src + row) * D + col] = dst[(n_src - 1) * D + col];
    }
}

extern "C" void fill_repeat_last_row_bf16_forward(__nv_bfloat16* dst, int n_src, int target_len, int D, cudaStream_t stream) {
    int pad = target_len - n_src;
    if (pad <= 0) return;
    int total = pad * D;
    int t = 256; int b = (total + t - 1)/t; if (b<1) b=1;
    fill_repeat_last_k<__nv_bfloat16><<<b, t, 0, stream>>>(dst, n_src, target_len, D);
}
extern "C" void fill_repeat_last_row_f32_forward(float* dst, int n_src, int target_len, int D, cudaStream_t stream) {
    int pad = target_len - n_src;
    if (pad <= 0) return;
    int total = pad * D;
    int t = 256; int b = (total + t - 1)/t; if (b<1) b=1;
    fill_repeat_last_k<float><<<b, t, 0, stream>>>(dst, n_src, target_len, D);
}
