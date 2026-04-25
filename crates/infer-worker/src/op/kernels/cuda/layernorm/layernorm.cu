#include "layernorm.h"
#include <math.h>

// ==================== F32 LayerNorm ====================
// One warp (32 threads) per row. Uses warp shuffle for reduction.
// For cols > 1024, use multi-pass; for cols <= 1024, single-pass.

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Shared-memory block reduction for large cols
__device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

__global__ void layernorm_f32_kernel(float* output, const float* input, int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Pass 1: compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += in_row[i];
    }
    sum = block_reduce_sum(sum, smem);
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = sum / (float)cols;
    __syncthreads();
    float mean = s_mean;

    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = in_row[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, smem);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(var_sum / (float)cols + eps);
    __syncthreads();
    float rstd = s_rstd;

    // Pass 3: normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = (in_row[i] - mean) * rstd;
    }
}

void layernorm_f32_forward(float* output, const float* input, int rows, int cols, float eps, cudaStream_t stream) {
    int threads = (cols < 1024) ? ((cols + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    int smem_size = (threads / 32) * sizeof(float);
    layernorm_f32_kernel<<<rows, threads, smem_size, stream>>>(output, input, rows, cols, eps);
}

// ==================== BF16 LayerNorm ====================

__global__ void layernorm_bf16_kernel(__nv_bfloat16* output, const __nv_bfloat16* input, int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* in_row = input + row * cols;
    __nv_bfloat16* out_row = output + row * cols;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += __bfloat162float(in_row[i]);
    }
    sum = block_reduce_sum(sum, smem);
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = sum / (float)cols;
    __syncthreads();
    float mean = s_mean;

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = __bfloat162float(in_row[i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, smem);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(var_sum / (float)cols + eps);
    __syncthreads();
    float rstd = s_rstd;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = __float2bfloat16((__bfloat162float(in_row[i]) - mean) * rstd);
    }
}

void layernorm_bf16_forward(__nv_bfloat16* output, const __nv_bfloat16* input, int rows, int cols, float eps, cudaStream_t stream) {
    int threads = (cols < 1024) ? ((cols + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    int smem_size = (threads / 32) * sizeof(float);
    layernorm_bf16_kernel<<<rows, threads, smem_size, stream>>>(output, input, rows, cols, eps);
}

// ==================== F16 LayerNorm ====================

__global__ void layernorm_f16_kernel(__half* output, const __half* input, int rows, int cols, float eps) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* in_row = input + row * cols;
    __half* out_row = output + row * cols;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += __half2float(in_row[i]);
    }
    sum = block_reduce_sum(sum, smem);
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = sum / (float)cols;
    __syncthreads();
    float mean = s_mean;

    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = __half2float(in_row[i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, smem);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(var_sum / (float)cols + eps);
    __syncthreads();
    float rstd = s_rstd;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = __float2half((__half2float(in_row[i]) - mean) * rstd);
    }
}

void layernorm_f16_forward(__half* output, const __half* input, int rows, int cols, float eps, cudaStream_t stream) {
    int threads = (cols < 1024) ? ((cols + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    int smem_size = (threads / 32) * sizeof(float);
    layernorm_f16_kernel<<<rows, threads, smem_size, stream>>>(output, input, rows, cols, eps);
}
