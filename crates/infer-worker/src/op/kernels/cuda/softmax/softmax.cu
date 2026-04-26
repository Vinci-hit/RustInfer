#include "softmax.h"
#include <cfloat>

// Softmax kernel: 每行一个 block，warp shuffle reduction
// 支持任意列数

__global__ void softmax_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    // Pass 1: find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += nthreads) {
        local_max = fmaxf(local_max, in_row[i]);
    }
    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    // Block reduce max
    __shared__ float warp_vals[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_vals[warp_id] = local_max;
    __syncthreads();
    __shared__ float row_max;
    if (tid == 0) {
        float m = -FLT_MAX;
        int nwarps = nthreads / 32;
        for (int i = 0; i < nwarps; i++) m = fmaxf(m, warp_vals[i]);
        row_max = m;
    }
    __syncthreads();

    // Pass 2: exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += nthreads) {
        float v = expf(in_row[i] - row_max);
        out_row[i] = v;
        local_sum += v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (lane_id == 0) warp_vals[warp_id] = local_sum;
    __syncthreads();
    __shared__ float row_sum;
    if (tid == 0) {
        float s = 0.0f;
        int nwarps = nthreads / 32;
        for (int i = 0; i < nwarps; i++) s += warp_vals[i];
        row_sum = s;
    }
    __syncthreads();

    // Pass 3: normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < cols; i += nthreads) {
        out_row[i] *= inv_sum;
    }
}

void softmax_f32_forward(float* output, const float* input, int rows, int cols, cudaStream_t stream) {
    int threads = (cols < 256) ? 128 : 256;
    softmax_f32_kernel<<<rows, threads, 0, stream>>>(output, input, cols);
}

// BF16 版本 (内部 f32 累加)
__global__ void softmax_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const __nv_bfloat16* in_row = input + row * cols;
    __nv_bfloat16* out_row = output + row * cols;

    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += nthreads)
        local_max = fmaxf(local_max, __bfloat162float(in_row[i]));
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    __shared__ float warp_vals[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_vals[warp_id] = local_max;
    __syncthreads();
    __shared__ float row_max;
    if (tid == 0) {
        float m = -FLT_MAX;
        int nwarps = nthreads / 32;
        for (int i = 0; i < nwarps; i++) m = fmaxf(m, warp_vals[i]);
        row_max = m;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += nthreads) {
        float v = expf(__bfloat162float(in_row[i]) - row_max);
        out_row[i] = __float2bfloat16(v);
        local_sum += v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (lane_id == 0) warp_vals[warp_id] = local_sum;
    __syncthreads();
    __shared__ float row_sum;
    if (tid == 0) {
        float s = 0.0f;
        int nwarps = nthreads / 32;
        for (int i = 0; i < nwarps; i++) s += warp_vals[i];
        row_sum = s;
    }
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < cols; i += nthreads) {
        out_row[i] = __float2bfloat16(__bfloat162float(out_row[i]) * inv_sum);
    }
}

void softmax_bf16_forward(__nv_bfloat16* output, const __nv_bfloat16* input, int rows, int cols, cudaStream_t stream) {
    int threads = (cols < 256) ? 128 : 256;
    softmax_bf16_kernel<<<rows, threads, 0, stream>>>(output, input, cols);
}
