#include "groupnorm.h"
#include <cmath>

// GroupNorm CUDA kernel
// Launch: grid(batch, num_groups), block(256)
// 每个 block 处理一个 (batch, group) 对

__global__ void groupnorm_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int channels, int spatial, int channels_per_group, float eps)
{
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int c_start = group_idx * channels_per_group;
    int group_size = channels_per_group * spatial;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // === Pass 1: 计算 mean ===
    float local_sum = 0.0f;
    for (int i = tid; i < group_size; i += nthreads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        local_sum += input[(batch_idx * channels + c) * spatial + s];
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);

    // Block reduce: 每个 warp 的 lane0 写入 shared memory
    __shared__ float warp_sums[8]; // 256/32 = 8 warps
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    // 由 thread 0 做最终 reduce
    __shared__ float mean_val;
    if (tid == 0) {
        float total = 0.0f;
        int nwarps = nthreads / 32;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        mean_val = total / (float)group_size;
    }
    __syncthreads();

    // === Pass 2: 计算 variance ===
    float local_var = 0.0f;
    for (int i = tid; i < group_size; i += nthreads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        float diff = input[(batch_idx * channels + c) * spatial + s] - mean_val;
        local_var += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_var += __shfl_xor_sync(0xffffffff, local_var, offset);
    if (lane_id == 0) warp_sums[warp_id] = local_var;
    __syncthreads();

    __shared__ float rstd_val;
    if (tid == 0) {
        float total = 0.0f;
        int nwarps = nthreads / 32;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        rstd_val = rsqrtf(total / (float)group_size + eps);
    }
    __syncthreads();

    // === Pass 3: 归一化 + affine ===
    for (int i = tid; i < group_size; i += nthreads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        int idx = (batch_idx * channels + c) * spatial + s;
        float val = (input[idx] - mean_val) * rstd_val;
        output[idx] = val * weight[c] + bias[c];
    }
}

void groupnorm_f32_forward(float* output, const float* input, const float* weight, const float* bias,
    int batch, int channels, int spatial, int num_groups, float eps, cudaStream_t stream)
{
    int channels_per_group = channels / num_groups;
    dim3 grid(batch, num_groups);
    dim3 block(256);
    groupnorm_f32_kernel<<<grid, block, 0, stream>>>(
        output, input, weight, bias, channels, spatial, channels_per_group, eps
    );
}

// === BF16 版本 (内部用 f32 累加) ===

__global__ void groupnorm_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    int channels, int spatial, int channels_per_group, float eps)
{
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int c_start = group_idx * channels_per_group;
    int group_size = channels_per_group * spatial;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float local_sum = 0.0f;
    for (int i = tid; i < group_size; i += nthreads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        local_sum += __bfloat162float(input[(batch_idx * channels + c) * spatial + s]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);

    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    __shared__ float mean_val;
    if (tid == 0) {
        float total = 0.0f;
        int nwarps = nthreads / 32;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        mean_val = total / (float)group_size;
    }
    __syncthreads();

    float local_var = 0.0f;
    for (int i = tid; i < group_size; i += nthreads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        float diff = __bfloat162float(input[(batch_idx * channels + c) * spatial + s]) - mean_val;
        local_var += diff * diff;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_var += __shfl_xor_sync(0xffffffff, local_var, offset);
    if (lane_id == 0) warp_sums[warp_id] = local_var;
    __syncthreads();

    __shared__ float rstd_val;
    if (tid == 0) {
        float total = 0.0f;
        int nwarps = nthreads / 32;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        rstd_val = rsqrtf(total / (float)group_size + eps);
    }
    __syncthreads();

    for (int i = tid; i < group_size; i += nthreads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        int idx = (batch_idx * channels + c) * spatial + s;
        float val = (__bfloat162float(input[idx]) - mean_val) * rstd_val;
        output[idx] = __float2bfloat16(val * __bfloat162float(weight[c]) + __bfloat162float(bias[c]));
    }
}

void groupnorm_bf16_forward(__nv_bfloat16* output, const __nv_bfloat16* input,
    const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    int batch, int channels, int spatial, int num_groups, float eps, cudaStream_t stream)
{
    int channels_per_group = channels / num_groups;
    dim3 grid(batch, num_groups);
    dim3 block(256);
    groupnorm_bf16_kernel<<<grid, block, 0, stream>>>(
        output, input, weight, bias, channels, spatial, channels_per_group, eps
    );
}
