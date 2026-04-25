#include "upsample.h"

// Upsample nearest 2x: 每个输出像素从最近的输入像素复制
// 一个 thread 处理一个输出像素

__global__ void upsample_nearest_2x_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int channels, int h_in, int w_in, int h_out, int w_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * h_out * w_out; // per-batch elements
    int batch_idx = blockIdx.y;

    if (idx >= total) return;

    int w = idx % w_out;
    int h = (idx / w_out) % h_out;
    int c = idx / (h_out * w_out);

    int ih = h / 2;
    int iw = w / 2;

    int in_idx = (batch_idx * channels + c) * h_in * w_in + ih * w_in + iw;
    int out_idx = (batch_idx * channels + c) * h_out * w_out + h * w_out + w;

    output[out_idx] = input[in_idx];
}

void upsample_nearest_2x_f32_forward(float* output, const float* input,
    int batch, int channels, int h_in, int w_in, cudaStream_t stream)
{
    int h_out = h_in * 2;
    int w_out = w_in * 2;
    int total_per_batch = channels * h_out * w_out;
    int threads = 256;
    int blocks_x = (total_per_batch + threads - 1) / threads;
    dim3 grid(blocks_x, batch);
    upsample_nearest_2x_f32_kernel<<<grid, threads, 0, stream>>>(
        output, input, channels, h_in, w_in, h_out, w_out
    );
}

__global__ void upsample_nearest_2x_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    int channels, int h_in, int w_in, int h_out, int w_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * h_out * w_out;
    int batch_idx = blockIdx.y;

    if (idx >= total) return;

    int w = idx % w_out;
    int h = (idx / w_out) % h_out;
    int c = idx / (h_out * w_out);

    int ih = h / 2;
    int iw = w / 2;

    int in_idx = (batch_idx * channels + c) * h_in * w_in + ih * w_in + iw;
    int out_idx = (batch_idx * channels + c) * h_out * w_out + h * w_out + w;

    output[out_idx] = input[in_idx];
}

void upsample_nearest_2x_bf16_forward(__nv_bfloat16* output, const __nv_bfloat16* input,
    int batch, int channels, int h_in, int w_in, cudaStream_t stream)
{
    int h_out = h_in * 2;
    int w_out = w_in * 2;
    int total_per_batch = channels * h_out * w_out;
    int threads = 256;
    int blocks_x = (total_per_batch + threads - 1) / threads;
    dim3 grid(blocks_x, batch);
    upsample_nearest_2x_bf16_kernel<<<grid, threads, 0, stream>>>(
        output, input, channels, h_in, w_in, h_out, w_out
    );
}
