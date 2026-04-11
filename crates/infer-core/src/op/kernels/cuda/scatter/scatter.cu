#include "scatter.h"
#include <cuda_bf16.h>

// BF16 vectorized scatter kernel using float4 (8 bf16 elements at once)
__global__ void scatter_bf16_vec8_kernel(
    float4* __restrict__ dst,
    const float4* __restrict__ src,
    const int* __restrict__ pos,  // pointer to position value
    int kvdim,       // dimension size (in bf16 elements)
    int num_vec4     // number of float4 elements to copy
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec4) {
        // Read position from device memory and calculate offset
        int position = *pos;
        int offset = (position * kvdim) / 8;  // offset in float4 elements
        dst[offset + idx] = src[idx];
    }
}

void scatter_kernel_bf16(
    __nv_bfloat16* dst,
    const __nv_bfloat16* src,
    int* pos,
    int kvdim,
    int max_seq_len,
    cudaStream_t stream
)
{
    // Use float4 for vectorized copy (8 bf16 elements = 16 bytes = 1 float4)
    // kvdim should be divisible by 8 for optimal performance
    int num_vec4 = kvdim / 8;

    auto* dst_f4 = reinterpret_cast<float4*>(dst);
    auto* src_f4 = reinterpret_cast<const float4*>(src);

    const int threads_per_block = 256;
    int blocks = (num_vec4 + threads_per_block - 1) / threads_per_block;

    // Pass pos pointer to kernel, which will read the value inside
    scatter_bf16_vec8_kernel<<<blocks, threads_per_block, 0, stream>>>(
        dst_f4, src_f4, pos, kvdim, num_vec4
    );
}

// F32 vectorized scatter kernel using float4 (4 f32 elements at once)
__global__ void scatter_f32_vec4_kernel(
    float4* __restrict__ dst,
    const float4* __restrict__ src,
    const int* __restrict__ pos,  // pointer to position value
    int kvdim,       // dimension size (in f32 elements)
    int num_vec4
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec4) {
        // Read position from device memory and calculate offset
        int position = *pos;
        int offset = (position * kvdim) / 4;  // offset in float4 elements
        dst[offset + idx] = src[idx];
    }
}

void scatter_kernel_f32(
    float* dst,
    const float* src,
    int* pos,
    int kvdim,
    int max_seq_len,
    cudaStream_t stream
)
{
    // Use float4 for vectorized copy (4 f32 elements)
    int num_vec4 = kvdim / 4;

    auto* dst_f4 = reinterpret_cast<float4*>(dst);
    auto* src_f4 = reinterpret_cast<const float4*>(src);

    const int threads_per_block = 256;
    int blocks = (num_vec4 + threads_per_block - 1) / threads_per_block;

    // Pass pos pointer to kernel, which will read the value inside
    scatter_f32_vec4_kernel<<<blocks, threads_per_block, 0, stream>>>(
        dst_f4, src_f4, pos, kvdim, num_vec4
    );
}

// Fused scatter for K and V caches simultaneously
// Each thread copies one float4 (8 bf16) to both K and V cache at the same position.
// Saves one kernel launch per layer.
__global__ void scatter_kv_bf16_vec8_kernel(
    float4* __restrict__ dst_k,
    const float4* __restrict__ src_k,
    float4* __restrict__ dst_v,
    const float4* __restrict__ src_v,
    const int* __restrict__ pos,
    int kvdim,
    int num_vec4
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec4) {
        int position = *pos;
        int offset = (position * kvdim) / 8;
        dst_k[offset + idx] = src_k[idx];
        dst_v[offset + idx] = src_v[idx];
    }
}

void scatter_kv_kernel_bf16(
    __nv_bfloat16* dst_k,
    const __nv_bfloat16* src_k,
    __nv_bfloat16* dst_v,
    const __nv_bfloat16* src_v,
    int* pos,
    int kvdim,
    int max_seq_len,
    cudaStream_t stream
)
{
    int num_vec4 = kvdim / 8;

    auto* dst_k_f4 = reinterpret_cast<float4*>(dst_k);
    auto* src_k_f4 = reinterpret_cast<const float4*>(src_k);
    auto* dst_v_f4 = reinterpret_cast<float4*>(dst_v);
    auto* src_v_f4 = reinterpret_cast<const float4*>(src_v);

    const int threads_per_block = 256;
    int blocks = (num_vec4 + threads_per_block - 1) / threads_per_block;

    scatter_kv_bf16_vec8_kernel<<<blocks, threads_per_block, 0, stream>>>(
        dst_k_f4, src_k_f4, dst_v_f4, src_v_f4, pos, kvdim, num_vec4
    );
}
