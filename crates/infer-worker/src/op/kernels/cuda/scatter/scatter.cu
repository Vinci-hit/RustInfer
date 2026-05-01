#include "scatter.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

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





// ============= FP16 variants (auto-generated from BF16) =============

__global__ void scatter_fp16_vec8_kernel(
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

extern "C" void scatter_kernel_fp16(
    __half* dst,
    const __half* src,
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
    scatter_fp16_vec8_kernel<<<blocks, threads_per_block, 0, stream>>>(
        dst_f4, src_f4, pos, kvdim, num_vec4
    );
}

__global__ void scatter_kv_fp16_vec8_kernel(
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

extern "C" void scatter_kv_kernel_fp16(
    __half* dst_k,
    const __half* src_k,
    __half* dst_v,
    const __half* src_v,
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

    scatter_kv_fp16_vec8_kernel<<<blocks, threads_per_block, 0, stream>>>(
        dst_k_f4, src_k_f4, dst_v_f4, src_v_f4, pos, kvdim, num_vec4
    );
}



// ============================================================================
// Batched scatter_kv：一次 kernel launch 写 B 行 K/V 到 B 个不同 cache
// dst_k_ptrs / dst_v_ptrs 是设备上保存的 B 个指针数组（每个指向一个 cache 起点）
// src_k, src_v 可以是非连续（如 fused qkv 的 k/v 段），通过
// src_{k,v}_row_stride（元素单位）和 src_{k,v}_col_offset（元素单位）寻址。
// ============================================================================

__global__ void scatter_kv_batch_bf16_vec8_kernel(
    __nv_bfloat16** __restrict__ dst_k_ptrs,
    __nv_bfloat16** __restrict__ dst_v_ptrs,
    const __nv_bfloat16* __restrict__ src_k_base,
    const __nv_bfloat16* __restrict__ src_v_base,
    const int* __restrict__ positions,
    int kvdim,
    int num_vec4,             // kvdim / 8
    int src_k_row_stride,     // elements (bf16)
    int src_v_row_stride
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq = blockIdx.y;
    if (idx >= num_vec4) return;

    const int pos = positions[seq];
    const int dst_offset_vec = (pos * kvdim) / 8;

    // Each seq's row starts at seq * row_stride (in bf16 elements) → /8 for float4
    const int src_k_row_vec = (seq * src_k_row_stride) / 8;
    const int src_v_row_vec = (seq * src_v_row_stride) / 8;

    float4* dst_k_f4 = reinterpret_cast<float4*>(dst_k_ptrs[seq]);
    float4* dst_v_f4 = reinterpret_cast<float4*>(dst_v_ptrs[seq]);
    const float4* src_k_f4 = reinterpret_cast<const float4*>(src_k_base);
    const float4* src_v_f4 = reinterpret_cast<const float4*>(src_v_base);

    dst_k_f4[dst_offset_vec + idx] = src_k_f4[src_k_row_vec + idx];
    dst_v_f4[dst_offset_vec + idx] = src_v_f4[src_v_row_vec + idx];
}

extern "C" void scatter_kv_batch_kernel_bf16(
    __nv_bfloat16** dst_k_ptrs,
    __nv_bfloat16** dst_v_ptrs,
    const __nv_bfloat16* src_k,
    const __nv_bfloat16* src_v,
    const int* positions,
    int batch_size,
    int kvdim,
    int src_k_row_stride,
    int src_v_row_stride,
    cudaStream_t stream)
{
    int num_vec4 = kvdim / 8;
    const int threads_per_block = 256;
    int blocks_x = (num_vec4 + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_x, batch_size);
    scatter_kv_batch_bf16_vec8_kernel<<<grid, threads_per_block, 0, stream>>>(
        dst_k_ptrs, dst_v_ptrs, src_k, src_v, positions,
        kvdim, num_vec4, src_k_row_stride, src_v_row_stride
    );
}

__global__ void scatter_kv_batch_fp16_vec8_kernel(
    __half** __restrict__ dst_k_ptrs,
    __half** __restrict__ dst_v_ptrs,
    const __half* __restrict__ src_k_base,
    const __half* __restrict__ src_v_base,
    const int* __restrict__ positions,
    int kvdim,
    int num_vec4,
    int src_k_row_stride,
    int src_v_row_stride
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq = blockIdx.y;
    if (idx >= num_vec4) return;

    const int pos = positions[seq];
    const int dst_offset_vec = (pos * kvdim) / 8;
    const int src_k_row_vec = (seq * src_k_row_stride) / 8;
    const int src_v_row_vec = (seq * src_v_row_stride) / 8;

    float4* dst_k_f4 = reinterpret_cast<float4*>(dst_k_ptrs[seq]);
    float4* dst_v_f4 = reinterpret_cast<float4*>(dst_v_ptrs[seq]);
    const float4* src_k_f4 = reinterpret_cast<const float4*>(src_k_base);
    const float4* src_v_f4 = reinterpret_cast<const float4*>(src_v_base);

    dst_k_f4[dst_offset_vec + idx] = src_k_f4[src_k_row_vec + idx];
    dst_v_f4[dst_offset_vec + idx] = src_v_f4[src_v_row_vec + idx];
}

extern "C" void scatter_kv_batch_kernel_fp16(
    __half** dst_k_ptrs,
    __half** dst_v_ptrs,
    const __half* src_k,
    const __half* src_v,
    const int* positions,
    int batch_size,
    int kvdim,
    int src_k_row_stride,
    int src_v_row_stride,
    cudaStream_t stream)
{
    int num_vec4 = kvdim / 8;
    const int threads_per_block = 256;
    int blocks_x = (num_vec4 + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_x, batch_size);
    scatter_kv_batch_fp16_vec8_kernel<<<grid, threads_per_block, 0, stream>>>(
        dst_k_ptrs, dst_v_ptrs, src_k, src_v, positions,
        kvdim, num_vec4, src_k_row_stride, src_v_row_stride
    );
}
