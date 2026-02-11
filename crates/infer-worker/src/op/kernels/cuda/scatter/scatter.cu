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

// ============================================================================
// Scatter KV到paged cache的kernel (仅BF16)
// ============================================================================

/// Scatter K,V到paged KV cache
/// 输入K,V: [batch_size, kv_dim]
/// 输出cache: [num_blocks, block_size, kv_dim]
/// slot_mapping: [batch_size] - 每个K/V对应的slot索引
__global__ void scatter_kv_bf16_kernel(
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const int* __restrict__ slot_mapping,
    int kv_dim,
    int block_size,
    int batch_size)
{
    // 每个线程处理一个K/V对中的一个元素
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (batch_idx >= batch_size || elem_idx >= kv_dim) {
        return;
    }

    // 从slot_mapping获取该批次应该写入的slot
    int slot = slot_mapping[batch_idx];

    // 计算在paged cache中的位置
    // cache shape: [num_blocks, block_size, kv_dim]
    // slot对应的block和offset
    int block_id = slot / block_size;
    int offset_in_block = slot % block_size;
    int cache_idx = (block_id * block_size + offset_in_block) * kv_dim + elem_idx;

    // 从input中读取元素
    int input_idx = batch_idx * kv_dim + elem_idx;

    // 写入K和V到各自的cache
    k_cache[cache_idx] = key[input_idx];
    v_cache[cache_idx] = value[input_idx];
}

extern "C"
void scatter_kv_kernel_bf16(
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const int* slot_mapping,
    int kv_dim,
    int block_size,
    int batch_size,
    cudaStream_t stream)
{
    // 总元素数 = batch_size * kv_dim
    int total_elements = batch_size * kv_dim;
    const int threads_per_block = 256;

    // grid: (num_blocks_x, batch_size)
    int grid_x = (kv_dim + threads_per_block - 1) / threads_per_block;
    dim3 grid(grid_x, batch_size);
    dim3 block(threads_per_block);

    scatter_kv_bf16_kernel<<<grid, block, 0, stream>>>(
        k_cache, v_cache, key, value, slot_mapping, kv_dim, block_size, batch_size
    );
}
