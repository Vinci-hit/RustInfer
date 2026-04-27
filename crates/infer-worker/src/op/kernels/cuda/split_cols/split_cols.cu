#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

/// Split columns from a [rows, total_cols] matrix into a [rows, dst_cols] matrix.
/// Copies columns [col_offset, col_offset + dst_cols) from src to dst.
__global__ void split_cols_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int rows,
    int total_cols,
    int col_offset,
    int dst_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * dst_cols;
    if (idx >= total_elements) return;

    int r = idx / dst_cols;
    int c = idx % dst_cols;
    dst[idx] = src[r * total_cols + col_offset + c];
}

extern "C" void split_cols_bf16(
    const __nv_bfloat16* src,
    __nv_bfloat16* dst,
    int rows,
    int total_cols,
    int col_offset,
    int dst_cols,
    cudaStream_t stream
) {
    int total_elements = rows * dst_cols;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    split_cols_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
        src, dst, rows, total_cols, col_offset, dst_cols
    );
}


// ============= FP16 variants (auto-generated from BF16) =============

__global__ void split_cols_fp16_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int rows,
    int total_cols,
    int col_offset,
    int dst_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * dst_cols;
    if (idx >= total_elements) return;

    int r = idx / dst_cols;
    int c = idx % dst_cols;
    dst[idx] = src[r * total_cols + col_offset + c];
}

extern "C" void split_cols_fp16(
    const __half* src,
    __half* dst,
    int rows,
    int total_cols,
    int col_offset,
    int dst_cols,
    cudaStream_t stream
) {
    int total_elements = rows * dst_cols;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    split_cols_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
        src, dst, rows, total_cols, col_offset, dst_cols
    );
}


// ============= F32 variant (used by CPU-vs-CUDA correctness tests) =============

__global__ void split_cols_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int rows,
    int total_cols,
    int col_offset,
    int dst_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * dst_cols;
    if (idx >= total_elements) return;

    int r = idx / dst_cols;
    int c = idx % dst_cols;
    dst[idx] = src[r * total_cols + col_offset + c];
}

extern "C" void split_cols_f32(
    const float* src,
    float* dst,
    int rows,
    int total_cols,
    int col_offset,
    int dst_cols,
    cudaStream_t stream
) {
    int total_elements = rows * dst_cols;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    split_cols_f32_kernel<<<grid_size, block_size, 0, stream>>>(
        src, dst, rows, total_cols, col_offset, dst_cols
    );
}
