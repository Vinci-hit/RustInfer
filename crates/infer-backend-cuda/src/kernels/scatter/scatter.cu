#include "scatter.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ============================================================================
// Unified scatter kernels via templates.
// All scatter operations are pure memory moves — the only difference is the
// element width which determines how many elements fit in one float4 (16 bytes).
//   - 16-bit types (bf16, fp16): 8 elements per float4
//   - 32-bit types (f32):        4 elements per float4
// ============================================================================

template <typename T>
struct ElemsPerVec4 { static constexpr int value = 16 / sizeof(T); };

// --- Single-row scatter: dst[pos, :] = src[0, :] ---
template <typename T>
__global__ void scatter_vec_kernel(
    float4* __restrict__ dst,
    const float4* __restrict__ src,
    const int* __restrict__ pos,
    int kvdim,
    int num_vec4
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec4) {
        constexpr int elems = ElemsPerVec4<T>::value;
        int position = *pos;
        int offset = (position * kvdim) / elems;
        dst[offset + idx] = src[idx];
    }
}

// --- Single-row fused K+V scatter ---
template <typename T>
__global__ void scatter_kv_vec_kernel(
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
        constexpr int elems = ElemsPerVec4<T>::value;
        int position = *pos;
        int offset = (position * kvdim) / elems;
        dst_k[offset + idx] = src_k[idx];
        dst_v[offset + idx] = src_v[idx];
    }
}

// ============================================================================
// Batched scatter_kv — optimized for BOTH prefill (batch=1-3) and decode.
//
// Key optimizations vs naive implementation:
//   1. float4 vectorized load/store (16 bytes = 8 bf16 per transaction)
//   2. 2D grid: dim3(blocks_x, batch_size) — blocks_x = ceil(num_vec4/BLOCK)
//      ensures multiple CTAs per sequence to fill SMs even with batch=1
//   3. No integer division in inner loop — threadIdx is used directly as the
//      element index within the row (constant-divisor /8 is compiled to shift)
//
// Grid sizing for SM utilization:
//   kvdim=128  → num_vec4=16,  blocks_x=1  (16 threads of work per seq)
//   kvdim=1024 → num_vec4=128, blocks_x=4  (128 elements, 32 threads/block)
//   kvdim=4096 → num_vec4=512, blocks_x=16
//   With batch=1: total blocks = blocks_x (up to 16 for large kvdim)
//   With batch=3: total blocks = 3 * blocks_x (up to 48)
// ============================================================================

template <typename T>
__global__ void scatter_kv_batch_vec_kernel(
    T** __restrict__ dst_k_ptrs,
    T** __restrict__ dst_v_ptrs,
    const T* __restrict__ src_k_base,
    const T* __restrict__ src_v_base,
    const int* __restrict__ positions,
    int num_vec4,
    int src_k_row_stride_vec,   // src_k_row_stride / elems_per_vec4
    int src_v_row_stride_vec    // src_v_row_stride / elems_per_vec4
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq = blockIdx.y;
    if (idx >= num_vec4) return;

    const int pos = positions[seq];
    // pos * num_vec4: position offset in float4 units (no division, just multiply)
    const int dst_offset_vec = pos * num_vec4;

    const int src_k_offset = seq * src_k_row_stride_vec + idx;
    const int src_v_offset = seq * src_v_row_stride_vec + idx;

    float4* __restrict__ dst_k_f4 = reinterpret_cast<float4*>(dst_k_ptrs[seq]);
    float4* __restrict__ dst_v_f4 = reinterpret_cast<float4*>(dst_v_ptrs[seq]);
    const float4* __restrict__ src_k_f4 = reinterpret_cast<const float4*>(src_k_base);
    const float4* __restrict__ src_v_f4 = reinterpret_cast<const float4*>(src_v_base);

    // ILP: issue both loads before stores
    float4 k_val = src_k_f4[src_k_offset];
    float4 v_val = src_v_f4[src_v_offset];
    dst_k_f4[dst_offset_vec + idx] = k_val;
    dst_v_f4[dst_offset_vec + idx] = v_val;
}

// ============================================================================
// Extern "C" API — thin wrappers
// ============================================================================

void scatter_kernel_bf16(
    __nv_bfloat16* dst,
    const __nv_bfloat16* src,
    int* pos,
    int kvdim,
    int max_seq_len,
    cudaStream_t stream)
{
    int num_vec4 = kvdim / 8;
    const int threads = 256;
    int blocks = (num_vec4 + threads - 1) / threads;
    scatter_vec_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float4*>(dst),
        reinterpret_cast<const float4*>(src),
        pos, kvdim, num_vec4);
}

void scatter_kernel_f32(
    float* dst,
    const float* src,
    int* pos,
    int kvdim,
    int max_seq_len,
    cudaStream_t stream)
{
    int num_vec4 = kvdim / 4;
    const int threads = 256;
    int blocks = (num_vec4 + threads - 1) / threads;
    scatter_vec_kernel<float><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float4*>(dst),
        reinterpret_cast<const float4*>(src),
        pos, kvdim, num_vec4);
}

void scatter_kv_kernel_bf16(
    __nv_bfloat16* dst_k,
    const __nv_bfloat16* src_k,
    __nv_bfloat16* dst_v,
    const __nv_bfloat16* src_v,
    int* pos,
    int kvdim,
    int max_seq_len,
    cudaStream_t stream)
{
    int num_vec4 = kvdim / 8;
    const int threads = 256;
    int blocks = (num_vec4 + threads - 1) / threads;
    scatter_kv_vec_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float4*>(dst_k),
        reinterpret_cast<const float4*>(src_k),
        reinterpret_cast<float4*>(dst_v),
        reinterpret_cast<const float4*>(src_v),
        pos, kvdim, num_vec4);
}

extern "C" void scatter_kernel_fp16(
    __half* dst,
    const __half* src,
    int* pos,
    int kvdim,
    int max_seq_len,
    cudaStream_t stream)
{
    int num_vec4 = kvdim / 8;
    const int threads = 256;
    int blocks = (num_vec4 + threads - 1) / threads;
    scatter_vec_kernel<__half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float4*>(dst),
        reinterpret_cast<const float4*>(src),
        pos, kvdim, num_vec4);
}

extern "C" void scatter_kv_kernel_fp16(
    __half* dst_k,
    const __half* src_k,
    __half* dst_v,
    const __half* src_v,
    int* pos,
    int kvdim,
    int max_seq_len,
    cudaStream_t stream)
{
    int num_vec4 = kvdim / 8;
    const int threads = 256;
    int blocks = (num_vec4 + threads - 1) / threads;
    scatter_kv_vec_kernel<__half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<float4*>(dst_k),
        reinterpret_cast<const float4*>(src_k),
        reinterpret_cast<float4*>(dst_v),
        reinterpret_cast<const float4*>(src_v),
        pos, kvdim, num_vec4);
}

// --- Batched K+V scatter ---
// Block size = 32 (1 warp) to maximize block count and SM occupancy.
// With 32 threads/block:
//   kvdim=128  → num_vec4=16,  blocks_x=1,  batch=1 → 1 block
//   kvdim=1024 → num_vec4=128, blocks_x=4,  batch=1 → 4 blocks
//   kvdim=4096 → num_vec4=512, blocks_x=16, batch=1 → 16 blocks
//   kvdim=8192 → num_vec4=1024,blocks_x=32, batch=1 → 32 blocks
// Each block is tiny (32 threads) so the HW can pack many onto each SM.

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
    constexpr int elems = 8;  // bf16: 8 per float4
    int num_vec4 = kvdim / elems;
    // 32 threads = 1 warp: minimum for coalesced access, maximizes blocks_x
    constexpr int threads_per_block = 32;
    int blocks_x = (num_vec4 + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_x, batch_size);

    scatter_kv_batch_vec_kernel<__nv_bfloat16><<<grid, threads_per_block, 0, stream>>>(
        dst_k_ptrs, dst_v_ptrs, src_k, src_v, positions,
        num_vec4,
        src_k_row_stride / elems,
        src_v_row_stride / elems);
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
    constexpr int elems = 8;  // fp16: 8 per float4
    int num_vec4 = kvdim / elems;
    constexpr int threads_per_block = 32;
    int blocks_x = (num_vec4 + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_x, batch_size);

    scatter_kv_batch_vec_kernel<__half><<<grid, threads_per_block, 0, stream>>>(
        dst_k_ptrs, dst_v_ptrs, src_k, src_v, positions,
        num_vec4,
        src_k_row_stride / elems,
        src_v_row_stride / elems);
}
