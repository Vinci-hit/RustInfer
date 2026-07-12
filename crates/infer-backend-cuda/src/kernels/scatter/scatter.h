#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// Scatter kernel for BF16 data type
// Copies src (shape: [1, kvdim]) to dst at offset position (dst[pos, :] = src[0, :])
void scatter_kernel_bf16(
    __nv_bfloat16* dst,           // destination tensor pointer
    const __nv_bfloat16* src,     // source tensor pointer (1, kvdim)
    int* pos,                       // position offset in the destination
    int kvdim,                     // dimension size
    int max_seq_len,              // maximum sequence length (for bounds checking)
    cudaStream_t stream
);

// Scatter kernel for F32 data type
void scatter_kernel_f32(
    float* dst,
    const float* src,
    int* pos,
    int kvdim,
    int max_seq_len,
    cudaStream_t stream
);

// Fused scatter for K and V caches simultaneously (BF16)
// Saves one kernel launch per layer in decode phase
void scatter_kv_kernel_bf16(
    __nv_bfloat16* dst_k,
    const __nv_bfloat16* src_k,
    __nv_bfloat16* dst_v,
    const __nv_bfloat16* src_v,
    int* pos,
    int kvdim,
    int max_seq_len,
    cudaStream_t stream
);

// Batched scatter_kv: B 行 K/V 一次 launch 写到 B 个不同 cache 的各自 position。
// dst_{k,v}_ptrs 是 device memory 上的指针数组（每 element 为 cache 起始指针），
// src_{k,v} 起点 + row_stride（元素单位）指定每行起始（可用于 fused qkv 非连续 slice）
void scatter_kv_batch_kernel_bf16(
    __nv_bfloat16** dst_k_ptrs,
    __nv_bfloat16** dst_v_ptrs,
    const __nv_bfloat16* src_k,
    const __nv_bfloat16* src_v,
    const int* positions,
    int batch_size,
    int kvdim,
    int src_k_row_stride,
    int src_v_row_stride,
    cudaStream_t stream
);

void scatter_kv_batch_kernel_fp16(
    __half** dst_k_ptrs,
    __half** dst_v_ptrs,
    const __half* src_k,
    const __half* src_v,
    const int* positions,
    int batch_size,
    int kvdim,
    int src_k_row_stride,
    int src_v_row_stride,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
