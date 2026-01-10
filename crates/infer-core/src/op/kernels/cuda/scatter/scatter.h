#include <cuda_runtime.h>
#include <cuda_bf16.h>

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

#ifdef __cplusplus
}
#endif
