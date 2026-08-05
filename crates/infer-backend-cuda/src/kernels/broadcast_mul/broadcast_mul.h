#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// dst[row * D + col] = a[row * D + col] * b[col]
// a: [rows, D], b: [D], dst: [rows, D]
void broadcast_mul_f32_forward(float* dst, const float* a, const float* b, int rows, int D, cudaStream_t stream);
void broadcast_mul_bf16_forward(__nv_bfloat16* dst, const __nv_bfloat16* a, const __nv_bfloat16* b, int rows, int D, cudaStream_t stream);
void broadcast_mul_f16_forward(__half* dst, const __half* a, const __half* b, int rows, int D, cudaStream_t stream);

// a[row * row_stride + col] += b[col]
// a: strided [rows, D], b: contiguous [D]
void broadcast_add_inplace_f32_forward(float* a, const float* b, int rows, int D, int row_stride, cudaStream_t stream);
void broadcast_add_inplace_bf16_forward(__nv_bfloat16* a, const __nv_bfloat16* b, int rows, int D, int row_stride, cudaStream_t stream);
void broadcast_add_inplace_f16_forward(__half* a, const __half* b, int rows, int D, int row_stride, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
