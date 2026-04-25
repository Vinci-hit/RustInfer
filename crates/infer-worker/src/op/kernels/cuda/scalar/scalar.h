#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

void scalar_mul_f32_forward(float* dst, const float* src, float val, int n, cudaStream_t stream);
void scalar_mul_bf16_forward(__nv_bfloat16* dst, const __nv_bfloat16* src, float val, int n, cudaStream_t stream);
void scalar_mul_f16_forward(__half* dst, const __half* src, float val, int n, cudaStream_t stream);

void scalar_add_f32_forward(float* dst, const float* src, float val, int n, cudaStream_t stream);
void scalar_add_bf16_forward(__nv_bfloat16* dst, const __nv_bfloat16* src, float val, int n, cudaStream_t stream);
void scalar_add_f16_forward(__half* dst, const __half* src, float val, int n, cudaStream_t stream);

void silu_inplace_f32_forward(float* data, int n, cudaStream_t stream);
void silu_inplace_bf16_forward(__nv_bfloat16* data, int n, cudaStream_t stream);
void silu_inplace_f16_forward(__half* data, int n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
