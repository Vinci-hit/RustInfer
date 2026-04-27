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

void tanh_inplace_f32_forward(float* data, int n, cudaStream_t stream);
void tanh_inplace_bf16_forward(__nv_bfloat16* data, int n, cudaStream_t stream);
void tanh_inplace_f16_forward(__half* data, int n, cudaStream_t stream);

// ===================== scalar_mul_inplace_from_dev =====================
// x[i] *= *d_scalar, 系数从 device [1] f32 读取。
// 用于 CUDA Graph capture: kernel 参数内不含随 step 变化的 host 值。
void scalar_mul_inplace_from_dev_f32_forward (float*        x, const float* d_scalar, int n, cudaStream_t stream);
void scalar_mul_inplace_from_dev_bf16_forward(__nv_bfloat16* x, const float* d_scalar, int n, cudaStream_t stream);
void scalar_mul_inplace_from_dev_f16_forward (__half*       x, const float* d_scalar, int n, cudaStream_t stream);

// ===================== sinusoid_embedding_from_dev =====================
// 输入  d_t       : [1] f32 device (host 侧已经乘过 t_scale)
//       dim        : 频率维度（必须为偶数；半半拆分：前半 cos，后半 sin）
// 输出  d_out_bf16 : [1, dim] bf16 device
//       d_out_f16  : [1, dim] f16  device
//       d_out_f32  : [1, dim] f32  device
void sinusoid_embedding_from_dev_f32_forward (float*         d_out, const float* d_t, int dim, cudaStream_t stream);
void sinusoid_embedding_from_dev_bf16_forward(__nv_bfloat16* d_out, const float* d_t, int dim, cudaStream_t stream);
void sinusoid_embedding_from_dev_f16_forward (__half*        d_out, const float* d_t, int dim, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
