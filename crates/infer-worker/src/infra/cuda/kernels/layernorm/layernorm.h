#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// LayerNorm without affine: output[i] = (input[i] - mean) / sqrt(var + eps)
// input/output: [rows, cols], each row normalized independently
void layernorm_f32_forward(float* output, const float* input, int rows, int cols, float eps, cudaStream_t stream);
void layernorm_bf16_forward(__nv_bfloat16* output, const __nv_bfloat16* input, int rows, int cols, float eps, cudaStream_t stream);
void layernorm_f16_forward(__half* output, const __half* input, int rows, int cols, float eps, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
