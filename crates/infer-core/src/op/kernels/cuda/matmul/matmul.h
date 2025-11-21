#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @brief 执行 SGEMV: y = input * weight (alpha=1, beta=0), 采用 float4 向量化访存。
///        这是一个 out-of-place 版本。
void sgemv_cu_fp32x4(
    const float* input,
    const float* weight,
    float* output,
    int M,
    int K,
    cudaStream_t stream
);

void sgemm_naive_f32_cu(
    const float* a,
    const float* b,
    float* c,
    int M, // rows of A
    int N, // cols of B
    int K, // cols of A / rows of B
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif