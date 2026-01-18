#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
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

void gemm_cublaslt_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* c,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    cublasLtHandle_t handle,
    void* workspace, size_t workspaceSize
);

#ifdef __cplusplus
}
#endif