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

/// @brief BF16 GEMV: y = W * x, where W is [N, K] and x is [1, K], output y is [1, N].
///        Uses bf16x8 vectorized loads with FP32 accumulation and warp-level reduction.
///        Optimized for decode phase (M=1) where cublasLt has excessive overhead.
void hgemv_bf16_cu(
    const __nv_bfloat16* input,   // [1, K]
    const __nv_bfloat16* weight,  // [N, K]
    __nv_bfloat16* output,        // [1, N]
    int N,
    int K,
    cudaStream_t stream
);

// INT4 quantized GEMV (decode, M=1) — K-packed format, BF16
void kpack_gemv_cu(
    const void* input, const void* weight_packed, const void* weight_zero_point,
    const void* weight_scale, void* output,
    int N, int K, int group_size, cudaStream_t stream
);

// INT4 quantized GEMM (prefill, M>1) — K-packed format, BF16
void kpack_gemm_cu(
    const void* input, const void* weight_packed, const void* weight_zero_point,
    const void* weight_scale, void* output,
    int M, int N, int K, int group_size, cudaStream_t stream
);

// Initializes Hopper CUTLASS kernel attributes and hardware metadata outside
// stream capture. A no-op on builds targeting a non-Hopper architecture.
int fp8_block_matmul_init_cu(int device_id);

// Dynamic block-scaled W8A8 matmul, BF16 activation/output. Weight is raw
// row-major [N,K] bytes and float32 scale_inv is row-major [N/128,K/128].
// Return path: 1=W8A8 GEMV, 2=accelerated CUTLASS, 3=W8A8 CUDA fallback; <0=error.
int fp8_block_matmul_bf16_cu(
    const void* input, const void* weight, const void* weight_scale_inv,
    void* output, int M, int N, int K, int scale_cols,
    int block_n, int block_k, int device_id,
    void* workspace, size_t workspace_size, cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
