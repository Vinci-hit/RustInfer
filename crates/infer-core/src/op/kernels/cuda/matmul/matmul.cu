#include <cub/block/block_reduce.cuh>
#include "matmul.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#define CHECK_CUBLAS(func) { \
    cublasStatus_t status = (func); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS API failed at line %d with error: %d\n", \
               __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}
template <int THREAD_PER_BLOCK>
__global__ void sgemv_kernel_cu_fp32x4(
    const float* input,
    const float* weight,
    float* output,
    int M,
    int K
) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  const int tid = threadIdx.x;
  const int start_row = blockIdx.x;
  if (start_row >= K){
    return;
  }
  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_num * pack_size;

  auto input_float4_ptr = reinterpret_cast<const float4 *>(input);
  auto weight_float4_ptr = reinterpret_cast<const float4 *>(weight + start_row * M);
  sdata[tid] = 0;
#pragma unroll
  for(int i = tid;i<pack_num;i+= blockDim.x){
    float4 input_float4 = *(input_float4_ptr + i);
    float4 weight_float4 = *(weight_float4_ptr + i);
    float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                   input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
    sdata[tid] += part_sum;
  }
    
  for(int i = pack_off + tid;i<M;i += blockDim.x){
    sdata[tid] += input[i] * weight[start_row * M + i];
  }
  __syncthreads();

  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;
  float part_sum = BlockReduce(temp).Sum(sdata[tid]);
  __syncthreads();

  if (tid == 0) {
    output[start_row] = part_sum;
  }
  __syncthreads();
    
}

// input是一列M，weight是KxM，也就是其实是weight @ input
void sgemv_cu_fp32x4(
    const float* input,
    const float* weight,
    float* output,
    int M,
    int K,
    cudaStream_t stream
) {
    constexpr int thread_per_block = 128;
    sgemv_kernel_cu_fp32x4<thread_per_block><<<K, thread_per_block, 0, stream>>>(input,weight,output,M,K);
}

__global__ void sgemm_naive_f32_transpose_b_kernel(
    const float *a, 
    const float *b, 
    float *c, 
    int M, // A 的行数
    int N, // B 的行数 (也是 B^T 的列数)
    int K  // A 的列数 (也是 B 的列数)
) {
    int n_out = blockIdx.x * blockDim.x + threadIdx.x; // C 的列索引
    int m_out = blockIdx.y * blockDim.y + threadIdx.y; // C 的行索引

    // 边界检查，C 的形状是 [M, N]
    if (m_out < M && n_out < N) {
        float psum = 0.0;
        
        // 循环点积的长度是 K
        for (int k = 0; k < K; k++) {
            // 从 A 中获取第 m_out 行, 第 k 列的元素
            float a_val = a[m_out * K + k];

            // **核心修改**:
            // 从 B 中获取第 n_out 行, 第 k 列的元素。
            // 这等价于从 B^T 中获取第 k 行, 第 n_out 列的元素。
            float b_val = b[n_out * K + k];

            psum += a_val * b_val;
        }
        
        // 将结果写入 C 的 [m_out, n_out] 位置
        c[m_out * N + n_out] = psum;
    }
}

extern "C" void sgemm_naive_f32_cu(
    const float* a,
    const float* b,
    float* c,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    // 定义 block 的大小
    dim3 threads_per_block(16, 16);
    
    // 计算 grid 的大小
    dim3 blocks_per_grid(
        (N + threads_per_block.x - 1) / threads_per_block.x,
        (M + threads_per_block.y - 1) / threads_per_block.y
    );

    sgemm_naive_f32_transpose_b_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        a, b, c, M, N, K
    );
}
void gemm_cublasLt_AxBT_RowMajor_bf16(
    cublasLtHandle_t ltHandle,
    int M, int N, int K,
    const __nv_bfloat16 *d_A, // shape: [M, K]
    const __nv_bfloat16 *d_B, // shape: [N, K]
    __nv_bfloat16 *d_C,       // shape: [M, N] output
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream)
{
    int m_gemm = N;
    int n_gemm = M;
    int k_gemm = K;

    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    // Matrix Layouts
    // Arg1 (B): 物理内存看作 ColMajor [K, N], leading dim = K
    cublasLtMatrixLayout_t Adesc = NULL;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, k_gemm, m_gemm, k_gemm));

    // Arg2 (A): 物理内存看作 ColMajor [K, M], leading dim = K
    cublasLtMatrixLayout_t Bdesc = NULL;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, k_gemm, n_gemm, k_gemm));

    // Result (C): 物理内存看作 ColMajor [N, M], leading dim = N
    cublasLtMatrixLayout_t Cdesc = NULL;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m_gemm, n_gemm, m_gemm));

    // Preference
    cublasLtMatmulPreference_t preference = NULL;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // Heuristic
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    // 参数顺序：Handle, Desc, Mat1(Left), Mat2(Right), C, D, Pref, Count, Result, ResultCount
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        printf("cuBLASLt: No algorithm found!\n");
        exit(1);
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // Execute
    // Inputs swapped: A_param = d_B, B_param = d_A
    CHECK_CUBLAS(cublasLtMatmul(ltHandle,
                                operationDesc,
                                &alpha,
                                d_B, Adesc,
                                d_A, Bdesc,
                                &beta,
                                d_C, Cdesc,
                                d_C, Cdesc,
                                &heuristicResult.algo,
                                workspace,
                                workspaceSize,
                                stream));

    // Cleanup
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
}
void gemm_cublaslt_bf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    cublasLtHandle_t handle,
    void* workspace, size_t workspaceSize
) {
    gemm_cublasLt_AxBT_RowMajor_bf16(handle, M, N, K, A, B, C, workspace, workspaceSize,stream);
}