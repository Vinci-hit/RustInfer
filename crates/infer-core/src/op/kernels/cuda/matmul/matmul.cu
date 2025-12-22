#include <cub/block/block_reduce.cuh>
#include "matmul.h"

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
void fast_bf16_gemm_atbt(
    cublasLtHandle_t ltHandle,
    int m, int n, int k,
    const __nv_bfloat16 *A, // M x K
    const __nv_bfloat16 *B, // N x K
    __nv_bfloat16 *C,       // M x N
    void *workspace, size_t workspaceSize)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    // 1. 创建描述符
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_16BF);

    // 设置转置：A 为 N (M*K)，B 为 T (N*K)
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    // 2. 矩阵布局描述符 (假设是 Row-Major)
    // 注意：cuBLASLt 默认是 Column-Major，处理 Row-Major 时通常通过交换 A,B 并调整转置实现
    // 或者简单理解为：C(M,N) = A(M,K) * B^T(K,N)
    cublasLtMatrixLayout_t adesc = NULL, bdesc = NULL, cdesc = NULL;
    cublasLtMatrixLayoutCreate(&adesc, CUDA_R_16BF, m, k, k); // M x K, LDA=K
    cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_16BF, n, k, k); // N x K, LDB=K
    cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_16BF, m, n, n); // M x N, LDC=N

    // 3. 设置启发式搜索偏好
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

    // 4. 获取最优算法
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, adesc, bdesc, cdesc, cdesc, preference, 1, &heuristicResult, &returnedResults);

    if (returnedResults == 0) {
        // 处理错误：未找到合适算法
        return;
    }

    // 5. 执行矩阵乘法
    cublasLtMatmul(ltHandle,
                   operationDesc,
                   &alpha,
                   A, adesc,
                   B, bdesc,
                   &beta,
                   C, cdesc,
                   C, cdesc,
                   &heuristicResult.algo,
                   workspace,
                   workspaceSize,
                   0);

    // 释放资源
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(adesc);
    cublasLtMatrixLayoutDestroy(bdesc);
    cublasLtMatrixLayoutDestroy(cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
}
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
) {
    fast_bf16_gemm_atbt(handle, M, N, K, a, b, c, workspace, workspaceSize);
}