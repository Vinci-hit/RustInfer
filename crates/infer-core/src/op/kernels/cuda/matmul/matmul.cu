#include <cub/block/block_reduce.cuh>
#include "matmul.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
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
// ============================================================================
// BF16 GEMV kernel v3 for decode phase (M=1)
// y[n] = dot(W[n,:], x[:])  where W is [N, K], x is [1, K], y is [1, N]
//
// Design: 1 warp (32 threads) computes 1 row, 1 block has 8 warps.
// No shared memory: input vector (8KB for K=4096) fits in L1 cache (128KB)
// and is implicitly cached across warps/blocks on same SM. This avoids
// __syncthreads() overhead and saves shared memory for higher occupancy.
//
// NCU-validated improvements over v2 (N=11008, K=4096, A10 sm_86):
//   - L1/TEX hit rate: 8.3% -> 49.8% (input vector cached in L1)
//   - Achieved occupancy: 87% -> 91.6%
//   - DRAM throughput: 92.6% -> 93.9%
//   - Duration: 185us -> 181us (~2% faster)
//   - No __syncthreads() needed
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum_gemv(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 6)
hgemv_bf16_v3_kernel(
    const __nv_bfloat16* __restrict__ input,   // [K]
    const __nv_bfloat16* __restrict__ weight,  // [N, K] row-major
    __nv_bfloat16* __restrict__ output,        // [N]
    const int N,
    const int K
) {
    const int lane_id = threadIdx.x & 31;
    const int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);

    if (row >= N) return;

    const int pack_num = K >> 3;  // K / 8, each pack = 8 bf16 = float4
    const float4* __restrict__ input_f4 = reinterpret_cast<const float4*>(input);
    const float4* __restrict__ weight_f4 = reinterpret_cast<const float4*>(weight + row * K);

    float sum = 0.0f;

    for (int i = lane_id; i < pack_num; i += 32) {
        float4 x = __ldg(input_f4 + i);
        float4 w = __ldg(weight_f4 + i);

        const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&x);
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&w);

        sum += __bfloat162float(xb[0]) * __bfloat162float(wb[0]);
        sum += __bfloat162float(xb[1]) * __bfloat162float(wb[1]);
        sum += __bfloat162float(xb[2]) * __bfloat162float(wb[2]);
        sum += __bfloat162float(xb[3]) * __bfloat162float(wb[3]);
        sum += __bfloat162float(xb[4]) * __bfloat162float(wb[4]);
        sum += __bfloat162float(xb[5]) * __bfloat162float(wb[5]);
        sum += __bfloat162float(xb[6]) * __bfloat162float(wb[6]);
        sum += __bfloat162float(xb[7]) * __bfloat162float(wb[7]);
    }

    sum = warp_reduce_sum_gemv(sum);

    if (lane_id == 0) {
        output[row] = __float2bfloat16(sum);
    }
}

extern "C" void hgemv_bf16_cu(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int N,
    int K,
    cudaStream_t stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;  // 256

    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    hgemv_bf16_v3_kernel<WARPS_PER_BLOCK><<<grid, THREADS, 0, stream>>>(
        input, weight, output, N, K);
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

// ============================================================================
// FP16 variants (for AWQ / float16 models)
// ============================================================================

// --- FP16 cublasLt GEMM: C = A @ B^T, row-major, all FP16 ---
void gemm_cublasLt_AxBT_RowMajor_fp16(
    cublasLtHandle_t ltHandle,
    int M, int N, int K,
    const half *d_A,
    const half *d_B,
    half *d_C,
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

    cublasLtMatrixLayout_t Adesc = NULL;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, k_gemm, m_gemm, k_gemm));
    cublasLtMatrixLayout_t Bdesc = NULL;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, k_gemm, n_gemm, k_gemm));
    cublasLtMatrixLayout_t Cdesc = NULL;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m_gemm, n_gemm, m_gemm));

    cublasLtMatmulPreference_t preference = NULL;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) { printf("cuBLASLt FP16: No algorithm found!\n"); exit(1); }

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasLtMatmul(ltHandle, operationDesc, &alpha,
                                d_B, Adesc, d_A, Bdesc, &beta,
                                d_C, Cdesc, d_C, Cdesc,
                                &heuristicResult.algo, workspace, workspaceSize, stream));

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
}

// --- FP16 GEMV kernel (same structure as BF16 v3) ---
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 6)
hgemv_fp16_v3_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int N,
    const int K
) {
    const int lane_id = threadIdx.x & 31;
    const int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;

    const int pack_num = K >> 3;
    const float4* __restrict__ input_f4 = reinterpret_cast<const float4*>(input);
    const float4* __restrict__ weight_f4 = reinterpret_cast<const float4*>(weight + row * K);

    float sum = 0.0f;
    for (int i = lane_id; i < pack_num; i += 32) {
        float4 x = __ldg(input_f4 + i);
        float4 w = __ldg(weight_f4 + i);
        const half* xh = reinterpret_cast<const half*>(&x);
        const half* wh = reinterpret_cast<const half*>(&w);
        sum += __half2float(xh[0]) * __half2float(wh[0]);
        sum += __half2float(xh[1]) * __half2float(wh[1]);
        sum += __half2float(xh[2]) * __half2float(wh[2]);
        sum += __half2float(xh[3]) * __half2float(wh[3]);
        sum += __half2float(xh[4]) * __half2float(wh[4]);
        sum += __half2float(xh[5]) * __half2float(wh[5]);
        sum += __half2float(xh[6]) * __half2float(wh[6]);
        sum += __half2float(xh[7]) * __half2float(wh[7]);
    }
    sum = warp_reduce_sum_gemv(sum);
    if (lane_id == 0) output[row] = __float2half(sum);
}

extern "C" void hgemv_fp16_cu(
    const half* input, const half* weight, half* output,
    int N, int K, cudaStream_t stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    hgemv_fp16_v3_kernel<WARPS_PER_BLOCK><<<grid, THREADS, 0, stream>>>(input, weight, output, N, K);
}

extern "C" void gemm_cublaslt_fp16(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    cudaStream_t stream, cublasLtHandle_t handle,
    void* workspace, size_t workspaceSize
) {
    gemm_cublasLt_AxBT_RowMajor_fp16(handle, M, N, K, A, B, C, workspace, workspaceSize, stream);
}

// ============================================================================
//  INT4 GEMV (decode, M=1) — K-packed, FP16 magic number dequant, BF16 I/O
//
//  Weight layout (compressed-tensors / pack-quantized format):
//    weight_packed:     [N, K/8]        (int32) — 8 consecutive K-position INT4 per int32
//    weight_zero_point: [N/8, num_groups] (int32) — zero points packed along N
//    weight_scale:      [N, num_groups]   (bf16)  — per-group scale factors
//
//  Magic number dequant: embed INT4 nibble into FP16 mantissa bits.
//    0x6400 = fp16(1024.0). OR nibble → fp16(1024 + nibble).
//    Subtract fp16(1024 + zero_point) → dequant in FP pipeline (no int shift/mask per element).
//    Reduces integer ALU from 16 to 7 ops per int32 word.
// ============================================================================

// Extract 8 nibbles from one int32 into 4 x half2, using FP16 magic number trick.
// Each half2 contains two non-adjacent nibbles: (n0,n4), (n1,n5), (n2,n6), (n3,n7).
// magic_zp = half2(1024+zp, 1024+zp) is precomputed per-group.
// scale_h2 = half2(scale, scale) is precomputed per-group.
__device__ __forceinline__ void dequant_8xint4_magic(
    uint32_t word,
    half2 magic_zp,  // half2(1024+zp, 1024+zp)
    half2 scale_h2,  // half2(scale, scale)
    half2 &out01,    // dequant(n0, n4)
    half2 &out23,    // dequant(n1, n5)
    half2 &out45,    // dequant(n2, n6)
    half2 &out67     // dequant(n3, n7)
) {
    static constexpr uint32_t MAGIC = 0x64006400u;  // two fp16 1024.0
    static constexpr uint32_t MASK  = 0x000F000Fu;  // low 4 bits of each halfword

    // 4 shift + 4 mask + 4 or = 12 int ops for 8 values (vs 16 before)
    uint32_t p04 = ((word      ) & MASK) | MAGIC;
    uint32_t p15 = ((word >>  4) & MASK) | MAGIC;
    uint32_t p26 = ((word >>  8) & MASK) | MAGIC;
    uint32_t p37 = ((word >> 12) & MASK) | MAGIC;

    // FP16 sub + mul: 8 values in 4 half2 ops each = 8 FP ops
    out01 = __hmul2(__hsub2(*reinterpret_cast<half2*>(&p04), magic_zp), scale_h2);
    out23 = __hmul2(__hsub2(*reinterpret_cast<half2*>(&p15), magic_zp), scale_h2);
    out45 = __hmul2(__hsub2(*reinterpret_cast<half2*>(&p26), magic_zp), scale_h2);
    out67 = __hmul2(__hsub2(*reinterpret_cast<half2*>(&p37), magic_zp), scale_h2);
}

template <int WARPS_PER_BLOCK, bool GROUP_SIZE_IS_POW2>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 4)
kpack_gemv_kernel(
    const __nv_bfloat16* __restrict__ input,        // [K]
    const int32_t* __restrict__ weight_packed,       // [N, K/8]
    const int32_t* __restrict__ weight_zero_point,   // [N/8, num_groups]
    const __nv_bfloat16* __restrict__ weight_scale,  // [N, num_groups]
    __nv_bfloat16* __restrict__ output,              // [N]
    const int N, const int K, const int group_size,
    const int group_shift
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int K_packed = K >> 3;
    const int num_groups = K / group_size;

    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const int32_t* wp_row = weight_packed + row * K_packed;
    const __nv_bfloat16* sc_row = weight_scale + row * num_groups;

    const int zp_row_packed = row >> 3;
    const int zp_bit_offset = (row & 7) * 4;
    const int32_t* zp_row = weight_zero_point + zp_row_packed * num_groups;

    const int4* input_i4 = reinterpret_cast<const int4*>(input);

    float acc = 0.0f;
    int kp = lane_id;

    // Main loop with 4x unroll
    for (; kp + 3 * 32 < K_packed; kp += 4 * 32) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int kpu = kp + u * 32;
            int k_base = kpu * 8;
            int g = GROUP_SIZE_IS_POW2 ? (k_base >> group_shift) : (k_base / group_size);

            // Load scale and zero point, build magic constants
            float scale_f = __bfloat162float(__ldg(&sc_row[g]));
            int32_t zp_packed = __ldg(&zp_row[g]);
            int zero = (zp_packed >> zp_bit_offset) & 0xF;

            half scale_h = __float2half(scale_f);
            half magic_zp_val = __float2half(1024.0f + (float)zero);
            half2 scale_h2 = __half2half2(scale_h);
            half2 magic_zp = __half2half2(magic_zp_val);

            // Load packed weight word
            int32_t word = __ldg(&wp_row[kpu]);

            // Dequant 8 nibbles via magic number trick → 4 x half2
            // out layout: (n0,n4), (n1,n5), (n2,n6), (n3,n7)
            half2 d04, d15, d26, d37;
            dequant_8xint4_magic(word, magic_zp, scale_h2, d04, d15, d26, d37);

            // Load input as int4 (8 bf16 values)
            int4 in = __ldg(&input_i4[kpu]);
            const __nv_bfloat16* inp = reinterpret_cast<const __nv_bfloat16*>(&in);

            // Accumulate: reorder dequant results to match sequential k order
            // d04 = (w0, w4), d15 = (w1, w5), d26 = (w2, w6), d37 = (w3, w7)
            float w0 = __half2float(__low2half(d04));
            float w1 = __half2float(__low2half(d15));
            float w2 = __half2float(__low2half(d26));
            float w3 = __half2float(__low2half(d37));
            float w4 = __half2float(__high2half(d04));
            float w5 = __half2float(__high2half(d15));
            float w6 = __half2float(__high2half(d26));
            float w7 = __half2float(__high2half(d37));

            acc += w0 * __bfloat162float(inp[0]);
            acc += w1 * __bfloat162float(inp[1]);
            acc += w2 * __bfloat162float(inp[2]);
            acc += w3 * __bfloat162float(inp[3]);
            acc += w4 * __bfloat162float(inp[4]);
            acc += w5 * __bfloat162float(inp[5]);
            acc += w6 * __bfloat162float(inp[6]);
            acc += w7 * __bfloat162float(inp[7]);
        }
    }

    // Remainder loop (same magic dequant)
    for (; kp < K_packed; kp += 32) {
        int k_base = kp * 8;
        int g = GROUP_SIZE_IS_POW2 ? (k_base >> group_shift) : (k_base / group_size);

        float scale_f = __bfloat162float(__ldg(&sc_row[g]));
        int32_t zp_packed = __ldg(&zp_row[g]);
        int zero = (zp_packed >> zp_bit_offset) & 0xF;

        half scale_h = __float2half(scale_f);
        half2 scale_h2 = __half2half2(scale_h);
        half2 magic_zp = __half2half2(__float2half(1024.0f + (float)zero));

        int32_t word = __ldg(&wp_row[kp]);
        half2 d04, d15, d26, d37;
        dequant_8xint4_magic(word, magic_zp, scale_h2, d04, d15, d26, d37);

        float w0 = __half2float(__low2half(d04));
        float w1 = __half2float(__low2half(d15));
        float w2 = __half2float(__low2half(d26));
        float w3 = __half2float(__low2half(d37));
        float w4 = __half2float(__high2half(d04));
        float w5 = __half2float(__high2half(d15));
        float w6 = __half2float(__high2half(d26));
        float w7 = __half2float(__high2half(d37));

        acc += w0 * __bfloat162float(__ldg(&input[k_base + 0]));
        acc += w1 * __bfloat162float(__ldg(&input[k_base + 1]));
        acc += w2 * __bfloat162float(__ldg(&input[k_base + 2]));
        acc += w3 * __bfloat162float(__ldg(&input[k_base + 3]));
        acc += w4 * __bfloat162float(__ldg(&input[k_base + 4]));
        acc += w5 * __bfloat162float(__ldg(&input[k_base + 5]));
        acc += w6 * __bfloat162float(__ldg(&input[k_base + 6]));
        acc += w7 * __bfloat162float(__ldg(&input[k_base + 7]));
    }

    acc = warp_reduce_sum_gemv(acc);
    if (lane_id == 0) {
        output[row] = __float2bfloat16(acc);
    }
}

// ============================================================================
//  INT4 GEMM (prefill, M>1) — K-packed, FP16 magic dequant, BF16 I/O
// ============================================================================
#define INT4_GEMM_BX 16
#define INT4_GEMM_BY 16

extern "C" __global__ void kpack_gemm_kernel(
    const __nv_bfloat16* __restrict__ input,         // [M, K]
    const int32_t* __restrict__ weight_packed,        // [N, K/8]
    const int32_t* __restrict__ weight_zero_point,    // [N/8, num_groups]
    const __nv_bfloat16* __restrict__ weight_scale,   // [N, num_groups]
    __nv_bfloat16* __restrict__ output,               // [M, N]
    int M, int N, int K, int group_size
) {
    const int row = blockIdx.y * INT4_GEMM_BY + threadIdx.y;  // M dim
    const int col = blockIdx.x * INT4_GEMM_BX + threadIdx.x;  // N dim
    if (row >= M || col >= N) return;

    const int K_packed = K / 8;
    const int num_groups = K / group_size;

    const int32_t* wp_row = weight_packed + col * K_packed;
    const __nv_bfloat16* sc_row = weight_scale + col * num_groups;

    const int zp_col_packed = col >> 3;
    const int zp_bit_offset = (col & 7) * 4;
    const int32_t* zp_row = weight_zero_point + zp_col_packed * num_groups;

    float acc = 0.0f;

    for (int kp = 0; kp < K_packed; kp++) {
        int k_base = kp * 8;
        int g = k_base / group_size;

        float scale_f = __bfloat162float(sc_row[g]);
        int32_t zp_packed = zp_row[g];
        int zero = (zp_packed >> zp_bit_offset) & 0xF;

        half scale_h = __float2half(scale_f);
        half2 scale_h2 = __half2half2(scale_h);
        half2 magic_zp = __half2half2(__float2half(1024.0f + (float)zero));

        int32_t word = wp_row[kp];
        half2 d04, d15, d26, d37;
        dequant_8xint4_magic(word, magic_zp, scale_h2, d04, d15, d26, d37);

        float w0 = __half2float(__low2half(d04));
        float w1 = __half2float(__low2half(d15));
        float w2 = __half2float(__low2half(d26));
        float w3 = __half2float(__low2half(d37));
        float w4 = __half2float(__high2half(d04));
        float w5 = __half2float(__high2half(d15));
        float w6 = __half2float(__high2half(d26));
        float w7 = __half2float(__high2half(d37));

        acc += w0 * __bfloat162float(input[row * K + k_base + 0]);
        acc += w1 * __bfloat162float(input[row * K + k_base + 1]);
        acc += w2 * __bfloat162float(input[row * K + k_base + 2]);
        acc += w3 * __bfloat162float(input[row * K + k_base + 3]);
        acc += w4 * __bfloat162float(input[row * K + k_base + 4]);
        acc += w5 * __bfloat162float(input[row * K + k_base + 5]);
        acc += w6 * __bfloat162float(input[row * K + k_base + 6]);
        acc += w7 * __bfloat162float(input[row * K + k_base + 7]);
    }
    output[row * N + col] = __float2bfloat16(acc);
}

// ============================================================================
//  INT4 C-linkage wrappers
// ============================================================================

extern "C" void kpack_gemv_cu(
    const void* input, const void* weight_packed, const void* weight_zero_point,
    const void* weight_scale, void* output,
    int N, int K, int group_size, cudaStream_t stream
) {
    constexpr int WARPS = 4;
    int grid_x = (N + WARPS - 1) / WARPS;
    const bool group_size_is_pow2 = group_size > 0 && ((group_size & (group_size - 1)) == 0);
    const int group_shift = group_size_is_pow2 ? __builtin_ctz(group_size) : 0;

    if (group_size_is_pow2) {
        kpack_gemv_kernel<WARPS, true><<<grid_x, WARPS * 32, 0, stream>>>(
            (const __nv_bfloat16*)input, (const int32_t*)weight_packed,
            (const int32_t*)weight_zero_point, (const __nv_bfloat16*)weight_scale,
            (__nv_bfloat16*)output, N, K, group_size, group_shift);
    } else {
        kpack_gemv_kernel<WARPS, false><<<grid_x, WARPS * 32, 0, stream>>>(
            (const __nv_bfloat16*)input, (const int32_t*)weight_packed,
            (const int32_t*)weight_zero_point, (const __nv_bfloat16*)weight_scale,
            (__nv_bfloat16*)output, N, K, group_size, 0);
    }
}

extern "C" void kpack_gemm_cu(
    const void* input, const void* weight_packed, const void* weight_zero_point,
    const void* weight_scale, void* output,
    int M, int N, int K, int group_size, cudaStream_t stream
) {
    dim3 block(INT4_GEMM_BX, INT4_GEMM_BY);
    dim3 grid((N + INT4_GEMM_BX - 1) / INT4_GEMM_BX, (M + INT4_GEMM_BY - 1) / INT4_GEMM_BY);
    kpack_gemm_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)input, (const int32_t*)weight_packed,
        (const int32_t*)weight_zero_point, (const __nv_bfloat16*)weight_scale,
        (__nv_bfloat16*)output, M, N, K, group_size);
}
