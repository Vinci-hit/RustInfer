#include "rope.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cuda_bf16.h>
// 宏定义：用于处理 CUDA 核函数中的错误检查，在实际生产代码中推荐使用
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            /* 在 kernel 中不能直接返回错误码，通常通过某种机制报告 */           \
        }                                                                         \
    } while (0)

// --- CUDA Kernel (F32版本) ---
/**
 * @brief RoPE 核心旋转操作的 CUDA 核函数 (F32版本)。
 * 
 * 每个线程处理一个维度对 (i, i+1)，即旋转向量的一个元素对。
 * 我们假设这个 kernel 是针对 Batch size=1, Sequence length=1 的单个向量调用的。
 * 如果要处理更大的 Batch 或 Sequence，需要修改启动配置和索引逻辑。
 *
 * @param dim Q/K 向量的总旋转维度。
 * @param kv_dim K 向量旋转的维度。
 * @param head_size Attention Head 的大小。
 * @param input_q Q 张量的设备指针 (可读写)。
 * @param input_k K 张量的设备指针 (可读写)。
 * @param pos 当前位置索引 (单个 i32)。
 * @param sin_cache 正弦缓存 (只读)。
 * @param cos_cache 余弦缓存 (只读)。
 */
__global__ void rope_rotate_kernel(
    const int dim,
    const int kv_dim,
    const int head_size,
    float* __restrict__ input_q,
    float* __restrict__ input_k,
    const int pos,
    const float* __restrict__ sin_cache,
    const float* __restrict__ cos_cache)
{
    // 每个线程处理一个head_size的元素
    // 每个block处理一个dim，y轴为seq_len
    int start_head_id = blockIdx.x * head_size;
    int seq_pos = blockIdx.y;

    // 旋转操作的维度索引 i = 2 * thread_idx
    int abs_pos = pos + seq_pos;
    int q_start = seq_pos * dim;
    int k_start = seq_pos * kv_dim;
    for (int i = 0; i < head_size / 2; i ++) {
        float sin_val = sin_cache[abs_pos * head_size + i*2]; // sin(val)
        float cos_val = cos_cache[abs_pos * head_size + i*2]; // cos(val)
        int q_idx_j = q_start + start_head_id + i;
        int q_idx_j1 = q_start + start_head_id + i + head_size / 2;
        float v0_q = input_q[q_idx_j];
        float v1_q = input_q[q_idx_j1];
        input_q[q_idx_j] = v0_q * cos_val - v1_q * sin_val;
        input_q[q_idx_j1] = v0_q * sin_val + v1_q * cos_val;
        if (start_head_id < kv_dim) {
            int k_idx_j = k_start + start_head_id + i;
            int k_idx_j1 = k_start + start_head_id + i + head_size / 2;
            float v0_k = input_k[k_idx_j];
            float v1_k = input_k[k_idx_j1];
            input_k[k_idx_j] = v0_k * cos_val - v1_k * sin_val;
            input_k[k_idx_j1] = v0_k * sin_val + v1_k * cos_val;
        }
    }
}


// --- CUDA Kernel (BF16版本) ---
/**
 * @brief RoPE 核心旋转操作的 CUDA 核函数 (BF16版本)。
 * 
 * 每个线程处理一个维度对 (i, i+1)，即旋转向量的一个元素对。
 * 我们假设这个 kernel 是针对 Batch size=1, Sequence length=1 的单个向量调用的。
 * 如果要处理更大的 Batch 或 Sequence，需要修改启动配置和索引逻辑。
 *
 * @param dim Q/K 向量的总旋转维度。
 * @param kv_dim K 向量旋转的维度。
 * @param head_size Attention Head 的大小。
 * @param input_q Q 张量的设备指针 (可读写)。
 * @param input_k K 张量的设备指针 (可读写)。
 * @param pos 当前位置索引 (单个 i32)。
 * @param sin_cache 正弦缓存 (只读)。
 * @param cos_cache 余弦缓存 (只读)。
 */
__global__ void rope_rotate_kernel_bf16x8(
    const int dim,
    const int kv_dim,
    const int head_size,
    __nv_bfloat16* __restrict__ input_q,
    __nv_bfloat16* __restrict__ input_k,
    const int pos,
    const int seq_len,
    const __nv_bfloat16* __restrict__ sin_cache,
    const __nv_bfloat16* __restrict__ cos_cache)
{
    // 每个线程处理一个head_size的元素
    // 每个block处理一个dim，y轴为seq_len
    int start_head_id = blockIdx.x * head_size;
    int seq_pos = blockIdx.y;

    // 旋转操作的维度索引 i = 2 * thread_idx
    int abs_pos = pos + seq_pos;
    int q_start = seq_pos * dim;
    int k_start = seq_pos * kv_dim;
    constexpr int elems_per_thread = 8;
    for (int i = 0; i < head_size / 2; i += elems_per_thread) {
#pragma unroll
        for (int j = 0; j < elems_per_thread; j ++)
        {
            __nv_bfloat16 sin_val = sin_cache[abs_pos * head_size + i + j]; // sin(val)
            __nv_bfloat16 cos_val = cos_cache[abs_pos * head_size + i + j]; // cos(val)
            int q_idx_j = q_start + start_head_id + i + j;
            int q_idx_j1 = q_start + start_head_id + i + head_size / 2 + j;
            __nv_bfloat16 v0_q = input_q[q_idx_j];
            __nv_bfloat16 v1_q = input_q[q_idx_j1];
            __nv_bfloat16 result0 = v0_q * cos_val - v1_q * sin_val;
            __nv_bfloat16 result1 = v0_q * sin_val + v1_q * cos_val;
            input_q[q_idx_j] = result0;
            input_q[q_idx_j1] = result1;
            if (start_head_id < kv_dim) {
                int k_idx_j = k_start + start_head_id + i;
                int k_idx_j1 = k_start + start_head_id + i + head_size / 2;
                __nv_bfloat16 v0_k = input_k[k_idx_j];
                __nv_bfloat16 v1_k = input_k[k_idx_j1];
                __nv_bfloat16 result0_k = v0_k * cos_val - v1_k * sin_val;
                __nv_bfloat16 result1_k = v0_k * sin_val + v1_k * cos_val;
                input_k[k_idx_j] = result0_k;
                input_k[k_idx_j1] = result1_k;
            }
        }
    }
}
void rope_kernel_cu_bf16(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    __nv_bfloat16* input_q,
    __nv_bfloat16* input_k,
    int32_t input_pos,
    int32_t seq_len,
    __nv_bfloat16* sin_cache,
    __nv_bfloat16* cos_cache,
    cudaStream_t stream)
{
    int threads_per_block = head_size/2;
    int num_blocks = dim / head_size; // 必定能整除
    // 每个thread处理一个head_size，每个block处理一个dim，y轴为seq_len
    dim3 grid(num_blocks, seq_len);
    // 3. 启动核函数
    rope_rotate_kernel_bf16x8<<<grid, threads_per_block, 0, stream>>>(
        dim,
        kv_dim,
        head_size,
        input_q,
        input_k,
        input_pos,
        seq_len,
        sin_cache,
        cos_cache
    );

    // 4. 检查是否有核函数启动错误
    CUDA_CHECK(cudaGetLastError());
}

// --- FFI 包装函数 (Host Function) ---
void rope_kernel_cu(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    float* input_q,
    float* input_k,
    int32_t input_pos,
    int32_t seq_len,
    const float* sin_cache,
    const float* cos_cache,
    cudaStream_t stream)
{
    int threads_per_block = head_size/2;
    int num_blocks = dim / head_size; // 必定能整除
    // 每个thread处理一个head_size，每个block处理一个dim，y轴为seq_len
    dim3 grid(num_blocks, seq_len);
    // 3. 启动核函数
    rope_rotate_kernel<<<grid, threads_per_block, 0, stream>>>(
        dim,
        kv_dim,
        head_size,
        input_q,
        input_k,
        input_pos,
        sin_cache,
        cos_cache
    );

    // 4. 检查是否有核函数启动错误
    CUDA_CHECK(cudaGetLastError());
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // 确保不越界
  if (idx >= head_size) return; 
  
  int head_dim = idx; // idx 直接就是 head_dim，因为我们每个线程处理一个 head_dim
  
  // 预先计算一次 base_power，避免循环中重复计算
  float base = 500000.0f;
  float head_size_f = (float)head_size;
  float head_dim_f = (float)head_dim;
  
  // exponent = head_dim / head_size
  float exponent = head_dim_f / head_size_f;
  
  // freq = 1.0f / pow(10000.0f, exponent)
  float freq = 1.0f / powf(base, exponent);

  // 循环 pos 维度
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float val = (float)pos * freq;
    
    // 由于 sinf 和 cosf 在 CUDA 中有 __device__ 实现，可以直接调用
    float fcr = cosf(val);
    float fci = sinf(val);
    
    // 写入缓存
    int cache_idx = pos * head_size + head_dim;
    sin_cache[cache_idx] = fci;
    cos_cache[cache_idx] = fcr;
  }
}

__global__ void sin_cos_calc_bf16(int head_size, int max_seq_len,
                                  __nv_bfloat16* sin_cache,
                                  __nv_bfloat16* cos_cache) {
    // 每个线程处理一个“偶数维度对”的索引 k（k = 0, 1, ..., head_size/2 - 1）
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    int half_head = head_size / 2;

    if (k >= half_head) return;

    // 对应的原始偶数维度索引：dim = 2 * k
    int dim = 2 * k;

    float base = 500000.0f;
    float exponent = (float)dim / (float)head_size;
    float freq = 1.0f / powf(base, exponent);

    // 遍历所有位置
    for (int pos = 0; pos < max_seq_len; ++pos) {
        float val = (float)pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        // 写入压缩后的缓存：每个 pos 只占 half_head 个元素
        int cache_idx = pos * half_head + k;
        sin_cache[cache_idx] = __float2bfloat16(fci);
        cos_cache[cache_idx] = __float2bfloat16(fcr);
    }
}


// --- FFI 包装函数 (Host Function) ---
void sin_cos_cache_calc_cu(
    int32_t head_size,
    int32_t max_seq_len,
    float* sin_cache,
    float* cos_cache,
    cudaStream_t stream) 
{
    // 启动配置：1 个 Block，head_size 个 Threads
    int threads = head_size;

    sin_cos_calc<<<1, threads, 0, stream>>>(
        head_size, 
        max_seq_len, 
        sin_cache, 
        cos_cache
    );
    
    // 检查核函数启动错误
    CUDA_CHECK(cudaGetLastError());
}

void sin_cos_cache_calc_cu_bf16(
    int32_t head_size,
    int32_t max_seq_len,
    __nv_bfloat16* sin_cache,
    __nv_bfloat16* cos_cache,
    cudaStream_t stream) 
{
    // 启动配置：1 个 Block，head_size 个 Threads
    int threads = head_size;

    sin_cos_calc_bf16<<<1, threads, 0, stream>>>(
        head_size, 
        max_seq_len, 
        sin_cache, 
        cos_cache
    );
    
    // 检查核函数启动错误
    CUDA_CHECK(cudaGetLastError());
}