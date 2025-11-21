#include "rope.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

// 宏定义：用于处理 CUDA 核函数中的错误检查，在实际生产代码中推荐使用
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            /* 在 kernel 中不能直接返回错误码，通常通过某种机制报告 */           \
        }                                                                         \
    } while (0)

// --- CUDA Kernel ---
/**
 * @brief RoPE 核心旋转操作的 CUDA 核函数。
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
    const int seq_len,
    const float* __restrict__ sin_cache,
    const float* __restrict__ cos_cache)
{
    // 每个线程处理一个维度对 (i, i+1)
    // 线程索引 i 从 0 到 dim/2 - 1
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_pos = blockIdx.y;
    // 我们只需要 dim/2 个线程，因为每个线程处理两个元素 (i 和 i+1)
    if (thread_idx * 2 >= dim) {
        return;
    }

    // 旋转操作的维度索引 i = 2 * thread_idx
    int i = thread_idx * 2;
    
    // --- 1. 计算缓存索引和加载 sin/cos ---
    // Caching index: pos * head_size + head_dim
    int head_dim = i % head_size;
    int cache_idx = (pos + seq_pos) * head_size + head_dim;
    // 加载 sin 和 cos 值
    float fci = sin_cache[cache_idx]; // sin(val)
    float fcr = cos_cache[cache_idx]; // cos(val)
    // --- 2. 旋转 Query (Q) 向量 ---
    auto some_ptr = reinterpret_cast<float2*>(input_q + (seq_pos * dim) + i);
    
    // 应用旋转: v0' = v0 * cos - v1 * sin, v1' = v0 * sin + v1 * cos
    float2 some = *some_ptr;
    (*some_ptr).x    = some.x * fcr - some.y * fci;
    (*some_ptr).y    = some.x * fci + some.y * fcr;
    
    // --- 3. 旋转 Key (K) 向量 ---
    // 只有当 i < kv_dim 时才旋转 K
    if (i < kv_dim) {
        auto some_ptr = reinterpret_cast<float2*>(input_k + (seq_pos * kv_dim) + i);
        float2 some = *some_ptr;
        (*some_ptr).x    = some.x * fcr - some.y * fci;
        (*some_ptr).y    = some.x * fci + some.y * fcr;
    }
}


// --- FFI 包装函数 (Host Function) ---
extern "C" void rope_kernel_cu(
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
    const int THREADS_PER_BLOCK = 256;
    int num_rotations = dim / 2;
    int num_blocks = (num_rotations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid(num_blocks, seq_len);
    // 3. 启动核函数
    rope_rotate_kernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
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


// --- FFI 包装函数 (Host Function) ---
extern "C" void sin_cos_cache_calc_cu(
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