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
    const int* pos,
    const float* __restrict__ sin_cache,
    const float* __restrict__ cos_cache)
{
    // 每个线程处理一个head_size的元素
    // 每个block处理一个dim，y轴为seq_len
    int start_head_id = blockIdx.x * head_size;
    int seq_pos = blockIdx.y;

    // 旋转操作的维度索引 i = 2 * thread_idx
    int abs_pos = *pos + seq_pos;
    
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
__global__ void rope_rotate_kernel_llama3_bf16(
    __nv_bfloat16* __restrict__ input_q,      // [seq_len, num_heads, head_size]
    __nv_bfloat16* __restrict__ input_k,      // [seq_len, num_kv_heads, head_size]
    const __nv_bfloat16* __restrict__ sin_cache, // [max_seq_len, head_size / 2]
    const __nv_bfloat16* __restrict__ cos_cache, // [max_seq_len, head_size / 2]
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int* pos_offset,                     // 当前 batch 的起始位置偏移
    const int seq_len                         // 当前处理的序列长度
) {
    // 1. 维度计算
    const int tid = threadIdx.x;              // 0 -> half_head - 1
    const int q_head_idx = blockIdx.x;        // 当前是第几个 Q head
    const int seq_idx = blockIdx.y;           // 当前是序列中的第几个 token
    const int half_head = head_size / 2;
    const int group_size = num_heads / num_kv_heads; // GQA 分组大小

    if (tid >= half_head || q_head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }

    // 2. 计算 RoPE Cache 的绝对位置索引
    // Llama 3 通常预计算好了 sin/cos，形状为 [max_seq, half_head]
    int abs_pos = *pos_offset + seq_idx;
    float sin_val = __bfloat162float(sin_cache[abs_pos * half_head + tid]);
    float cos_val = __bfloat162float(cos_cache[abs_pos * half_head + tid]);

    // 3. 处理 Query (Q)
    // 计算当前线程对应的 Q 的两个分量位置
    // 数据布局假设为: [seq, num_heads, head_size]
    int q_base = (seq_idx * num_heads + q_head_idx) * head_size;
    int idx_1 = q_base + tid;
    int idx_2 = q_base + tid + half_head;

    float q1 = __bfloat162float(input_q[idx_1]);
    float q2 = __bfloat162float(input_q[idx_2]);

    // RoPE 核心旋转公式
    input_q[idx_1] = __float2bfloat16(q1 * cos_val - q2 * sin_val);
    input_q[idx_2] = __float2bfloat16(q1 * sin_val + q2 * cos_val);

    // 4. 处理 Key (K) - GQA 逻辑
    // 只有当当前的 Q head 索引是一个组的第一个时，才处理 K，防止重复计算
    if (q_head_idx % group_size == 0) {
        int kv_head_idx = q_head_idx / group_size;
        int k_base = (seq_idx * num_kv_heads + kv_head_idx) * head_size;
        
        int k_idx_1 = k_base + tid;
        int k_idx_2 = k_base + tid + half_head;

        float k1 = __bfloat162float(input_k[k_idx_1]);
        float k2 = __bfloat162float(input_k[k_idx_2]);

        input_k[k_idx_1] = __float2bfloat16(k1 * cos_val - k2 * sin_val);
        input_k[k_idx_2] = __float2bfloat16(k1 * sin_val + k2 * cos_val);
    }
}

void rope_kernel_cu_bf16(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    __nv_bfloat16* input_q,
    __nv_bfloat16* input_k,
    int32_t* input_pos,
    int32_t seq_len,
    __nv_bfloat16* sin_cache,
    __nv_bfloat16* cos_cache,
    cudaStream_t stream)
{
    // 每个线程处理一对元素，所以 Block 大小为 head_size / 2
    // 假设 head_size = 128 (Llama-3), threads = 64，一个 Warp 也就搞定了，效率很高
    int threads_per_block = head_size / 2;
    
    // Grid.x = Head 的数量
    int num_heads = dim / head_size; 
    
    // Grid.y = Sequence Length
    dim3 grid(num_heads, seq_len);
    rope_rotate_kernel_llama3_bf16<<<grid, threads_per_block, 0, stream>>>(
        input_q,
        input_k,
        sin_cache,
        cos_cache,
        num_heads,
        kv_dim / head_size,
        head_size,
        input_pos,
        seq_len
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("RoPE Kernel Failed: %s\n", cudaGetErrorString(err));
    }
}

// --- FFI 包装函数 (Host Function) ---
void rope_kernel_cu(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    float* input_q,
    float* input_k,
    int32_t* input_pos,
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

__global__ void sin_cos_calc(int head_size, int max_seq_len, float base, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // 确保不越界
  if (idx >= head_size) return;

  int head_dim = idx; // idx 直接就是 head_dim，因为我们每个线程处理一个 head_dim

  // 预先计算一次 base_power，避免循环中重复计算
  float head_size_f = (float)head_size;
  float head_dim_f = (float)head_dim;

  // exponent = head_dim / head_size
  float exponent = head_dim_f / head_size_f;

  // freq = 1.0f / pow(base, exponent)
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

__global__ void sin_cos_calc_bf16(int head_size, int max_seq_len, float base,
                                  __nv_bfloat16* sin_cache,
                                  __nv_bfloat16* cos_cache) {
    // 每个线程处理一个"偶数维度对"的索引 k（k = 0, 1, ..., head_size/2 - 1）
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    int half_head = head_size / 2;

    if (k >= half_head) return;

    // 对应的原始偶数维度索引：dim = 2 * k
    int dim = 2 * k;

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
    float rope_base,
    float* sin_cache,
    float* cos_cache,
    cudaStream_t stream)
{
    // 启动配置：1 个 Block，head_size 个 Threads
    int threads = head_size;

    sin_cos_calc<<<1, threads, 0, stream>>>(
        head_size,
        max_seq_len,
        rope_base,
        sin_cache,
        cos_cache
    );

    // 检查核函数启动错误
    CUDA_CHECK(cudaGetLastError());
}

void sin_cos_cache_calc_cu_bf16(
    int32_t head_size,
    int32_t max_seq_len,
    float rope_base,
    __nv_bfloat16* sin_cache,
    __nv_bfloat16* cos_cache,
    cudaStream_t stream)
{

    // 启动配置：1 个 Block，head_size 个 Threads
    int threads = head_size;

    sin_cos_calc_bf16<<<1, threads, 0, stream>>>(
        head_size,
        max_seq_len,
        rope_base,
        sin_cache,
        cos_cache
    );

    // 检查核函数启动错误
    CUDA_CHECK(cudaGetLastError());
}