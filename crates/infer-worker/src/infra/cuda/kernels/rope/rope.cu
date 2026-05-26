#include "rope.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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


// --- CUDA Kernel (BF16 版本) ---
//
// RoPE 的唯一 BF16 内核：按"每行一个绝对位置"语义运行。
//
//   abs_pos = positions[seq_idx]
//   q_row_stride / k_row_stride 用于支持非连续输入（如 qkv.slice 的 q、k 段）
//
// 所有 caller（不管是 prefill 的 seq_len 行还是 decode 的 1/B 行）都走这条路径；
// prefill 的 caller 负责在 host 上把 [p, p+1, ..., p+seq_len-1] 写到 positions[]。
__global__ void rope_rotate_kernel_llama3_bf16(
    __nv_bfloat16* __restrict__ input_q,
    __nv_bfloat16* __restrict__ input_k,
    const __nv_bfloat16* __restrict__ sin_cache,
    const __nv_bfloat16* __restrict__ cos_cache,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int* __restrict__ positions, // [seq_len]，每行绝对位置
    const int seq_len,
    const int q_row_stride,            // elements (bf16) per row in input_q
    const int k_row_stride             // elements (bf16) per row in input_k
) {
    const int tid = threadIdx.x;
    const int q_head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int half_head = head_size / 2;
    const int group_size = num_heads / num_kv_heads;

    if (tid >= half_head || q_head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }

    int abs_pos = positions[seq_idx];
    float sin_val = __bfloat162float(sin_cache[abs_pos * half_head + tid]);
    float cos_val = __bfloat162float(cos_cache[abs_pos * half_head + tid]);

    int q_base = seq_idx * q_row_stride + q_head_idx * head_size;
    int idx_1 = q_base + tid;
    int idx_2 = q_base + tid + half_head;

    float q1 = __bfloat162float(input_q[idx_1]);
    float q2 = __bfloat162float(input_q[idx_2]);

    input_q[idx_1] = __float2bfloat16(q1 * cos_val - q2 * sin_val);
    input_q[idx_2] = __float2bfloat16(q1 * sin_val + q2 * cos_val);

    if (q_head_idx % group_size == 0) {
        int kv_head_idx = q_head_idx / group_size;
        int k_base = seq_idx * k_row_stride + kv_head_idx * head_size;
        int k_idx_1 = k_base + tid;
        int k_idx_2 = k_base + tid + half_head;

        float k1 = __bfloat162float(input_k[k_idx_1]);
        float k2 = __bfloat162float(input_k[k_idx_2]);

        input_k[k_idx_1] = __float2bfloat16(k1 * cos_val - k2 * sin_val);
        input_k[k_idx_2] = __float2bfloat16(k1 * sin_val + k2 * cos_val);
    }
}

extern "C" void rope_kernel_cu_bf16(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    __nv_bfloat16* input_q,
    __nv_bfloat16* input_k,
    const int32_t* positions,
    int32_t seq_len,
    int32_t q_row_stride,
    int32_t k_row_stride,
    __nv_bfloat16* sin_cache,
    __nv_bfloat16* cos_cache,
    cudaStream_t stream)
{
    int threads_per_block = head_size / 2;
    int num_heads = dim / head_size;
    dim3 grid(num_heads, seq_len);

    rope_rotate_kernel_llama3_bf16<<<grid, threads_per_block, 0, stream>>>(
        input_q, input_k, sin_cache, cos_cache,
        num_heads, kv_dim / head_size, head_size,
        positions, seq_len, q_row_stride, k_row_stride
    );
    CUDA_CHECK(cudaGetLastError());
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

__global__ void sin_cos_calc(int head_size, int max_seq_len, float rope_theta, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // 确保不越界
  if (idx >= head_size) return; 
  
  int head_dim = idx; // idx 直接就是 head_dim，因为我们每个线程处理一个 head_dim
  
  float base = rope_theta;
  float head_size_f = (float)head_size;
  float head_dim_f = (float)head_dim;

  // exponent = head_dim / head_size
  float exponent = head_dim_f / head_size_f;

  // freq = 1.0f / pow(rope_theta, exponent)
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

__global__ void sin_cos_calc_bf16(int head_size, int max_seq_len, float rope_theta,
                                  __nv_bfloat16* sin_cache,
                                  __nv_bfloat16* cos_cache,
                                  float factor, float low_freq_factor, float high_freq_factor,
                                  float original_max_pos_emb) {
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    int half_head = head_size / 2;

    if (k >= half_head) return;

    int dim = 2 * k;
    float exponent = (float)dim / (float)head_size;
    float freq = 1.0f / powf(rope_theta, exponent);

    // Apply llama3 rope scaling if factor > 1
    if (factor > 1.0f) {
        float old_context_len = original_max_pos_emb;
        float low_freq_wavelen = old_context_len / low_freq_factor;
        float high_freq_wavelen = old_context_len / high_freq_factor;
        float wavelen = 2.0f * 3.14159265358979323846f / freq;

        if (wavelen > low_freq_wavelen) {
            freq = freq / factor;
        } else if (wavelen >= high_freq_wavelen) {
            float smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            freq = (1.0f - smooth) * freq / factor + smooth * freq;
        }
    }

    for (int pos = 0; pos < max_seq_len; ++pos) {
        float val = (float)pos * freq;
        int cache_idx = pos * half_head + k;
        sin_cache[cache_idx] = __float2bfloat16(sinf(val));
        cos_cache[cache_idx] = __float2bfloat16(cosf(val));
    }
}


// --- FFI 包装函数 (Host Function) ---
extern "C" void sin_cos_cache_calc_cu(
    int32_t head_size,
    int32_t max_seq_len,
    float rope_theta,
    float* sin_cache,
    float* cos_cache,
    cudaStream_t stream)
{
    // 启动配置：1 个 Block，head_size 个 Threads
    int threads = head_size;

    sin_cos_calc<<<1, threads, 0, stream>>>(
        head_size,
        max_seq_len,
        rope_theta,
        sin_cache,
        cos_cache
    );
    
    // 检查核函数启动错误
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void sin_cos_cache_calc_cu_bf16(
    int32_t head_size,
    int32_t max_seq_len,
    float rope_theta,
    __nv_bfloat16* sin_cache,
    __nv_bfloat16* cos_cache,
    float factor, float low_freq_factor, float high_freq_factor,
    float original_max_pos_emb,
    cudaStream_t stream)
{
    int threads = head_size;

    sin_cos_calc_bf16<<<1, threads, 0, stream>>>(
        head_size,
        max_seq_len,
        rope_theta,
        sin_cache,
        cos_cache,
        factor, low_freq_factor, high_freq_factor, original_max_pos_emb
    );

    CUDA_CHECK(cudaGetLastError());
}




// ============= FP16 variants =============

__global__ void rope_rotate_kernel_llama3_fp16(
    __half* __restrict__ input_q,
    __half* __restrict__ input_k,
    const __half* __restrict__ sin_cache,
    const __half* __restrict__ cos_cache,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int* __restrict__ positions,
    const int seq_len,
    const int q_row_stride,
    const int k_row_stride
) {
    const int tid = threadIdx.x;
    const int q_head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int half_head = head_size / 2;
    const int group_size = num_heads / num_kv_heads;

    if (tid >= half_head || q_head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }

    int abs_pos = positions[seq_idx];
    float sin_val = __half2float(sin_cache[abs_pos * half_head + tid]);
    float cos_val = __half2float(cos_cache[abs_pos * half_head + tid]);

    int q_base = seq_idx * q_row_stride + q_head_idx * head_size;
    int idx_1 = q_base + tid;
    int idx_2 = q_base + tid + half_head;

    float q1 = __half2float(input_q[idx_1]);
    float q2 = __half2float(input_q[idx_2]);

    input_q[idx_1] = __float2half(q1 * cos_val - q2 * sin_val);
    input_q[idx_2] = __float2half(q1 * sin_val + q2 * cos_val);

    if (q_head_idx % group_size == 0) {
        int kv_head_idx = q_head_idx / group_size;
        int k_base = seq_idx * k_row_stride + kv_head_idx * head_size;
        int k_idx_1 = k_base + tid;
        int k_idx_2 = k_base + tid + half_head;

        float k1 = __half2float(input_k[k_idx_1]);
        float k2 = __half2float(input_k[k_idx_2]);

        input_k[k_idx_1] = __float2half(k1 * cos_val - k2 * sin_val);
        input_k[k_idx_2] = __float2half(k1 * sin_val + k2 * cos_val);
    }
}

extern "C" void rope_kernel_cu_fp16(
    int32_t dim,
    int32_t kv_dim,
    int32_t head_size,
    __half* input_q,
    __half* input_k,
    const int32_t* positions,
    int32_t seq_len,
    int32_t q_row_stride,
    int32_t k_row_stride,
    __half* sin_cache,
    __half* cos_cache,
    cudaStream_t stream)
{
    int threads_per_block = head_size / 2;
    int num_heads = dim / head_size;
    dim3 grid(num_heads, seq_len);

    rope_rotate_kernel_llama3_fp16<<<grid, threads_per_block, 0, stream>>>(
        input_q, input_k, sin_cache, cos_cache,
        num_heads, kv_dim / head_size, head_size,
        positions, seq_len, q_row_stride, k_row_stride
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void sin_cos_calc_fp16(int head_size, int max_seq_len, float rope_theta,
                                  __half* sin_cache,
                                  __half* cos_cache,
                                  float factor, float low_freq_factor, float high_freq_factor,
                                  float original_max_pos_emb) {
    int k = threadIdx.x + blockDim.x * blockIdx.x;
    int half_head = head_size / 2;

    if (k >= half_head) return;

    int dim = 2 * k;
    float exponent = (float)dim / (float)head_size;
    float freq = 1.0f / powf(rope_theta, exponent);

    if (factor > 1.0f) {
        float old_context_len = original_max_pos_emb;
        float low_freq_wavelen = old_context_len / low_freq_factor;
        float high_freq_wavelen = old_context_len / high_freq_factor;
        float wavelen = 2.0f * 3.14159265358979323846f / freq;

        if (wavelen > low_freq_wavelen) {
            freq = freq / factor;
        } else if (wavelen >= high_freq_wavelen) {
            float smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            freq = (1.0f - smooth) * freq / factor + smooth * freq;
        }
    }

    for (int pos = 0; pos < max_seq_len; ++pos) {
        float val = (float)pos * freq;
        int cache_idx = pos * half_head + k;
        sin_cache[cache_idx] = __float2half(sinf(val));
        cos_cache[cache_idx] = __float2half(cosf(val));
    }
}

extern "C" void sin_cos_cache_calc_cu_fp16(
    int32_t head_size,
    int32_t max_seq_len,
    float rope_theta,
    __half* sin_cache,
    __half* cos_cache,
    float factor, float low_freq_factor, float high_freq_factor,
    float original_max_pos_emb,
    cudaStream_t stream)
{
    int threads = head_size;

    sin_cos_calc_fp16<<<1, threads, 0, stream>>>(
        head_size,
        max_seq_len,
        rope_theta,
        sin_cache,
        cos_cache,
        factor, low_freq_factor, high_freq_factor, original_max_pos_emb
    );

    CUDA_CHECK(cudaGetLastError());
}

