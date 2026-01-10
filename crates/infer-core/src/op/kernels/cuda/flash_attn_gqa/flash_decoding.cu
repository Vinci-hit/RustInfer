#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "flash_attn_gqa.h"

// Warp 内求和
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ----------------------------------------------------------------------
// 修正后的 Block Reduce Sum (修复了跨 Warp 广播问题)
// ----------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_sum(float val, float* shared_mem) {
    // 1. Warp 内先归约
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_sum(val);

    // 2. 每个 Warp 的第一个线程把结果写入 Shared Memory
    if (lane == 0) shared_mem[wid] = val;
    __syncthreads(); // 等待所有 Warp 写完

    // 3. 让第一个 Warp 把 Shared Memory 里的结果再归约一次
    // 只有 Warp 0 的前 few 个线程需要干活 (假设 BlockDim 128 -> 4 Warps)
    val = (threadIdx.x < (blockDim.x / 32)) ? shared_mem[lane] : 0.0f;
    
    if (wid == 0) {
        val = warp_reduce_sum(val);
        // --- 修复开始 ---
        // 4. Warp 0 的线程 0 拿到了最终总和，必须把它写回 Shared Memory [0]
        if (lane == 0) shared_mem[0] = val;
    }
    
    // 5. 等待 Warp 0 写回
    __syncthreads();

    // 6. 所有线程从 Shared Memory [0] 读取最终结果
    return shared_mem[0];
    // --- 修复结束 ---
}

__global__ void simple_gqa_decoding_native_layout_kernel(
    const float* __restrict__ Q,       
    const float* __restrict__ K_cache, 
    const float* __restrict__ V_cache, 
    float* __restrict__ Output,        
    int* kv_seq_len_ptr,
    int head_dim,
    int num_kv_heads,                  
    int group_size,                    
    float sm_scale
) {
    extern __shared__ float smem[]; 
    int kv_seq_len = *kv_seq_len_ptr + 1;
    int q_head_idx = blockIdx.x; 
    int tid = threadIdx.x;
    
    int kv_head_idx = q_head_idx / group_size;

    if (tid >= head_dim) return;

    // 加载 Q
    float q_val = Q[q_head_idx * head_dim + tid];

    // Stride 计算
    int stride_kv_seq = num_kv_heads * head_dim;
    int current_head_offset = kv_head_idx * head_dim + tid;

    float m_prev = -INFINITY;
    float d_prev = 0.0f;
    float acc_output = 0.0f;

    // Time Loop
    for (int t = 0; t < kv_seq_len; ++t) {
        int curr_kv_idx = t * stride_kv_seq + current_head_offset;
        
        // QK^T
        float k_val = K_cache[curr_kv_idx];
        float score_part = q_val * k_val;
        
        // 这里的 block_reduce_sum 必须保证所有线程拿到一样的值
        float score = block_reduce_sum(score_part, smem);
        score *= sm_scale;

        // Online Softmax
        float m_curr = fmaxf(m_prev, score);
        float scale_prev = __expf(m_prev - m_curr);
        float scale_curr = __expf(score - m_curr);

        d_prev = d_prev * scale_prev + scale_curr;

        // Output Accumulate
        float v_val = V_cache[curr_kv_idx];
        // 注意：这里 acc_output 是向量，score/prob 是标量。
        // 因为 score 对所有 tid 都是一样的 (广播了)，所以逻辑正确。
        acc_output = acc_output * scale_prev + v_val * scale_curr;

        m_prev = m_curr;
    }

    // 归一化
    acc_output = acc_output / d_prev;
    Output[q_head_idx * head_dim + tid] = acc_output;
}

void flash_decoding_cu(
    const float* q_ptr,
    const float* k_ptr,
    const float* v_ptr,
    float* o_ptr,
    int32_t q_seq_len,   
    int32_t* kv_seq_len,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    cudaStream_t stream)
{
    // 显式忽略未使用的参数
    (void)q_seq_len;
    
    dim3 grid(num_q_heads); 
    
    // 确保线程数足够 (向上取整到 32 倍数)
    int threads = (head_dim + 31) / 32 * 32;
    dim3 block(threads);

    // Shared Memory 只用于存放 Warp 归约的中间值 (只需 threads/32 个 float)
    // 比如 128 线程 -> 4 个 float
    size_t smem_size = (threads / 32) * sizeof(float);

    float sm_scale = 1.0f / sqrtf((float)head_dim);
    int group_size = num_q_heads / num_kv_heads;

    // --- 修正调用参数 ---
    // 不要传 kv_seq_len + 1，除非你的 buffer 真的预留了。
    // 通常 kv_seq_len 代表当前 KCache 有效长度。传 +1 会读到未初始化的内存(NaN/INF/Zero)，导致乱码。
    simple_gqa_decoding_native_layout_kernel<<<grid, block, smem_size, stream>>>(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        kv_seq_len,
        head_dim,
        num_kv_heads,
        group_size,
        sm_scale
    );

    CUDA_CHECK(cudaGetLastError());
}