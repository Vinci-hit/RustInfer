// In your .cu file for CUDA kernels

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>
#include "sampler.h"

// ------------------- F32 版本 -------------------
void argmax_cu_f32_ffi(
    const float* logits_ptr,
    int vocab_size,
    int* result_ptr_gpu, // << 接收 GPU 指针
    cudaStream_t stream
) {
    // 使用 stream 创建 Thrust 执行策略
    auto policy = thrust::cuda::par.on(stream);

    // 将裸指针包装成 Thrust 的 device_ptr
    thrust::device_ptr<const float> d_logits(logits_ptr);
    thrust::device_ptr<int> d_result(result_ptr_gpu);

    // 使用 Thrust 找到最大元素的迭代器
    auto max_elem_it = thrust::max_element(policy, d_logits, d_logits + vocab_size);
    
    // 计算索引并将其写入到 GPU 上的结果指针
    *d_result = thrust::distance(d_logits, max_elem_it);

}


// ------------------- BF16 版本 -------------------
#include <cuda_bf16.h>
#include <cub/cub.cuh> // 仅使用 BlockReduce，这是 header-only 的

// 1. 定义 Kernel
__global__ void argmax_kernel_bf16(
    const __nv_bfloat16* __restrict__ input, 
    int n, 
    int* __restrict__ output_idx
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程维护当前的局部最大值和索引
    // 初始化为极小值
    float max_val = -3.40282e38f; // -FLT_MAX
    int max_idx = -1;

    // Grid-Stride Loop: 处理数据量 > 线程数的情况
    for (int i = gid; i < n; i += gridDim.x * blockDim.x) {
        float val = __bfloat162float(input[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    // Block 内归约 (使用 CUB BlockReduce)
    // 定义 Pair 类型: <数值, 索引>。注意 CUB ArgMax 默认找 Key 最大，我们这里需要自定义
    // 为了简单，我们手动做 Block Reduce
    
    // 共享内存
    using BlockReduce = cub::BlockReduce<cub::KeyValuePair<int, float>, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // 创建当前线程的 KV 对 (注意：CUB ArgMax 比较器有点绕，我们直接用 reduce 找最大 float)
    // 更简单的办法：把 (val, idx) 作为一个对象，自定义 max 操作
    
    cub::KeyValuePair<int, float> thread_data;
    thread_data.key = max_idx; // 索引
    thread_data.value = max_val; // 数值

    // 归约操作符：返回 Value 更大的那个；如果 Value 相等，返回 Key (Index) 更小的
    auto argmax_op = [](const cub::KeyValuePair<int, float>& a, const cub::KeyValuePair<int, float>& b) {
        if (a.value != b.value) return (a.value > b.value) ? a : b;
        return (a.key < b.key) ? a : b; // 索引越小越优先
    };

    // 执行 Block Reduce
    // 结果在 threadIdx.x == 0 的线程中
    cub::KeyValuePair<int, float> block_result = BlockReduce(temp_storage).Reduce(thread_data, argmax_op);

    // 2. Block 间归约 (由于 ArgMax 很难用 atomicMax 实现，
    // 对于 LLM 的 vocab_size (比如 32000/128000)，通常一个 Block (256/1024线程) 就能处理完一行的 ArgMax)
    // 所以这里我们可以只启动 **1个 Block** 即可！
    
    if (tid == 0) {
        *output_idx = block_result.key;
    }
}

void argmax_cu_bf16_ffi(
    const __nv_bfloat16* logits_ptr, 
    int vocab_size,
    int* result_ptr_gpu, 
    cudaStream_t stream
) {
    // 针对 LLM Vocab Size (e.g. 32k ~ 128k) 的优化配置
    // 一个 Block 256 线程，每个线程处理约 128~500 个元素，效率很高，无需多 Block 级联
    const int threads = 256;
    const int blocks = 1; 

    argmax_kernel_bf16<<<blocks, threads, 0, stream>>>(logits_ptr, vocab_size, result_ptr_gpu);

}

// ------------------- Batch BF16 版本 -------------------
// 对 batch 中每一行独立做 argmax
extern "C"
void argmax_batch_cu_bf16_ffi(
    const __nv_bfloat16* logits_ptr, // [batch_size, vocab_size]
    int batch_size,
    int vocab_size,
    int* output_ptr_gpu,             // [batch_size]
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = 1;

    for (int i = 0; i < batch_size; i++) {
        argmax_kernel_bf16<<<blocks, threads, 0, stream>>>(
            logits_ptr + i * vocab_size,
            vocab_size,
            output_ptr_gpu + i
        );
    }
}