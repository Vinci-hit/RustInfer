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


// ------------------- BF16 版本 (Two-phase parallel argmax) -------------------
#include <cuda_bf16.h>
#include <cub/cub.cuh>

// Static device arrays for inter-block communication
__device__ float d_partial_vals[1024];
__device__ int   d_partial_idxs[1024];

// Phase 1: Each block reduces a chunk of the input, writes (max_val, max_idx) to d_partial_*
__global__ void argmax_phase1_bf16(
    const __nv_bfloat16* __restrict__ input,
    int n
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float max_val = -3.40282e38f;
    int max_idx = -1;

    for (int i = gid; i < n; i += stride) {
        float val = __bfloat162float(input[i]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    using BlockReduce = cub::BlockReduce<cub::KeyValuePair<int, float>, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    cub::KeyValuePair<int, float> thread_data{max_idx, max_val};
    auto argmax_op = [](const cub::KeyValuePair<int, float>& a, const cub::KeyValuePair<int, float>& b) {
        return (a.value >= b.value) ? a : b;
    };

    auto block_result = BlockReduce(temp_storage).Reduce(thread_data, argmax_op);

    if (tid == 0) {
        d_partial_vals[blockIdx.x] = block_result.value;
        d_partial_idxs[blockIdx.x] = block_result.key;
    }
}

// Phase 2: Single block reduces partial results → final answer
__global__ void argmax_phase2(
    int num_blocks,
    int* __restrict__ output_idx
) {
    int tid = threadIdx.x;

    float max_val = -3.40282e38f;
    int max_idx = -1;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float val = d_partial_vals[i];
        if (val > max_val) {
            max_val = val;
            max_idx = d_partial_idxs[i];
        }
    }

    using BlockReduce = cub::BlockReduce<cub::KeyValuePair<int, float>, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    cub::KeyValuePair<int, float> thread_data{max_idx, max_val};
    auto argmax_op = [](const cub::KeyValuePair<int, float>& a, const cub::KeyValuePair<int, float>& b) {
        return (a.value >= b.value) ? a : b;
    };

    auto block_result = BlockReduce(temp_storage).Reduce(thread_data, argmax_op);

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
    const int threads = 256;
    // Each thread processes ~4 elements for good occupancy
    int num_blocks = (vocab_size + threads * 4 - 1) / (threads * 4);
    if (num_blocks > 1024) num_blocks = 1024;
    if (num_blocks < 1) num_blocks = 1;

    argmax_phase1_bf16<<<num_blocks, threads, 0, stream>>>(logits_ptr, vocab_size);
    argmax_phase2<<<1, threads, 0, stream>>>(num_blocks, result_ptr_gpu);
}