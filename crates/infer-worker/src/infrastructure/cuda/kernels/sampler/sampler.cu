// In your .cu file for CUDA kernels

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>
#include "sampler.h"

// ------------------- F32 版本 -------------------
void argmax_cu_f32_ffi(
    const float* logits_ptr,
    int vocab_size,
    int* result_ptr_gpu, // << 接收 GPU 指针
    float *workspace,
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
    *d_result = max_elem_it - d_logits;

}


// ------------------- BF16 版本 (Two-phase parallel argmax) -------------------
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

constexpr int ARGMAX_THREADS = 256;
constexpr int ARGMAX_MAX_BLOCKS_PER_ROW = 170;
constexpr int ARGMAX_WORKSPACE_BUFS_PER_BATCH = 3;

// ------------------- type convert -------------------

template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<float>(float x) {
    return x;
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

// ------------------- argmax op -------------------

struct ArgMaxPairOp {
    __device__ __forceinline__
    cub::KeyValuePair<int, float> operator()(
        const cub::KeyValuePair<int, float>& a,
        const cub::KeyValuePair<int, float>& b
    ) const {
        if (a.value > b.value) return a;
        if (b.value > a.value) return b;

        if (a.key < 0) return b;
        if (b.key < 0) return a;
        return (a.key <= b.key) ? a : b;
    }
};

// Phase 1: each block reduces a chunk, writes partial result to workspace
__global__ void argmax_phase1_bf16(
    const __nv_bfloat16* __restrict__ input,
    int batch_size,
    int vocab_size,
    int num_blocks_per_row,
    __nv_bfloat16* __restrict__ workspace_bf16
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    int block_in_row = blockIdx.x;

    float max_val = -3.40282e38f;
    int max_idx = -1;

    const __nv_bfloat16* row_ptr =
        input + static_cast<size_t>(row) * vocab_size;

    int stride = num_blocks_per_row * blockDim.x;

    for (int i = gid; i < vocab_size; i += stride) {
        float val = to_float<__nv_bfloat16>(row_ptr[i]);
        if (val > max_val || (val == max_val && (max_idx < 0 || i < max_idx))) {
            max_val = val;
            max_idx = i;
        }
    }

    using BlockReduce = cub::BlockReduce<cub::KeyValuePair<int, float>, ARGMAX_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    cub::KeyValuePair<int, float> thread_data{max_idx, max_val};

    cub::KeyValuePair<int, float> block_result =
        BlockReduce(temp_storage).Reduce(thread_data, ArgMaxPairOp{});

    if (tid == 0) {

        auto* partial_vals = workspace_bf16 + row * 512;
        auto* partial_idxs = reinterpret_cast<int*>(partial_vals + 170);

        partial_vals[block_in_row] = __float2bfloat16(block_result.value);
        partial_idxs[block_in_row] = block_result.key;
    }
}

// Phase 2: single block per row reduces partial results
__global__ void argmax_phase2_bf16(
    int batch_size,
    int num_blocks_per_row,
    int* __restrict__ output_idx,
    const __nv_bfloat16* __restrict__ workspace_bf16
) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    float max_val = -3.40282e38f;
    int max_idx = -1;

    auto* partial_vals = workspace_bf16 + row * 512;
    const auto* partial_idxs = reinterpret_cast<const int*>(partial_vals + 170);


    for (int i = tid; i < num_blocks_per_row; i += blockDim.x) {
        float val = __bfloat162float(partial_vals[i]);
        int idx = partial_idxs[i];

        if (val > max_val || (val == max_val && idx >= 0 && (max_idx < 0 || idx < max_idx))) {
            max_val = val;
            max_idx = idx;
        }
    }

    using BlockReduce = cub::BlockReduce<cub::KeyValuePair<int, float>, ARGMAX_THREADS>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    cub::KeyValuePair<int, float> thread_data{max_idx, max_val};

    cub::KeyValuePair<int, float> block_result =
        BlockReduce(temp_storage).Reduce(thread_data, ArgMaxPairOp{});

    if (tid == 0) {
        output_idx[row] = block_result.key;
    }
}

void argmax_cu_bf16_ffi(
    const __nv_bfloat16* logits_ptr,
    int batch_size,
    int vocab_size,
    int* result_ptr_gpu,
    float* workspace,
    cudaStream_t stream
) {
    if (batch_size <= 0 || vocab_size <= 0) {
        return;
    }

    int num_blocks_per_row =
        (vocab_size + ARGMAX_THREADS * 4 - 1) / (ARGMAX_THREADS * 4);

    if (num_blocks_per_row > ARGMAX_MAX_BLOCKS_PER_ROW) {
        num_blocks_per_row = ARGMAX_MAX_BLOCKS_PER_ROW;
    }

    auto* workspace_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);

    dim3 block(ARGMAX_THREADS);
    dim3 grid1(num_blocks_per_row, batch_size);

    argmax_phase1_bf16<<<grid1, block, 0, stream>>>(
        logits_ptr,
        batch_size,
        vocab_size,
        num_blocks_per_row,
        workspace_bf16
    );

    dim3 grid2(batch_size);

    argmax_phase2_bf16<<<grid2, block, 0, stream>>>(
        batch_size,
        num_blocks_per_row,
        result_ptr_gpu,
        workspace_bf16
    );
}

extern "C" void argmax_cu_fp16_ffi(
    const __half* logits_ptr,
    int vocab_size,
    int* result_ptr_gpu,
    float *workspace,
    cudaStream_t stream
) {
    const int threads = 256;

}