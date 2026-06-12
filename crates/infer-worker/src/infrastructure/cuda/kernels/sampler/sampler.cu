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
    const int * selected_rows_device,
    int vocab_size,
    int num_blocks_per_row,
    __nv_bfloat16* __restrict__ workspace_bf16
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = selected_rows_device == nullptr ? blockIdx.y : selected_rows_device[blockIdx.y];
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

        auto* partial_vals = workspace_bf16 + blockIdx.y * 512;
        auto* partial_idxs = reinterpret_cast<int*>(partial_vals + 170);

        partial_vals[block_in_row] = __float2bfloat16(block_result.value);
        partial_idxs[block_in_row] = block_result.key;
    }
}

// Phase 2: single block per row reduces partial results
__global__ void argmax_phase2_bf16(
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
struct ArgMax {
    __nv_bfloat16 val;
    uint32_t idx;

    __device__ __forceinline__
    ArgMax()
        : val(__ushort_as_bfloat16((unsigned short)0xff80u)),
          idx(0xffffffffu) {}

    __device__ __forceinline__
    ArgMax(__nv_bfloat16 v, uint32_t i)
        : val(v), idx(i) {}
};

struct ArgMax2 {
    __nv_bfloat162 val;
    uint2 idx;

    __device__ __forceinline__
    ArgMax2()
        : val(__nv_bfloat162(
              __ushort_as_bfloat16((unsigned short)0xff80u),
              __ushort_as_bfloat16((unsigned short)0xff80u))),
          idx(make_uint2(0xffffffffu, 0xffffffffu)) {}

    __device__ __forceinline__
    ArgMax2(__nv_bfloat162 v, uint2 i)
        : val(v), idx(i) {}
};

union BF16Bits {
    __nv_bfloat16 bf16;
    uint16_t u16;
};

union BF162Bits {
    __nv_bfloat162 bf162;
    uint32_t u32;
};

__device__ __forceinline__
uint16_t bf16_as_u16(__nv_bfloat16 x) {
    BF16Bits v;
    v.bf16 = x;
    return v.u16;
}

__device__ __forceinline__
__nv_bfloat16 u16_as_bf16(uint16_t x) {
    BF16Bits v;
    v.u16 = x;
    return v.bf16;
}

__device__ __forceinline__
uint32_t bf162_as_u32(__nv_bfloat162 x) {
    BF162Bits v;
    v.bf162 = x;
    return v.u32;
}

__device__ __forceinline__
__nv_bfloat162 u32_as_bf162(uint32_t x) {
    BF162Bits v;
    v.u32 = x;
    return v.bf162;
}

__device__ __forceinline__
void argmax_update_hge(
    ArgMax& best,
    const ArgMax& cand
) {
    uint32_t mask = __hge(cand.val, best.val) ? 0xffffffffu : 0u;

    uint32_t best_bits = bf16_as_u16(best.val);
    uint32_t cand_bits = bf16_as_u16(cand.val);

    uint32_t new_bits = (cand_bits & mask) | (best_bits & ~mask);
    best.val = u16_as_bf16(static_cast<uint16_t>(new_bits));

    best.idx = mask ? cand.idx : best.idx;
}

__device__ __forceinline__
void argmax2_update_hge(
    ArgMax2& best,
    const __nv_bfloat162 cand_val,
    const uint2 cand_idx
) {
    uint32_t mask = __hge2_mask(cand_val, best.val);

    uint32_t best_bits = bf162_as_u32(best.val);
    uint32_t cand_bits = bf162_as_u32(cand_val);

    uint32_t new_bits = (cand_bits & mask) | (best_bits & ~mask);
    best.val = u32_as_bf162(new_bits);

    uint32_t lo_update = mask & 0x0000ffffu;
    uint32_t hi_update = mask & 0xffff0000u;

    best.idx.x = lo_update ? cand_idx.x : best.idx.x;
    best.idx.y = hi_update ? cand_idx.y : best.idx.y;
}

__device__ __forceinline__
void argmax2_update_hge(
    ArgMax2& best,
    const ArgMax2& cand
) {
    argmax2_update_hge(best, cand.val, cand.idx);
}

__device__ __forceinline__
ArgMax argmax2_to_argmax_hge(const ArgMax2& v) {
    uint32_t bits = bf162_as_u32(v.val);

    __nv_bfloat16 lo = u16_as_bf16(static_cast<uint16_t>(bits & 0x0000ffffu));
    __nv_bfloat16 hi = u16_as_bf16(static_cast<uint16_t>((bits >> 16) & 0x0000ffffu));

    ArgMax ret(lo, v.idx.x);
    ArgMax cand(hi, v.idx.y);
    argmax_update_hge(ret, cand);
    return ret;
}

__device__ __forceinline__
ArgMax warp_argmax_reduce_hge(ArgMax v) {
    constexpr uint32_t FULL_MASK = 0xffffffffu;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        ArgMax other;

        uint32_t val_bits = bf16_as_u16(v.val);
        uint32_t other_bits = __shfl_down_sync(FULL_MASK, val_bits, offset);
        other.val = u16_as_bf16(static_cast<uint16_t>(other_bits));

        other.idx = __shfl_down_sync(FULL_MASK, v.idx, offset);

        argmax_update_hge(v, other);
    }

    return v;
}

__device__ __forceinline__
ArgMax2 warp_argmax2_reduce_hge(ArgMax2 v) {
    constexpr uint32_t FULL_MASK = 0xffffffffu;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        ArgMax2 other;

        uint32_t val_bits = bf162_as_u32(v.val);
        uint32_t other_bits = __shfl_down_sync(FULL_MASK, val_bits, offset);
        other.val = u32_as_bf162(other_bits);

        other.idx.x = __shfl_down_sync(FULL_MASK, v.idx.x, offset);
        other.idx.y = __shfl_down_sync(FULL_MASK, v.idx.y, offset);

        argmax2_update_hge(v, other);
    }

    return v;
}
#define One_thread_process_count 4

__global__ void argmax_phase1_bf162(
    const __nv_bfloat16* __restrict__ logits_ptr,
    const int* __restrict__ selected_rows_device,
    int vocab_size,
    int num_blocks_per_row,
    __nv_bfloat16* __restrict__ workspace_bf16
) {
    int tid = threadIdx.x;
    int row = blockIdx.y;
    int block_per_row = blockIdx.x;

    int src_row = selected_rows_device ? selected_rows_device[row] : row;
    const __nv_bfloat16* row_logits = logits_ptr + src_row * vocab_size;

    ArgMax2 part_max;

#pragma unroll
    for (int i = 0; i < One_thread_process_count; ++i) {
        int target_idx =
            block_per_row * ARGMAX_THREADS * One_thread_process_count * 2
            + i * ARGMAX_THREADS * 2
            + tid * 2;

        if (target_idx + 1 < vocab_size) {
            const __nv_bfloat162* row_logits2 =
                reinterpret_cast<const __nv_bfloat162*>(row_logits);

            __nv_bfloat162 cand_val = row_logits2[target_idx >> 1];
            uint2 cand_idx = make_uint2(
                static_cast<uint32_t>(target_idx),
                static_cast<uint32_t>(target_idx + 1));

            argmax2_update_hge(part_max, cand_val, cand_idx);
        } else if (target_idx < vocab_size) {
            ArgMax tmp = argmax2_to_argmax_hge(part_max);
            ArgMax cand(row_logits[target_idx], static_cast<uint32_t>(target_idx));
            argmax_update_hge(tmp, cand);

            part_max = ArgMax2(
                __nv_bfloat162(tmp.val, __ushort_as_bfloat16((unsigned short)0xff80u)),
                make_uint2(tmp.idx, 0xffffffffu));
        }
    }

    __shared__ ArgMax2 smem[ARGMAX_THREADS];

    smem[tid] = part_max;
    __syncthreads();

#pragma unroll
    for (int stride = ARGMAX_THREADS >> 1; stride > 32; stride >>= 1) {
        if (tid < stride) {
            argmax2_update_hge(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        ArgMax2 v = smem[tid];

        if (ARGMAX_THREADS >= 64) {
            argmax2_update_hge(v, smem[tid + 32]);
        }

        v = warp_argmax2_reduce_hge(v);

        if (tid == 0) {
            ArgMax out = argmax2_to_argmax_hge(v);

            __nv_bfloat16* partial_vals = workspace_bf16 + row * 512;
            int* partial_idxs = reinterpret_cast<int*>(partial_vals + 170);

            partial_vals[block_per_row] = out.val;
            partial_idxs[block_per_row] = static_cast<int>(out.idx);
        }
    }
}
__global__ void argmax_phase2_bf16_hge(
    int num_blocks_per_row,
    int* __restrict__ output_idx,
    const __nv_bfloat16* __restrict__ workspace_bf16
) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const __nv_bfloat16* partial_vals = workspace_bf16 + row * 512;
    const int* partial_idxs = reinterpret_cast<const int*>(partial_vals + 170);

    ArgMax thread_max;

    for (int i = tid; i < num_blocks_per_row; i += blockDim.x) {
        int idx = partial_idxs[i];

        if (idx >= 0) {
            ArgMax cand(partial_vals[i], static_cast<uint32_t>(idx));
            argmax_update_hge(thread_max, cand);
        }
    }

    __shared__ ArgMax smem[ARGMAX_THREADS];

    smem[tid] = thread_max;
    __syncthreads();

#pragma unroll
    for (int stride = ARGMAX_THREADS >> 1; stride > 32; stride >>= 1) {
        if (tid < stride) {
            argmax_update_hge(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        ArgMax v = smem[tid];

        if (ARGMAX_THREADS >= 64) {
            argmax_update_hge(v, smem[tid + 32]);
        }

        v = warp_argmax_reduce_hge(v);

        if (tid == 0) {
            output_idx[row] = static_cast<int>(v.idx);
        }
    }
}


void argmax_cu_bf16_ffi(
    const __nv_bfloat16* logits_ptr,
    const int* selected_rows_device,
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
        (vocab_size + ARGMAX_THREADS * One_thread_process_count * 2 - 1)
        / (ARGMAX_THREADS * One_thread_process_count * 2);

    if (num_blocks_per_row > ARGMAX_MAX_BLOCKS_PER_ROW) {
        num_blocks_per_row = ARGMAX_MAX_BLOCKS_PER_ROW;
    }

    auto* workspace_bf16 = reinterpret_cast<__nv_bfloat16*>(workspace);

    dim3 block(ARGMAX_THREADS);
    dim3 grid1(num_blocks_per_row, batch_size);

    argmax_phase1_bf162<<<grid1, block, 0, stream>>>(
        logits_ptr,
        selected_rows_device,
        vocab_size,
        num_blocks_per_row,
        workspace_bf16
    );

    dim3 grid2(batch_size);

    argmax_phase2_bf16_hge<<<grid2, block, 0, stream>>>(
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