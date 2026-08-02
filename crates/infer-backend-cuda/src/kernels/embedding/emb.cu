#include "emb.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// --- CUDA Kernel (BF16版本) ---
__global__ void embedding_kernel_bf16x8(
    float4* output,
    const int* input_token_ids,
    const float4* weight,
    int dim_units, // 这里 dim 是 float4 的个数 (原始 dim / 8)
    int vocab_size
) {
    const int token_idx = blockIdx.x;
    float4* out_row = output + token_idx * dim_units;
    int32_t token = input_token_ids[token_idx];
    if (token < 0 || token >= vocab_size) {
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = threadIdx.x; i < dim_units; i += blockDim.x) {
            out_row[i] = zero;
        }
        return;
    }

    const float4* wei_row = weight + token * dim_units;

    for (int i = threadIdx.x; i < dim_units; i += blockDim.x) {
        out_row[i] = wei_row[i];
    }
}

void embedding_kernel_cu_bf16x8(
    __nv_bfloat16* output,
    const int* input_token_ids,
    const __nv_bfloat16* weight,
    int token_len,
    int dim,
    int vocab_size,
    cudaStream_t stream
) {
    int dim_units = dim / 8; // float4 的数量
    
    int threads_per_block = (dim_units > 1024) ? 1024 : dim_units;
    
    // Grid 大小等于 Token 数量
    dim3 grid(token_len);
    dim3 block(threads_per_block);

    auto out_f4 = reinterpret_cast<float4*>(output);
    auto weight_f4 = reinterpret_cast<const float4*>(weight);

    embedding_kernel_bf16x8<<<grid, block, 0, stream>>>(
        out_f4, input_token_ids, weight_f4, dim_units, vocab_size
    );
}

__global__ void embedding_kernel(
    float4* output,
    const int* input_token_ids,
    const float4* weight,
    int dim, int vocab_size
) {
    const int token_idx = blockIdx.x;
    float4* output_ptr_start = output + token_idx * dim;
    int32_t token = input_token_ids[token_idx];
    if (token < 0 || token >= vocab_size) {
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
            output_ptr_start[i] = zero;
        }
        return;
    }

    const float4* weight_ptr_start = weight + token * dim;

    for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        output_ptr_start[i] = weight_ptr_start[i];
    }
}

#include <cstdio>
void embedding_kernel_cu_fp32x4(
    float* output,
    const int* input_token_ids,
    const float* weight,
    int token_len,
    int dim,
    int vocab_size,
    cudaStream_t stream
) {
    
    constexpr int32_t thread_num = 128;
    
    // --- 类型转换 ---
    float4* out_f4 = reinterpret_cast<float4*>(output);
    const float4* weight_f4 = reinterpret_cast<const float4*>(weight);
    dim /= 4;
    // --- 启动内核 ---
    embedding_kernel<<<token_len, thread_num, 0, stream>>>(
        out_f4, input_token_ids, weight_f4, dim, vocab_size
    );
}




// ============= FP16 variants (auto-generated from BF16) =============

__global__ void embedding_kernel_fp16x8(
    float4* output,
    const int* input_token_ids,
    const float4* weight,
    int dim_units, // 这里 dim 是 float4 的个数 (原始 dim / 8)
    int vocab_size
) {
    const int token_idx = blockIdx.x;
    float4* out_row = output + token_idx * dim_units;
    int32_t token = input_token_ids[token_idx];
    if (token < 0 || token >= vocab_size) {
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = threadIdx.x; i < dim_units; i += blockDim.x) {
            out_row[i] = zero;
        }
        return;
    }

    const float4* wei_row = weight + token * dim_units;

    for (int i = threadIdx.x; i < dim_units; i += blockDim.x) {
        out_row[i] = wei_row[i];
    }
}

extern "C" void embedding_kernel_cu_fp16x8(
    __half* output,
    const int* input_token_ids,
    const __half* weight,
    int token_len,
    int dim,
    int vocab_size,
    cudaStream_t stream
) {
    int dim_units = dim / 8; // float4 的数量
    
    int threads_per_block = (dim_units > 1024) ? 1024 : dim_units;
    
    // Grid 大小等于 Token 数量
    dim3 grid(token_len);
    dim3 block(threads_per_block);

    auto out_f4 = reinterpret_cast<float4*>(output);
    auto weight_f4 = reinterpret_cast<const float4*>(weight);

    embedding_kernel_fp16x8<<<grid, block, 0, stream>>>(
        out_f4, input_token_ids, weight_f4, dim_units, vocab_size
    );
}

template <typename Vec>
__global__ void vocab_embedding_kernel(
    Vec* output,
    const int* input_token_ids,
    const Vec* weight,
    int dim_units,
    int vocab_start,
    int local_vocab_size
) {
    const int token_idx = blockIdx.x;
    Vec* out_row = output + token_idx * dim_units;
    const int64_t token = input_token_ids[token_idx];
    const int64_t start = vocab_start;
    const int64_t end = start + local_vocab_size;
    if (token < start || token >= end) {
        const Vec zero{};
        for (int i = threadIdx.x; i < dim_units; i += blockDim.x) {
            out_row[i] = zero;
        }
        return;
    }
    const int local_token = static_cast<int>(token - start);
    const Vec* weight_row = weight + local_token * dim_units;
    for (int i = threadIdx.x; i < dim_units; i += blockDim.x) {
        out_row[i] = weight_row[i];
    }
}

template <typename Scalar>
__global__ void vocab_embedding_scalar_kernel(
    Scalar* output,
    const int* input_token_ids,
    const Scalar* weight,
    int dim,
    int vocab_start,
    int local_vocab_size
) {
    const int token_idx = blockIdx.x;
    Scalar* out_row = output + token_idx * dim;
    const int64_t token = input_token_ids[token_idx];
    const int64_t start = vocab_start;
    const int64_t end = start + local_vocab_size;
    if (token < start || token >= end) {
        const Scalar zero{};
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            out_row[i] = zero;
        }
        return;
    }
    const int local_token = static_cast<int>(token - start);
    const Scalar* weight_row = weight + local_token * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = weight_row[i];
    }
}

template <typename Scalar, int ELEMENTS_PER_VEC>
void launch_vocab_embedding(
    Scalar* output,
    const int* input_token_ids,
    const Scalar* weight,
    int token_len,
    int dim,
    int vocab_start,
    int local_vocab_size,
    cudaStream_t stream
) {
    if (dim % ELEMENTS_PER_VEC == 0) {
        const int dim_units = dim / ELEMENTS_PER_VEC;
        const int threads = dim_units > 1024 ? 1024 : dim_units;
        vocab_embedding_kernel<<<token_len, threads, 0, stream>>>(
            reinterpret_cast<float4*>(output),
            input_token_ids,
            reinterpret_cast<const float4*>(weight),
            dim_units,
            vocab_start,
            local_vocab_size
        );
    } else {
        const int threads = dim > 1024 ? 1024 : dim;
        vocab_embedding_scalar_kernel<<<token_len, threads, 0, stream>>>(
            output,
            input_token_ids,
            weight,
            dim,
            vocab_start,
            local_vocab_size
        );
    }
}

extern "C" void vocab_embedding_kernel_cu_bf16x8(
    __nv_bfloat16* output,
    const int* input_token_ids,
    const __nv_bfloat16* weight,
    int token_len,
    int dim,
    int vocab_start,
    int local_vocab_size,
    cudaStream_t stream
) {
    launch_vocab_embedding<__nv_bfloat16, 8>(
        output, input_token_ids, weight, token_len, dim,
        vocab_start, local_vocab_size, stream
    );
}

extern "C" void vocab_embedding_kernel_cu_fp16x8(
    __half* output,
    const int* input_token_ids,
    const __half* weight,
    int token_len,
    int dim,
    int vocab_start,
    int local_vocab_size,
    cudaStream_t stream
) {
    launch_vocab_embedding<__half, 8>(
        output, input_token_ids, weight, token_len, dim,
        vocab_start, local_vocab_size, stream
    );
}

extern "C" void vocab_embedding_kernel_cu_fp32x4(
    float* output,
    const int* input_token_ids,
    const float* weight,
    int token_len,
    int dim,
    int vocab_start,
    int local_vocab_size,
    cudaStream_t stream
) {
    launch_vocab_embedding<float, 4>(
        output, input_token_ids, weight, token_len, dim,
        vocab_start, local_vocab_size, stream
    );
}
