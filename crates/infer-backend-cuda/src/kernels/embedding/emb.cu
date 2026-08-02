#include "emb.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

template <typename Unit>
__global__ void embedding_lookup_kernel(
    Unit* output,
    const int* input_token_ids,
    const Unit* weight,
    int row_units,
    int vocab_start,
    int local_vocab_size
) {
    const int token_idx = blockIdx.x;
    const size_t row_stride = static_cast<size_t>(row_units);
    Unit* output_row = output + static_cast<size_t>(token_idx) * row_stride;

    const int64_t token = input_token_ids[token_idx];
    const int64_t start = vocab_start;
    const int64_t end = start + static_cast<int64_t>(local_vocab_size);
    if (token < start || token >= end) {
        const Unit zero{};
        for (int unit = threadIdx.x; unit < row_units; unit += blockDim.x) {
            output_row[unit] = zero;
        }
        return;
    }

    const size_t local_token = static_cast<size_t>(token - start);
    const Unit* weight_row = weight + local_token * row_stride;
    for (int unit = threadIdx.x; unit < row_units; unit += blockDim.x) {
        output_row[unit] = weight_row[unit];
    }
}

template <typename Scalar, int ELEMENTS_PER_VECTOR>
void launch_embedding_lookup(
    Scalar* output,
    const int* input_token_ids,
    const Scalar* weight,
    int token_len,
    int dim,
    int vocab_start,
    int local_vocab_size,
    cudaStream_t stream
) {
    if (token_len <= 0 || dim <= 0) {
        return;
    }

    if (dim % ELEMENTS_PER_VECTOR == 0) {
        const int row_units = dim / ELEMENTS_PER_VECTOR;
        const int threads = row_units > 1024 ? 1024 : row_units;
        embedding_lookup_kernel<<<token_len, threads, 0, stream>>>(
            reinterpret_cast<float4*>(output),
            input_token_ids,
            reinterpret_cast<const float4*>(weight),
            row_units,
            vocab_start,
            local_vocab_size
        );
    } else {
        const int threads = dim > 1024 ? 1024 : dim;
        embedding_lookup_kernel<<<token_len, threads, 0, stream>>>(
            output,
            input_token_ids,
            weight,
            dim,
            vocab_start,
            local_vocab_size
        );
    }
}

extern "C" void embedding_kernel_cu_bf16(
    __nv_bfloat16* output,
    const int* input_token_ids,
    const __nv_bfloat16* weight,
    int token_len,
    int dim,
    int vocab_start,
    int local_vocab_size,
    cudaStream_t stream
) {
    launch_embedding_lookup<__nv_bfloat16, 8>(
        output, input_token_ids, weight, token_len, dim,
        vocab_start, local_vocab_size, stream
    );
}

extern "C" void embedding_kernel_cu_fp16(
    __half* output,
    const int* input_token_ids,
    const __half* weight,
    int token_len,
    int dim,
    int vocab_start,
    int local_vocab_size,
    cudaStream_t stream
) {
    launch_embedding_lookup<__half, 8>(
        output, input_token_ids, weight, token_len, dim,
        vocab_start, local_vocab_size, stream
    );
}

extern "C" void embedding_kernel_cu_fp32(
    float* output,
    const int* input_token_ids,
    const float* weight,
    int token_len,
    int dim,
    int vocab_start,
    int local_vocab_size,
    cudaStream_t stream
) {
    launch_embedding_lookup<float, 4>(
        output, input_token_ids, weight, token_len, dim,
        vocab_start, local_vocab_size, stream
    );
}
