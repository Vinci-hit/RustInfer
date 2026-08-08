#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace {

__global__ void grouped_expert_gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weights,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ expert_offsets,
    size_t output_elements,
    int experts,
    int out_features,
    int in_features)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;

    const int row = static_cast<int>(index / out_features);
    const int output_feature = static_cast<int>(index % out_features);
    int expert = -1;
    for (int candidate = 0; candidate < experts; ++candidate) {
        if (row >= expert_offsets[candidate] && row < expert_offsets[candidate + 1]) {
            expert = candidate;
            break;
        }
    }
    if (expert < 0) {
        output[index] = __float2bfloat16_rn(0.0f);
        return;
    }

    const size_t input_base = static_cast<size_t>(row) * in_features;
    const size_t weight_base =
        (static_cast<size_t>(expert) * out_features + output_feature) * in_features;
    float accumulator = 0.0f;
    for (int feature = 0; feature < in_features; ++feature) {
        accumulator += __bfloat162float(input[input_base + feature]) *
            __bfloat162float(weights[weight_base + feature]);
    }
    output[index] = __float2bfloat16_rn(accumulator);
}

} // namespace

extern "C" void grouped_expert_gemm_bf16(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weights,
    __nv_bfloat16* output,
    const int* expert_offsets,
    int rows,
    int experts,
    int out_features,
    int in_features,
    cudaStream_t stream)
{
    const size_t output_elements = static_cast<size_t>(rows) * out_features;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((output_elements + threads - 1) / threads);
    grouped_expert_gemm_bf16_kernel<<<blocks, threads, 0, stream>>>(
        input,
        weights,
        output,
        expert_offsets,
        output_elements,
        experts,
        out_features,
        in_features);
}
