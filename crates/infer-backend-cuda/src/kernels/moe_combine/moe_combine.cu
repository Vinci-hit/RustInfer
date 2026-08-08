#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace {

__global__ void accumulate_weighted_routes_bf16_kernel(
    const __nv_bfloat16* __restrict__ expert_output,
    const int* __restrict__ source_tokens,
    const float* __restrict__ route_weights,
    float* __restrict__ accumulator,
    size_t elements,
    int tokens,
    int hidden)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    const int route = static_cast<int>(index / hidden);
    const int column = static_cast<int>(index % hidden);
    const int token = source_tokens[route];
    if (token < 0 || token >= tokens) return;
    atomicAdd(
        accumulator + static_cast<size_t>(token) * hidden + column,
        route_weights[route] * __bfloat162float(expert_output[index]));
}

__global__ void write_combined_bf16_kernel(
    const float* __restrict__ accumulator,
    __nv_bfloat16* __restrict__ output,
    size_t elements)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = __float2bfloat16_rn(accumulator[index]);
}

} // namespace

extern "C" int moe_combine_bf16(
    const __nv_bfloat16* expert_output,
    const int* source_tokens,
    const float* route_weights,
    __nv_bfloat16* output,
    float* accumulator,
    int routes,
    int tokens,
    int hidden,
    cudaStream_t stream)
{
    const size_t output_elements = static_cast<size_t>(tokens) * hidden;
    cudaError_t status = cudaMemsetAsync(
        accumulator,
        0,
        output_elements * sizeof(float),
        stream);
    if (status != cudaSuccess) return static_cast<int>(status);

    constexpr int threads = 256;
    const size_t route_elements = static_cast<size_t>(routes) * hidden;
    const int route_blocks = static_cast<int>((route_elements + threads - 1) / threads);
    accumulate_weighted_routes_bf16_kernel<<<route_blocks, threads, 0, stream>>>(
        expert_output,
        source_tokens,
        route_weights,
        accumulator,
        route_elements,
        tokens,
        hidden);
    status = cudaGetLastError();
    if (status != cudaSuccess) return static_cast<int>(status);

    const int output_blocks = static_cast<int>((output_elements + threads - 1) / threads);
    write_combined_bf16_kernel<<<output_blocks, threads, 0, stream>>>(
        accumulator,
        output,
        output_elements);
    return static_cast<int>(cudaGetLastError());
}
