#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace {

// Correctness-first route planner. One thread produces deterministic stable
// expert-major ordering without atomics or host round-trips.
__global__ void build_route_plan_kernel(
    const int* __restrict__ expert_ids,
    const float* __restrict__ expert_weights,
    int* __restrict__ source_tokens,
    float* __restrict__ route_weights,
    int* __restrict__ expert_offsets,
    int tokens,
    int top_k,
    int experts)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int routes = tokens * top_k;
    int output_row = 0;
    expert_offsets[0] = 0;
    for (int expert = 0; expert < experts; ++expert) {
        for (int route = 0; route < routes; ++route) {
            if (expert_ids[route] == expert) {
                source_tokens[output_row] = route / top_k;
                route_weights[output_row] = expert_weights[route];
                ++output_row;
            }
        }
        expert_offsets[expert + 1] = output_row;
    }

    // Router output is contractually in range, so output_row == routes. Keep
    // malformed direct calls memory-safe without inventing an expert mapping.
    for (int route = output_row; route < routes; ++route) {
        source_tokens[route] = 0;
        route_weights[route] = 0.0f;
    }
}

__global__ void gather_token_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int* __restrict__ source_tokens,
    __nv_bfloat16* __restrict__ permuted_input,
    size_t elements,
    int hidden)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    const int output_row = static_cast<int>(index / hidden);
    const int column = static_cast<int>(index % hidden);
    const int source_token = source_tokens[output_row];
    permuted_input[index] = input[static_cast<size_t>(source_token) * hidden + column];
}

} // namespace

extern "C" void moe_permute_tokens_bf16(
    const __nv_bfloat16* input,
    const int* expert_ids,
    const float* expert_weights,
    __nv_bfloat16* permuted_input,
    int* source_tokens,
    float* route_weights,
    int* expert_offsets,
    int tokens,
    int hidden,
    int top_k,
    int experts,
    cudaStream_t stream)
{
    build_route_plan_kernel<<<1, 1, 0, stream>>>(
        expert_ids,
        expert_weights,
        source_tokens,
        route_weights,
        expert_offsets,
        tokens,
        top_k,
        experts);

    const size_t routes = static_cast<size_t>(tokens) * top_k;
    const size_t elements = routes * hidden;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    gather_token_rows_bf16_kernel<<<blocks, threads, 0, stream>>>(
        input,
        source_tokens,
        permuted_input,
        elements,
        hidden);
}
