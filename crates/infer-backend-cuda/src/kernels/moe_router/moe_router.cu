#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <climits>

namespace {

constexpr int kMaxTopK = 32;

__device__ __forceinline__ bool route_precedes(
    float score,
    int expert,
    float current_score,
    int current_expert)
{
    return score > current_score ||
        (score == current_score && expert < current_expert);
}

__device__ __forceinline__ float canonical_score(__nv_bfloat16 value)
{
    const float score = __bfloat162float(value);
    if (isfinite(score)) return score;
    return score > 0.0f ? FLT_MAX : -FLT_MAX;
}

// Correctness-first router: one CUDA thread owns one token row. The component
// boundary and output layout are stable; a parallel implementation can replace
// this kernel later without changing callers.
__global__ void moe_route_topk_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int* __restrict__ expert_ids,
    float* __restrict__ expert_weights,
    int rows,
    int experts,
    int top_k,
    bool renormalize)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float selected_scores[kMaxTopK];
    int selected_experts[kMaxTopK];
    #pragma unroll
    for (int route = 0; route < kMaxTopK; ++route) {
        selected_scores[route] = -FLT_MAX;
        selected_experts[route] = INT_MAX;
    }

    const __nv_bfloat16* row_logits = logits + static_cast<size_t>(row) * experts;
    float row_max = -FLT_MAX;
    for (int expert = 0; expert < experts; ++expert) {
        const float score = canonical_score(row_logits[expert]);
        row_max = fmaxf(row_max, score);

        int insert_at = top_k;
        for (int route = 0; route < top_k; ++route) {
            if (route_precedes(
                    score,
                    expert,
                    selected_scores[route],
                    selected_experts[route])) {
                insert_at = route;
                break;
            }
        }
        if (insert_at < top_k) {
            for (int route = top_k - 1; route > insert_at; --route) {
                selected_scores[route] = selected_scores[route - 1];
                selected_experts[route] = selected_experts[route - 1];
            }
            selected_scores[insert_at] = score;
            selected_experts[insert_at] = expert;
        }
    }

    float denominator = 0.0f;
    const float normalizer = renormalize ? selected_scores[0] : row_max;
    if (renormalize) {
        for (int route = 0; route < top_k; ++route) {
            denominator += expf(selected_scores[route] - normalizer);
        }
    } else {
        for (int expert = 0; expert < experts; ++expert) {
            const float score = canonical_score(row_logits[expert]);
            denominator += expf(score - normalizer);
        }
    }

    const size_t output_base = static_cast<size_t>(row) * top_k;
    for (int route = 0; route < top_k; ++route) {
        expert_ids[output_base + route] = selected_experts[route];
        expert_weights[output_base + route] =
            expf(selected_scores[route] - normalizer) / denominator;
    }
}

} // namespace

extern "C" void moe_route_topk_bf16(
    const __nv_bfloat16* logits,
    int* expert_ids,
    float* expert_weights,
    int rows,
    int experts,
    int top_k,
    int renormalize,
    cudaStream_t stream)
{
    constexpr int threads = 128;
    const int blocks = (rows + threads - 1) / threads;
    moe_route_topk_bf16_kernel<<<blocks, threads, 0, stream>>>(
        logits,
        expert_ids,
        expert_weights,
        rows,
        experts,
        top_k,
        renormalize != 0);
}
