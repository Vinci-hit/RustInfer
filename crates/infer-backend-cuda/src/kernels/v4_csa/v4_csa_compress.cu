#include "../v4_common/compression.cuh"

namespace {
using namespace rustinfer::v4;
constexpr int R = 4;

__device__ Pool load(const float* state, int group, int d) {
    const float* p = state + group * 3 * D + d;
    return {p[0], p[D], p[2 * D]};
}
__device__ void save(float* state, int group, int d, Pool p) {
    float* dst = state + group * 3 * D + d;
    dst[0] = p.m; dst[D] = p.z; dst[2 * D] = p.n;
}

// Half-block statistics use the same token order in prefill and decode.
// Before this call: group 0 = previous block's first half; groups 1/2 =
// the prefix of the current block's first/second half. State is read-only
// throughout the parallel output kernel, including its neighboring CTAs.
__device__ Pool aggregate(const float* values, const float* gates, const float* ape,
                          const float* state, int start, long long end,
                          int block, int half, int d) {
    if (block < 0) return Pool::empty();
    const int first_block = start / R;
    if (block < first_block) return load(state, 0, d); // only previous first half
    const int begin = block == first_block ? start % R : 0;
    Pool p = begin ? load(state, 1 + half, d) : Pool::empty();
    const long long base = static_cast<long long>(block) * R;
    const int count = static_cast<int>(min(static_cast<long long>(R), end - base));
    for (int j = begin; j < count; ++j) {
        const size_t src = static_cast<size_t>(base + j - start) * (2 * D) + half * D + d;
        p.add(values[src], gates[src] + ape[j * (2 * D) + half * D + d]);
    }
    return p;
}

__global__ void decode(const float* __restrict__ values, const float* __restrict__ gates,
                       const float* __restrict__ ape, const float* __restrict__ norm,
                       const float* __restrict__ rope, const int* __restrict__ start_ptr,
                       float* __restrict__ state, __nv_bfloat16* __restrict__ cache,
                       int capacity, float eps) {
    const int start = *start_ptr, t = threadIdx.x;
    if (!valid_range<R>(start, 1, capacity)) return;
    const int slot = start % R;
    Pool output[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int d = t + i * Threads;
        Pool first = slot ? load(state, 1, d) : Pool::empty();
        Pool second = slot ? load(state, 2, d) : Pool::empty();
        first.add(values[d], gates[d] + ape[slot * (2 * D) + d]);
        second.add(values[D + d], gates[D + d] + ape[slot * (2 * D) + D + d]);
        if (slot == R - 1) {
            const Pool prev = start / R ? load(state, 0, d) : Pool::empty();
            output[i] = prev.merge(second);
            save(state, 0, d, first);
            first = second = Pool::empty();
        } else if (start == 0) {
            save(state, 0, d, Pool::empty());
        }
        save(state, 1, d, first);
        save(state, 2, d, second);
    }
    if (slot == R - 1) finish(output[0], output[1], norm, rope, cache, start / R, eps);
}

__global__ void complete_blocks(
    const float* __restrict__ values, const float* __restrict__ gates,
    const float* __restrict__ ape, const float* __restrict__ norm,
    const float* __restrict__ rope, const int* __restrict__ start_ptr,
    const float* __restrict__ state, __nv_bfloat16* __restrict__ cache,
    int tokens, int capacity, float eps) {
    const int start = *start_ptr, t = threadIdx.x;
    if (!valid_range<R>(start, tokens, capacity)) return;
    const long long end = static_cast<long long>(start) + tokens;
    const int block = start / R + blockIdx.x;
    if (block >= end / R) return;
    Pool p[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int d = t + i * Threads;
        const Pool prev = aggregate(values, gates, ape, state, start, end, block - 1, 0, d);
        const Pool curr = aggregate(values, gates, ape, state, start, end, block, 1, d);
        p[i] = prev.merge(curr);
    }
    finish(p[0], p[1], norm, rope, cache, block, eps);
}

// Stream-ordered state commit after ALL prefix readers have finished. This
// recomputes at most eight tokens' statistics, without a scratch allocation.
__global__ void tail(const float* __restrict__ values, const float* __restrict__ gates,
                     const float* __restrict__ ape, const int* __restrict__ start_ptr,
                     float* __restrict__ state, int tokens, int capacity) {
    const int start = *start_ptr, t = threadIdx.x;
    if (!valid_range<R>(start, tokens, capacity)) return;
    const long long end = static_cast<long long>(start) + tokens;
    const int block = static_cast<int>(end / R);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int d = t + i * Threads;
        const Pool prev = aggregate(values, gates, ape, state, start, end, block - 1, 0, d);
        const Pool first = aggregate(values, gates, ape, state, start, end, block, 0, d);
        const Pool second = aggregate(values, gates, ape, state, start, end, block, 1, d);
        save(state, 0, d, prev);
        save(state, 1, d, first);
        save(state, 2, d, second);
    }
}
} // namespace

extern "C" int rustinfer_v4_csa_compress(
    const float* values, const float* gates, const float* ape, const float* norm,
    const float* rope, const int* start, float* state, __nv_bfloat16* cache,
    int tokens, int capacity, float eps, cudaStream_t stream) {
    if (tokens == 1) {
        decode<<<1, Threads, 0, stream>>>(values, gates, ape, norm, rope, start, state, cache, capacity, eps);
    } else {
        complete_blocks<<<tokens / R + 1, Threads, 0, stream>>>(
            values, gates, ape, norm, rope, start, state, cache, tokens, capacity, eps);
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) return static_cast<int>(status);
        tail<<<1, Threads, 0, stream>>>(values, gates, ape, start, state, tokens, capacity);
    }
    return static_cast<int>(cudaGetLastError());
}
