#include "../v4_common/compression.cuh"

namespace {
using namespace rustinfer::v4;
constexpr int R = 128;

// Per-channel online softmax: maximum, denominator and weighted numerator.
// Only 3*512 floats persist, instead of two 128*512 projection buffers.

__device__ Pool initial(const float* state, int d, bool resume) {
    return resume ? Pool{state[d], state[D + d], state[2 * D + d]}
                  : Pool{-CUDART_INF_F, 0.0f, 0.0f};
}

__device__ void save(float* state, int d, Pool p, bool reset) {
    state[d] = reset ? -CUDART_INF_F : p.m;
    state[D + d] = reset ? 0.0f : p.z;
    state[2 * D + d] = reset ? 0.0f : p.n;
}

__global__ void decode(const float* __restrict__ values, const float* __restrict__ gates,
                       const float* __restrict__ ape, const float* __restrict__ norm,
                       const float* __restrict__ rope, const int* __restrict__ start_ptr,
                       float* __restrict__ state, __nv_bfloat16* __restrict__ cache,
                       int capacity, float eps) {
    const int start = *start_ptr, t = threadIdx.x;
    if (!valid_range<R>(start, 1, capacity)) return;
    const int slot = start % R;
    Pool p[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int d = t + i * Threads;
        p[i] = initial(state, d, slot != 0);
        p[i].add(values[d], gates[d] + ape[slot * D + d]);
        save(state, d, p[i], slot == R - 1);
    }
    if (slot == R - 1) finish(p[0], p[1], norm, rope, cache, start / R, eps);
}

// Every completed block is independent except the first, which may resume a
// prefix. State remains read-only until a second, stream-ordered kernel runs.
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
    const int begin = blockIdx.x == 0 ? start % R : 0;
    const size_t first = static_cast<long long>(block) * R + begin - start;
    Pool p[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) p[i] = initial(state, t + i * Threads, begin != 0);
    for (int j = begin; j < R; ++j) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int d = t + i * Threads;
            const size_t src = (first + j - begin) * D + d;
            p[i].add(values[src], gates[src] + ape[j * D + d]);
        }
    }
    finish(p[0], p[1], norm, rope, cache, block, eps);
}

__global__ void tail(const float* __restrict__ values, const float* __restrict__ gates,
                     const float* __restrict__ ape, const int* __restrict__ start_ptr,
                     float* __restrict__ state, int tokens, int capacity) {
    const int start = *start_ptr, t = threadIdx.x;
    if (!valid_range<R>(start, tokens, capacity)) return;
    const long long end = static_cast<long long>(start) + tokens;
    const int count = end % R;
    const bool resume = start / R == end / R && start % R != 0;
    const int begin = resume ? start % R : 0;
    const size_t first = resume ? 0 : tokens - count;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int d = t + i * Threads;
        Pool p = initial(state, d, resume);
        for (int j = begin; j < count; ++j) {
            const size_t src = (first + j - begin) * D + d;
            p.add(values[src], gates[src] + ape[j * D + d]);
        }
        save(state, d, p, count == 0);
    }
}
} // namespace

extern "C" int rustinfer_v4_hca_compress(
    const float* values, const float* gates, const float* ape, const float* norm,
    const float* rope, const int* start, float* state, __nv_bfloat16* cache,
    int tokens, int capacity, float eps, cudaStream_t stream) {
    if (tokens == 1) {
        decode<<<1, Threads, 0, stream>>>(values, gates, ape, norm, rope, start, state, cache, capacity, eps);
    } else {
        complete_blocks<<<tokens / R + 1, Threads, 0, stream>>>(
            values, gates, ape, norm, rope, start, state, cache, tokens, capacity, eps);
        auto status = cudaGetLastError();
        if (status != cudaSuccess) return static_cast<int>(status);
        tail<<<1, Threads, 0, stream>>>(values, gates, ape, start, state, tokens, capacity);
    }
    return static_cast<int>(cudaGetLastError());
}
