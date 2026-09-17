#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>
#include <climits>

namespace {
constexpr int D = 512, R = 4, RD = 64, Threads = 256;

__device__ bool valid(int start, int tokens, int capacity) {
    return start >= 0 && start <= INT_MAX - (tokens - 1)
        && (static_cast<long long>(start) + tokens) / R <= capacity;
}

struct Pool {
    float m, z, n;
    __device__ static Pool empty() { return {-CUDART_INF_F, 0.0f, 0.0f}; }
    __device__ void add(float value, float score) {
        const float e = expf(-fabsf(score - m));
        const float a = score > m ? e : 1.0f;
        const float b = score > m ? 1.0f : e;
        m = fmaxf(m, score);
        z = fmaf(z, a, b);
        n = fmaf(n, a, value * b);
    }
    __device__ Pool merge(Pool other) const {
        if (z == 0.0f) return other;
        const float e = expf(-fabsf(m - other.m));
        const float a = other.m > m ? e : 1.0f;
        const float b = other.m > m ? 1.0f : e;
        return {fmaxf(m, other.m), fmaf(z, a, other.z * b),
                fmaf(n, a, other.n * b)};
    }
};

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

// Preserve the unquantized BF16 reference's pool/norm/RoPE rounding boundaries.
__device__ void finish(Pool p0, Pool p1, const float* weight, const float* rope,
                       __nv_bfloat16* cache, int block, float eps) {
    const int t = threadIdx.x;
    __shared__ float v[D];
    using Reduce = cub::BlockReduce<float, Threads>;
    __shared__ typename Reduce::TempStorage reduction;
    __shared__ float inv;
    const float a = __bfloat162float(__float2bfloat16_rn(p0.n / p0.z));
    const float b = __bfloat162float(__float2bfloat16_rn(p1.n / p1.z));
    const float sum = Reduce(reduction).Sum(a * a + b * b);
    if (t == 0) inv = rsqrtf(sum / D + eps);
    __syncthreads();
    v[t] = __bfloat162float(__float2bfloat16_rn((a * inv) * weight[t]));
    v[t + Threads] = __bfloat162float(__float2bfloat16_rn((b * inv) * weight[t + Threads]));
    __syncthreads();
    const int d = t * 2;
    float x = v[d], y = v[d + 1];
    if (d >= D - RD) {
        const size_t i = static_cast<size_t>(block) * RD + d - (D - RD);
        const float c = rope[i], s = rope[i + 1];
        const float xc = x * c, ys = y * s, xs = x * s, yc = y * c;
        x = __fsub_rn(xc, ys);
        y = __fadd_rn(xs, yc);
    }
    reinterpret_cast<__nv_bfloat162*>(cache + static_cast<size_t>(block) * D)[t]
        = __floats2bfloat162_rn(x, y);
}

__global__ void decode(const float* __restrict__ values, const float* __restrict__ gates,
                       const float* __restrict__ ape, const float* __restrict__ norm,
                       const float* __restrict__ rope, const int* __restrict__ start_ptr,
                       float* __restrict__ state, __nv_bfloat16* __restrict__ cache,
                       int capacity, float eps) {
    const int start = *start_ptr, t = threadIdx.x;
    if (!valid(start, 1, capacity)) return;
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
    if (!valid(start, tokens, capacity)) return;
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
    if (!valid(start, tokens, capacity)) return;
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
