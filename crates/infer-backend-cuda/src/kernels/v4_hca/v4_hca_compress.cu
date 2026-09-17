#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>
#include <climits>

namespace {
constexpr int D = 512, R = 128, RD = 64, Threads = 256;

__device__ bool valid(int start, int tokens, int capacity) {
    return start >= 0 && start <= INT_MAX - (tokens - 1)
        && (static_cast<long long>(start) + tokens) / R <= capacity;
}

// Per-channel online softmax: maximum, denominator and weighted numerator.
// Only 3*512 floats persist, instead of two 128*512 projection buffers.
struct Pool {
    float m, z, n;
    __device__ void add(float value, float score) {
        const float e = expf(-fabsf(score - m));
        const float a = score > m ? e : 1.0f;
        const float b = score > m ? 1.0f : e;
        m = fmaxf(m, score);
        z = fmaf(z, a, b);
        n = fmaf(n, a, value * b);
    }
};

__device__ Pool initial(const float* state, int d, bool resume) {
    return resume ? Pool{state[d], state[D + d], state[2 * D + d]}
                  : Pool{-CUDART_INF_F, 0.0f, 0.0f};
}

__device__ void save(float* state, int d, Pool p, bool reset) {
    state[d] = reset ? -CUDART_INF_F : p.m;
    state[D + d] = reset ? 0.0f : p.z;
    state[2 * D + d] = reset ? 0.0f : p.n;
}

// Match the BF16 reference boundaries: pool -> BF16 -> FP32 RMSNorm ->
// BF16 -> FP32 interleaved partial RoPE -> BF16. RoPE table row b already
// encodes absolute position b*128, including the caller's YaRN settings.
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
        // Explicit products preserve the reference's separate FP32 operations.
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
    if (!valid(start, tokens, capacity)) return;
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
    if (!valid(start, tokens, capacity)) return;
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
