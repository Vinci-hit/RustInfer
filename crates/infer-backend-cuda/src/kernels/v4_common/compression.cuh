#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>
#include <climits>

namespace rustinfer::v4 {
constexpr int D = 512, RD = 64, Threads = 256;
template<int Ratio>
__device__ __forceinline__ bool valid_range(int start, int tokens, int capacity) {
    return start >= 0 && start <= INT_MAX - (tokens - 1)
        && (static_cast<long long>(start) + tokens) / Ratio <= capacity;
}
// Per-channel online softmax; shared by both compressor schedules.
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

// Pool -> BF16 -> FP32 RMSNorm -> BF16 -> FP32 RoPE -> BF16.
__device__ __forceinline__ void finish(Pool p0, Pool p1, const float* weight, const float* rope,
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

} // namespace rustinfer::v4
