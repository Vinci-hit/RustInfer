#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <math_constants.h>

namespace {
constexpr int D = 512, W = 128, Threads = 256, Warps = 8;
constexpr float Scale = 0.04419417382415922f;
struct Maximum {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

__device__ const __nv_bfloat16* row_at(int j, int local, int slot,
                                      const __nv_bfloat16* kv,
                                      const __nv_bfloat16* cache,
                                      const __nv_bfloat16* compressed) {
    if (j >= local) return compressed + static_cast<size_t>(j - local) * D;
    return j == slot ? kv : cache + j * D;
}

// One CTA per head; stream local and compressed KV through the SAME online
// softmax. No scores/workspace proportional to context length is materialized.
__global__ void decode(const __nv_bfloat16* __restrict__ query,
                       const __nv_bfloat16* __restrict__ kv,
                       const float* __restrict__ sink, const int* __restrict__ position,
                       const __nv_bfloat16* __restrict__ compressed,
                       __nv_bfloat16* __restrict__ cache,
                       __nv_bfloat16* __restrict__ output, int capacity) {
    const int t = threadIdx.x, lane = t & 31, warp = t / 32, head = blockIdx.x;
    const int pos = *position;
    const int blocks = (static_cast<long long>(pos) + 1) / W;
    if (pos < 0 || blocks > capacity) {
        for (int d = t; d < D; d += Threads)
            output[head * D + d] = __float2bfloat16_rn(CUDART_NAN_F);
        return;
    }
    const int slot = pos % W, local = pos < W ? pos + 1 : W, count = local + blocks;
    float2 q[8], acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        q[i] = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(query + head * D)[lane + 32 * i]);
        acc[i] = make_float2(0.0f, 0.0f);
    }
    __shared__ float probabilities[W], partial[Warps][D];
    __shared__ float maximum, total, factor;
    using Reduce = cub::BlockReduce<float, Threads>;
    __shared__ typename Reduce::TempStorage reduction;
    if (t == 0) { maximum = sink[head]; total = 1.0f; }
    __syncthreads();
    for (int base = 0; base < count; base += W) {
        const int size = min(W, count - base);
        for (int j = warp; j < size; j += Warps) {
            const auto* row = row_at(base + j, local, slot, kv, cache, compressed);
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float2 k = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(row)[lane + 32 * i]);
                dot = fmaf(q[i].x, k.x, dot);
                dot = fmaf(q[i].y, k.y, dot);
            }
            #pragma unroll
            for (int delta = 16; delta; delta >>= 1) dot += __shfl_down_sync(0xffffffff, dot, delta);
            if (lane == 0) probabilities[j] = dot * Scale;
        }
        __syncthreads();
        float m = t < size ? probabilities[t] : -CUDART_INF_F;
        if (t == 0) m = fmaxf(m, maximum);
        m = Reduce(reduction).Reduce(m, Maximum());
        if (t == 0) { factor = expf(maximum - m); maximum = m; }
        __syncthreads();
        float p = t < size ? expf(probabilities[t] - maximum) : 0.0f;
        if (t < W) probabilities[t] = p;
        const float z = Reduce(reduction).Sum(p);
        if (t == 0) total = total * factor + z;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 8; ++i) { acc[i].x *= factor; acc[i].y *= factor; }
        for (int j = warp; j < size; j += Warps) {
            const auto* row = row_at(base + j, local, slot, kv, cache, compressed);
            const float p = probabilities[j];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float2 v = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(row)[lane + 32 * i]);
                acc[i].x = fmaf(p, v.x, acc[i].x);
                acc[i].y = fmaf(p, v.y, acc[i].y);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        partial[warp][2 * lane + 64 * i] = acc[i].x;
        partial[warp][2 * lane + 64 * i + 1] = acc[i].y;
    }
    __syncthreads();
    for (int d = t; d < D; d += Threads) {
        float value = 0.0f;
        #pragma unroll
        for (int w = 0; w < Warps; ++w) value += partial[w][d];
        output[head * D + d] = __float2bfloat16_rn(value / total);
        // Every CTA bypasses this slot, so the fused append is race-free.
        if (head == 0) cache[slot * D + d] = kv[d];
    }
}
} // namespace

extern "C" int rustinfer_v4_hca_decode_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* kv, const float* sink,
    const int* position, const __nv_bfloat16* compressed,
    __nv_bfloat16* cache, __nv_bfloat16* output, int heads, int capacity, cudaStream_t stream) {
    decode<<<heads, Threads, 0, stream>>>(query, kv, sink, position, compressed, cache, output, capacity);
    return static_cast<int>(cudaGetLastError());
}
