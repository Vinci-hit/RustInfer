#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_reduce.cuh>

namespace {
constexpr int kDim = 512;
constexpr int kWindow = 128;
constexpr int kWarps = 8;
constexpr int kThreads = 32 * kWarps;
constexpr float kScale = 0.04419417382415922f; // 1/sqrt(512)
struct Maximum {
    __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

// One CTA per Q head. KV is shared across heads through L2; each warp handles
// a disjoint subset of the small window. No global scores or scratch buffers.
__global__ void decode(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ new_kv,
    const float* __restrict__ sink,
    const int* __restrict__ position,
    __nv_bfloat16* __restrict__ cache,
    __nv_bfloat16* __restrict__ output) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid / 32;
    const int head = blockIdx.x;
    const int pos = *position;
    if (pos < 0) {
        for (int d = tid; d < kDim; d += kThreads)
            output[head * kDim + d] = __float2bfloat16_rn(CUDART_NAN_F);
        return;
    }
    const int slot = pos & (kWindow - 1);
    const int count = pos < kWindow ? pos + 1 : kWindow;
    float2 q[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        q[i] = __bfloat1622float2(
            reinterpret_cast<const __nv_bfloat162*>(query + head * kDim)[lane + 32 * i]);

    __shared__ float probabilities[kWindow];
    __shared__ float partial[kWarps][kDim];
    using Reduce = cub::BlockReduce<float, kThreads>;
    __shared__ typename Reduce::TempStorage reduction;
    __shared__ float maximum, inverse_sum;

    // Physical-slot order is sufficient: weighted attention is permutation
    // invariant. Until the first wrap, only slots [0, pos] are valid.
    for (int j = warp; j < count; j += kWarps) {
        // EVERY head bypasses the slot being overwritten. Only CTA 0 commits
        // it below, so the fused append cannot race another CTA's cache reads.
        const __nv_bfloat16* row = j == slot ? new_kv : cache + j * kDim;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float2 kv = __bfloat1622float2(
                reinterpret_cast<const __nv_bfloat162*>(row)[lane + 32 * i]);
            dot = fmaf(q[i].x, kv.x, dot);
            dot = fmaf(q[i].y, kv.y, dot);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        if (lane == 0) probabilities[j] = dot * kScale;
    }
    __syncthreads();
    float local_max = tid < count ? probabilities[tid] : -CUDART_INF_F;
    if (tid == 0) local_max = fmaxf(local_max, sink[head]);
    const float block_max = Reduce(reduction).Reduce(local_max, Maximum());
    if (tid == 0) maximum = block_max;
    __syncthreads();
    float weight = tid < count ? expf(probabilities[tid] - maximum) : 0.0f;
    if (tid < kWindow) probabilities[tid] = weight;
    if (tid == 0) weight += expf(sink[head] - maximum);
    const float total = Reduce(reduction).Sum(weight);
    if (tid == 0) inverse_sum = 1.0f / total;
    __syncthreads();

    float2 acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = make_float2(0.0f, 0.0f);
    for (int j = warp; j < count; j += kWarps) {
        const float p = probabilities[j] * inverse_sum;
        const __nv_bfloat16* row = j == slot ? new_kv : cache + j * kDim;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float2 v = __bfloat1622float2(
                reinterpret_cast<const __nv_bfloat162*>(row)[lane + 32 * i]);
            acc[i].x = fmaf(p, v.x, acc[i].x);
            acc[i].y = fmaf(p, v.y, acc[i].y);
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        partial[warp][2 * lane + 64 * i] = acc[i].x;
        partial[warp][2 * lane + 64 * i + 1] = acc[i].y;
    }
    __syncthreads();
    for (int d = tid; d < kDim; d += kThreads) {
        float value = 0.0f;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) value += partial[w][d];
        output[head * kDim + d] = __float2bfloat16_rn(value);
        if (head == 0) cache[slot * kDim + d] = new_kv[d];
    }
}
} // namespace

extern "C" int rustinfer_v4_swa_decode_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* new_kv,
    const float* sink, const int* position,
    __nv_bfloat16* cache, __nv_bfloat16* output,
    int heads, cudaStream_t stream) {
    decode<<<heads, kThreads, 0, stream>>>(query, new_kv, sink, position, cache, output);
    return static_cast<int>(cudaGetLastError());
}
