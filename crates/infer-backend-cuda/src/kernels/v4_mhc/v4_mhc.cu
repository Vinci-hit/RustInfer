#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace {
constexpr int Streams = 4;
constexpr int Chunk = 256;
constexpr int Rows = 4;

__device__ __forceinline__ float sum_warp(float x) {
    for (int d = 16; d; d /= 2) x += __shfl_down_sync(0xffffffff, x, d);
    return x;
}

// Split K supplies parallelism even for a single decode token. Four token rows
// reuse each FP32 weight load; no weight downcast, atomics or temporary X copy.
// Scratch is [N, ceil(4*D/Chunk), Outputs+1], with RMS sum in the last column.
template<int Outputs>
__global__ void project(const __nv_bfloat16* x, const float* weight,
                        float* partial, int n, int dim, int parts) {
    const int part = blockIdx.x % parts;
    const int row = (blockIdx.x / parts) * Rows;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int width = Streams * dim;
    constexpr int PerWarp = (Outputs + 7) / 8;
    float acc[Rows][PerWarp] = {};
    float norm[Rows] = {};
    #pragma unroll
    for (int step = 0; step < Chunk / 32; ++step) {
        const int k = part * Chunk + step * 32 + lane;
        float values[Rows];
        #pragma unroll
        for (int r = 0; r < Rows; ++r) {
            values[r] = row + r < n && k < width
                ? __bfloat162float(x[static_cast<size_t>(row + r) * width + k]) : 0.f;
            if (warp == 0) norm[r] = fmaf(values[r], values[r], norm[r]);
        }
        #pragma unroll
        for (int j = 0; j < PerWarp; ++j) {
            const int col = warp + 8 * j;
            const float w = col < Outputs && k < width ? weight[col * width + k] : 0.f;
            #pragma unroll
            for (int r = 0; r < Rows; ++r) acc[r][j] = fmaf(values[r], w, acc[r][j]);
        }
    }
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        const size_t out = (static_cast<size_t>(row + r) * parts + part) * (Outputs + 1);
        #pragma unroll
        for (int j = 0; j < PerWarp; ++j) {
            const float value = sum_warp(acc[r][j]);
            if (lane == 0 && row + r < n && warp + 8 * j < Outputs)
                partial[out + warp + 8 * j] = value;
        }
        if (warp == 0) {
            const float value = sum_warp(norm[r]);
            if (lane == 0 && row + r < n) partial[out + Outputs] = value;
        }
    }
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

template<bool Head>
__global__ void finish(const __nv_bfloat16* x, const float* partial,
                       const float* scale, const float* base,
                       __nv_bfloat16* collapsed, float* post, float* comb,
                       int dim, int parts, float norm_eps, float hc_eps, int iters) {
    constexpr int Outputs = Head ? 4 : 24;
    __shared__ float mixes[Outputs + 1];
    __shared__ float pre[4];
    const int t = threadIdx.x;
    const size_t row = blockIdx.x;
    if (t <= Outputs) {
        float total = 0.f;
        for (int p = 0; p < parts; ++p)
            total += partial[(row * parts + p) * (Outputs + 1) + t];
        mixes[t] = total;
    }
    __syncthreads();
    const float inv = rsqrtf(mixes[Outputs] / (Streams * dim) + norm_eps);
    if (t < 4) {
        pre[t] = sigmoid(__fadd_rn(__fmul_rn(mixes[t] * inv, scale[0]), base[t])) + hc_eps;
        if constexpr (!Head)
            post[row * 4 + t] = 2.f * sigmoid(__fadd_rn(__fmul_rn(mixes[4 + t] * inv, scale[1]), base[4 + t]));
    }
    if constexpr (!Head) {
        if (t < 32) {
            // Lanes 0..15 own a row-major 4x4 matrix. All 32 lanes participate
            // in shuffles; XOR 1/2 stays in a row, XOR 4/8 stays in a column.
            float v = t < 16 ? __fadd_rn(__fmul_rn(mixes[8 + t] * inv, scale[2]), base[8 + t]) : 0.f;
            float m = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
            v = expf(v - m);
            float z = v + __shfl_xor_sync(0xffffffff, v, 1);
            z += __shfl_xor_sync(0xffffffff, z, 2);
            v = v / z + hc_eps;
            for (int iteration = 0; iteration < iters; ++iteration) {
                if (iteration) {
                    z = v + __shfl_xor_sync(0xffffffff, v, 1);
                    z += __shfl_xor_sync(0xffffffff, z, 2);
                    v /= z + hc_eps;
                }
                z = v + __shfl_xor_sync(0xffffffff, v, 4);
                z += __shfl_xor_sync(0xffffffff, z, 8);
                v /= z + hc_eps;
            }
            if (t < 16) comb[row * 16 + t] = v;
        }
    }
    __syncthreads();
    for (int d = t; d < dim; d += blockDim.x) {
        float value = 0.f;
        #pragma unroll
        for (int i = 0; i < Streams; ++i)
            value = __fadd_rn(value, __fmul_rn(pre[i], __bfloat162float(x[(row * Streams + i) * dim + d])));
        collapsed[row * dim + d] = __float2bfloat16_rn(value);
    }
}

__device__ __forceinline__ float bf_round(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__global__ void expand(const __nv_bfloat16* residual, const __nv_bfloat16* branch,
                       const float* post, const float* comb, __nv_bfloat16* output,
                       int dim, int tiles) {
    const size_t row = blockIdx.x / tiles;
    const int d = (blockIdx.x % tiles) * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float r[Streams];
    #pragma unroll
    for (int i = 0; i < Streams; ++i)
        r[i] = __bfloat162float(residual[(row * Streams + i) * dim + d]);
    const float x = __bfloat162float(branch[row * dim + d]);
    #pragma unroll
    for (int j = 0; j < Streams; ++j) {
        float mixed = 0.f;
        #pragma unroll
        for (int i = 0; i < Streams; ++i)
            mixed = __fadd_rn(mixed, __fmul_rn(bf_round(comb[row * 16 + i * Streams + j]), r[i]));
        // Match the existing Transformers BF16 reference: cast coefficients,
        // round the residual matmul and branch product separately, then add.
        const float placed = bf_round(__fmul_rn(bf_round(post[row * Streams + j]), x));
        output[(row * Streams + j) * dim + d] = __float2bfloat16_rn(__fadd_rn(bf_round(mixed), placed));
    }
}
} // namespace

extern "C" int rustinfer_v4_mhc_pre_bf16(
    const __nv_bfloat16* x, const float* weight, const float* scale, const float* base,
    float* partial, __nv_bfloat16* collapsed, float* post, float* comb,
    int n, int dim, int head, float norm_eps, float hc_eps, int iters, cudaStream_t stream) {
    const int parts = (Streams * dim + Chunk - 1) / Chunk;
    const unsigned blocks = static_cast<unsigned>((static_cast<size_t>(n) + Rows - 1) / Rows * parts);
    if (head) project<4><<<blocks, 256, 0, stream>>>(x, weight, partial, n, dim, parts);
    else project<24><<<blocks, 256, 0, stream>>>(x, weight, partial, n, dim, parts);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (head) finish<true><<<n, 128, 0, stream>>>(x, partial, scale, base, collapsed, post, comb, dim, parts, norm_eps, hc_eps, iters);
    else finish<false><<<n, 128, 0, stream>>>(x, partial, scale, base, collapsed, post, comb, dim, parts, norm_eps, hc_eps, iters);
    return cudaGetLastError();
}

extern "C" int rustinfer_v4_mhc_post_bf16(
    const __nv_bfloat16* residual, const __nv_bfloat16* branch,
    const float* post, const float* comb, __nv_bfloat16* output,
    int n, int dim, cudaStream_t stream) {
    const int tiles = (dim + 255) / 256;
    expand<<<static_cast<unsigned>(static_cast<size_t>(n) * tiles), 256, 0, stream>>>(residual, branch, post, comb, output, dim, tiles);
    return cudaGetLastError();
}
