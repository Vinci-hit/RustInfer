#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>
#include <climits>

namespace {
namespace wm = nvcuda::wmma;
constexpr int D = 512, W = 128, M = 16, Threads = 128;
constexpr int Ld = D + 16; // aligned padding reduces shared-memory bank conflicts
constexpr float Scale = 0.04419417382415922f;

struct __align__(32) Shared {
    __nv_bfloat16 q[M * Ld];
    __nv_bfloat16 kv[M * Ld];
    union {
        float scores[M * W];
        __nv_bfloat16 probabilities[M * W * 2];
    };
    float partial[4 * M * M];
    float maximum[M], total[M], factor[M];
}; // 46,272 bytes; no dynamic shared-memory opt-in or global workspace

__device__ __forceinline__ bool valid_start(int start, int tokens, int capacity) {
    return start >= 0 && start <= INT_MAX - (tokens - 1)
        && (static_cast<long long>(start) + tokens) / W <= capacity;
}

// Each query has its own window. History is immutable until a second kernel
// commits the chunk, so queries at the front cannot lose needed old KV rows.
__device__ void stage_kv(Shared& s, const __nv_bfloat16* chunk,
                        const __nv_bfloat16* cache, const __nv_bfloat16* compressed,
                        int start, int first, int local, int count, int base) {
    for (int pair = threadIdx.x; pair < M * D / 2; pair += Threads) {
        const int row = pair / (D / 2), col = (pair % (D / 2)) * 2;
        __nv_bfloat162 value = __float2bfloat162_rn(0.0f);
        if (base + row < count) {
            const int j = base + row;
            const int absolute = first + (j < local ? j : 0);
            const __nv_bfloat16* src = j >= local
                ? compressed + static_cast<size_t>(j - local) * D
                : (absolute < start ? cache + (absolute & (W - 1)) * D
                   : chunk + static_cast<size_t>(absolute - start) * D);
            value = *reinterpret_cast<const __nv_bfloat162*>(src + col);
        }
        *reinterpret_cast<__nv_bfloat162*>(s.kv + row * Ld + col) = value;
    }
}

__global__ void prefill(const __nv_bfloat16* __restrict__ query,
                        const __nv_bfloat16* __restrict__ chunk,
                        const float* __restrict__ sink, const int* __restrict__ start_ptr,
                        const __nv_bfloat16* __restrict__ cache,
                        const __nv_bfloat16* __restrict__ compressed,
                        __nv_bfloat16* __restrict__ output, int tokens, int heads, int capacity) {
    const int token = blockIdx.x, head0 = blockIdx.y * M;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid & 31;
    const int start = *start_ptr;
    const size_t offset = (static_cast<size_t>(token) * heads + head0) * D;
    if (!valid_start(start, tokens, capacity)) {
        for (int i = tid; i < M * D; i += Threads)
            if (head0 + i / D < heads) output[offset + i] = __float2bfloat16_rn(CUDART_NAN_F);
        return;
    }
    const int position = start + token;
    const int first = position > W - 1 ? position - (W - 1) : 0;
    const int local = position - first + 1;
    const int count = local + (static_cast<long long>(position) + 1) / W;
    __shared__ Shared s;
    for (int pair = tid; pair < M * D / 2; pair += Threads) {
        const int row = pair / (D / 2), col = (pair % (D / 2)) * 2;
        const __nv_bfloat162 value = head0 + row < heads
            ? *reinterpret_cast<const __nv_bfloat162*>(query + offset + row * D + col)
            : __float2bfloat162_rn(0.0f);
        *reinterpret_cast<__nv_bfloat162*>(s.q + row * Ld + col) = value;
    }
    if (tid < M) {
        s.maximum[tid] = head0 + tid < heads ? sink[head0 + tid] : 0.0f;
        s.total[tid] = 1.0f;
    }
    __syncthreads();

    using A = wm::fragment<wm::matrix_a, 16, 16, 16, __nv_bfloat16, wm::row_major>;
    using Bcol = wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::col_major>;
    using Brow = wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::row_major>;
    using Acc = wm::fragment<wm::accumulator, 16, 16, 16, float>;
    Acc out[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) wm::fill_fragment(out[i], 0.0f);
    // Local KV then all causally visible compressed KV, in 128-row tiles.
    // Running max/denominator and output accumulation span BOTH pools.
    for (int tile = 0; tile < count; tile += W) {
        const int size = min(W, count - tile);
        for (int i = tid; i < M * W; i += Threads) s.scores[i] = -CUDART_INF_F;
        __syncthreads();
        for (int base = 0; base < size; base += M) {
            stage_kv(s, chunk, cache, compressed, start, first, local, count, tile + base);
            __syncthreads();
            Acc scores;
            wm::fill_fragment(scores, 0.0f);
            // Four warps split the 512-dimensional dot product into 128 each.
            for (int k = warp * 128; k < (warp + 1) * 128; k += 16) {
                A q; Bcol kvec;
                wm::load_matrix_sync(q, s.q + k, Ld);
                wm::load_matrix_sync(kvec, s.kv + k, Ld);
                wm::mma_sync(scores, q, kvec, scores);
            }
            wm::store_matrix_sync(s.partial + warp * M * M, scores, M, wm::mem_row_major);
            __syncthreads();
            for (int i = tid; i < M * M; i += Threads) {
                if (base + i % M < size) {
                    float sum = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) sum += s.partial[w * M * M + i];
                    s.scores[(i / M) * W + base + i % M] = sum * Scale;
                }
            }
            __syncthreads();
        }

        // One warp owns a complete row, including both passes of softmax. Read
        // every FP32 score before reusing its storage for BF16 high/residual parts.
        for (int row = warp; row < M; row += 4) {
            float scores[4], maximum = s.maximum[row];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scores[j] = s.scores[row * W + lane + 32 * j];
                maximum = fmaxf(maximum, scores[j]);
            }
            #pragma unroll
            for (int delta = 16; delta; delta >>= 1)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
            float total = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scores[j] = expf(scores[j] - maximum);
                total += scores[j];
            }
            #pragma unroll
            for (int delta = 16; delta; delta >>= 1)
                total += __shfl_xor_sync(0xffffffff, total, delta);
            if (lane == 0) {
                const float factor = expf(s.maximum[row] - maximum);
                s.factor[row] = factor;
                s.maximum[row] = maximum;
                s.total[row] = s.total[row] * factor + total;
            }
            __syncwarp();
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p = scores[j];
                const __nv_bfloat16 hi = __float2bfloat16_rn(p);
                const __nv_bfloat16 lo = __float2bfloat16_rn(p - __bfloat162float(hi));
                s.probabilities[row * (2 * W) + lane + 32 * j] = hi;
                s.probabilities[row * (2 * W) + W + lane + 32 * j] = lo;
            }
        }
        __syncthreads();
        // WMMA fragment element layout is opaque. Rescale via a row-major shared
        // tile instead of relying on undocumented lane-to-row mappings.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float* tmp = s.partial + warp * M * M;
            wm::store_matrix_sync(tmp, out[i], M, wm::mem_row_major);
            __syncwarp();
            for (int e = lane; e < M * M; e += 32) tmp[e] *= s.factor[e / M];
            __syncwarp();
            wm::load_matrix_sync(out[i], tmp, M, wm::mem_row_major);
            __syncwarp();
        }
        for (int base = 0; base < size; base += M) {
            stage_kv(s, chunk, cache, compressed, start, first, local, count, tile + base);
            __syncthreads();
            A hi, lo;
            wm::load_matrix_sync(hi, s.probabilities + base, 2 * W);
            wm::load_matrix_sync(lo, s.probabilities + W + base, 2 * W);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                Brow v;
                wm::load_matrix_sync(v, s.kv + (warp + 4 * i) * 16, Ld);
                wm::mma_sync(out[i], hi, v, out[i]);
                wm::mma_sync(out[i], lo, v, out[i]);
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        wm::store_matrix_sync(s.partial + warp * M * M, out[i], M, wm::mem_row_major);
        __syncwarp();
        for (int e = lane; e < M * M; e += 32) {
            const int row = e / M, col = (warp + 4 * i) * 16 + e % M;
            if (head0 + row < heads)
                output[offset + row * D + col] = __float2bfloat16_rn(s.partial[warp * M * M + e] / s.total[row]);
        }
        __syncwarp();
    }
}

__global__ void commit(const __nv_bfloat16* __restrict__ chunk,
                       const int* __restrict__ start_ptr,
                       __nv_bfloat16* __restrict__ cache, int tokens, int capacity) {
    const int start = *start_ptr;
    if (!valid_start(start, tokens, capacity)) return;
    const int row = (tokens > W ? tokens - W : 0) + blockIdx.x;
    const int slot = (start + row) & (W - 1);
    for (int pair = threadIdx.x; pair < D / 2; pair += blockDim.x)
        reinterpret_cast<__nv_bfloat162*>(cache + slot * D)[pair] =
            reinterpret_cast<const __nv_bfloat162*>(chunk + static_cast<size_t>(row) * D)[pair];
}
} // namespace

extern "C" int rustinfer_v4_hca_prefill_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* chunk,
    const float* sink, const int* start, const __nv_bfloat16* compressed, __nv_bfloat16* cache,
    __nv_bfloat16* output, int tokens, int heads, int capacity, cudaStream_t stream) {
    prefill<<<dim3(tokens, (heads + M - 1) / M), Threads, 0, stream>>>(
        query, chunk, sink, start, cache, compressed, output, tokens, heads, capacity);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    commit<<<tokens < W ? tokens : W, Threads, 0, stream>>>(chunk, start, cache, tokens, capacity);
    return static_cast<int>(cudaGetLastError());
}
