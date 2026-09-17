#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>
#include <climits>

// Shared memory layout and tile arithmetic; SWA/HCA keep separate softmax
// schedules so specialization adds no runtime mode checks in the hot loops.
namespace rustinfer::v4::attention {
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
};
struct __align__(32) OnlineShared : Shared {
    float maximum[M], total[M], factor[M];
};
static_assert(sizeof(Shared) == 46080 && sizeof(OnlineShared) == 46272);

template<bool Compressed>
__device__ __forceinline__ bool valid_start(int start, int tokens, int capacity) {
    if (start < 0 || start > INT_MAX - (tokens - 1)) return false;
    if constexpr (Compressed)
        return (static_cast<long long>(start) + tokens) / W <= capacity;
    return true;
}
template<bool Compressed, class Storage>
__device__ __forceinline__ void stage_kv(Storage& s, const __nv_bfloat16* chunk,
                        const __nv_bfloat16* cache, const __nv_bfloat16* compressed,
                        int start, int first, int local, int count, int base) {
    for (int pair = threadIdx.x; pair < M * D / 2; pair += Threads) {
        const int row = pair / (D / 2), col = (pair % (D / 2)) * 2;
        __nv_bfloat162 value = __float2bfloat162_rn(0.0f);
        if (base + row < count) {
            const int j = base + row;
            const __nv_bfloat16* src;
            if (Compressed && j >= local) {
                src = compressed + static_cast<size_t>(j - local) * D;
            } else {
                const int absolute = first + j;
                src = absolute < start ? cache + (absolute & (W - 1)) * D
                    : chunk + static_cast<size_t>(absolute - start) * D;
            }
            value = *reinterpret_cast<const __nv_bfloat162*>(src + col);
        }
        *reinterpret_cast<__nv_bfloat162*>(s.kv + row * Ld + col) = value;
    }
}
template<class Storage>
__device__ __forceinline__ void stage_query(Storage& s, const __nv_bfloat16* query,
                                           size_t offset, int head0, int heads) {
    const int tid = threadIdx.x;
    for (int pair = tid; pair < M * D / 2; pair += Threads) {
        const int row = pair / (D / 2), col = (pair % (D / 2)) * 2;
        const __nv_bfloat162 value = head0 + row < heads
            ? *reinterpret_cast<const __nv_bfloat162*>(query + offset + row * D + col)
            : __float2bfloat162_rn(0.0f);
        *reinterpret_cast<__nv_bfloat162*>(s.q + row * Ld + col) = value;
    }
}
using A = wm::fragment<wm::matrix_a, 16, 16, 16, __nv_bfloat16, wm::row_major>;
using Bcol = wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::col_major>;
using Brow = wm::fragment<wm::matrix_b, 16, 16, 16, __nv_bfloat16, wm::row_major>;
using Acc = wm::fragment<wm::accumulator, 16, 16, 16, float>;
template<class Storage>
__device__ __forceinline__ void qk_tile(Storage& s, int base, int size) {
    const int tid = threadIdx.x, warp = tid / 32;
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
template<bool Compressed>
__global__ void commit(const __nv_bfloat16* __restrict__ chunk,
                       const int* __restrict__ start_ptr,
                       __nv_bfloat16* __restrict__ cache, int tokens, int capacity) {
    const int start = *start_ptr;
    if (!valid_start<Compressed>(start, tokens, capacity)) return;
    const int row = (tokens > W ? tokens - W : 0) + blockIdx.x;
    const int slot = (start + row) & (W - 1);
    for (int pair = threadIdx.x; pair < D / 2; pair += blockDim.x)
        reinterpret_cast<__nv_bfloat162*>(cache + slot * D)[pair] =
            reinterpret_cast<const __nv_bfloat162*>(chunk + static_cast<size_t>(row) * D)[pair];
}
} // namespace rustinfer::v4::attention
