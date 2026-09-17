#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>
#include <climits>

namespace {
namespace wm = nvcuda::wmma;
constexpr int D = 128, M = 16, Tile = 64, Threads = 128, Ld = D + 16;

template<int QRows>
struct __align__(32) Shared {
    __nv_bfloat16 query[QRows * Ld];
    __nv_bfloat16 keys[Tile * Ld];
    float dot[4 * M * M];
    float weights[QRows];
};

// A CTA owns one token and 64 candidate keys; each warp owns 16 candidates.
// The 16x16 matrix product uses heads as its M axis, so even N=1 uses Tensor
// Cores. Stage keys once and reuse them for all head tiles; never materialize
// [N,H,C] in global memory or require a cross-CTA reduction/atomic operation.
template<int CachedHeads>
__global__ void scores(const __nv_bfloat16* __restrict__ query,
                       const __nv_bfloat16* __restrict__ keys,
                       const float* __restrict__ weights,
                       const int* __restrict__ start_ptr,
                       float* __restrict__ output,
                       int tokens, int heads, int capacity, int tiles) {
    const int token = blockIdx.x / tiles, base = (blockIdx.x % tiles) * Tile;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int start = *start_ptr;
    const bool valid = start >= 0 && start <= INT_MAX - (tokens - 1)
        && (static_cast<long long>(start) + tokens) / 4 <= capacity;
    const int visible = valid ? static_cast<int>((static_cast<long long>(start) + token + 1) / 4) : 0;
    const size_t offset = static_cast<size_t>(token) * capacity;
    if (!valid || base >= visible) {
        if (tid < Tile && base + tid < capacity)
            output[offset + base + tid] = valid ? -CUDART_INF_F : CUDART_NAN_F;
        return;
    }

    __shared__ Shared<CachedHeads ? CachedHeads : M> s;
    for (int pair = tid; pair < Tile * D / 2; pair += Threads) {
        const int row = pair / (D / 2), col = (pair % (D / 2)) * 2;
        const __nv_bfloat162 v = base + row < visible
            ? *reinterpret_cast<const __nv_bfloat162*>(keys + static_cast<size_t>(base + row) * D + col)
            : __float2bfloat162_rn(0.0f);
        *reinterpret_cast<__nv_bfloat162*>(s.keys + row * Ld + col) = v;
    }
    auto stage_query = [&](int head0) {
        constexpr int QRows = CachedHeads ? CachedHeads : M;
        for (int pair = tid; pair < QRows * D / 2; pair += Threads) {
            const int row = pair / (D / 2), col = (pair % (D / 2)) * 2;
            const size_t src = (static_cast<size_t>(token) * heads + head0 + row) * D + col;
            const __nv_bfloat162 v = head0 + row < heads
                ? *reinterpret_cast<const __nv_bfloat162*>(query + src)
                : __float2bfloat162_rn(0.0f);
            *reinterpret_cast<__nv_bfloat162*>(s.query + row * Ld + col) = v;
        }
        if (tid < QRows)
            s.weights[tid] = head0 + tid < heads
                ? weights[static_cast<size_t>(token) * heads + head0 + tid] : 0.0f;
    };
    if constexpr (CachedHeads) {
        // With up to 64 heads Q and K fit together below 48 KiB. Publish them
        // once; subsequent head tiles require only warp-local synchronization.
        stage_query(0);
        __syncthreads();
    }
    using A = wm::fragment<wm::matrix_a, M, M, M, __nv_bfloat16, wm::row_major>;
    using B = wm::fragment<wm::matrix_b, M, M, M, __nv_bfloat16, wm::col_major>;
    using Acc = wm::fragment<wm::accumulator, M, M, M, float>;
    float total = 0.0f;
    for (int head0 = 0; head0 < heads; head0 += M) {
        if constexpr (!CachedHeads) {
            stage_query(head0);
            __syncthreads();
        }
        const int qrow = CachedHeads ? head0 : 0;
        Acc acc;
        wm::fill_fragment(acc, 0.0f);
        #pragma unroll
        for (int k = 0; k < D; k += M) {
            A a; B b;
            wm::load_matrix_sync(a, s.query + qrow * Ld + k, Ld);
            wm::load_matrix_sync(b, s.keys + warp * M * Ld + k, Ld);
            wm::mma_sync(acc, a, b, acc);
        }
        // Use the documented row-major store instead of assuming a fragment
        // lane layout. Each lane reduces eight heads for one candidate.
        wm::store_matrix_sync(s.dot + warp * M * M, acc, M, wm::mem_row_major);
        __syncwarp();
        float partial = 0.0f;
        #pragma unroll
        for (int h = 0; h < M / 2; ++h) {
            const int row = (lane / M) * (M / 2) + h;
            const float dot = s.dot[warp * M * M + row * M + lane % M];
            partial = fmaf(fmaxf(dot, 0.0f), s.weights[qrow + row], partial);
        }
        total += partial + __shfl_xor_sync(0xffffffff, partial, M);
        if constexpr (CachedHeads) __syncwarp(); // protect this warp's dot tile
        else __syncthreads(); // protect Q/weights before the next overwrite
    }
    const int key = base + warp * M + lane;
    if (lane < M && key < capacity)
        output[offset + key] = key < visible ? total : -CUDART_INF_F;
}
} // namespace

extern "C" int rustinfer_v4_indexer_scores_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* keys,
    const float* weights, const int* start, float* output,
    int tokens, int heads, int capacity, cudaStream_t stream) {
    const int tiles = (capacity + Tile - 1) / Tile;
    // Rust validates this flattened grid against the CUDA x-dimension limit.
    const unsigned grid = static_cast<unsigned>(static_cast<size_t>(tokens) * tiles);
    if (heads <= 16)
        scores<16><<<grid, Threads, 0, stream>>>(query, keys, weights, start, output, tokens, heads, capacity, tiles);
    else if (heads <= 32)
        scores<32><<<grid, Threads, 0, stream>>>(query, keys, weights, start, output, tokens, heads, capacity, tiles);
    // On sm_89 the cached 64-head path lowers small-grid latency, but its
    // 40 KiB footprint reduces CTA residency on larger grids. Keep the
    // 26 KiB streaming path for throughput; this threshold is shape-only
    // and graph-safe, measured on the local 16GB development GPU.
    else if (heads <= 64 && grid <= 128)
        scores<64><<<grid, Threads, 0, stream>>>(query, keys, weights, start, output, tokens, heads, capacity, tiles);
    else
        scores<0><<<grid, Threads, 0, stream>>>(query, keys, weights, start, output, tokens, heads, capacity, tiles);
    return static_cast<int>(cudaGetLastError());
}
