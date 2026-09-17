#include "../v4_common/attention.cuh"

namespace {
using namespace rustinfer::v4::attention;

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
    if (!valid_start<true>(start, tokens, capacity)) {
        for (int i = tid; i < M * D; i += Threads)
            if (head0 + i / D < heads) output[offset + i] = __float2bfloat16_rn(CUDART_NAN_F);
        return;
    }
    const int position = start + token;
    const int first = position > W - 1 ? position - (W - 1) : 0;
    const int local = position - first + 1;
    const int count = local + (static_cast<long long>(position) + 1) / W;
    __shared__ OnlineShared s;
    stage_query(s, query, offset, head0, heads);

    if (tid < M) {
        s.maximum[tid] = head0 + tid < heads ? sink[head0 + tid] : 0.0f;
        s.total[tid] = 1.0f;
    }
    __syncthreads();

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
            stage_kv<true>(s, chunk, cache, compressed, start, first, local, count, tile + base);
            __syncthreads();
            qk_tile(s, base, size);
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
            stage_kv<true>(s, chunk, cache, compressed, start, first, local, count, tile + base);
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

} // namespace

extern "C" int rustinfer_v4_hca_prefill_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* chunk,
    const float* sink, const int* start, const __nv_bfloat16* compressed, __nv_bfloat16* cache,
    __nv_bfloat16* output, int tokens, int heads, int capacity, cudaStream_t stream) {
    prefill<<<dim3(tokens, (heads + M - 1) / M), Threads, 0, stream>>>(
        query, chunk, sink, start, cache, compressed, output, tokens, heads, capacity);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    commit<true><<<tokens < W ? tokens : W, Threads, 0, stream>>>(chunk, start, cache, tokens, capacity);
    return static_cast<int>(cudaGetLastError());
}
