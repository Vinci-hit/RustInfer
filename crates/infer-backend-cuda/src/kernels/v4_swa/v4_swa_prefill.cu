#include "../v4_common/attention.cuh"

namespace {
using namespace rustinfer::v4::attention;

__global__ void prefill(const __nv_bfloat16* __restrict__ query,
                        const __nv_bfloat16* __restrict__ chunk,
                        const float* __restrict__ sink, const int* __restrict__ start_ptr,
                        const __nv_bfloat16* __restrict__ cache,
                        __nv_bfloat16* __restrict__ output, int tokens, int heads) {
    const int token = blockIdx.x, head0 = blockIdx.y * M;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid & 31;
    const int start = *start_ptr;
    const size_t offset = (static_cast<size_t>(token) * heads + head0) * D;
    if (!valid_start<false>(start, tokens, 0)) {
        for (int i = tid; i < M * D; i += Threads)
            if (head0 + i / D < heads) output[offset + i] = __float2bfloat16_rn(CUDART_NAN_F);
        return;
    }
    const int position = start + token;
    const int first = position > W - 1 ? position - (W - 1) : 0;
    const int count = position - first + 1;
    __shared__ Shared s;
    stage_query(s, query, offset, head0, heads);
    for (int i = tid; i < M * W; i += Threads) s.scores[i] = -CUDART_INF_F;
    __syncthreads();

    for (int base = 0; base < count; base += M) {
        stage_kv<false>(s, chunk, cache, nullptr, start, first, count, count, base);
        __syncthreads();
        qk_tile(s, base, count);
    }

    // One warp owns a complete row, including both passes of softmax. Read
    // every FP32 score before reusing its storage for BF16 high/residual parts.
    for (int row = warp; row < M; row += 4) {
        float scores[4], maximum = head0 + row < heads ? sink[head0 + row] : 0.0f;
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
        const float sink_score = head0 + row < heads ? sink[head0 + row] : 0.0f;
        const float inv = 1.0f / (total + expf(sink_score - maximum));
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float p = scores[j] * inv;
            const __nv_bfloat16 hi = __float2bfloat16_rn(p);
            const __nv_bfloat16 lo = __float2bfloat16_rn(p - __bfloat162float(hi));
            s.probabilities[row * (2 * W) + lane + 32 * j] = hi;
            s.probabilities[row * (2 * W) + W + lane + 32 * j] = lo;
        }
    }
    __syncthreads();
    Acc out[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) wm::fill_fragment(out[i], 0.0f);
    for (int base = 0; base < count; base += M) {
        stage_kv<false>(s, chunk, cache, nullptr, start, first, count, count, base);
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
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        wm::store_matrix_sync(s.partial + warp * M * M, out[i], M, wm::mem_row_major);
        __syncwarp();
        for (int e = lane; e < M * M; e += 32) {
            const int row = e / M, col = (warp + 4 * i) * 16 + e % M;
            if (head0 + row < heads)
                output[offset + row * D + col] = __float2bfloat16_rn(s.partial[warp * M * M + e]);
        }
        __syncwarp();
    }
}

} // namespace

extern "C" int rustinfer_v4_swa_prefill_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* chunk,
    const float* sink, const int* start, __nv_bfloat16* cache,
    __nv_bfloat16* output, int tokens, int heads, cudaStream_t stream) {
    prefill<<<dim3(tokens, (heads + M - 1) / M), Threads, 0, stream>>>(
        query, chunk, sink, start, cache, output, tokens, heads);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    commit<false><<<tokens < W ? tokens : W, Threads, 0, stream>>>(chunk, start, cache, tokens, 0);
    return static_cast<int>(cudaGetLastError());
}
