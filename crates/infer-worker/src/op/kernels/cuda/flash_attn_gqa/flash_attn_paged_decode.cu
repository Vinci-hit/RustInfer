// flash_attn_paged_decode.cu
// -----------------------------------------------------------------------------
// Paged decode attention over a global KV pool.
//
// Optimized over the original correctness-first kernel by using a full CTA per
// (request, q_head): row_max is reduced across KV tokens, and output head_dim
// lanes are computed in parallel. This keeps the public ABI and no-workspace
// contract unchanged while avoiding the original single-thread serial path.
// -----------------------------------------------------------------------------

#include "flash_attn_gqa.h"
#include "flash_attn_paged_common.cuh"

#include <cstdio>

namespace flash_paged_decode {

static constexpr int kBlockThreads = 256;

__device__ __forceinline__ void token_to_block(
    int token,
    int block_size,
    int& block_idx,
    int& block_off)
{
    if (block_size == 16) {
        block_idx = token >> 4;
        block_off = token & 15;
    } else if (block_size == 32) {
        block_idx = token >> 5;
        block_off = token & 31;
    } else {
        block_idx = token / block_size;
        block_off = token - block_idx * block_size;
    }
}

template <typename Elem>
__device__ __forceinline__ float qk_score(
    const Elem* __restrict__ q_row,
    const Elem* __restrict__ k_row,
    int head_dim)
{
    float score = 0.0f;
    #pragma unroll 4
    for (int d = 0; d < head_dim; ++d) {
        score += PagedET<Elem>::to_f(q_row[d]) * PagedET<Elem>::to_f(k_row[d]);
    }
    return score;
}

template <typename Elem>
__global__ void paged_decode_parallel_kernel(
    const Elem* __restrict__ q,
    int64_t qsb,
    int64_t qsh,
    const Elem* __restrict__ k_pool,
    const Elem* __restrict__ v_pool,
    Elem* __restrict__ o,
    int64_t osb,
    int64_t osh,
    const uint32_t* __restrict__ block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* __restrict__ kv_lens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale)
{
    const int b = blockIdx.x;
    const int qh = blockIdx.y;
    const int tid = threadIdx.x;

    Elem* o_row = o + static_cast<int64_t>(b) * osb + static_cast<int64_t>(qh) * osh;
    const int kv_len = kv_lens[b];
    if (kv_len <= 0) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            o_row[d] = PagedET<Elem>::from_f(0.0f);
        }
        return;
    }

    const int kvh = qh / (num_q_heads / num_kv_heads);
    const Elem* q_row = q + static_cast<int64_t>(b) * qsb + static_cast<int64_t>(qh) * qsh;
    const uint32_t* block_table = block_tables + static_cast<int64_t>(b) * max_blocks_per_seq;

    __shared__ float s_red[kBlockThreads];

    float local_max = -INFINITY;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        int block_idx, block_off;
        token_to_block(t, block_size, block_idx, block_off);
        const uint32_t physical_block = block_table[block_idx];
        const Elem* k_row = paged_kv_row(
            k_pool, physical_block, block_off, block_size, num_kv_heads, kvh, head_dim);
        local_max = fmaxf(local_max, qk_score(q_row, k_row, head_dim) * softmax_scale);
    }
    s_red[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_red[tid] = fmaxf(s_red[tid], s_red[tid + offset]);
        }
        __syncthreads();
    }
    const float row_max = s_red[0];

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float denom = 0.0f;
        float acc = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            int block_idx, block_off;
            token_to_block(t, block_size, block_idx, block_off);
            const uint32_t physical_block = block_table[block_idx];
            const Elem* k_row = paged_kv_row(
                k_pool, physical_block, block_off, block_size, num_kv_heads, kvh, head_dim);
            const Elem* v_row = paged_kv_row(
                v_pool, physical_block, block_off, block_size, num_kv_heads, kvh, head_dim);
            const float score = qk_score(q_row, k_row, head_dim) * softmax_scale;
            const float w = __expf(score - row_max);
            denom += w;
            acc += w * PagedET<Elem>::to_f(v_row[d]);
        }
        o_row[d] = PagedET<Elem>::from_f(denom == 0.0f ? 0.0f : acc / denom);
    }
}

template <typename Elem>
static inline cudaError_t launch_dispatch(
    const Elem* q,
    int64_t qsb,
    int64_t qsh,
    const Elem* k_pool,
    const Elem* v_pool,
    Elem* o,
    int64_t osb,
    int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    int batch,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float softmax_scale,
    cudaStream_t stream)
{
    if (batch <= 0) return cudaSuccess;
    paged_decode_parallel_kernel<Elem><<<dim3(batch, num_q_heads), dim3(kBlockThreads), 0, stream>>>(
        q, qsb, qsh, k_pool, v_pool, o, osb, osh,
        block_tables, max_blocks_per_seq, block_size, kv_lens,
        num_q_heads, num_kv_heads, head_dim, softmax_scale);
    return cudaGetLastError();
}

}  // namespace flash_paged_decode

extern "C" void launch_flash_attn_paged_decode_bf16(
    const __nv_bfloat16* q, int64_t qsb, int64_t qsh,
    const __nv_bfloat16* k_pool,
    const __nv_bfloat16* v_pool,
          __nv_bfloat16* o, int64_t osb, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream)
{
    cudaError_t err = flash_paged_decode::launch_dispatch<__nv_bfloat16>(
        q, qsb, qsh, k_pool, v_pool, o, osb, osh,
        block_tables, max_blocks_per_seq, block_size, kv_lens,
        batch, num_q_heads, num_kv_heads, head_dim, softmax_scale, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[flash_attn_paged_decode_bf16] launch error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void launch_flash_attn_paged_decode_fp16(
    const __half* q, int64_t qsb, int64_t qsh,
    const __half* k_pool,
    const __half* v_pool,
          __half* o, int64_t osb, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream)
{
    cudaError_t err = flash_paged_decode::launch_dispatch<__half>(
        q, qsb, qsh, k_pool, v_pool, o, osb, osh,
        block_tables, max_blocks_per_seq, block_size, kv_lens,
        batch, num_q_heads, num_kv_heads, head_dim, softmax_scale, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[flash_attn_paged_decode_fp16] launch error: %s\n", cudaGetErrorString(err));
    }
}
