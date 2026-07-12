#include "./qkv_norm_rope_scatter.h"
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

namespace {

constexpr int THREADS_PER_WARP = 32;
constexpr int MAX_HEADS_PER_BLOCK = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void q_norm_rope_head128(
    __nv_bfloat16* __restrict__ base,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ sin_cache,
    const __nv_bfloat16* __restrict__ cos_cache,
    int pos,
    float eps,
    int lane_id)
{
    const int i0 = lane_id;
    const int i1 = lane_id + 32;
    const int j0 = lane_id + 64;
    const int j1 = lane_id + 96;

    const float x0 = __bfloat162float(base[i0]);
    const float x1 = __bfloat162float(base[i1]);
    const float y0 = __bfloat162float(base[j0]);
    const float y1 = __bfloat162float(base[j1]);

    float sum = 0.0f;
    sum = __fmaf_rn(x0, x0, sum);
    sum = __fmaf_rn(x1, x1, sum);
    sum = __fmaf_rn(y0, y0, sum);
    sum = __fmaf_rn(y1, y1, sum);

    const float total = warp_reduce_sum(sum);
    const float inv_rms = rsqrtf(total * (1.0f / 128.0f) + eps);

    const float nx0 = __bfloat162float(__float2bfloat16(x0 * __bfloat162float(weight[i0]) * inv_rms));
    const float nx1 = __bfloat162float(__float2bfloat16(x1 * __bfloat162float(weight[i1]) * inv_rms));
    const float ny0 = __bfloat162float(__float2bfloat16(y0 * __bfloat162float(weight[j0]) * inv_rms));
    const float ny1 = __bfloat162float(__float2bfloat16(y1 * __bfloat162float(weight[j1]) * inv_rms));

    const float s0 = __bfloat162float(sin_cache[(long long)pos * 64 + i0]);
    const float c0 = __bfloat162float(cos_cache[(long long)pos * 64 + i0]);
    const float s1 = __bfloat162float(sin_cache[(long long)pos * 64 + i1]);
    const float c1 = __bfloat162float(cos_cache[(long long)pos * 64 + i1]);

    base[i0] = __float2bfloat16(nx0 * c0 - ny0 * s0);
    base[j0] = __float2bfloat16(nx0 * s0 + ny0 * c0);
    base[i1] = __float2bfloat16(nx1 * c1 - ny1 * s1);
    base[j1] = __float2bfloat16(nx1 * s1 + ny1 * c1);
}

__device__ __forceinline__ void k_norm_rope_scatter_head128(
    const __nv_bfloat16* __restrict__ base,
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ sin_cache,
    const __nv_bfloat16* __restrict__ cos_cache,
    int pos,
    float eps,
    int lane_id)
{
    const int i0 = lane_id;
    const int i1 = lane_id + 32;
    const int j0 = lane_id + 64;
    const int j1 = lane_id + 96;

    const float x0 = __bfloat162float(base[i0]);
    const float x1 = __bfloat162float(base[i1]);
    const float y0 = __bfloat162float(base[j0]);
    const float y1 = __bfloat162float(base[j1]);

    float sum = 0.0f;
    sum = __fmaf_rn(x0, x0, sum);
    sum = __fmaf_rn(x1, x1, sum);
    sum = __fmaf_rn(y0, y0, sum);
    sum = __fmaf_rn(y1, y1, sum);

    const float total = warp_reduce_sum(sum);
    const float inv_rms = rsqrtf(total * (1.0f / 128.0f) + eps);

    const float nx0 = __bfloat162float(__float2bfloat16(x0 * __bfloat162float(weight[i0]) * inv_rms));
    const float nx1 = __bfloat162float(__float2bfloat16(x1 * __bfloat162float(weight[i1]) * inv_rms));
    const float ny0 = __bfloat162float(__float2bfloat16(y0 * __bfloat162float(weight[j0]) * inv_rms));
    const float ny1 = __bfloat162float(__float2bfloat16(y1 * __bfloat162float(weight[j1]) * inv_rms));

    const float s0 = __bfloat162float(sin_cache[(long long)pos * 64 + i0]);
    const float c0 = __bfloat162float(cos_cache[(long long)pos * 64 + i0]);
    const float s1 = __bfloat162float(sin_cache[(long long)pos * 64 + i1]);
    const float c1 = __bfloat162float(cos_cache[(long long)pos * 64 + i1]);

    dst[i0] = __float2bfloat16(nx0 * c0 - ny0 * s0);
    dst[j0] = __float2bfloat16(nx0 * s0 + ny0 * c0);
    dst[i1] = __float2bfloat16(nx1 * c1 - ny1 * s1);
    dst[j1] = __float2bfloat16(nx1 * s1 + ny1 * c1);
}

__device__ __forceinline__ void copy_head128(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int lane_id)
{
    const int offset = lane_id << 2;
    const uint2 vec = *reinterpret_cast<const uint2*>(src + offset);
    *reinterpret_cast<uint2*>(dst + offset) = vec;
}

__device__ __forceinline__ void q_norm_rope_one_warp(
    __nv_bfloat16* __restrict__ base,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ sin_cache,
    const __nv_bfloat16* __restrict__ cos_cache,
    int pos,
    int head_dim,
    float eps,
    int lane_id)
{
    float sum = 0.0f;
    for (int i = lane_id; i < head_dim; i += THREADS_PER_WARP) {
        const float x = __bfloat162float(base[i]);
        sum = __fmaf_rn(x, x, sum);
    }

    const float total = warp_reduce_sum(sum);
    const float inv_rms = rsqrtf(total / float(head_dim) + eps);

    const int half_head = head_dim >> 1;
    for (int i = lane_id; i < half_head; i += THREADS_PER_WARP) {
        const float s = __bfloat162float(sin_cache[(long long)pos * half_head + i]);
        const float c = __bfloat162float(cos_cache[(long long)pos * half_head + i]);
        const float n0 = __bfloat162float(__float2bfloat16(__bfloat162float(base[i]) * __bfloat162float(weight[i]) * inv_rms));
        const float n1 = __bfloat162float(__float2bfloat16(__bfloat162float(base[i + half_head]) * __bfloat162float(weight[i + half_head]) * inv_rms));
        base[i] = __float2bfloat16(n0 * c - n1 * s);
        base[i + half_head] = __float2bfloat16(n0 * s + n1 * c);
    }
}

__device__ __forceinline__ void k_norm_rope_scatter_one_warp(
    const __nv_bfloat16* __restrict__ base,
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ sin_cache,
    const __nv_bfloat16* __restrict__ cos_cache,
    int pos,
    int head_dim,
    float eps,
    int lane_id)
{
    float sum = 0.0f;
    for (int i = lane_id; i < head_dim; i += THREADS_PER_WARP) {
        const float x = __bfloat162float(base[i]);
        sum = __fmaf_rn(x, x, sum);
    }

    const float total = warp_reduce_sum(sum);
    const float inv_rms = rsqrtf(total / float(head_dim) + eps);

    const int half_head = head_dim >> 1;
    for (int i = lane_id; i < half_head; i += THREADS_PER_WARP) {
        const float s = __bfloat162float(sin_cache[(long long)pos * half_head + i]);
        const float c = __bfloat162float(cos_cache[(long long)pos * half_head + i]);
        const float n0 = __bfloat162float(__float2bfloat16(__bfloat162float(base[i]) * __bfloat162float(weight[i]) * inv_rms));
        const float n1 = __bfloat162float(__float2bfloat16(__bfloat162float(base[i + half_head]) * __bfloat162float(weight[i + half_head]) * inv_rms));
        dst[i] = __float2bfloat16(n0 * c - n1 * s);
        dst[i + half_head] = __float2bfloat16(n0 * s + n1 * c);
    }
}

__global__ __launch_bounds__(1024)
void qkv_norm_rope_scatter_kernel_bf16(
    __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ sin_cache,
    const __nv_bfloat16* __restrict__ cos_cache,
    const int* __restrict__ positions,
    __nv_bfloat16* __restrict__ k_pool,
    __nv_bfloat16* __restrict__ v_pool,
    const unsigned int* __restrict__ block_tables,
    const int* __restrict__ seq_positions,
    const int* __restrict__ seq_starts,
    const int* __restrict__ seq_lens,
    int num_tokens,
    int head_num,
    int kv_head_num,
    int head_dim,
    int kv_dim,
    long long q_row_stride,
    long long k_row_stride,
    long long v_row_stride,
    int max_blocks_per_seq,
    int block_size,
    float q_eps,
    float k_eps)
{
    const int seq = blockIdx.x;
    const int path = blockIdx.y;
    const int token_stride = gridDim.z;
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int lane_id = threadIdx.x & (THREADS_PER_WARP - 1);

    const int len = seq_lens[seq];
    if (len <= 0) return;

    const int start = seq_starts[seq];
    const int base_pos = seq_positions[seq];

    for (int t = blockIdx.z; t < len; t += token_stride) {
        const int row = start + t;
        if (row >= num_tokens) continue;

        const int logical_pos = base_pos + t;
        const int blk_idx = logical_pos / block_size;
        const int blk_off = logical_pos - blk_idx * block_size;
        const unsigned int physical_block = block_tables[seq * max_blocks_per_seq + blk_idx];
        const size_t dst_offset = (static_cast<size_t>(physical_block) * block_size + blk_off) * kv_dim;

        if (path == 0) {
            if (warp_id >= head_num) continue;
            const int pos = positions[row];
            __nv_bfloat16* q_head = q + (long long)row * q_row_stride + (long long)warp_id * head_dim;
            if (head_dim == 128) {
                q_norm_rope_head128(q_head, q_weight, sin_cache, cos_cache, pos, q_eps, lane_id);
            } else {
                q_norm_rope_one_warp(q_head, q_weight, sin_cache, cos_cache, pos, head_dim, q_eps, lane_id);
            }
        } else if (path == 1) {
            if (warp_id >= kv_head_num) continue;
            const int pos = positions[row];
            const __nv_bfloat16* k_head = k + (long long)row * k_row_stride + (long long)warp_id * head_dim;
            __nv_bfloat16* k_dst = k_pool + dst_offset + (long long)warp_id * head_dim;
            if (head_dim == 128) {
                k_norm_rope_scatter_head128(k_head, k_dst, k_weight, sin_cache, cos_cache, pos, k_eps, lane_id);
            } else {
                k_norm_rope_scatter_one_warp(k_head, k_dst, k_weight, sin_cache, cos_cache, pos, head_dim, k_eps, lane_id);
            }
        } else {
            if (warp_id >= kv_head_num) continue;
            const __nv_bfloat16* v_head = v + (long long)row * v_row_stride + (long long)warp_id * head_dim;
            __nv_bfloat16* v_dst = v_pool + dst_offset + (long long)warp_id * head_dim;
            if (head_dim == 128) {
                copy_head128(v_head, v_dst, lane_id);
            } else {
                for (int i = lane_id; i < head_dim; i += THREADS_PER_WARP) {
                    v_dst[i] = v_head[i];
                }
            }
        }
    }
}

} // namespace

extern "C" void qkv_norm_rope_scatter_bf16(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* sin_cache,
    const __nv_bfloat16* cos_cache,
    const int* positions,
    __nv_bfloat16* k_pool,
    __nv_bfloat16* v_pool,
    const unsigned int* block_tables,
    const int* seq_positions,
    const int* seq_starts,
    const int* seq_lens,
    int num_tokens,
    int batch,
    int head_num,
    int kv_head_num,
    int head_dim,
    int kv_dim,
    long long q_row_stride,
    long long k_row_stride,
    long long v_row_stride,
    int max_blocks_per_seq,
    int block_size,
    float q_eps,
    float k_eps,
    cudaStream_t stream)
{
    if (batch <= 0 || num_tokens <= 0) return;

    const int max_heads = head_num > kv_head_num ? head_num : kv_head_num;
    if (max_heads > MAX_HEADS_PER_BLOCK) {
        printf("qkv_norm_rope_scatter_bf16: max_heads=%d exceeds %d\n", max_heads, MAX_HEADS_PER_BLOCK);
        return;
    }

    int token_blocks = 1;
    if (batch < 32) {
        const int by_batch = 80 / batch;
        token_blocks = by_batch < 1 ? 1 : (by_batch > 32 ? 32 : by_batch);
    }

    dim3 grid(batch, 3, token_blocks);
    dim3 block(max_heads * THREADS_PER_WARP);
    qkv_norm_rope_scatter_kernel_bf16<<<grid, block, 0, stream>>>(
        q, k, v,
        q_weight, k_weight,
        sin_cache, cos_cache, positions,
        k_pool, v_pool,
        block_tables, seq_positions, seq_starts, seq_lens,
        num_tokens,
        head_num, kv_head_num, head_dim, kv_dim,
        q_row_stride, k_row_stride, v_row_stride,
        max_blocks_per_seq, block_size,
        q_eps, k_eps);
    CUDA_CHECK(cudaGetLastError());
}
