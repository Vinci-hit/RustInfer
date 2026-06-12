#include "qkv_norm_rope_scatter.h"
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

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
    (void)q; (void)k; (void)v;
    (void)q_weight; (void)k_weight;
    (void)sin_cache; (void)cos_cache; (void)positions;
    (void)k_pool; (void)v_pool; (void)block_tables;
    (void)seq_positions; (void)seq_starts; (void)seq_lens;
    (void)num_tokens; (void)batch;
    (void)head_num; (void)kv_head_num; (void)head_dim; (void)kv_dim;
    (void)q_row_stride; (void)k_row_stride; (void)v_row_stride;
    (void)max_blocks_per_seq; (void)block_size;
    (void)q_eps; (void)k_eps; (void)stream;
    CUDA_CHECK(cudaGetLastError());
}
