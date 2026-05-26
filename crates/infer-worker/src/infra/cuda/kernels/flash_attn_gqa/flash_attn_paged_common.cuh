#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cmath>

// Common helpers for paged attention kernels.
// K/V pool layout: [num_blocks, block_size, num_kv_heads, head_dim].

template <class Elem> struct PagedET;

template <> struct PagedET<__nv_bfloat16> {
    __device__ __forceinline__ static float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
    __device__ __forceinline__ static __nv_bfloat16 from_f(float x) { return __float2bfloat16_rn(x); }
};

template <> struct PagedET<__half> {
    __device__ __forceinline__ static float to_f(__half x) { return __half2float(x); }
    __device__ __forceinline__ static __half from_f(float x) { return __float2half_rn(x); }
};

template <typename Elem>
__device__ __forceinline__ const Elem* paged_kv_row(
    const Elem* pool,
    uint32_t physical_block,
    int block_off,
    int block_size,
    int num_kv_heads,
    int kv_head,
    int head_dim)
{
    return pool + ((static_cast<size_t>(physical_block) * block_size + block_off) * num_kv_heads + kv_head) * head_dim;
}
