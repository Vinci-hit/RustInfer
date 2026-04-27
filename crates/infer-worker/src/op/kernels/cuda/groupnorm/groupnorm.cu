#include "groupnorm.h"
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ============================================================================
// GroupNorm CUDA kernels
//
// Two launch strategies depending on `group_size = channels_per_group * spatial`:
//
//   1. Small groups (group_size <= 8192 elements):
//      One block per (batch, group), 256 threads. Three-pass scalar algorithm.
//      This path is used by DiT / text-encoder GN calls where spatial is small
//      and channels-per-group is small — launch overhead dominates, not BW.
//
//   2. Large groups (group_size > 8192):
//      VAE decoder at 128×128 / 256×256 hits this path. We switch to a
//      vectorized 2-pass Welford kernel:
//        - 512 threads per block, one block per (batch, group).
//        - Reads / writes 8 bf16 (or 4 f32) elements per transaction via
//          16-byte uint4 / float4 loads.
//        - Pass 1: single sweep computes sum and sum-of-squares via Welford
//          style accumulation → `mean`, `rstd` in shared memory.
//        - Pass 2: re-read input, apply `(x - mean) * rstd * w[c] + b[c]`,
//          optionally fused with SiLU (`y = y * sigmoid(y)`).
//      Saves one full tensor read vs. the 3-pass kernel and unlocks ~4× more
//      bandwidth via vectorized LDG/STG.
//
// The groupnorm+silu fused variants avoid an extra 2× tensor traffic (separate
// SiLU kernel would read + write the output again), which is the dominant
// bottleneck in the VAE decoder's late stages at 256×256.
// ============================================================================

// ---- Device helpers -------------------------------------------------------

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float block_reduce_sum_512(float v, float* shm) {
    // warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    if (lane_id == 0) shm[warp_id] = v;
    __syncthreads();
    // final reduce in first warp (we have 512/32 = 16 warp sums)
    if (warp_id == 0) {
        float s = (threadIdx.x < 16) ? shm[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xffffffff, s, offset);
        if (threadIdx.x == 0) shm[0] = s;
    }
    __syncthreads();
    return shm[0];
}

__device__ __forceinline__ float block_reduce_sum_256(float v, float* shm) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    if (lane_id == 0) shm[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        float s = (threadIdx.x < 8) ? shm[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xffffffff, s, offset);
        if (threadIdx.x == 0) shm[0] = s;
    }
    __syncthreads();
    return shm[0];
}

// ============================================================================
// Legacy 3-pass scalar kernels (small groups)
// ============================================================================

__global__ void groupnorm_f32_kernel_small(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int channels, int spatial, int channels_per_group, float eps, int fuse_silu)
{
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int c_start = group_idx * channels_per_group;
    int group_size = channels_per_group * spatial;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* in_base = input + batch_idx * channels * spatial;
    float* out_base = output + batch_idx * channels * spatial;

    // === Pass 1: sum ===
    float local_sum = 0.0f;
    for (int i = tid; i < group_size; i += nthreads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        local_sum += in_base[c * spatial + s];
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);

    __shared__ float warp_sums[8];
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    __shared__ float mean_val, rstd_val;
    if (tid == 0) {
        float total = 0.0f;
        int nwarps = nthreads / 32;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        mean_val = total / (float)group_size;
    }
    __syncthreads();

    // === Pass 2: variance ===
    float local_var = 0.0f;
    for (int i = tid; i < group_size; i += nthreads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        float d = in_base[c * spatial + s] - mean_val;
        local_var += d * d;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_var += __shfl_xor_sync(0xffffffff, local_var, offset);
    if (lane_id == 0) warp_sums[warp_id] = local_var;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        int nwarps = nthreads / 32;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        rstd_val = rsqrtf(total / (float)group_size + eps);
    }
    __syncthreads();

    // === Pass 3: normalize + affine (+ optional silu) ===
    for (int i = tid; i < group_size; i += nthreads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        int idx = c * spatial + s;
        float v = (in_base[idx] - mean_val) * rstd_val;
        v = v * weight[c] + bias[c];
        if (fuse_silu) v = silu_f32(v);
        out_base[idx] = v;
    }
}

__global__ void groupnorm_bf16_kernel_small(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    int channels, int spatial, int channels_per_group, float eps, int fuse_silu)
{
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int c_start = group_idx * channels_per_group;
    int group_size = channels_per_group * spatial;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const __nv_bfloat16* in_base = input + batch_idx * channels * spatial;
    __nv_bfloat16* out_base = output + batch_idx * channels * spatial;

    float local_sum = 0.0f;
    for (int i = tid; i < group_size; i += nthreads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        local_sum += __bfloat162float(in_base[c * spatial + s]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);

    __shared__ float warp_sums[8];
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    __shared__ float mean_val, rstd_val;
    if (tid == 0) {
        float total = 0.0f;
        int nwarps = nthreads / 32;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        mean_val = total / (float)group_size;
    }
    __syncthreads();

    float local_var = 0.0f;
    for (int i = tid; i < group_size; i += nthreads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        float d = __bfloat162float(in_base[c * spatial + s]) - mean_val;
        local_var += d * d;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_var += __shfl_xor_sync(0xffffffff, local_var, offset);
    if (lane_id == 0) warp_sums[warp_id] = local_var;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        int nwarps = nthreads / 32;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        rstd_val = rsqrtf(total / (float)group_size + eps);
    }
    __syncthreads();

    for (int i = tid; i < group_size; i += nthreads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        int idx = c * spatial + s;
        float v = (__bfloat162float(in_base[idx]) - mean_val) * rstd_val;
        v = v * __bfloat162float(weight[c]) + __bfloat162float(bias[c]);
        if (fuse_silu) v = silu_f32(v);
        out_base[idx] = __float2bfloat16(v);
    }
}

// ============================================================================
// Vectorized 2-pass Welford kernels (large groups, spatial % 8 == 0)
//
// One block per (batch, group), 512 threads, each transaction 8 bf16 / 4 f32.
// Pass 1: compute sum & sum-sq in a single sweep (saves 1 read vs. 3-pass).
// Pass 2: load, normalize, affine, optional silu, store.
// Weight/bias are fetched per-element (resolved once per channel slab — cache
// hits after the first warp), so no extra shared-memory staging needed.
// ============================================================================

// Each thread processes `VEC` elements per transaction.
// For bf16: VEC=8 → 16B load (uint4).
// For f32 : VEC=4 → 16B load (float4).

// NOTE: `spatial` is assumed a multiple of VEC at the caller level.
// The launcher falls back to the scalar small-group kernel otherwise.

__global__ void groupnorm_f32_kernel_vec(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int channels, int spatial, int channels_per_group, float eps, int fuse_silu)
{
    constexpr int VEC = 4;
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int c_start = group_idx * channels_per_group;
    int group_size = channels_per_group * spatial;
    int n_vec = group_size / VEC;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const float* in_base = input + (batch_idx * channels + c_start) * spatial;
    float* out_base = output + (batch_idx * channels + c_start) * spatial;
    int spatial_vec = spatial / VEC;

    __shared__ float shm[16];
    __shared__ float mean_val, rstd_val;

    // ---- Pass 1: sum & sum-sq in one sweep ----
    float s1 = 0.0f, s2 = 0.0f;
    for (int idx = tid; idx < n_vec; idx += nthreads) {
        int c = idx / spatial_vec;       // channel within group
        int sv = idx % spatial_vec;      // vector-index within channel
        float4 v = reinterpret_cast<const float4*>(in_base + c * spatial)[sv];
        s1 += v.x + v.y + v.z + v.w;
        s2 += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    float gsum = block_reduce_sum_512(s1, shm);
    __syncthreads();
    float gsq = block_reduce_sum_512(s2, shm);
    if (tid == 0) {
        float inv_n = 1.0f / (float)group_size;
        float mean = gsum * inv_n;
        float var = gsq * inv_n - mean * mean;
        if (var < 0.0f) var = 0.0f;
        mean_val = mean;
        rstd_val = rsqrtf(var + eps);
    }
    __syncthreads();

    // ---- Pass 2: normalize + affine + optional silu ----
    float m = mean_val;
    float rs = rstd_val;
    for (int idx = tid; idx < n_vec; idx += nthreads) {
        int c = idx / spatial_vec;
        int sv = idx % spatial_vec;
        int cg = c_start + c;
        float w = weight[cg];
        float bi = bias[cg];
        float4 v = reinterpret_cast<const float4*>(in_base + c * spatial)[sv];
        float4 o;
        float a = (v.x - m) * rs * w + bi;
        float b = (v.y - m) * rs * w + bi;
        float cc = (v.z - m) * rs * w + bi;
        float d = (v.w - m) * rs * w + bi;
        if (fuse_silu) { a = silu_f32(a); b = silu_f32(b); cc = silu_f32(cc); d = silu_f32(d); }
        o.x = a; o.y = b; o.z = cc; o.w = d;
        reinterpret_cast<float4*>(out_base + c * spatial)[sv] = o;
    }
}

__global__ void groupnorm_bf16_kernel_vec(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    int channels, int spatial, int channels_per_group, float eps, int fuse_silu)
{
    constexpr int VEC = 8;  // 8 bf16 = 16 bytes per transaction
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int c_start = group_idx * channels_per_group;
    int group_size = channels_per_group * spatial;
    int n_vec = group_size / VEC;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const __nv_bfloat16* in_base  = input  + (batch_idx * channels + c_start) * spatial;
    __nv_bfloat16*       out_base = output + (batch_idx * channels + c_start) * spatial;
    int spatial_vec = spatial / VEC;

    __shared__ float shm[16];
    __shared__ float mean_val, rstd_val;

    // ---- Pass 1: sum & sum-sq ----
    float s1 = 0.0f, s2 = 0.0f;
    for (int idx = tid; idx < n_vec; idx += nthreads) {
        int c  = idx / spatial_vec;
        int sv = idx % spatial_vec;
        const uint4* src4 = reinterpret_cast<const uint4*>(in_base + c * spatial);
        uint4 raw = src4[sv];
        __nv_bfloat162 ab = *reinterpret_cast<__nv_bfloat162*>(&raw.x);
        __nv_bfloat162 cd = *reinterpret_cast<__nv_bfloat162*>(&raw.y);
        __nv_bfloat162 ef = *reinterpret_cast<__nv_bfloat162*>(&raw.z);
        __nv_bfloat162 gh = *reinterpret_cast<__nv_bfloat162*>(&raw.w);
        float2 fab = __bfloat1622float2(ab);
        float2 fcd = __bfloat1622float2(cd);
        float2 fef = __bfloat1622float2(ef);
        float2 fgh = __bfloat1622float2(gh);
        s1 += fab.x + fab.y + fcd.x + fcd.y + fef.x + fef.y + fgh.x + fgh.y;
        s2 += fab.x*fab.x + fab.y*fab.y + fcd.x*fcd.x + fcd.y*fcd.y
            + fef.x*fef.x + fef.y*fef.y + fgh.x*fgh.x + fgh.y*fgh.y;
    }
    float gsum = block_reduce_sum_512(s1, shm);
    __syncthreads();
    float gsq  = block_reduce_sum_512(s2, shm);
    if (tid == 0) {
        float inv_n = 1.0f / (float)group_size;
        float mean = gsum * inv_n;
        float var = gsq * inv_n - mean * mean;
        if (var < 0.0f) var = 0.0f;
        mean_val = mean;
        rstd_val = rsqrtf(var + eps);
    }
    __syncthreads();

    // ---- Pass 2: normalize + affine (+ optional silu) ----
    float m  = mean_val;
    float rs = rstd_val;
    for (int idx = tid; idx < n_vec; idx += nthreads) {
        int c  = idx / spatial_vec;
        int sv = idx % spatial_vec;
        int cg = c_start + c;
        float w  = __bfloat162float(weight[cg]);
        float bi = __bfloat162float(bias[cg]);
        const uint4* src4 = reinterpret_cast<const uint4*>(in_base + c * spatial);
        uint4 raw = src4[sv];
        __nv_bfloat162 ab = *reinterpret_cast<__nv_bfloat162*>(&raw.x);
        __nv_bfloat162 cd = *reinterpret_cast<__nv_bfloat162*>(&raw.y);
        __nv_bfloat162 ef = *reinterpret_cast<__nv_bfloat162*>(&raw.z);
        __nv_bfloat162 gh = *reinterpret_cast<__nv_bfloat162*>(&raw.w);
        float2 fab = __bfloat1622float2(ab);
        float2 fcd = __bfloat1622float2(cd);
        float2 fef = __bfloat1622float2(ef);
        float2 fgh = __bfloat1622float2(gh);
        float v0 = (fab.x - m) * rs * w + bi;
        float v1 = (fab.y - m) * rs * w + bi;
        float v2 = (fcd.x - m) * rs * w + bi;
        float v3 = (fcd.y - m) * rs * w + bi;
        float v4 = (fef.x - m) * rs * w + bi;
        float v5 = (fef.y - m) * rs * w + bi;
        float v6 = (fgh.x - m) * rs * w + bi;
        float v7 = (fgh.y - m) * rs * w + bi;
        if (fuse_silu) {
            v0 = silu_f32(v0); v1 = silu_f32(v1);
            v2 = silu_f32(v2); v3 = silu_f32(v3);
            v4 = silu_f32(v4); v5 = silu_f32(v5);
            v6 = silu_f32(v6); v7 = silu_f32(v7);
        }
        __nv_bfloat162 out_ab = __float22bfloat162_rn(make_float2(v0, v1));
        __nv_bfloat162 out_cd = __float22bfloat162_rn(make_float2(v2, v3));
        __nv_bfloat162 out_ef = __float22bfloat162_rn(make_float2(v4, v5));
        __nv_bfloat162 out_gh = __float22bfloat162_rn(make_float2(v6, v7));
        uint4 packed;
        packed.x = *reinterpret_cast<unsigned int*>(&out_ab);
        packed.y = *reinterpret_cast<unsigned int*>(&out_cd);
        packed.z = *reinterpret_cast<unsigned int*>(&out_ef);
        packed.w = *reinterpret_cast<unsigned int*>(&out_gh);
        reinterpret_cast<uint4*>(out_base + c * spatial)[sv] = packed;
    }
}

// ============================================================================
// Launchers
// ============================================================================

static inline bool pick_vec_path(int spatial, int channels_per_group, int vec_elems) {
    // Need both spatial and channel-slab to be vec-aligned so int indexing
    // `c * spatial / VEC` stays lane-perfect.
    return (spatial % vec_elems == 0)
        && (spatial >= vec_elems * 8)
        && (channels_per_group >= 1)
        && ((size_t)channels_per_group * spatial >= (size_t)(vec_elems * 512));
}

static inline void launch_groupnorm_f32(
    float* output, const float* input, const float* weight, const float* bias,
    int batch, int channels, int spatial, int num_groups, float eps,
    int fuse_silu, cudaStream_t stream)
{
    int channels_per_group = channels / num_groups;
    dim3 grid(batch, num_groups);
    if (pick_vec_path(spatial, channels_per_group, 4)) {
        dim3 block(512);
        groupnorm_f32_kernel_vec<<<grid, block, 0, stream>>>(
            output, input, weight, bias, channels, spatial, channels_per_group, eps, fuse_silu);
    } else {
        dim3 block(256);
        groupnorm_f32_kernel_small<<<grid, block, 0, stream>>>(
            output, input, weight, bias, channels, spatial, channels_per_group, eps, fuse_silu);
    }
}

static inline void launch_groupnorm_bf16(
    __nv_bfloat16* output, const __nv_bfloat16* input,
    const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    int batch, int channels, int spatial, int num_groups, float eps,
    int fuse_silu, cudaStream_t stream)
{
    int channels_per_group = channels / num_groups;
    dim3 grid(batch, num_groups);
    if (pick_vec_path(spatial, channels_per_group, 8)) {
        dim3 block(512);
        groupnorm_bf16_kernel_vec<<<grid, block, 0, stream>>>(
            output, input, weight, bias, channels, spatial, channels_per_group, eps, fuse_silu);
    } else {
        dim3 block(256);
        groupnorm_bf16_kernel_small<<<grid, block, 0, stream>>>(
            output, input, weight, bias, channels, spatial, channels_per_group, eps, fuse_silu);
    }
}

// ---- Public C ABI ----------------------------------------------------------

extern "C" void groupnorm_f32_forward(
    float* output, const float* input, const float* weight, const float* bias,
    int batch, int channels, int spatial, int num_groups, float eps, cudaStream_t stream)
{
    launch_groupnorm_f32(output, input, weight, bias, batch, channels, spatial,
                         num_groups, eps, /*fuse_silu=*/0, stream);
}

extern "C" void groupnorm_bf16_forward(
    __nv_bfloat16* output, const __nv_bfloat16* input,
    const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    int batch, int channels, int spatial, int num_groups, float eps, cudaStream_t stream)
{
    launch_groupnorm_bf16(output, input, weight, bias, batch, channels, spatial,
                          num_groups, eps, /*fuse_silu=*/0, stream);
}

extern "C" void groupnorm_silu_f32_forward(
    float* output, const float* input, const float* weight, const float* bias,
    int batch, int channels, int spatial, int num_groups, float eps, cudaStream_t stream)
{
    launch_groupnorm_f32(output, input, weight, bias, batch, channels, spatial,
                         num_groups, eps, /*fuse_silu=*/1, stream);
}

extern "C" void groupnorm_silu_bf16_forward(
    __nv_bfloat16* output, const __nv_bfloat16* input,
    const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    int batch, int channels, int spatial, int num_groups, float eps, cudaStream_t stream)
{
    launch_groupnorm_bf16(output, input, weight, bias, batch, channels, spatial,
                          num_groups, eps, /*fuse_silu=*/1, stream);
}
