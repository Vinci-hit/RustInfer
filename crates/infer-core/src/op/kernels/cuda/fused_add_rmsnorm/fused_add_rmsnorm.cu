#include "fused_add_rmsnorm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Warp-level reduction via shuffle (no shared memory needed within a warp)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused RMSNorm + Residual Add (BF16) - Optimized
// residual[i] += input[i];  norm_out[i] = rmsnorm(residual[i], weight)
template <int BLOCK_DIM_X>
__global__ void fused_add_rmsnorm_bf16_kernel(
    __nv_bfloat16* __restrict__ norm_output,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    int dim,
    float eps
) {
    constexpr int NUM_WARPS = BLOCK_DIM_X / 32;

    const int row_offset = blockIdx.x * dim;
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    // 128-bit vectorized pointers (8 bf16 = 1 uint4/float4)
    const uint4* res_u4 = reinterpret_cast<const uint4*>(residual + row_offset);
    uint4* res_out_u4 = reinterpret_cast<uint4*>(residual + row_offset);
    const uint4* in_u4 = reinterpret_cast<const uint4*>(input + row_offset);
    const uint4* w_u4 = reinterpret_cast<const uint4*>(weight);
    uint4* out_u4 = reinterpret_cast<uint4*>(norm_output + row_offset);

    const int vec_count = dim >> 3;

    // Pass 1: residual += input, compute sum of squares
    float sum = 0.0f;
    for (int i = tid; i < vec_count; i += BLOCK_DIM_X) {
        uint4 r_raw = res_u4[i];
        uint4 inp_raw = in_u4[i];

        __nv_bfloat162* r_b2 = reinterpret_cast<__nv_bfloat162*>(&r_raw);
        const __nv_bfloat162* in_b2 = reinterpret_cast<const __nv_bfloat162*>(&inp_raw);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            r_b2[j] = __hadd2(r_b2[j], in_b2[j]);
            float2 f2 = __bfloat1622float2(r_b2[j]);
            sum = __fmaf_rn(f2.x, f2.x, sum);
            sum = __fmaf_rn(f2.y, f2.y, sum);
        }

        res_out_u4[i] = r_raw;
    }

    // Block-level warp shuffle reduction (minimal shared memory)
    sum = warp_reduce_sum(sum);

    __shared__ float smem[NUM_WARPS + 1];
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            smem[NUM_WARPS] = rsqrtf(val / float(dim) + eps);
        }
    }
    __syncthreads();

    // Pass 2: norm_output = residual * inv_rms * weight
    float f_inv_rms = smem[NUM_WARPS];
    __nv_bfloat162 scale = __floats2bfloat162_rn(f_inv_rms, f_inv_rms);

    for (int i = tid; i < vec_count; i += BLOCK_DIM_X) {
        uint4 r_raw = res_u4[i];
        uint4 w_raw = w_u4[i];

        __nv_bfloat162* r_b2 = reinterpret_cast<__nv_bfloat162*>(&r_raw);
        __nv_bfloat162* w_b2 = reinterpret_cast<__nv_bfloat162*>(&w_raw);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            r_b2[j] = __hmul2(__hmul2(r_b2[j], scale), w_b2[j]);
        }

        out_u4[i] = r_raw;
    }
}

void fused_add_rmsnorm_kernel_cu_bf16(
    __nv_bfloat16* norm_output,
    __nv_bfloat16* residual,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    int rows, int dim, float eps,
    cudaStream_t stream
) {
    constexpr int threads_num = 256;
    fused_add_rmsnorm_bf16_kernel<threads_num><<<rows, threads_num, 0, stream>>>(
        norm_output, residual, input, weight, dim, eps
    );
}





// ============= FP16 variants (auto-generated from BF16) =============

template <int BLOCK_DIM_X>
__global__ void fused_add_rmsnorm_fp16_kernel(
    __half* __restrict__ norm_output,
    __half* __restrict__ residual,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    int dim,
    float eps
) {
    constexpr int NUM_WARPS = BLOCK_DIM_X / 32;

    const int row_offset = blockIdx.x * dim;
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    // 128-bit vectorized pointers (8 bf16 = 1 uint4/float4)
    const uint4* res_u4 = reinterpret_cast<const uint4*>(residual + row_offset);
    uint4* res_out_u4 = reinterpret_cast<uint4*>(residual + row_offset);
    const uint4* in_u4 = reinterpret_cast<const uint4*>(input + row_offset);
    const uint4* w_u4 = reinterpret_cast<const uint4*>(weight);
    uint4* out_u4 = reinterpret_cast<uint4*>(norm_output + row_offset);

    const int vec_count = dim >> 3;

    // Pass 1: residual += input, compute sum of squares
    float sum = 0.0f;
    for (int i = tid; i < vec_count; i += BLOCK_DIM_X) {
        uint4 r_raw = res_u4[i];
        uint4 inp_raw = in_u4[i];

        half2* r_b2 = reinterpret_cast<half2*>(&r_raw);
        const half2* in_b2 = reinterpret_cast<const half2*>(&inp_raw);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            r_b2[j] = __hadd2(r_b2[j], in_b2[j]);
            float2 f2 = __half22float2(r_b2[j]);
            sum = __fmaf_rn(f2.x, f2.x, sum);
            sum = __fmaf_rn(f2.y, f2.y, sum);
        }

        res_out_u4[i] = r_raw;
    }

    // Block-level warp shuffle reduction (minimal shared memory)
    sum = warp_reduce_sum(sum);

    __shared__ float smem[NUM_WARPS + 1];
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            smem[NUM_WARPS] = rsqrtf(val / float(dim) + eps);
        }
    }
    __syncthreads();

    // Pass 2: norm_output = residual * inv_rms * weight
    float f_inv_rms = smem[NUM_WARPS];
    half2 scale = __floats2half2_rn(f_inv_rms, f_inv_rms);

    for (int i = tid; i < vec_count; i += BLOCK_DIM_X) {
        uint4 r_raw = res_u4[i];
        uint4 w_raw = w_u4[i];

        half2* r_b2 = reinterpret_cast<half2*>(&r_raw);
        half2* w_b2 = reinterpret_cast<half2*>(&w_raw);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            r_b2[j] = __hmul2(__hmul2(r_b2[j], scale), w_b2[j]);
        }

        out_u4[i] = r_raw;
    }
}

extern "C" void fused_add_rmsnorm_kernel_cu_fp16(
    __half* norm_output,
    __half* residual,
    const __half* input,
    const __half* weight,
    int rows, int dim, float eps,
    cudaStream_t stream
) {
    constexpr int threads_num = 256;
    fused_add_rmsnorm_fp16_kernel<threads_num><<<rows, threads_num, 0, stream>>>(
        norm_output, residual, input, weight, dim, eps
    );
}

