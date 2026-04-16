#include <cub/block/block_reduce.cuh>
#include "fused_add_rmsnorm.h"
#include <cuda_bf16.h>

// Fused RMSNorm + Residual Add (BF16)
// residual[i] += input[i]; norm_out[i] = rmsnorm(residual[i], weight)
__global__ void fused_add_rmsnorm_bf16_kernel(
    __nv_bfloat16* __restrict__ norm_output,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    int dim,
    float eps
) {
    const int row_offset = blockIdx.x * dim;
    const int tid = threadIdx.x;

    float4* res_f4 = reinterpret_cast<float4*>(residual + row_offset);
    const float4* in_f4 = reinterpret_cast<const float4*>(input + row_offset);
    const float4* w_f4 = reinterpret_cast<const float4*>(weight);
    float4* out_f4 = reinterpret_cast<float4*>(norm_output + row_offset);

    const int vec_count = dim / 8;

    float sum = 0.0f;
    for (int i = tid; i < vec_count; i += blockDim.x) {
        float4 r = res_f4[i];
        float4 inp = in_f4[i];

        __nv_bfloat162* r_b2 = reinterpret_cast<__nv_bfloat162*>(&r);
        const __nv_bfloat162* in_b2 = reinterpret_cast<const __nv_bfloat162*>(&inp);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            r_b2[j] = __hadd2(r_b2[j], in_b2[j]);
            float x0 = __bfloat162float(r_b2[j].x);
            float x1 = __bfloat162float(r_b2[j].y);
            sum += x0 * x0 + x1 * x1;
        }

        res_f4[i] = r;
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float total_sum = BlockReduce(temp_storage).Sum(sum);

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(total_sum / float(dim) + eps);
    }
    __syncthreads();

    float f_inv_rms = inv_rms;
    for (int i = tid; i < vec_count; i += blockDim.x) {
        float4 r = res_f4[i];
        float4 w = w_f4[i];

        __nv_bfloat162* r_b2 = reinterpret_cast<__nv_bfloat162*>(&r);
        __nv_bfloat162* w_b2 = reinterpret_cast<__nv_bfloat162*>(&w);

        __nv_bfloat162 scale = __floats2bfloat162_rn(f_inv_rms, f_inv_rms);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            r_b2[j] = __hmul2(__hmul2(r_b2[j], scale), w_b2[j]);
        }

        out_f4[i] = r;
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
    fused_add_rmsnorm_bf16_kernel<<<rows, threads_num, 0, stream>>>(
        norm_output, residual, input, weight, dim, eps
    );
}
