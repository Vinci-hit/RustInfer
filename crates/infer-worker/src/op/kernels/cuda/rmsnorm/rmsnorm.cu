#include <cub/block/block_reduce.cuh>
#include "rmsnorm.h"
#include <cuda_bf16.h>
__global__ void rmsnorm_bf16_optimized(
    __nv_bfloat16* __restrict__ output, 
    const __nv_bfloat16* __restrict__ input, 
    const __nv_bfloat16* __restrict__ weight, 
    int dim, 
    float eps
) {
    // 使用外部传入的 block_sum 或在内部做
    const int offset = blockIdx.x * dim;
    const int tid = threadIdx.x;

    // 假设 dim 是 8 的倍数，直接使用 reinterpret_cast 提升访存效率
    const float4* in_ptr = reinterpret_cast<const float4*>(input + offset);
    const float4* weight_ptr = reinterpret_cast<const float4*>(weight);
    float4* out_ptr = reinterpret_cast<float4*>(output + offset);

    float sum = 0.0f;
    const int loops = dim / (blockDim.x * 8); 

    // 1. 计算平方和 (利用 bf162 指令提升吞吐)
    for (int i = tid; i < dim / 8; i += blockDim.x) {
        float4 tmp_in = in_ptr[i];
        __nv_bfloat162* b162_vals = reinterpret_cast<__nv_bfloat162*>(&tmp_in);
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // __hmul2 在较新 GPU 上有专用硬件指令
            __nv_bfloat162 squared = __hmul2(b162_vals[j], b162_vals[j]);
            sum += (__bfloat162float(squared.x) + __bfloat162float(squared.y));
        }
    }

    // 2. Block Reduce
    typedef cub::BlockReduce<float, 128> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float total_sum = BlockReduce(temp_storage).Sum(sum);

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(total_sum / float(dim) + eps);
    }
    __syncthreads();

    // 3. 应用缩放并写回
    float f_inv_rms = inv_rms;
    for (int i = tid; i < dim / 8; i += blockDim.x) {
        float4 tmp_in = in_ptr[i];
        float4 tmp_weight = weight_ptr[i];
        
        __nv_bfloat162* in_b162 = reinterpret_cast<__nv_bfloat162*>(&tmp_in);
        __nv_bfloat162* weight_b162 = reinterpret_cast<__nv_bfloat162*>(&tmp_weight);
        
        __nv_bfloat162 scale_bf162 = __floats2bfloat162_rn(f_inv_rms, f_inv_rms);
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // x = (x * inv_rms) * weight
            in_b162[j] = __hmul2(__hmul2(in_b162[j], scale_bf162), weight_b162[j]);
        }
        out_ptr[i] = tmp_in;
    }
}

void rmsnorm_kernel_cu_bf16x8(__nv_bfloat16* output, __nv_bfloat16* input, __nv_bfloat16* weight, int row, int dim, float eps, cudaStream_t stream) {
    constexpr int threads_num = 128;
    rmsnorm_bf16_optimized<<<row, threads_num, 0, stream>>>(output, input, weight, dim, eps);
}
__global__ void row_rmsnorm_f32_dim(float* output, float* input, float* weight, int row, int dim, float eps){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    float * block_in = input + bid * dim;
    float * block_out = output + bid * dim;
    
    constexpr int pack_size = 4;
    const int pack_num = dim / pack_size;
    const int pack_off = pack_num * pack_size;
    const int remain_elements = dim - pack_off;
    float sum = 0.0f;
    float4* in_pack = reinterpret_cast<float4*>(block_in);
        
    for (int i = tid; i < pack_num; i+=blockDim.x)
    {
        float4 in_float4 = *(in_pack + i);
        sum += in_float4.x * in_float4.x;
        sum += in_float4.y * in_float4.y;
        sum += in_float4.z * in_float4.z;
        sum += in_float4.w * in_float4.w;
    }
    if (remain_elements > 0 && tid < remain_elements){
        int idx = pack_off + tid;
        float val = block_in[idx];
        sum += val * val;
    }
    
    __syncthreads();
    using BlockReduce = cub::BlockReduce<float, 128>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;
    const float scale = rsqrtf(sum / static_cast<float>(dim) + eps);
    float4* wei_pack = reinterpret_cast<float4*>(weight);
    float4* out_pack = reinterpret_cast<float4*>(block_out);
    for (int i = tid; i < pack_num; i += blockDim.x) {
        float4 in_float4 = *(in_pack + i);
        float4 wei_float4 = *(wei_pack + i);
        *(out_pack + i) =
            make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                        scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
    }
    for (int i = pack_off + tid; i < dim; i += blockDim.x) {
        block_out[i] = weight[i] * block_in[i] * scale;
    }
}

void rmsnorm_kernel_cu_dim(float* output, float* input, float* weight, int row, int dim, float eps, CUstream_st* stream) {
    constexpr int threads_num = 128;
    row_rmsnorm_f32_dim<<<row, threads_num, 0, stream>>>(output, input,  weight, row, dim, eps);
}
