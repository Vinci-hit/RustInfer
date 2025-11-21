#include <cub/block/block_reduce.cuh>
#include "rmsnorm.h"
__global__ void row_rmsnorm_f32_dim(float* output, float* input, float* weight, int row, int dim, float eps){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    if (bid >= row)
    {
        return;
    }
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
