#include <cub/block/block_reduce.cuh>
#include "rmsnorm.h"
#include <cuda_bf16.h>
// 定义一个转换器 Union，自动处理对齐
union Bfloat16Pack {
    float4 f4;
    __nv_bfloat162 bf162[4];
};

__global__ void rmsnorm_bf16x8(
    __nv_bfloat16* __restrict__ output, 
    const __nv_bfloat16* __restrict__ input, 
    const __nv_bfloat16* __restrict__ weight, 
    int dim, 
    float eps // 注意：这里保持 float，精度更好
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int offset = bid * dim;

    // 强制转为 float4 指针读取 (要求 input/output 首地址 16 字节对齐)
    const float4* input_pack = reinterpret_cast<const float4*>(input + offset);
    float4* output_pack = reinterpret_cast<float4*>(output + offset);
    const float4* weight_pack = reinterpret_cast<const float4*>(weight);

    const int pack_num = dim / 8;
    float sum = 0.0f;

    // --- Phase 1: 平方和 ---
    for (int i = tid; i < pack_num; i += blockDim.x) {
        // 1. 读取 128 bit 数据
        float4 load_val = input_pack[i];
        
        // 2. [修复] 使用 union 进行安全的类型转换
        Bfloat16Pack pack;
        pack.f4 = load_val;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // pack.bf162[j] 直接访问寄存器
            __nv_bfloat162 res = __hmul2(pack.bf162[j], pack.bf162[j]);
            sum += __low2float(res) + __high2float(res);
        }
    }

    // --- Phase 2: Reduce ---
    using BlockReduce = cub::BlockReduce<float, 128>;
    __shared__ typename BlockReduce::TempStorage temp;
    
    // BlockReduce 只在 Thread 0 返回有效 Sum，其他线程可能是未定义的
    float block_sum = BlockReduce(temp).Sum(sum);

    __shared__ float shared_inv_scale;
    if (threadIdx.x == 0) {
        // RMSNorm 公式: 1 / sqrt(mean(x^2) + eps)
        shared_inv_scale = rsqrtf(block_sum / float(dim) + eps);
    }
    __syncthreads();

    const __nv_bfloat16 inv_scale_bf16 = __float2bfloat16(shared_inv_scale);
    const __nv_bfloat162 scale_2 = {inv_scale_bf16, inv_scale_bf16};

    // --- Phase 3: 应用 Norm 和 Weight ---
    for (int i = tid; i < pack_num; i += blockDim.x) {
        float4 in_val = input_pack[i];
        float4 wei_val = weight_pack[i];

        // [修复] 使用 union
        Bfloat16Pack in_pack_u, wei_pack_u, out_pack_u;
        in_pack_u.f4 = in_val;
        wei_pack_u.f4 = wei_val;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Norm: x * scale
            __nv_bfloat162 norm = __hmul2(in_pack_u.bf162[j], scale_2);
            // Output: norm * weight
            out_pack_u.bf162[j] = __hmul2(norm, wei_pack_u.bf162[j]);
        }

        // 写回
        output_pack[i] = out_pack_u.f4;
    }
}

void rmsnorm_kernel_cu_bf16x8(__nv_bfloat16* output, __nv_bfloat16* input, __nv_bfloat16* weight, int row, int dim, float eps, cudaStream_t stream) {
    // 安全检查：dim 必须是 8 的倍数，否则 pack_num 会截断导致计算错误
    if (dim % 8 != 0) {
        printf("Error: dim must be multiple of 8\n");
        return;
    }
    constexpr int threads_num = 128;
    rmsnorm_bf16x8<<<row, threads_num, 0, stream>>>(output, input, weight, dim, eps);
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
