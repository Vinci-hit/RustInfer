// RMSNorm CUDA kernels（通用 strided 版本）。
//
// 数据视图：input/output 抽象为 3-D `[outer0, outer1, dim]` 的 strided view，
// 最后一维 dense（element stride=1）；前两维 stride 由调用方提供。Kernel
// 启动时 grid.x = outer0 * outer1，每个 block 处理一个 norm 行。
//
//   row_idx = blockIdx.x
//   o0 = row_idx / outer1
//   o1 = row_idx % outer1
//   row_offset = o0 * stride0 + o1 * stride1   (element 单位)
//   row_in  = input  + row_offset
//   row_out = output + row_offset
//
// 用法举例：
//   dense 2-D `[rows, dim]`     → outer0=rows, outer1=1, stride0=dim, stride1=0
//   1-D `[dim]`                 → outer0=1,    outer1=1, stride0=0, stride1=0
//   dense 3-D `[B, S, dim]`     → outer0=B*S, outer1=1, stride0=dim, stride1=0
//   strided `qkv.narrow(...)`，按 head 切：[T, head_num, head_dim]，stride
//     0=cols, stride1=head_dim → outer0=T, outer1=head_num, stride0=cols,
//     stride1=head_dim
//
// 要求：
//   * dim 是 8 的倍数（half）/ 4 的倍数（f32）
//   * stride0、stride1 都是 8 的倍数（half 路径，保证 float4 16-byte 对齐）
//   * input / output 同 layout（forward 别名 buffer 时 stride 可以不同；
//     in-place 时 output==input 同 stride）
//
// 模板化：half kernel（bf16/fp16）共用一份 `rmsnorm_half_kernel<HalfT>`，f32
// 单独一份；dispatch wrapper 实例化 3 个 extern "C"。

#include <cub/block/block_reduce.cuh>
#include "rmsnorm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

template <class T> struct HalfTraits;
template <> struct HalfTraits<__nv_bfloat16> {
    using Vec2 = __nv_bfloat162;
    static __device__ __forceinline__ Vec2 mul2(Vec2 a, Vec2 b) { return __hmul2(a, b); }
    static __device__ __forceinline__ float2 to_f2(Vec2 v)      { return __bfloat1622float2(v); }
    static __device__ __forceinline__ Vec2  f2_to(float a, float b) { return __floats2bfloat162_rn(a, b); }
};
template <> struct HalfTraits<__half> {
    using Vec2 = half2;
    static __device__ __forceinline__ Vec2 mul2(Vec2 a, Vec2 b) { return __hmul2(a, b); }
    static __device__ __forceinline__ float2 to_f2(Vec2 v)      { return __half22float2(v); }
    static __device__ __forceinline__ Vec2  f2_to(float a, float b) { return __floats2half2_rn(a, b); }
};

template <class HalfT>
__global__ void rmsnorm_half_kernel(
    HalfT* __restrict__ output,
    const HalfT* __restrict__ input,
    const HalfT* __restrict__ weight,
    int dim,
    int outer1,
    long long stride0,
    long long stride1,
    long long out_stride0,
    long long out_stride1,
    float eps)
{
    using Traits = HalfTraits<HalfT>;
    using Vec2   = typename Traits::Vec2;

    const int row     = blockIdx.x;
    const int o0      = (outer1 == 1) ? row : (row / outer1);
    const int o1      = (outer1 == 1) ? 0   : (row % outer1);
    const long long in_off  = (long long)o0 * stride0     + (long long)o1 * stride1;
    const long long out_off = (long long)o0 * out_stride0 + (long long)o1 * out_stride1;

    const int tid = threadIdx.x;

    const float4* in_ptr     = reinterpret_cast<const float4*>(input  + in_off);
    const float4* weight_ptr = reinterpret_cast<const float4*>(weight);
          float4* out_ptr    = reinterpret_cast<      float4*>(output + out_off);

    const int vec_count = dim / 8;

    float sum = 0.0f;
    for (int i = tid; i < vec_count; i += blockDim.x) {
        float4 raw = in_ptr[i];
        Vec2*  v2  = reinterpret_cast<Vec2*>(&raw);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = Traits::to_f2(v2[j]);
            sum += f.x * f.x + f.y * f.y;
        }
    }

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float total = BlockReduce(temp_storage).Sum(sum);

    __shared__ float inv_rms;
    if (tid == 0) inv_rms = rsqrtf(total / float(dim) + eps);
    __syncthreads();

    const Vec2 scale = Traits::f2_to(inv_rms, inv_rms);
    for (int i = tid; i < vec_count; i += blockDim.x) {
        float4 raw_in = in_ptr[i];
        float4 raw_w  = weight_ptr[i];
        Vec2*  in_v2  = reinterpret_cast<Vec2*>(&raw_in);
        Vec2*  w_v2   = reinterpret_cast<Vec2*>(&raw_w);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            in_v2[j] = Traits::mul2(Traits::mul2(in_v2[j], scale), w_v2[j]);
        }
        out_ptr[i] = raw_in;
    }
}

__global__ void rmsnorm_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int dim,
    int outer1,
    long long stride0,
    long long stride1,
    long long out_stride0,
    long long out_stride1,
    float eps)
{
    const int row = blockIdx.x;
    const int o0  = (outer1 == 1) ? row : (row / outer1);
    const int o1  = (outer1 == 1) ? 0   : (row % outer1);
    const long long in_off  = (long long)o0 * stride0     + (long long)o1 * stride1;
    const long long out_off = (long long)o0 * out_stride0 + (long long)o1 * out_stride1;

    const int tid = threadIdx.x;

    const float4* in_ptr  = reinterpret_cast<const float4*>(input  + in_off);
    const float4* w_ptr   = reinterpret_cast<const float4*>(weight);
          float4* out_ptr = reinterpret_cast<      float4*>(output + out_off);

    const int vec_count = dim / 4;

    float sum = 0.0f;
    for (int i = tid; i < vec_count; i += blockDim.x) {
        float4 v = in_ptr[i];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    using BlockReduce = cub::BlockReduce<float, 128>;
    __shared__ typename BlockReduce::TempStorage temp;
    float total = BlockReduce(temp).Sum(sum);

    __shared__ float inv_rms;
    if (tid == 0) inv_rms = rsqrtf(total / float(dim) + eps);
    __syncthreads();
    const float scale = inv_rms;

    for (int i = tid; i < vec_count; i += blockDim.x) {
        float4 v = in_ptr[i];
        float4 w = w_ptr[i];
        out_ptr[i] = make_float4(
            v.x * w.x * scale,
            v.y * w.y * scale,
            v.z * w.z * scale,
            v.w * w.w * scale);
    }
}

} // namespace

extern "C" void rmsnorm_kernel_cu_bf16x8(
    __nv_bfloat16* output, __nv_bfloat16* input, __nv_bfloat16* weight,
    int outer0, int outer1, int dim,
    long long in_stride0, long long in_stride1,
    long long out_stride0, long long out_stride1,
    float eps, cudaStream_t stream)
{
    constexpr int threads = 256;
    const int rows = outer0 * outer1;
    rmsnorm_half_kernel<__nv_bfloat16><<<rows, threads, 0, stream>>>(
        output, input, weight, dim, outer1,
        in_stride0, in_stride1, out_stride0, out_stride1, eps);
}

extern "C" void rmsnorm_kernel_cu_fp16x8(
    __half* output, __half* input, __half* weight,
    int outer0, int outer1, int dim,
    long long in_stride0, long long in_stride1,
    long long out_stride0, long long out_stride1,
    float eps, cudaStream_t stream)
{
    constexpr int threads = 256;
    const int rows = outer0 * outer1;
    rmsnorm_half_kernel<__half><<<rows, threads, 0, stream>>>(
        output, input, weight, dim, outer1,
        in_stride0, in_stride1, out_stride0, out_stride1, eps);
}

extern "C" void rmsnorm_kernel_cu_dim(
    float* output, float* input, float* weight,
    int outer0, int outer1, int dim,
    long long in_stride0, long long in_stride1,
    long long out_stride0, long long out_stride1,
    float eps, cudaStream_t stream)
{
    constexpr int threads = 128;
    const int rows = outer0 * outer1;
    rmsnorm_f32_kernel<<<rows, threads, 0, stream>>>(
        output, input, weight, dim, outer1,
        in_stride0, in_stride1, out_stride0, out_stride1, eps);
}
