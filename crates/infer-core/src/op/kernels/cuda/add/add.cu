#include "add.h"
#include <cuda_bf16.h>
__global__ void bf16_vec8_add_kernel(
    float4*  __restrict__ c,
    float4*  __restrict__ a,
    float4*  __restrict__ b,
    int N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += stride) {
        auto val_a = reinterpret_cast<__nv_bfloat162*>(&a[i]);
        auto val_b = reinterpret_cast<__nv_bfloat162*>(&b[i]);
        auto val_c = reinterpret_cast<__nv_bfloat162*>(&c[i]);
        val_c[0] = __hadd2(val_a[0], val_b[0]);
        val_c[1] = __hadd2(val_a[1], val_b[1]);
        val_c[2] = __hadd2(val_a[2], val_b[2]);
        val_c[3] = __hadd2(val_a[3], val_b[3]);
    }
}
void add_kernel_bf16x8(
    __nv_bfloat16* c,
    __nv_bfloat16* a,
    __nv_bfloat16* b,
    int num_elements,
    cudaStream_t stream
)
{
    constexpr int threads_per_block = 256;
    int num_sm = 0;
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device);
    const int blocks_per_grid = num_sm * 8;
    auto* c_f4 = reinterpret_cast<float4*>(c);
    auto* a_f4 = reinterpret_cast<float4*>(a);
    auto* b_f4 = reinterpret_cast<float4*>(b);
    bf16_vec8_add_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(c_f4, a_f4, b_f4, num_elements / 8);
}
__global__ void bf16_inplace_vec8_add_kernel(
    float4* a_and_c,
    float4* b,
    int N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < N; i += stride) {
        auto val_a = reinterpret_cast<__nv_bfloat162*>(&a_and_c[i]);
        auto val_b = reinterpret_cast<__nv_bfloat162*>(&b[i]);
        val_a[0] = __hadd2(val_a[0], val_b[0]);
        val_a[1] = __hadd2(val_a[1], val_b[1]);
        val_a[2] = __hadd2(val_a[2], val_b[2]);
        val_a[3] = __hadd2(val_a[3], val_b[3]);
    }
}
void add_inplace_kernel_bf16x8(
    __nv_bfloat16* a_and_c,
    __nv_bfloat16* b,
    int num_elements,
    cudaStream_t stream
)
{
    constexpr int threads_per_block = 256;
    int num_sm = 0;
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device);
    const int blocks_per_grid = num_sm * 8;
    auto* a_and_c_f4 = reinterpret_cast<float4*>(a_and_c);
    auto* b_f4 = reinterpret_cast<float4*>(b);
    bf16_inplace_vec8_add_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(a_and_c_f4, b_f4, num_elements / 8);
}

__global__ void add_kernel_float2(float2* c, const float2* a, const float2* b, int num_float2_elements) {
    // 我们的 grid-stride loop 是以 float2 为单位的
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_float2_elements; i += stride) {
        float2 val_a = a[i];
        float2 val_b = b[i];

        c[i].x = val_a.x + val_b.x;
        c[i].y = val_a.y + val_b.y;
    }
}

__global__ void add_inplace_kernel_float2(
    float2* a_and_c,
    const float2* b,
    int num_float2_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < num_float2_elements; i += stride) {
        a_and_c[i].x += b[i].x;
        a_and_c[i].y += b[i].y;
    }
}

void add_kernel_float2_forward(
    float* c,
    const float* a,
    const float* b,
    int num_elements,
    cudaStream_t stream
) {
    // 元素总数要除以 2，因为我们现在以 float2 为单位
    int num_float2_elements = num_elements / 2;

    const int threads_per_block = 256;
    int num_sm = 0;
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device);
    const int blocks_per_grid = num_sm * 8;
    float2* c_f2 = reinterpret_cast<float2*>(c);
    const float2* a_f2 = reinterpret_cast<const float2*>(a);
    const float2* b_f2 = reinterpret_cast<const float2*>(b);

    // --- 启动内核 ---
    if (stream){
        add_kernel_float2<<<blocks_per_grid, threads_per_block, 0, stream>>>(c_f2, a_f2, b_f2, num_float2_elements);
    }else{
        add_kernel_float2<<<blocks_per_grid, threads_per_block>>>(c_f2, a_f2, b_f2, num_float2_elements);
    }
}

void add_inplace_kernel_float2_forward(
    float* a_and_c,
    const float* b,
    int num_elements,
    cudaStream_t stream
) {
    int num_float2_elements = num_elements / 2;
    const int threads_per_block = 256;
    int num_sm = 0;
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device);
    const int blocks_per_grid = num_sm * 8;

    float2* a_and_c_f2 = reinterpret_cast<float2*>(a_and_c);
    const float2* b_f2 = reinterpret_cast<const float2*>(b);

    add_inplace_kernel_float2<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        a_and_c_f2, b_f2, num_float2_elements
    );
    
}