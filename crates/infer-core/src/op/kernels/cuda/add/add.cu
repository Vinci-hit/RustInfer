#include "add.h"

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