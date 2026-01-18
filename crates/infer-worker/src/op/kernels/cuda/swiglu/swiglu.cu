#include "swiglu.h"
// --- CUDA Kernel (BF16版本) ---
__global__ void swiglu_inplace_kernel_bf16x8(
    float4* __restrict__ input_output_x,
    const float4* __restrict__ input_y,
    int num_float4_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    __nv_bfloat162 one = {1.0, 1.0};
    for (; i < num_float4_elements; i += stride) {
        // a. 读取 x 的原始值
        auto x_bf162_vec4 = reinterpret_cast<__nv_bfloat162*>(&input_output_x[i]);
        // b. 读取 y 的值
        auto y_bf162_vec4 = reinterpret_cast<const __nv_bfloat162*>(&input_y[i]);
        for (int j =0;j<4;j++)
        {
            x_bf162_vec4[j] = x_bf162_vec4[j] * y_bf162_vec4[j] / (one + h2exp(-x_bf162_vec4[j]));
        }
        
    }
}


// ======================= 主机端 FFI 函数修改 =======================
// 函数名和签名被修改以反映其原地操作的特性
// 这是将要从 Rust 调用的 FFI 函数
void swiglu_inplace_cu_bf16x8(
    const __nv_bfloat16* input_y,      // <--- 只读的 y
    __nv_bfloat16* input_output_x, // <--- 可读写的 x
    int num_elements,
    cudaStream_t stream
) {
    int num_float4_elements = num_elements / 8;
    const int threads_per_block = 256;
    int num_sm = 0;
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device);
    const int blocks_per_grid = num_sm * 8;
    // --- 类型转换 (指针调整) ---
    float4* in_out_x_f4 = reinterpret_cast<float4*>(input_output_x);
    const float4* in_y_f4 = reinterpret_cast<const float4*>(input_y);

    // --- 启动原地内核 ---
    swiglu_inplace_kernel_bf16x8<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        in_out_x_f4, in_y_f4, num_float4_elements
    );
}
// ============================================================================
//  在这里填写您的 CUDA C++ 内核实现
// ============================================================================

/*
 * 优化思路建议：
 * 
 * 1.  向量化访存 (float4):
 *     - SwiGLU 是纯粹的逐元素操作，是向量化访存的完美应用场景。
 *     - 将输入和输出指针 reinterpret_cast 为 float4*。
 *     - 内核的 grid-stride loop 将以 float4 为单位进行迭代。
 *     - 这要求元素总数 `num_elements` 必须是 4 的倍数。
 * 
 * 2.  Sigmoid 实现:
 *     - `sigmoid(x) = 1.0f / (1.0f + expf(-x))`
 *     - `expf()` 是 CUDA 内置的、快速的单精度指数函数。
 *     - 当对 `float4` 操作时，你需要对每个分量（.x, .y, .z, .w）分别计算。
 * 
 * 3.  Grid-Stride Loop:
 *     - 采用 grid-stride loop 可以保证内核的健壮性和可扩展性。
 *     - `int idx = blockIdx.x * blockDim.x + threadIdx.x;`
 *     - `int stride = gridDim.x * blockDim.x;`
 *     - `for (int i = idx; i < num_float4_elements; i += stride) { ... }`
 */
__global__ void swiglu_kernel(
    float4* output,
    const float4* input_x,
    const float4* input_y,
    int num_float4_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; i < num_float4_elements; i += stride) {
        float4 x = input_x[i];
        float4 y = input_y[i];

        // 计算 swilu(x)
        x.x = x.x / (1.0f + expf(-x.x));
        x.y = x.y / (1.0f + expf(-x.y));
        x.z = x.z / (1.0f + expf(-x.z));
        x.w = x.w / (1.0f + expf(-x.w));

        // swilu(x) * y
        x.x *= y.x;
        x.y *= y.y;
        x.z *= y.z;
        x.w *= y.w;

        output[i] = x;
    }
}

__global__ void swiglu_inplace_kernel(
    float4* input_output_x, // <--- x 同时是输入和输出
    const float4* input_y,
    int num_float4_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; i < num_float4_elements; i += stride) {
        // a. 读取 x 的原始值
        float4 x_orig = input_output_x[i];
        
        // b. 读取 y 的值
        float4 y = input_y[i];

        // c. 计算 swilu(x) 的结果，可以直接复用 x_orig 变量
        //    (x * sigmoid(x) = x / (1 + exp(-x)))
        x_orig.x = x_orig.x / (1.0f + expf(-x_orig.x));
        x_orig.y = x_orig.y / (1.0f + expf(-x_orig.y));
        x_orig.z = x_orig.z / (1.0f + expf(-x_orig.z));
        x_orig.w = x_orig.w / (1.0f + expf(-x_orig.w));

        // d. swilu(x) * y
        x_orig.x *= y.x;
        x_orig.y *= y.y;
        x_orig.z *= y.z;
        x_orig.w *= y.w;

        // e. 将最终结果写回 x 的原始位置
        input_output_x[i] = x_orig;
    }
}


// ======================= 主机端 FFI 函数修改 =======================
// 函数名和签名被修改以反映其原地操作的特性
// 这是将要从 Rust 调用的 FFI 函数
extern "C" void swiglu_inplace_kernel_cu_fp32x4(
    const float* input_y,      // <--- 只读的 y
    float* input_output_x, // <--- 可读写的 x
    int num_elements,
    cudaStream_t stream
) {
    // 检查：确保元素数量是 4 的倍数，以便使用 float4
    if (num_elements % 4 != 0) {
        // 在生产代码中，这里应该返回一个错误码或记录一个错误
        // 为简单起见，我们直接返回
        return; 
    }
    
    int num_float4_elements = num_elements / 4;
    
    // --- 启动配置 (保持不变) ---
    const int threads_per_block = 256;
    int num_sm = 0;
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, device);
    const int blocks_per_grid = num_sm * 8;
    
    // --- 类型转换 (指针调整) ---
    float4* in_out_x_f4 = reinterpret_cast<float4*>(input_output_x);
    const float4* in_y_f4 = reinterpret_cast<const float4*>(input_y);

    // --- 启动原地内核 ---
    swiglu_inplace_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        in_out_x_f4, in_y_f4, num_float4_elements
    );
}