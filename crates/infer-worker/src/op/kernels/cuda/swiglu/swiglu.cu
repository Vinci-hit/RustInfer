#include "swiglu.h"
// --- CUDA Kernel (BF16版本) ---
__global__ void swiglu_inplace_kernel_bf16x8(
    float4* __restrict__ input_output_x,
    const float4* __restrict__ input_y,
    int num_float4_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < num_float4_elements; i += stride) {
        float4 x_f4 = input_output_x[i];
        float4 y_f4 = input_y[i];

        __nv_bfloat16* x_bf16 = reinterpret_cast<__nv_bfloat16*>(&x_f4);
        const __nv_bfloat16* y_bf16 = reinterpret_cast<const __nv_bfloat16*>(&y_f4);

        // Compute SwiGLU in FP32 for precision: silu(x) * y = x * sigmoid(x) * y
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float xf = __bfloat162float(x_bf16[j]);
            float yf = __bfloat162float(y_bf16[j]);
            float sigmoid_x = 1.0f / (1.0f + expf(-xf));
            x_bf16[j] = __float2bfloat16(xf * sigmoid_x * yf);
        }

        input_output_x[i] = x_f4;
    }
}


// ======================= 主机端 FFI 函数修改 =======================
void swiglu_inplace_cu_bf16x8(
    const __nv_bfloat16* input_y,
    __nv_bfloat16* input_output_x,
    int num_elements,
    cudaStream_t stream
) {
    int num_float4_elements = num_elements / 8;
    const int threads_per_block = 256;
    // Fixed grid size to avoid runtime cudaGetDevice/cudaDeviceGetAttribute overhead
    const int blocks_per_grid = (num_float4_elements + threads_per_block - 1) / threads_per_block;

    float4* in_out_x_f4 = reinterpret_cast<float4*>(input_output_x);
    const float4* in_y_f4 = reinterpret_cast<const float4*>(input_y);

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
    
    // --- 启动配置 ---
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_float4_elements + threads_per_block - 1) / threads_per_block;
    
    // --- 类型转换 (指针调整) ---
    float4* in_out_x_f4 = reinterpret_cast<float4*>(input_output_x);
    const float4* in_y_f4 = reinterpret_cast<const float4*>(input_y);

    // --- 启动原地内核 ---
    swiglu_inplace_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        in_out_x_f4, in_y_f4, num_float4_elements
    );
}




// ============= FP16 variants (auto-generated from BF16) =============

__global__ void swiglu_inplace_kernel_fp16x8(
    float4* __restrict__ input_output_x,
    const float4* __restrict__ input_y,
    int num_float4_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < num_float4_elements; i += stride) {
        float4 x_f4 = input_output_x[i];
        float4 y_f4 = input_y[i];

        __half* x_fp16 = reinterpret_cast<__half*>(&x_f4);
        const __half* y_fp16 = reinterpret_cast<const __half*>(&y_f4);

        // Compute SwiGLU in FP32 for precision: silu(x) * y = x * sigmoid(x) * y
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float xf = __half2float(x_fp16[j]);
            float yf = __half2float(y_fp16[j]);
            float sigmoid_x = 1.0f / (1.0f + expf(-xf));
            x_fp16[j] = __float2half(xf * sigmoid_x * yf);
        }

        input_output_x[i] = x_f4;
    }
}

extern "C" void swiglu_inplace_cu_fp16x8(
    const __half* input_y,
    __half* input_output_x,
    int num_elements,
    cudaStream_t stream
) {
    int num_float4_elements = num_elements / 8;
    const int threads_per_block = 256;
    // Fixed grid size to avoid runtime cudaGetDevice/cudaDeviceGetAttribute overhead
    const int blocks_per_grid = (num_float4_elements + threads_per_block - 1) / threads_per_block;

    float4* in_out_x_f4 = reinterpret_cast<float4*>(input_output_x);
    const float4* in_y_f4 = reinterpret_cast<const float4*>(input_y);

    swiglu_inplace_kernel_fp16x8<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        in_out_x_f4, in_y_f4, num_float4_elements
    );
}


// ============================================================================
// Strided SwiGLU inplace (BF16): 按 (num_rows, inner_dim) 2D 循环，支持非连续
// row_stride，避免调用方必须 split 独立 buffer。
//   dst = silu(x) * y = x*sigmoid(x) * y, 写入 x 的位置。
//   x_base + seq * x_row_stride + col_offset_x  (col in [0, inner_dim))
//   y_base + seq * y_row_stride + col_offset_y
//   inner_dim 必须是 8 的倍数（用 float4 读写）
// ============================================================================
__global__ void swiglu_inplace_strided_kernel_bf16x8(
    __nv_bfloat16* __restrict__ x_base,         // 整体 base 指针（未加 col_offset）
    const __nv_bfloat16* __restrict__ y_base,
    int num_rows,
    int inner_dim,           // elements per row to process
    int x_row_stride,        // elements per row in x
    int y_row_stride,
    int x_col_offset,
    int y_col_offset
) {
    int seq = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    if (seq >= num_rows) return;

    int num_vec4 = inner_dim / 8;

    __nv_bfloat16* x_row = x_base + (size_t)seq * x_row_stride + x_col_offset;
    const __nv_bfloat16* y_row = y_base + (size_t)seq * y_row_stride + y_col_offset;

    float4* x_f4 = reinterpret_cast<float4*>(x_row);
    const float4* y_f4 = reinterpret_cast<const float4*>(y_row);

    for (int i = idx; i < num_vec4; i += stride) {
        float4 x_v = x_f4[i];
        float4 y_v = y_f4[i];
        __nv_bfloat16* xbh = reinterpret_cast<__nv_bfloat16*>(&x_v);
        const __nv_bfloat16* ybh = reinterpret_cast<const __nv_bfloat16*>(&y_v);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float xf = __bfloat162float(xbh[j]);
            float yf = __bfloat162float(ybh[j]);
            float s = 1.0f / (1.0f + expf(-xf));
            xbh[j] = __float2bfloat16(xf * s * yf);
        }
        x_f4[i] = x_v;
    }
}

extern "C" void swiglu_inplace_strided_cu_bf16x8(
    __nv_bfloat16* x_base,
    const __nv_bfloat16* y_base,
    int num_rows,
    int inner_dim,
    int x_row_stride,
    int y_row_stride,
    int x_col_offset,
    int y_col_offset,
    cudaStream_t stream)
{
    int num_vec4 = inner_dim / 8;
    const int threads = 256;
    int blocks_x = (num_vec4 + threads - 1) / threads;
    dim3 grid(blocks_x, num_rows);
    swiglu_inplace_strided_kernel_bf16x8<<<grid, threads, 0, stream>>>(
        x_base, y_base, num_rows, inner_dim,
        x_row_stride, y_row_stride, x_col_offset, y_col_offset
    );
}

// ============================================================================
// swiglu_packed: gate_up [rows, 2*inter] → out [rows, inter]
//   out[r,d] = silu(gate_up[r, d]) * gate_up[r, inter + d]
// 一个 kernel 替代 2×split_cols + swiglu，省掉 2 次 kernel launch。
// ============================================================================
__global__ void swiglu_packed_kernel_bf16x8(
    const __nv_bfloat16* __restrict__ gate_up,  // [rows, 2*inter]
    __nv_bfloat16* __restrict__ out,            // [rows, inter]
    int rows,
    int inter          // half of total cols
) {
    // 每线程处理 8 个 bf16 (1 个 float4)
    // grid: (ceil(inter/8 / 256), rows)
    const int row = blockIdx.y;
    const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_vec = inter / 8;
    if (vec_idx >= num_vec) return;

    // gate = gate_up[row, 0..inter], up = gate_up[row, inter..2*inter]
    const int row_offset = row * (2 * inter);
    const float4* gate_row = reinterpret_cast<const float4*>(gate_up + row_offset);
    const float4* up_row   = reinterpret_cast<const float4*>(gate_up + row_offset + inter);
    float4* out_row        = reinterpret_cast<float4*>(out + row * inter);

    float4 g = gate_row[vec_idx];
    float4 u = up_row[vec_idx];

    __nv_bfloat16* gp = reinterpret_cast<__nv_bfloat16*>(&g);
    const __nv_bfloat16* up_p = reinterpret_cast<const __nv_bfloat16*>(&u);

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float xf = __bfloat162float(gp[j]);
        float yf = __bfloat162float(up_p[j]);
        float sigmoid_x = 1.0f / (1.0f + expf(-xf));
        gp[j] = __float2bfloat16(xf * sigmoid_x * yf);
    }

    out_row[vec_idx] = g;
}

extern "C" void swiglu_packed_cu_bf16(
    const __nv_bfloat16* gate_up,
    __nv_bfloat16* out,
    int rows,
    int inter,
    cudaStream_t stream)
{
    const int num_vec = inter / 8;
    const int threads = 256;
    int blocks_x = (num_vec + threads - 1) / threads;
    dim3 grid(blocks_x, rows);
    swiglu_packed_kernel_bf16x8<<<grid, threads, 0, stream>>>(
        gate_up, out, rows, inter);
}

