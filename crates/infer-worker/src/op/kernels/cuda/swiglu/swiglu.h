#include <cuda_runtime.h>
#include <cuda_bf16.h>
#ifdef __cplusplus
extern "C" {
#endif

/// @brief 执行 SwiGLU: Output = (X * sigmoid(X)) * Y, 采用 float4 向量化访存。
///
/// @param output     输出张量 Output 的设备指针。
/// @param input_x    输入张量 X 的设备指针。
/// @param input_y    输入张量 Y 的设备指针。
/// @param num_elements 张量中的元素总数 (以 float 为单位)。
/// @param stream     CUDA stream。
void swiglu_inplace_kernel_cu_fp32x4(
    const float* input_y,      // <--- 只读的 y
    float* input_output_x, // <--- 可读写的 x
    int num_elements,
    cudaStream_t stream
);

void swiglu_inplace_cu_bf16x8(
    const __nv_bfloat16* input_y,      // <--- 只读的 y
    __nv_bfloat16* input_output_x, // <--- 可读写的 x
    int num_elements,
    cudaStream_t stream
);

/// Strided inplace SwiGLU (BF16): 支持非连续 row_stride / col_offset（element 单位）。
/// 处理 num_rows × inner_dim 个 bf16 元素（inner_dim 必须是 8 的倍数）。
void swiglu_inplace_strided_cu_bf16x8(
    __nv_bfloat16* x_base,
    const __nv_bfloat16* y_base,
    int num_rows,
    int inner_dim,
    int x_row_stride,
    int y_row_stride,
    int x_col_offset,
    int y_col_offset,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif