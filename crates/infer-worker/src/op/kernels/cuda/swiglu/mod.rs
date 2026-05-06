// In src/op/kernels/cuda/swiglu.rs (或者您存放 CUDA 包装器的地方)

use crate::base::error::{Result, Error};
use crate::cuda::CudaConfig;
use crate::tensor::Tensor;

// ============================================================================
//  手动 FFI 声明 (已更新为原地版本)
// ============================================================================
unsafe extern "C" {
    fn swiglu_inplace_kernel_cu_fp32x4(
        input_y: *const f32,
        input_output_x: *mut f32, // <-- x 同时是输入和输出
        num_elements: i32,
        stream: crate::cuda::ffi::cudaStream_t,
    );
    fn swiglu_inplace_cu_bf16x8(
        input_y: *const half::bf16,      // <--- 只读的 y
        input_output_x: *mut half::bf16, // <--- 可读写的 x
        num_elements: i32,
        stream: crate::cuda::ffi::cudaStream_t,
    );

    fn swiglu_inplace_cu_fp16x8(
        input_y: *const half::f16,      // <--- 只读的 y
        input_output_x: *mut half::f16, // <--- 可读写的 x
        num_elements: i32,
        stream: crate::cuda::ffi::cudaStream_t,
    );

    fn swiglu_inplace_strided_cu_bf16x8(
        x_base: *mut half::bf16,
        y_base: *const half::bf16,
        num_rows: i32,
        inner_dim: i32,
        x_row_stride: i32,
        y_row_stride: i32,
        x_col_offset: i32,
        y_col_offset: i32,
        stream: crate::cuda::ffi::cudaStream_t,
    );

    fn swiglu_packed_cu_bf16(
        gate_up: *const half::bf16,
        out: *mut half::bf16,
        rows: i32,
        inter: i32,
        stream: crate::cuda::ffi::cudaStream_t,
    );
}

/// (原地版本) SwiGLU 的 CUDA 内核包装函数。
///
/// 计算 `x = (x * SiLU(x)) * y`，并将结果写回 `x`。
pub fn swiglu(
    input_y: &Tensor,            // <-- 只读的 y
    input_output_x: &mut Tensor, // <-- 可读写的 x
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    
    // --- 1. 获取 stream ---
    let stream = CudaConfig::resolve_stream(cuda_config);
    
    // --- 2. 检查前置条件 ---
    let num_elements = input_output_x.numel();
    // a) 元素数量必须是 8 的倍数 (对于 bf16x8 内核)
    if !num_elements.is_multiple_of(8) {
        return Err(Error::InvalidArgument(
            "CUDA SwiGLU kernel (bf16x8) requires element count to be a multiple of 8.".to_string()
        ).into());
    }

    // --- 3. 根据数据类型分发并调用 FFI ---
    let x_dtype = input_output_x.dtype();
    let y_dtype = input_y.dtype();
    
    // 检查输入数据类型匹配
    if x_dtype != y_dtype {
        return Err(Error::InvalidArgument(
            format!("SwiGLU requires x and y to have the same data type, but got x={:?}, y={:?}",
                    x_dtype, y_dtype)
        ).into());
    }

    match x_dtype {
        crate::base::DataType::F32 => {
            // --- F32 路径 ---
            let y_ptr = input_y.as_f32()?.data_ptr();
            let x_ptr = input_output_x.as_f32_mut()?.data_ptr_mut();

            unsafe {
                swiglu_inplace_kernel_cu_fp32x4(
                    y_ptr,
                    x_ptr,
                    num_elements as i32,
                    stream,
                );
            }
        }
        crate::base::DataType::BF16 => {
            // --- BF16 路径 ---
            let y_ptr = input_y.as_bf16()?.data_ptr();
            let x_ptr = input_output_x.as_bf16_mut()?.data_ptr_mut();

            unsafe {
                swiglu_inplace_cu_bf16x8(
                    y_ptr,
                    x_ptr,
                    num_elements as i32,
                    stream,
                );
            }
        }
        crate::base::DataType::F16 => {
            // --- FP16 路径 ---
            let y_ptr = input_y.as_f16()?.data_ptr();
            let x_ptr = input_output_x.as_f16_mut()?.data_ptr_mut();

            unsafe {
                swiglu_inplace_cu_fp16x8(
                    y_ptr,
                    x_ptr,
                    num_elements as i32,
                    stream,
                );
            }
        }
        _ => {
            return Err(Error::InvalidArgument(
                format!("Unsupported data type for CUDA SwiGLU: {:?}", x_dtype)
            ).into());
        }
    }

    Ok(())
}
/// Strided inplace SwiGLU (BF16)：支持非连续 row_stride + col_offset，避免 split_cols。
/// 参数 x_base / y_base 是 tensor 起点；实际访问位置为
///     x[seq, col] = x_base[seq * x_row_stride + x_col_offset + col]  (col in [0, inner_dim))
#[allow(clippy::too_many_arguments)]
pub unsafe fn swiglu_inplace_strided_bf16(
    x_base: &mut Tensor,
    y_base: &Tensor,
    num_rows: usize,
    inner_dim: usize,
    x_row_stride: usize,
    y_row_stride: usize,
    x_col_offset: usize,
    y_col_offset: usize,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    if !inner_dim.is_multiple_of(8) {
        return Err(Error::InvalidArgument(format!(
            "swiglu_inplace_strided_bf16: inner_dim ({}) must be multiple of 8", inner_dim
        )).into());
    }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let x_ptr = x_base.as_bf16_mut()?.data_ptr_mut();
    let y_ptr = y_base.as_bf16()?.data_ptr();
    unsafe {
        swiglu_inplace_strided_cu_bf16x8(
            x_ptr, y_ptr,
            num_rows as i32, inner_dim as i32,
            x_row_stride as i32, y_row_stride as i32,
            x_col_offset as i32, y_col_offset as i32,
            stream,
        );
    }
    Ok(())
}

/// Packed SwiGLU (BF16): gate_up [rows, 2*inter] → out [rows, inter]
///   out[r,d] = silu(gate_up[r,d]) * gate_up[r, inter+d]
/// `inter` must be a multiple of 8.
pub fn swiglu_packed_bf16(
    gate_up: &Tensor,
    out: &mut Tensor,
    rows: usize,
    inter: usize,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    if !inter.is_multiple_of(8) {
        return Err(Error::InvalidArgument(format!(
            "swiglu_packed_bf16: inter ({}) must be multiple of 8", inter
        )).into());
    }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let gate_up_ptr = gate_up.as_bf16()?.data_ptr();
    let out_ptr = out.as_bf16_mut()?.data_ptr_mut();
    unsafe {
        swiglu_packed_cu_bf16(
            gate_up_ptr,
            out_ptr,
            rows as i32,
            inter as i32,
            stream,
        );
    }
    Ok(())
}