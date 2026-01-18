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
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);
    
    // --- 2. 检查前置条件 ---
    let num_elements = input_output_x.num_elements();
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
            let y_ptr = input_y.as_f32()?.buffer().as_ptr() as *const f32;
            let x_ptr = input_output_x.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;

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
            let y_ptr = input_y.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let x_ptr = input_output_x.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;

            unsafe {
                swiglu_inplace_cu_bf16x8(
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