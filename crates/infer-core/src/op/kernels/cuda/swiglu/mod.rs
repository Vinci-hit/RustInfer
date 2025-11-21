// In src/op/kernels/cuda/swiglu.rs (或者您存放 CUDA 包装器的地方)

use crate::base::error::{Result, Error};
use crate::base::DataType; // 需要引入 DataType
use crate::cuda::{self, CudaConfig};
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
    // (如果支持 bf16，在这里添加 bf16 版本的 FFI 声明)
    // fn swiglu_inplace_kernel_cu_bf16x8(...);
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
    // a) 元素数量必须是 4 的倍数 (对于 fp32x4 内核)
    if num_elements % 4 != 0 {
        return Err(Error::InvalidArgument(
            "CUDA SwiGLU kernel (fp32x4) requires element count to be a multiple of 4.".to_string()
        ).into());
    }

    // --- 3. 根据数据类型分发并调用 FFI ---
    //    (这个模式与您之前的 embedding 适配器类似，非常健壮)
    if input_output_x.dtype() == DataType::F32 && input_y.dtype() == DataType::F32 {
        // --- F32 路径 ---
        
        // a. 获取指针
        let y_ptr = input_y.as_f32()?.buffer().as_ptr() as *const f32;
        let x_ptr = input_output_x.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;

        // b. 调用 FFI 函数
        unsafe {
            swiglu_inplace_kernel_cu_fp32x4(
                y_ptr,
                x_ptr,
                num_elements as i32,
                stream,
            );
        }
    } 
    else {
        return Err(Error::InvalidArgument(
            format!("Unsupported dtypes for CUDA SwiGLU: x={:?}, y={:?}",
                    input_output_x.dtype(), input_y.dtype())
        ).into());
    }

    Ok(())
}