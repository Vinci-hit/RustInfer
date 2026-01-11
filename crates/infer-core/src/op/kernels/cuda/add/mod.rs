use crate::base::error::{Result,Error};
use crate::tensor::{Tensor, TypedTensor};
use crate::cuda::{self, CudaConfig};

unsafe extern "C" {
    fn add_kernel_bf16x8(
        c: *mut half::bf16,
        a: *const half::bf16,
        b: *const half::bf16,
        num_elements: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn add_inplace_kernel_bf16x8(
        a_and_c: *mut half::bf16,
        b: *const half::bf16,
        num_elements: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    /// @brief 执行 C = A + B (out-of-place)
    ///        (函数签名必须与 add.h 中定义的完全一致)
    fn add_kernel_float2_forward(
        c: *mut f32,
        a: *const f32,
        b: *const f32,
        num_elements: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    /// @brief 执行 A = A + B (in-place)
    ///        (函数签名必须与 add.h 中定义的完全一致)
    fn add_inplace_kernel_float2_forward(
        a_and_c: *mut f32,
        b: *const f32,
        num_elements: i32,
        stream: cuda::ffi::cudaStream_t,
    );
}


pub fn add(
    input_a: &Tensor,
    input_b: &Tensor,
    output_c: &mut Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    // 1. 数据类型校验和分发
    let dtype = input_a.dtype();
    
    // 确保所有张量数据类型一致
    if input_a.dtype() != input_b.dtype() || input_a.dtype() != output_c.dtype() {
        return Err(Error::InvalidArgument(format!(
            "All tensors must have the same data type for addition. A: {:?}, B: {:?}, C: {:?}",
            input_a.dtype(), input_b.dtype(), output_c.dtype()
        )).into());
    }

    // 2. 严格的维度和长度校验
    let dims_a = input_a.shape();
    let dims_b = input_b.shape();
    let dims_c = output_c.shape();

    // 形状校验：所有 Tensor 的形状必须完全匹配
    if dims_a != dims_b || dims_a != dims_c {
        return Err(Error::InvalidArgument(format!(
            "Tensor shapes must match for addition. A: {:?}, B: {:?}, C: {:?}",
            dims_a, dims_b, dims_c
        )).into());
    }

    // 长度校验
    let num_elements = input_a.num_elements();
    if num_elements == 0 {
        return Err(Error::InvalidArgument("Input Tensor cannot be empty".into()).into());
    }

    // 3. 对齐校验 (仅对F32版本需要)
    if let Some(last_dim) = dims_a.last()
        && dtype == crate::base::DataType::F32 {
            assert!(last_dim % 2 == 0, "add_kernel_float2_forward requires last dim to be divisible by 2 (for float2 packing).");
        }
    
    // 4. 获取 CUDA stream
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);

    // 5. 根据数据类型调用相应的 FFI 函数
    match dtype {
        crate::base::DataType::F32 => {
            let input_a_typed = input_a.as_f32()?;
            let input_b_typed = input_b.as_f32()?;
            let output_c_typed: &mut TypedTensor<f32> = output_c.as_f32_mut()?;
            
            let a_ptr = input_a_typed.buffer().as_ptr() as *const f32;
            let b_ptr = input_b_typed.buffer().as_ptr() as *const f32;
            let c_ptr = output_c_typed.buffer_mut().as_mut_ptr() as *mut f32;

            unsafe {
                add_kernel_float2_forward(
                    c_ptr,
                    a_ptr,
                    b_ptr,
                    num_elements as i32,
                    stream,
                );
            }
        }
        crate::base::DataType::BF16 => {
            let input_a_typed = input_a.as_bf16()?;
            let input_b_typed = input_b.as_bf16()?;
            let output_c_typed: &mut TypedTensor<half::bf16> = output_c.as_bf16_mut()?;
            
            let a_ptr = input_a_typed.buffer().as_ptr() as *const half::bf16;
            let b_ptr = input_b_typed.buffer().as_ptr() as *const half::bf16;
            let c_ptr = output_c_typed.buffer_mut().as_mut_ptr() as *mut half::bf16;

            unsafe {
                add_kernel_bf16x8(
                    c_ptr,
                    a_ptr,
                    b_ptr,
                    num_elements as i32,
                    stream,
                );
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported data type for add CUDA kernel: {:?}", dtype
            )).into());
        }
    }
    
    Ok(())
}
/// 执行 A = A + B (in-place)
///
/// 统一接口顺序: Input/Output A, Input B, Output Placeholder/Unused, CudaConfig
pub fn add_inplace(
    input_output_a: &mut Tensor,
    input_b: &Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    // 1. 数据类型校验和分发
    let dtype = input_output_a.dtype();
    
    // 确保所有张量数据类型一致
    if input_output_a.dtype() != input_b.dtype() {
        return Err(Error::InvalidArgument(format!(
            "All tensors must have the same data type for in-place addition. A: {:?}, B: {:?}",
            input_output_a.dtype(), input_b.dtype()
        )).into());
    }

    // 2. 严格的维度和长度校验
    let dims_a = input_output_a.shape();
    let dims_b = input_b.shape();

    // 形状校验：A 和 B 的形状必须完全匹配
    if dims_a != dims_b {
        return Err(Error::InvalidArgument(format!(
            "Tensor shapes must match for in-place addition. A: {:?}, B: {:?}",
            dims_a, dims_b
        )).into());
    }
    
    let num_elements = input_output_a.num_elements();
    if num_elements == 0 {
        return Err(Error::InvalidArgument("Input Tensor cannot be empty".into()).into());
    }

    // 3. 对齐校验 (仅对F32版本需要)
    if let Some(last_dim) = dims_a.last()
        && dtype == crate::base::DataType::F32 {
            assert!(last_dim % 2 == 0, "add_inplace_kernel_float2_forward requires last dim to be divisible by 2.");
        }

    // 4. 获取 CUDA stream
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);

    // 5. 根据数据类型调用相应的 FFI 函数
    match dtype {
        crate::base::DataType::F32 => {
            let input_output_a_typed: &mut TypedTensor<f32> = input_output_a.as_f32_mut()?;
            let input_b_typed = input_b.as_f32()?;
            
            let a_and_c_ptr = input_output_a_typed.buffer_mut().as_mut_ptr() as *mut f32;
            let b_ptr = input_b_typed.buffer().as_ptr() as *const f32;

            unsafe {
                add_inplace_kernel_float2_forward(
                    a_and_c_ptr, // FFI 参数 a_and_c
                    b_ptr,       // FFI 参数 b
                    num_elements as i32,
                    stream,
                );
            }
        }
        crate::base::DataType::BF16 => {
            let input_output_a_typed: &mut TypedTensor<half::bf16> = input_output_a.as_bf16_mut()?;
            let input_b_typed = input_b.as_bf16()?;
            
            let a_and_c_ptr = input_output_a_typed.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let b_ptr = input_b_typed.buffer().as_ptr() as *const half::bf16;

            unsafe {
                add_inplace_kernel_bf16x8(
                    a_and_c_ptr, // FFI 参数 a_and_c
                    b_ptr,       // FFI 参数 b
                    num_elements as i32,
                    stream,
                );
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported data type for add_inplace CUDA kernel: {:?}", dtype
            )).into());
        }
    }
    
    Ok(())
}