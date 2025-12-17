use crate::base::error::{Result,Error};
use crate::tensor::{Tensor, TypedTensor};
use crate::cuda::{self, CudaConfig};

unsafe extern "C" {
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
    _cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    // 1. 获取 TypedTensor 并进行类型校验
    let input_a_typed = input_a.as_f32()?;
    let input_b_typed = input_b.as_f32()?;
    let output_c_typed: &mut TypedTensor<f32> = output_c.as_f32_mut()?;

    // 2. 严格的维度和长度校验
    let dims_a = input_a_typed.shape();
    let dims_b = input_b_typed.shape();
    let dims_c = output_c_typed.shape();

    // 形状校验：所有 Tensor 的形状必须完全匹配
    if dims_a != dims_b || dims_a != dims_c {
        return Err(Error::InvalidArgument(format!(
            "Tensor shapes must match for addition. A: {:?}, B: {:?}, C: {:?}",
            dims_a, dims_b, dims_c
        )).into());
    }

    // 长度校验
    let num_elements = input_a_typed.num_elements();
    if num_elements == 0 {
        return Err(Error::InvalidArgument("Input Tensor cannot be empty".into()).into());
    }

    // 3. 对齐校验
    if let Some(last_dim) = dims_a.last() {
        assert!(last_dim % 2 == 0, "add_kernel_float2_forward requires last dim to be divisible by 2 (for float2 packing).");
    }
    
    // 4. 获取 FFI 参数
    let a_ptr = input_a_typed.buffer().as_ptr() as *const f32;
    let b_ptr = input_b_typed.buffer().as_ptr() as *const f32;
    let c_ptr = output_c_typed.buffer_mut().as_mut_ptr() as *mut f32;
    let stream_ptr: crate::cuda::ffi::cudaStream_t = std::ptr::null_mut();

    // 5. 调用 FFI
    unsafe {
        add_kernel_float2_forward(
            c_ptr,
            a_ptr,
            b_ptr,
            num_elements as i32,
            stream_ptr,
        );
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
    // 1. 获取 TypedTensor 并进行类型校验
    let input_output_a_typed: &mut TypedTensor<f32> = input_output_a.as_f32_mut()?;
    let input_b_typed = input_b.as_f32()?;

    // 2. 严格的维度和长度校验
    let dims_a = input_output_a_typed.shape();
    let dims_b = input_b_typed.shape();

    // 形状校验：A 和 B 的形状必须完全匹配
    if dims_a != dims_b {
        return Err(Error::InvalidArgument(format!(
            "Tensor shapes must match for in-place addition. A: {:?}, B: {:?}",
            dims_a, dims_b
        )).into());
    }
    
    let num_elements = input_output_a_typed.num_elements();
    if num_elements == 0 {
        return Err(Error::InvalidArgument("Input Tensor cannot be empty".into()).into());
    }

    // 3. 对齐校验 (与 add 保持一致)
    if let Some(last_dim) = dims_a.last() {
        assert!(last_dim % 2 == 0, "add_inplace_kernel_float2_forward requires last dim to be divisible by 2.");
    }

    // 4. 获取 FFI 参数
    let a_and_c_ptr = input_output_a_typed.buffer_mut().as_mut_ptr() as *mut f32;
    let b_ptr = input_b_typed.buffer().as_ptr() as *const f32;
    let mut stream: crate::cuda::ffi::cudaStream_t = std::ptr::null_mut();

    if cuda_config.is_some() {
        stream = cuda_config.ok_or(Error::InvalidArgument("CudaConfig not provided".into()))?.stream;
    }

    // 5. 调用 FFI
    unsafe {
        add_inplace_kernel_float2_forward(
            a_and_c_ptr, // FFI 参数 a_and_c
            b_ptr,       // FFI 参数 b
            num_elements as i32,
            stream,
        );
    }
    
    Ok(())
}