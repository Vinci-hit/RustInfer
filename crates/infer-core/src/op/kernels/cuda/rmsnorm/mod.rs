use crate::base::error::{Error, Result};
use crate::cuda::config::CudaConfig;
use crate::tensor::{Tensor, TypedTensor};

// 假设我们会在 build.rs 中为 `rmsnorm_kernel_forward` 函数生成 FFI 绑定
unsafe extern "C" {
    fn rmsnorm_kernel_cu_dim(
        output: *mut f32,
        input: *const f32,
        weight: *const f32,
        rows: i32,
        cols: i32,
        eps: f32,
        stream:crate::cuda::ffi::cudaStream_t,
    );
}

/// RMSNorm 的 CUDA 内核包装函数
pub fn rmsnorm(input: &Tensor, weight: &Tensor, output: &mut Tensor, cuda_config:Option<&CudaConfig>) -> Result<()> {
    let input_typed = input.as_f32()?;
    let weight_typed = weight.as_f32()?;
    let output_typed:&mut TypedTensor<f32> = output.as_f32_mut()?;
    let dim = weight_typed.shape()[0];
    let rows = input_typed.num_elements() / dim;
    if dim % 16 != 0 {
        return Err(Error::InvalidArgument("rmsnrom input last dim must 对齐16字节！未实现无对齐的版本！".to_string()).into());
    }
    let input_ptr = input_typed.buffer().as_ptr() as *const f32;
    let weight_ptr = weight_typed.buffer().as_ptr() as *const f32;
    let output_ptr = output_typed.buffer_mut().as_mut_ptr() as *mut f32;
    let mut stream :crate::cuda::ffi::cudaStream_t = std::ptr::null_mut();
    if cuda_config.is_some(){
        stream = cuda_config.ok_or(Error::InvalidArgument("CudaConfig not provided".into()))?.stream;
    }
    unsafe {
        rmsnorm_kernel_cu_dim(
            output_ptr,
            input_ptr,
            weight_ptr,
            rows as i32,
            dim as i32,
            1e-6, // epsilon
            stream
        );
    }

    Ok(())
}