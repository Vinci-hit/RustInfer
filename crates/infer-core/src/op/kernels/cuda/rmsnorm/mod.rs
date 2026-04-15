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
    fn rmsnorm_kernel_cu_bf16x8(
        output: *mut half::bf16,
        input: *const half::bf16,
        weight: *const half::bf16,
        rows: i32,
        cols: i32,
        eps: f32,
        stream:crate::cuda::ffi::cudaStream_t,
    );
    fn fused_add_rmsnorm_kernel_cu_bf16(
        norm_output: *mut half::bf16,
        residual: *mut half::bf16,
        input: *const half::bf16,
        weight: *const half::bf16,
        rows: i32,
        dim: i32,
        eps: f32,
        stream: crate::cuda::ffi::cudaStream_t,
    );
}

/// RMSNorm 的 CUDA 内核包装函数
pub fn rmsnorm(input: &Tensor, weight: &Tensor, output: &mut Tensor, eps: f32, cuda_config:Option<&CudaConfig>) -> Result<()> {
    let dim = weight.shape()[0];
    let rows = input.num_elements() / dim;
    
    // 获取 CUDA stream
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);
    // 根据输出数据类型进行分发
    let dtype = output.dtype();
    match dtype {
        crate::base::DataType::F32 => {
            let input_typed = input.as_f32()?;
            let weight_typed = weight.as_f32()?;
            let output_typed:&mut TypedTensor<f32> = output.as_f32_mut()?;

            // 检查对齐要求
            if !dim.is_multiple_of(16) {
                return Err(Error::InvalidArgument("RMSNorm f32 kernel requires dimension to be multiple of 16".to_string()).into());
            }
            
            let input_ptr = input_typed.buffer().as_ptr() as *const f32;
            let weight_ptr = weight_typed.buffer().as_ptr() as *const f32;
            let output_ptr = output_typed.buffer_mut().as_mut_ptr() as *mut f32;
            
            unsafe {
                rmsnorm_kernel_cu_dim(
                    output_ptr,
                    input_ptr,
                    weight_ptr,
                    rows as i32,
                    dim as i32,
                    eps,
                    stream
                );
            }
        }
        crate::base::DataType::BF16 => {
            let input_typed = input.as_bf16()?;
            let weight_typed = weight.as_bf16()?;
            let output_typed:&mut TypedTensor<half::bf16> = output.as_bf16_mut()?;
            // 检查对齐要求
            if !dim.is_multiple_of(16) {
                return Err(Error::InvalidArgument("RMSNorm bf16 kernel requires dimension to be multiple of 16".to_string()).into());
            }
            
            let input_ptr = input_typed.buffer().as_ptr() as *const half::bf16;
            let weight_ptr = weight_typed.buffer().as_ptr() as *const half::bf16;
            let output_ptr = output_typed.buffer_mut().as_mut_ptr() as *mut half::bf16;
            
            unsafe {
                rmsnorm_kernel_cu_bf16x8(
                    output_ptr,
                    input_ptr,
                    weight_ptr,
                    rows as i32,
                    dim as i32,
                    eps,
                    stream
                );
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported data type for RMSNorm CUDA kernel: {:?}", dtype
            )).into());
        }
    }

    Ok(())
}

/// Fused: residual += input; norm_output = rmsnorm(residual, weight)
/// BF16 only. Saves one kernel launch vs separate add + rmsnorm.
pub fn fused_add_rmsnorm(
    norm_output: &mut Tensor,
    residual: &mut Tensor,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let dim = weight.shape()[0];
    let rows = input.num_elements() / dim;
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);

    let norm_out_ptr = norm_output.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
    let res_ptr = residual.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
    let input_ptr = input.as_bf16()?.buffer().as_ptr() as *const half::bf16;
    let weight_ptr = weight.as_bf16()?.buffer().as_ptr() as *const half::bf16;

    unsafe {
        fused_add_rmsnorm_kernel_cu_bf16(
            norm_out_ptr,
            res_ptr,
            input_ptr,
            weight_ptr,
            rows as i32,
            dim as i32,
            eps,
            stream,
        );
    }
    Ok(())
}
