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

    fn rmsnorm_kernel_cu_fp16x8(
        output: *mut half::f16,
        input: *const half::f16,
        weight: *const half::f16,
        rows: i32,
        cols: i32,
        eps: f32,
        stream:crate::cuda::ffi::cudaStream_t,
    );
}

/// RMSNorm 的 CUDA 内核包装函数
pub fn rmsnorm(input: &Tensor, weight: &Tensor, output: &mut Tensor, eps: f32, cuda_config:Option<&CudaConfig>) -> Result<()> {
    let dim = weight.shape()[0];
    let rows = input.num_elements() / dim;
    
    // 获取 CUDA stream
    let stream = CudaConfig::resolve_stream(cuda_config);
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
        crate::base::DataType::F16 => {
            let input_typed = input.as_f16()?;
            let weight_typed = weight.as_f16()?;
            let output_typed:&mut TypedTensor<half::f16> = output.as_f16_mut()?;
            // 检查对齐要求
            if !dim.is_multiple_of(16) {
                return Err(Error::InvalidArgument("RMSNorm fp16 kernel requires dimension to be multiple of 16".to_string()).into());
            }
            
            let input_ptr = input_typed.buffer().as_ptr() as *const half::f16;
            let weight_ptr = weight_typed.buffer().as_ptr() as *const half::f16;
            let output_ptr = output_typed.buffer_mut().as_mut_ptr() as *mut half::f16;
            
            unsafe {
                rmsnorm_kernel_cu_fp16x8(
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

/// In-place variant of [`rmsnorm`]: `x = rmsnorm(x, weight, eps)`.
///
/// Uses the exact same CUDA kernels as [`rmsnorm`]; the kernels are
/// safe to invoke with `output == input` because each thread loads
/// `input[i]` into a register (via `float4`) *before* writing
/// `output[i]`, and the block reduction that computes the RMS scale
/// has already completed (with a `__syncthreads()`) by the time any
/// thread issues its store. No cross-thread read-after-write hazard
/// exists since each element is touched by exactly one thread on the
/// write pass.
pub fn rmsnorm_inplace(
    x: &mut Tensor,
    weight: &Tensor,
    eps: f32,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let dim = weight.shape()[0];
    let rows = x.num_elements() / dim;

    if !dim.is_multiple_of(16) {
        return Err(Error::InvalidArgument(
            "RMSNorm in-place kernel requires dimension to be multiple of 16".to_string(),
        ).into());
    }

    let stream = CudaConfig::resolve_stream(cuda_config);
    let dtype = x.dtype();
    match dtype {
        crate::base::DataType::F32 => {
            let x_typed: &mut TypedTensor<f32> = x.as_f32_mut()?;
            let w_typed = weight.as_f32()?;
            let ptr = x_typed.buffer_mut().as_mut_ptr() as *mut f32;
            let w_ptr = w_typed.buffer().as_ptr() as *const f32;
            unsafe {
                rmsnorm_kernel_cu_dim(ptr, ptr as *const f32, w_ptr, rows as i32, dim as i32, eps, stream);
            }
        }
        crate::base::DataType::BF16 => {
            let x_typed: &mut TypedTensor<half::bf16> = x.as_bf16_mut()?;
            let w_typed = weight.as_bf16()?;
            let ptr = x_typed.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let w_ptr = w_typed.buffer().as_ptr() as *const half::bf16;
            unsafe {
                rmsnorm_kernel_cu_bf16x8(ptr, ptr as *const half::bf16, w_ptr, rows as i32, dim as i32, eps, stream);
            }
        }
        crate::base::DataType::F16 => {
            let x_typed: &mut TypedTensor<half::f16> = x.as_f16_mut()?;
            let w_typed = weight.as_f16()?;
            let ptr = x_typed.buffer_mut().as_mut_ptr() as *mut half::f16;
            let w_ptr = w_typed.buffer().as_ptr() as *const half::f16;
            unsafe {
                rmsnorm_kernel_cu_fp16x8(ptr, ptr as *const half::f16, w_ptr, rows as i32, dim as i32, eps, stream);
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported data type for RMSNorm CUDA in-place kernel: {:?}", dtype
            )).into());
        }
    }
    Ok(())
}
