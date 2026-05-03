use crate::base::error::{Error, Result};
use crate::cuda::config::CudaConfig;
use crate::tensor::Tensor;

unsafe extern "C" {
    fn layernorm_f32_forward(output: *mut f32, input: *const f32, rows: i32, cols: i32, eps: f32, stream: crate::cuda::ffi::cudaStream_t);
    fn layernorm_bf16_forward(output: *mut half::bf16, input: *const half::bf16, rows: i32, cols: i32, eps: f32, stream: crate::cuda::ffi::cudaStream_t);
    fn layernorm_f16_forward(output: *mut half::f16, input: *const half::f16, rows: i32, cols: i32, eps: f32, stream: crate::cuda::ffi::cudaStream_t);
}

/// LayerNorm without affine parameters (CUDA).
/// input/output: [rows, cols], each row normalized to zero-mean unit-variance.
pub fn layernorm(input: &Tensor, output: &mut Tensor, eps: f32, cuda_config: Option<&CudaConfig>) -> Result<()> {
    let shape = input.shape();
    let cols = *shape.last().unwrap();
    let rows = (input.numel() / cols) as i32;
    let stream = CudaConfig::resolve_stream(cuda_config);

    match input.dtype() {
        crate::base::DataType::F32 => {
            let ip = input.as_f32()?.data_ptr();
            let op = output.as_f32_mut()?.data_ptr_mut();
            unsafe { layernorm_f32_forward(op, ip, rows, cols as i32, eps, stream); }
        }
        crate::base::DataType::BF16 => {
            let ip = input.as_bf16()?.data_ptr();
            let op = output.as_bf16_mut()?.data_ptr_mut();
            unsafe { layernorm_bf16_forward(op, ip, rows, cols as i32, eps, stream); }
        }
        crate::base::DataType::F16 => {
            let ip = input.as_f16()?.data_ptr();
            let op = output.as_f16_mut()?.data_ptr_mut();
            unsafe { layernorm_f16_forward(op, ip, rows, cols as i32, eps, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "CUDA layernorm: unsupported dtype {:?}", other
        )).into()),
    }
    Ok(())
}
