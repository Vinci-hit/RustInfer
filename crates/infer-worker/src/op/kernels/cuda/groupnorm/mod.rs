use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use crate::cuda::ffi;

unsafe extern "C" {
    fn groupnorm_f32_forward(output: *mut f32, input: *const f32, weight: *const f32, bias: *const f32,
        batch: i32, channels: i32, spatial: i32, num_groups: i32, eps: f32, stream: ffi::cudaStream_t);
    fn groupnorm_bf16_forward(output: *mut half::bf16, input: *const half::bf16, weight: *const half::bf16,
        bias: *const half::bf16, batch: i32, channels: i32, spatial: i32, num_groups: i32, eps: f32, stream: ffi::cudaStream_t);
}

#[cfg(feature = "cuda")]
pub fn groupnorm(
    input: &Tensor, weight: &Tensor, bias: &Tensor, output: &mut Tensor,
    num_groups: usize, eps: f32, stream: ffi::cudaStream_t,
) -> Result<()> {
    let shape = input.shape(); // [B, C, H, W] or [B, C, ...]
    let batch = shape[0] as i32;
    let channels = shape[1] as i32;
    let spatial: usize = shape[2..].iter().product();

    match input.dtype() {
        crate::base::DataType::F32 => unsafe {
            groupnorm_f32_forward(
                output.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32,
                input.as_f32()?.buffer().as_ptr() as *const f32,
                weight.as_f32()?.buffer().as_ptr() as *const f32,
                bias.as_f32()?.buffer().as_ptr() as *const f32,
                batch, channels, spatial as i32, num_groups as i32, eps, stream,
            );
            Ok(())
        },
        crate::base::DataType::BF16 => unsafe {
            groupnorm_bf16_forward(
                output.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16,
                input.as_bf16()?.buffer().as_ptr() as *const half::bf16,
                weight.as_bf16()?.buffer().as_ptr() as *const half::bf16,
                bias.as_bf16()?.buffer().as_ptr() as *const half::bf16,
                batch, channels, spatial as i32, num_groups as i32, eps, stream,
            );
            Ok(())
        },
        other => Err(Error::InvalidArgument(format!(
            "CUDA groupnorm: unsupported dtype {:?}", other
        )).into()),
    }
}
