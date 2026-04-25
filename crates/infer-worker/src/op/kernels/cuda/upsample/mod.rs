use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use crate::cuda::ffi;

unsafe extern "C" {
    fn upsample_nearest_2x_f32_forward(output: *mut f32, input: *const f32,
        batch: i32, channels: i32, h_in: i32, w_in: i32, stream: ffi::cudaStream_t);
    fn upsample_nearest_2x_bf16_forward(output: *mut half::bf16, input: *const half::bf16,
        batch: i32, channels: i32, h_in: i32, w_in: i32, stream: ffi::cudaStream_t);
}

#[cfg(feature = "cuda")]
pub fn upsample_nearest_2x(input: &Tensor, output: &mut Tensor, stream: ffi::cudaStream_t) -> Result<()> {
    let shape = input.shape(); // [B, C, H, W]
    let batch = shape[0] as i32;
    let channels = shape[1] as i32;
    let h_in = shape[2] as i32;
    let w_in = shape[3] as i32;

    match input.dtype() {
        crate::base::DataType::F32 => unsafe {
            upsample_nearest_2x_f32_forward(
                output.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32,
                input.as_f32()?.buffer().as_ptr() as *const f32,
                batch, channels, h_in, w_in, stream,
            );
            Ok(())
        },
        crate::base::DataType::BF16 => unsafe {
            upsample_nearest_2x_bf16_forward(
                output.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16,
                input.as_bf16()?.buffer().as_ptr() as *const half::bf16,
                batch, channels, h_in, w_in, stream,
            );
            Ok(())
        },
        other => Err(Error::InvalidArgument(format!(
            "CUDA upsample_nearest_2x: unsupported dtype {:?}", other
        )).into()),
    }
}
