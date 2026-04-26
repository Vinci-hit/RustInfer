use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use crate::cuda::ffi;

unsafe extern "C" {
    fn softmax_f32_forward(output: *mut f32, input: *const f32, rows: i32, cols: i32, stream: ffi::cudaStream_t);
    fn softmax_bf16_forward(output: *mut half::bf16, input: *const half::bf16, rows: i32, cols: i32, stream: ffi::cudaStream_t);
}

#[cfg(feature = "cuda")]
pub fn softmax(input: &Tensor, output: &mut Tensor, stream: ffi::cudaStream_t) -> Result<()> {
    let shape = input.shape();
    let last_dim = *shape.last().unwrap();
    let n_rows = (input.num_elements() / last_dim) as i32;
    let cols = last_dim as i32;

    match input.dtype() {
        crate::base::DataType::F32 => unsafe {
            softmax_f32_forward(
                output.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32,
                input.as_f32()?.buffer().as_ptr() as *const f32,
                n_rows, cols, stream,
            );
            Ok(())
        },
        crate::base::DataType::BF16 => unsafe {
            softmax_bf16_forward(
                output.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16,
                input.as_bf16()?.buffer().as_ptr() as *const half::bf16,
                n_rows, cols, stream,
            );
            Ok(())
        },
        other => Err(Error::InvalidArgument(format!(
            "CUDA softmax: unsupported dtype {:?}", other
        )).into()),
    }
}
