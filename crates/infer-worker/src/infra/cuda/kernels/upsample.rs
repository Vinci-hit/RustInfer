//! Upsample nearest 2× CUDA kernel wrapper.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn upsample_nearest_2x_f32_forward(output: *mut f32, input: *const f32, batch: i32, channels: i32, h_in: i32, w_in: i32, stream: cudaStream_t);
    fn upsample_nearest_2x_bf16_forward(output: *mut half::bf16, input: *const half::bf16, batch: i32, channels: i32, h_in: i32, w_in: i32, stream: cudaStream_t);
}

pub fn upsample_nearest_2x<T: Dtype>(input: &Tensor<T, Cuda>, output: &mut Tensor<T, Cuda>) -> OpResult<()> {
    let shape = input.shape().as_slice();
    let (batch, channels, h_in, w_in) = (shape[0] as i32, shape[1] as i32, shape[2] as i32, shape[3] as i32);
    let stream = input.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => upsample_nearest_2x_f32_forward(output.data_ptr_mut() as _, input.data_ptr() as _, batch, channels, h_in, w_in, stream),
            DataType::BF16 => upsample_nearest_2x_bf16_forward(output.data_ptr_mut() as _, input.data_ptr() as _, batch, channels, h_in, w_in, stream),
            _ => return Err(OpError::Kernel(format!("upsample_nearest_2x: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
