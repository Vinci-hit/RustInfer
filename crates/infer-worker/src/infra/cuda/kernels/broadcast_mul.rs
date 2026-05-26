//! Broadcast multiply CUDA kernel wrapper.
//! broadcast_mul: dst[i] = a[i] * b[i % D]  (b is [D], a is [rows, D])

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn broadcast_mul_f32_forward(dst: *mut f32, a: *const f32, b: *const f32, rows: i32, d: i32, stream: cudaStream_t);
    fn broadcast_mul_bf16_forward(dst: *mut half::bf16, a: *const half::bf16, b: *const half::bf16, rows: i32, d: i32, stream: cudaStream_t);
    fn broadcast_mul_f16_forward(dst: *mut half::f16, a: *const half::f16, b: *const half::f16, rows: i32, d: i32, stream: cudaStream_t);
}

/// In-place broadcast multiply: x[i,j] *= scale[j].
pub fn broadcast_mul_inplace<T: Dtype>(x: &mut Tensor<T, Cuda>, scale: &Tensor<T, Cuda>) -> OpResult<()> {
    let dim = scale.numel() as i32;
    let rows = (x.numel() as i32) / dim;
    let stream = x.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => broadcast_mul_f32_forward(x.data_ptr_mut() as _, x.data_ptr() as _, scale.data_ptr() as _, rows, dim, stream),
            DataType::BF16 => broadcast_mul_bf16_forward(x.data_ptr_mut() as _, x.data_ptr() as _, scale.data_ptr() as _, rows, dim, stream),
            DataType::F16 => broadcast_mul_f16_forward(x.data_ptr_mut() as _, x.data_ptr() as _, scale.data_ptr() as _, rows, dim, stream),
            _ => return Err(OpError::Kernel(format!("broadcast_mul: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
