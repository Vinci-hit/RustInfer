//! Element-wise multiply CUDA kernel wrapper.

use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn ewise_mul_f32_forward(
        dst: *mut f32,
        a: *const f32,
        b: *const f32,
        n: i32,
        stream: cudaStream_t,
    );
    fn ewise_mul_bf16_forward(
        dst: *mut half::bf16,
        a: *const half::bf16,
        b: *const half::bf16,
        n: i32,
        stream: cudaStream_t,
    );
    fn ewise_mul_f16_forward(
        dst: *mut half::f16,
        a: *const half::f16,
        b: *const half::f16,
        n: i32,
        stream: cudaStream_t,
    );
}

pub fn ewise_mul<T: Dtype>(
    stream: cudaStream_t,
    a: &Tensor<T, Cuda>,
    b: &Tensor<T, Cuda>,
    dst: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let n = a.numel() as i32;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => ewise_mul_f32_forward(
                dst.data_ptr_mut() as _,
                a.data_ptr() as _,
                b.data_ptr() as _,
                n,
                stream,
            ),
            DataType::BF16 => ewise_mul_bf16_forward(
                dst.data_ptr_mut() as _,
                a.data_ptr() as _,
                b.data_ptr() as _,
                n,
                stream,
            ),
            DataType::F16 => ewise_mul_f16_forward(
                dst.data_ptr_mut() as _,
                a.data_ptr() as _,
                b.data_ptr() as _,
                n,
                stream,
            ),
            _ => return Err(OpError::Kernel(format!("ewise_mul: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
