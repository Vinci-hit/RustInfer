//! SiLU + SwiGLU CUDA kernel wrappers.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    // SwiGLU: input_output_x = silu(input_output_x) * input_y, in place.
    // Signature in .cu: (input_y, input_output_x, num_elements, stream)
    fn swiglu_inplace_cu_bf16x8(y: *const half::bf16, x: *mut half::bf16, n: i32, stream: cudaStream_t);
    fn swiglu_inplace_cu_fp16x8(y: *const half::f16, x: *mut half::f16, n: i32, stream: cudaStream_t);
    fn swiglu_inplace_kernel_cu_fp32x4(y: *const f32, x: *mut f32, n: i32, stream: cudaStream_t);

    // SiLU in-place: x = silu(x).
    fn silu_inplace_bf16_forward(x: *mut half::bf16, n: i32, stream: cudaStream_t);
    fn silu_inplace_f16_forward(x: *mut half::f16, n: i32, stream: cudaStream_t);
    fn silu_inplace_f32_forward(x: *mut f32, n: i32, stream: cudaStream_t);
}

pub fn silu_inplace<T: Dtype>(x: &mut Tensor<T, Cuda>) -> OpResult<()> {
    let n = x.numel() as i32;
    let stream = x.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => silu_inplace_f32_forward(x.data_ptr_mut() as _, n, stream),
            DataType::BF16 => silu_inplace_bf16_forward(x.data_ptr_mut() as _, n, stream),
            DataType::F16 => silu_inplace_f16_forward(x.data_ptr_mut() as _, n, stream),
            _ => return Err(OpError::Kernel(format!("silu: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}

pub fn swiglu_inplace<T: Dtype>(x: &mut Tensor<T, Cuda>, gate: &Tensor<T, Cuda>) -> OpResult<()> {
    let n = x.numel() as i32;
    let stream = x.device().config.stream;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => swiglu_inplace_kernel_cu_fp32x4(gate.data_ptr() as _, x.data_ptr_mut() as _, n, stream),
            DataType::BF16 => swiglu_inplace_cu_bf16x8(gate.data_ptr() as _, x.data_ptr_mut() as _, n, stream),
            DataType::F16 => swiglu_inplace_cu_fp16x8(gate.data_ptr() as _, x.data_ptr_mut() as _, n, stream),
            _ => return Err(OpError::Kernel(format!("swiglu: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}
