//! LayerNorm CUDA kernel wrapper.
//! The .cu kernel computes (x - mean) / std. We apply weight*x + bias via
//! broadcast_mul + a simple bias-add loop on top.

use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};
use crate::Cuda;
use crate::ffi::cudaStream_t;

unsafe extern "C" {
    fn layernorm_f32_forward(
        output: *mut f32,
        input: *const f32,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn layernorm_bf16_forward(
        output: *mut half::bf16,
        input: *const half::bf16,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn layernorm_f16_forward(
        output: *mut half::f16,
        input: *const half::f16,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    );
}

// Reuse broadcast_mul from its module
unsafe extern "C" {
    fn broadcast_mul_f32_forward(
        dst: *mut f32,
        a: *const f32,
        b: *const f32,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn broadcast_mul_bf16_forward(
        dst: *mut half::bf16,
        a: *const half::bf16,
        b: *const half::bf16,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn broadcast_mul_f16_forward(
        dst: *mut half::f16,
        a: *const half::f16,
        b: *const half::f16,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    // broadcast_add (reuse the add kernel from broadcast_mul.cu)
    fn broadcast_add_inplace_f32_forward(
        a: *mut f32,
        b: *const f32,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn broadcast_add_inplace_bf16_forward(
        a: *mut half::bf16,
        b: *const half::bf16,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
    fn broadcast_add_inplace_f16_forward(
        a: *mut half::f16,
        b: *const half::f16,
        rows: i32,
        d: i32,
        stream: cudaStream_t,
    );
}

/// Full layernorm: normalize → scale by weight → add bias.
pub fn layernorm<T: Dtype>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    bias: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    eps: f32,
) -> OpResult<()> {
    let dim = *input.shape().as_slice().last().unwrap();
    let rows = (input.numel() / dim) as i32;
    let cols = dim as i32;

    unsafe {
        // Step 1: normalize (zero-mean, unit-variance)
        match T::DATA_TYPE {
            DataType::F32 => layernorm_f32_forward(
                output.data_ptr_mut() as _,
                input.data_ptr() as _,
                rows,
                cols,
                eps,
                stream,
            ),
            DataType::BF16 => layernorm_bf16_forward(
                output.data_ptr_mut() as _,
                input.data_ptr() as _,
                rows,
                cols,
                eps,
                stream,
            ),
            DataType::F16 => layernorm_f16_forward(
                output.data_ptr_mut() as _,
                input.data_ptr() as _,
                rows,
                cols,
                eps,
                stream,
            ),
            _ => return Err(OpError::Kernel(format!("layernorm: {:?}", T::DATA_TYPE))),
        }

        // Step 2: output *= weight (broadcast multiply)
        match T::DATA_TYPE {
            DataType::F32 => broadcast_mul_f32_forward(
                output.data_ptr_mut() as _,
                output.data_ptr() as _,
                weight.data_ptr() as _,
                rows,
                cols,
                stream,
            ),
            DataType::BF16 => broadcast_mul_bf16_forward(
                output.data_ptr_mut() as _,
                output.data_ptr() as _,
                weight.data_ptr() as _,
                rows,
                cols,
                stream,
            ),
            DataType::F16 => broadcast_mul_f16_forward(
                output.data_ptr_mut() as _,
                output.data_ptr() as _,
                weight.data_ptr() as _,
                rows,
                cols,
                stream,
            ),
            _ => {}
        }

        // Step 3: output += bias (broadcast add)
        match T::DATA_TYPE {
            DataType::F32 => broadcast_add_inplace_f32_forward(
                output.data_ptr_mut() as _,
                bias.data_ptr() as _,
                rows,
                cols,
                stream,
            ),
            DataType::BF16 => broadcast_add_inplace_bf16_forward(
                output.data_ptr_mut() as _,
                bias.data_ptr() as _,
                rows,
                cols,
                stream,
            ),
            DataType::F16 => broadcast_add_inplace_f16_forward(
                output.data_ptr_mut() as _,
                bias.data_ptr() as _,
                rows,
                cols,
                stream,
            ),
            _ => {}
        }
    }
    Ok(())
}
