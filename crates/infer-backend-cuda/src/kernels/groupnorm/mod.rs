//! GroupNorm CUDA kernel wrapper.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};

unsafe extern "C" {
    fn groupnorm_f32_forward(
        output: *mut f32,
        input: *const f32,
        weight: *const f32,
        bias: *const f32,
        batch: i32,
        channels: i32,
        spatial: i32,
        num_groups: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn groupnorm_bf16_forward(
        output: *mut half::bf16,
        input: *const half::bf16,
        weight: *const half::bf16,
        bias: *const half::bf16,
        batch: i32,
        channels: i32,
        spatial: i32,
        num_groups: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn groupnorm_silu_f32_forward(
        output: *mut f32,
        input: *const f32,
        weight: *const f32,
        bias: *const f32,
        batch: i32,
        channels: i32,
        spatial: i32,
        num_groups: i32,
        eps: f32,
        stream: cudaStream_t,
    );
    fn groupnorm_silu_bf16_forward(
        output: *mut half::bf16,
        input: *const half::bf16,
        weight: *const half::bf16,
        bias: *const half::bf16,
        batch: i32,
        channels: i32,
        spatial: i32,
        num_groups: i32,
        eps: f32,
        stream: cudaStream_t,
    );
}

pub fn groupnorm<T: Dtype>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    bias: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    num_groups: usize,
    eps: f32,
) -> OpResult<()> {
    let shape = input.shape().as_slice();
    let batch = shape[0] as i32;
    let channels = shape[1] as i32;
    let spatial: i32 = shape[2..].iter().product::<usize>() as i32;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => groupnorm_f32_forward(
                output.data_ptr_mut() as _,
                input.data_ptr() as _,
                weight.data_ptr() as _,
                bias.data_ptr() as _,
                batch,
                channels,
                spatial,
                num_groups as i32,
                eps,
                stream,
            ),
            DataType::BF16 => groupnorm_bf16_forward(
                output.data_ptr_mut() as _,
                input.data_ptr() as _,
                weight.data_ptr() as _,
                bias.data_ptr() as _,
                batch,
                channels,
                spatial,
                num_groups as i32,
                eps,
                stream,
            ),
            _ => return Err(OpError::Kernel(format!("groupnorm: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}

pub fn groupnorm_silu<T: Dtype>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    bias: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    num_groups: usize,
    eps: f32,
) -> OpResult<()> {
    let shape = input.shape().as_slice();
    let batch = shape[0] as i32;
    let channels = shape[1] as i32;
    let spatial: i32 = shape[2..].iter().product::<usize>() as i32;
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => groupnorm_silu_f32_forward(
                output.data_ptr_mut() as _,
                input.data_ptr() as _,
                weight.data_ptr() as _,
                bias.data_ptr() as _,
                batch,
                channels,
                spatial,
                num_groups as i32,
                eps,
                stream,
            ),
            DataType::BF16 => groupnorm_silu_bf16_forward(
                output.data_ptr_mut() as _,
                input.data_ptr() as _,
                weight.data_ptr() as _,
                bias.data_ptr() as _,
                batch,
                channels,
                spatial,
                num_groups as i32,
                eps,
                stream,
            ),
            _ => {
                return Err(OpError::Kernel(format!(
                    "groupnorm_silu: {:?}",
                    T::DATA_TYPE
                )));
            }
        }
    }
    Ok(())
}
