//! GroupNorm CUDA kernel wrapper.

use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::cuda::ffi::cudaStream_t;

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
    let stream = input.device().config.stream;
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
    let stream = input.device().config.stream;
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
