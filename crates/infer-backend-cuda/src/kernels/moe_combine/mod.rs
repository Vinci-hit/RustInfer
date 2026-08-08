//! BF16 weighted route combination with FP32 device accumulation.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use half::bf16;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn moe_combine_bf16(
        expert_output: *const bf16,
        source_tokens: *const i32,
        route_weights: *const f32,
        output: *mut bf16,
        accumulator: *mut f32,
        routes: i32,
        tokens: i32,
        hidden: i32,
        stream: cudaStream_t,
    ) -> i32;
}

pub fn combine_bf16(
    stream: cudaStream_t,
    expert_output: &Tensor<bf16, Cuda>,
    source_tokens: &Tensor<i32, Cuda>,
    route_weights: &Tensor<f32, Cuda>,
    output: &mut Tensor<bf16, Cuda>,
    accumulator: &mut Tensor<f32, Cuda>,
) -> OpResult<()> {
    for (name, contiguous) in [
        ("expert_output", expert_output.is_contiguous()),
        ("source_tokens", source_tokens.is_contiguous()),
        ("route_weights", route_weights.is_contiguous()),
        ("output", output.is_contiguous()),
        ("accumulator", accumulator.is_contiguous()),
    ] {
        if !contiguous {
            return Err(OpError::Shape(format!(
                "moe_combine {} must be contiguous",
                name
            )));
        }
    }

    let expert_shape = expert_output.shape().as_slice();
    let output_shape = output.shape().as_slice();
    if expert_shape.len() != 2 || expert_shape.contains(&0) {
        return Err(OpError::Shape(format!(
            "moe_combine expert_output must be non-empty [routes,hidden], got {:?}",
            expert_shape
        )));
    }
    if output_shape.len() != 2 || output_shape.contains(&0) {
        return Err(OpError::Shape(format!(
            "moe_combine output must be non-empty [tokens,hidden], got {:?}",
            output_shape
        )));
    }
    let routes = expert_shape[0];
    let hidden = expert_shape[1];
    let tokens = output_shape[0];
    if output_shape[1] != hidden
        || source_tokens.shape().as_slice() != [routes]
        || route_weights.shape().as_slice() != [routes]
        || accumulator.shape().as_slice() != output_shape
    {
        return Err(OpError::Shape(format!(
            "moe_combine incompatible expert {:?}, source {:?}, weights {:?}, output {:?}, accumulator {:?}",
            expert_shape,
            source_tokens.shape().as_slice(),
            route_weights.shape().as_slice(),
            output_shape,
            accumulator.shape().as_slice()
        )));
    }
    validate_grid(routes, hidden, "route scatter")?;
    validate_grid(tokens, hidden, "output cast")?;

    let routes = to_i32(routes, "route count")?;
    let tokens = to_i32(tokens, "token count")?;
    let hidden = to_i32(hidden, "hidden size")?;
    let status = unsafe {
        moe_combine_bf16(
            expert_output.data_ptr(),
            source_tokens.data_ptr(),
            route_weights.data_ptr(),
            output.data_ptr_mut(),
            accumulator.data_ptr_mut(),
            routes,
            tokens,
            hidden,
            stream,
        )
    };
    if status != 0 {
        return Err(OpError::Kernel(format!(
            "moe_combine CUDA launch failed with status {}",
            status
        )));
    }
    Ok(())
}

fn validate_grid(rows: usize, columns: usize, name: &str) -> OpResult<()> {
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| OpError::Shape(format!("moe_combine {} size overflows", name)))?;
    const THREADS: usize = 256;
    let blocks = elements.div_ceil(THREADS);
    let _ = to_i32(blocks, &format!("{} grid block count", name))?;
    Ok(())
}

fn to_i32(value: usize, name: &str) -> OpResult<i32> {
    i32::try_from(value).map_err(|_| OpError::Shape(format!("moe_combine {} exceeds i32", name)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16s(values: &[f32]) -> Vec<bf16> {
        values.iter().copied().map(bf16::from_f32).collect()
    }

    #[test]
    fn combine_weights_routes_and_clears_accumulator_between_calls() {
        let cuda = Cuda::new(0).unwrap();
        let expert_output = Tensor::from_host_slice(
            &bf16s(&[
                4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0,
            ]),
            [6, 2],
            &cuda,
        )
        .unwrap();
        let source_tokens = Tensor::from_host_slice(&[0i32, 2, 1, 2, 0, 1], [6], &cuda).unwrap();
        let route_weights =
            Tensor::from_host_slice(&[0.25f32, 0.5, 0.75, 0.5, 0.75, 0.25], [6], &cuda).unwrap();
        let mut output = Tensor::<bf16, Cuda>::zeros([3, 2], &cuda).unwrap();
        let mut accumulator = Tensor::<f32, Cuda>::zeros([3, 2], &cuda).unwrap();

        for _ in 0..2 {
            combine_bf16(
                cuda.config.stream,
                &expert_output,
                &source_tokens,
                &route_weights,
                &mut output,
                &mut accumulator,
            )
            .unwrap();
            let got = output
                .to_host_vec()
                .unwrap()
                .into_iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            assert_eq!(got, vec![28.0, 32.0, 26.0, 30.0, 20.0, 24.0]);
        }
    }
}
