//! Minimal BF16 MoE router for the staged Qwen3 bring-up.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use half::bf16;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

pub const MAX_TOP_K: usize = 32;

unsafe extern "C" {
    fn moe_route_topk_bf16(
        logits: *const bf16,
        expert_ids: *mut i32,
        expert_weights: *mut f32,
        rows: i32,
        experts: i32,
        top_k: i32,
        renormalize: i32,
        stream: cudaStream_t,
    );
}

pub fn route_topk_bf16(
    stream: cudaStream_t,
    logits: &Tensor<bf16, Cuda>,
    expert_ids: &mut Tensor<i32, Cuda>,
    expert_weights: &mut Tensor<f32, Cuda>,
    top_k: usize,
    renormalize: bool,
) -> OpResult<()> {
    let shape = logits.shape().as_slice();
    if shape.len() != 2 || shape.contains(&0) {
        return Err(OpError::Shape(format!(
            "moe_route_topk logits must be non-empty [rows,experts], got {:?}",
            shape
        )));
    }
    let rows = shape[0];
    let experts = shape[1];
    if top_k == 0 || top_k > experts || top_k > MAX_TOP_K {
        return Err(OpError::Shape(format!(
            "moe_route_topk top_k {} must be in 1..=min({}, {})",
            top_k, experts, MAX_TOP_K
        )));
    }
    let output_shape = [rows, top_k];
    if expert_ids.shape().as_slice() != output_shape
        || expert_weights.shape().as_slice() != output_shape
    {
        return Err(OpError::Shape(format!(
            "moe_route_topk outputs must both be {:?}, got ids {:?}, weights {:?}",
            output_shape,
            expert_ids.shape().as_slice(),
            expert_weights.shape().as_slice()
        )));
    }
    let rows = i32::try_from(rows)
        .map_err(|_| OpError::Shape("moe_route_topk row count exceeds i32".into()))?;
    let experts = i32::try_from(experts)
        .map_err(|_| OpError::Shape("moe_route_topk expert count exceeds i32".into()))?;
    let top_k = i32::try_from(top_k)
        .map_err(|_| OpError::Shape("moe_route_topk top_k exceeds i32".into()))?;

    unsafe {
        moe_route_topk_bf16(
            logits.data_ptr(),
            expert_ids.data_ptr_mut(),
            expert_weights.data_ptr_mut(),
            rows,
            experts,
            top_k,
            i32::from(renormalize),
            stream,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16s(values: &[f32]) -> Vec<bf16> {
        values.iter().copied().map(bf16::from_f32).collect()
    }

    #[test]
    fn route_topk_is_stable_and_renormalizes_selected_weights() {
        let cuda = Cuda::new(0).unwrap();
        let logits = Tensor::from_host_slice(
            &bf16s(&[
                1.0, 3.0, 2.0, 3.0, 0.0, // tie: expert 1 precedes 3
                0.0, 1.0, 2.0, 3.0, 4.0,
            ]),
            [2, 5],
            &cuda,
        )
        .unwrap();
        let mut ids = Tensor::<i32, Cuda>::zeros([2, 2], &cuda).unwrap();
        let mut weights = Tensor::<f32, Cuda>::zeros([2, 2], &cuda).unwrap();

        route_topk_bf16(cuda.config.stream, &logits, &mut ids, &mut weights, 2, true).unwrap();

        assert_eq!(ids.to_host_vec().unwrap(), vec![1, 3, 4, 3]);
        let got = weights.to_host_vec().unwrap();
        let expected = [0.5, 0.5, 0.7310586, 0.2689414];
        for (actual, expected) in got.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn route_topk_can_keep_full_softmax_probability_mass() {
        let cuda = Cuda::new(0).unwrap();
        let logits = Tensor::from_host_slice(&bf16s(&[0.0, 1.0, 2.0]), [1, 3], &cuda).unwrap();
        let mut ids = Tensor::<i32, Cuda>::zeros([1, 1], &cuda).unwrap();
        let mut weights = Tensor::<f32, Cuda>::zeros([1, 1], &cuda).unwrap();

        route_topk_bf16(
            cuda.config.stream,
            &logits,
            &mut ids,
            &mut weights,
            1,
            false,
        )
        .unwrap();

        assert_eq!(ids.to_host_vec().unwrap(), vec![2]);
        let expected = 2.0f32.exp() / (1.0 + 1.0f32.exp() + 2.0f32.exp());
        let actual = weights.to_host_vec().unwrap()[0];
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}
