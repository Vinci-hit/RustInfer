//! Correctness-first BF16 token permutation for sparse MoE routes.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use half::bf16;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn moe_permute_tokens_bf16(
        input: *const bf16,
        expert_ids: *const i32,
        expert_weights: *const f32,
        permuted_input: *mut bf16,
        source_tokens: *mut i32,
        route_weights: *mut f32,
        expert_offsets: *mut i32,
        tokens: i32,
        hidden: i32,
        top_k: i32,
        experts: i32,
        stream: cudaStream_t,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn permute_tokens_bf16(
    stream: cudaStream_t,
    input: &Tensor<bf16, Cuda>,
    expert_ids: &Tensor<i32, Cuda>,
    expert_weights: &Tensor<f32, Cuda>,
    permuted_input: &mut Tensor<bf16, Cuda>,
    source_tokens: &mut Tensor<i32, Cuda>,
    route_weights: &mut Tensor<f32, Cuda>,
    expert_offsets: &mut Tensor<i32, Cuda>,
) -> OpResult<()> {
    for (name, contiguous) in [
        ("input", input.is_contiguous()),
        ("expert_ids", expert_ids.is_contiguous()),
        ("expert_weights", expert_weights.is_contiguous()),
        ("permuted_input", permuted_input.is_contiguous()),
        ("source_tokens", source_tokens.is_contiguous()),
        ("route_weights", route_weights.is_contiguous()),
        ("expert_offsets", expert_offsets.is_contiguous()),
    ] {
        if !contiguous {
            return Err(OpError::Shape(format!(
                "moe_permute_tokens {} must be contiguous",
                name
            )));
        }
    }

    let input_shape = input.shape().as_slice();
    if input_shape.len() != 2 || input_shape.contains(&0) {
        return Err(OpError::Shape(format!(
            "moe_permute_tokens input must be non-empty [tokens,hidden], got {:?}",
            input_shape
        )));
    }
    let tokens = input_shape[0];
    let hidden = input_shape[1];
    let ids_shape = expert_ids.shape().as_slice();
    if ids_shape.len() != 2 || ids_shape[0] != tokens || ids_shape[1] == 0 {
        return Err(OpError::Shape(format!(
            "moe_permute_tokens expert_ids must be [tokens,top_k], got {:?}",
            ids_shape
        )));
    }
    let top_k = ids_shape[1];
    if expert_weights.shape().as_slice() != ids_shape {
        return Err(OpError::Shape(format!(
            "moe_permute_tokens expert_weights must be {:?}, got {:?}",
            ids_shape,
            expert_weights.shape().as_slice()
        )));
    }
    let offsets_shape = expert_offsets.shape().as_slice();
    if offsets_shape.len() != 1 || offsets_shape[0] < 2 {
        return Err(OpError::Shape(format!(
            "moe_permute_tokens expert_offsets must be [experts+1], got {:?}",
            offsets_shape
        )));
    }
    let experts = offsets_shape[0] - 1;
    if top_k > experts {
        return Err(OpError::Shape(format!(
            "moe_permute_tokens top_k {} exceeds experts {}",
            top_k, experts
        )));
    }
    let routes = tokens
        .checked_mul(top_k)
        .ok_or_else(|| OpError::Shape("moe_permute_tokens route count overflows".into()))?;
    let _ = to_i32(routes, "route count")?;
    if permuted_input.shape().as_slice() != [routes, hidden]
        || source_tokens.shape().as_slice() != [routes]
        || route_weights.shape().as_slice() != [routes]
    {
        return Err(OpError::Shape(format!(
            "moe_permute_tokens outputs must be input [{},{}], source/weights [{}], got {:?}, {:?}, {:?}",
            routes,
            hidden,
            routes,
            permuted_input.shape().as_slice(),
            source_tokens.shape().as_slice(),
            route_weights.shape().as_slice()
        )));
    }

    let tokens = to_i32(tokens, "token count")?;
    let hidden = to_i32(hidden, "hidden size")?;
    let top_k = to_i32(top_k, "top_k")?;
    let experts = to_i32(experts, "expert count")?;

    unsafe {
        moe_permute_tokens_bf16(
            input.data_ptr(),
            expert_ids.data_ptr(),
            expert_weights.data_ptr(),
            permuted_input.data_ptr_mut(),
            source_tokens.data_ptr_mut(),
            route_weights.data_ptr_mut(),
            expert_offsets.data_ptr_mut(),
            tokens,
            hidden,
            top_k,
            experts,
            stream,
        );
    }
    Ok(())
}

fn to_i32(value: usize, name: &str) -> OpResult<i32> {
    i32::try_from(value)
        .map_err(|_| OpError::Shape(format!("moe_permute_tokens {} exceeds i32", name)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16s(values: &[f32]) -> Vec<bf16> {
        values.iter().copied().map(bf16::from_f32).collect()
    }

    #[test]
    fn permutation_is_stable_and_preserves_empty_experts() {
        let cuda = Cuda::new(0).unwrap();
        let input =
            Tensor::from_host_slice(&bf16s(&[10.0, 11.0, 20.0, 21.0]), [2, 2], &cuda).unwrap();
        let ids = Tensor::from_host_slice(&[2i32, 0, 0, 2], [2, 2], &cuda).unwrap();
        let weights = Tensor::from_host_slice(&[0.6f32, 0.4, 0.7, 0.3], [2, 2], &cuda).unwrap();
        let mut output = Tensor::<bf16, Cuda>::zeros([4, 2], &cuda).unwrap();
        let mut source_tokens = Tensor::<i32, Cuda>::zeros([4], &cuda).unwrap();
        let mut route_weights = Tensor::<f32, Cuda>::zeros([4], &cuda).unwrap();
        let mut offsets = Tensor::<i32, Cuda>::zeros([4], &cuda).unwrap();

        permute_tokens_bf16(
            cuda.config.stream,
            &input,
            &ids,
            &weights,
            &mut output,
            &mut source_tokens,
            &mut route_weights,
            &mut offsets,
        )
        .unwrap();

        assert_eq!(offsets.to_host_vec().unwrap(), vec![0, 2, 2, 4]);
        assert_eq!(source_tokens.to_host_vec().unwrap(), vec![0, 1, 0, 1]);
        assert_eq!(
            route_weights.to_host_vec().unwrap(),
            vec![0.4, 0.7, 0.6, 0.3]
        );
        let got = output
            .to_host_vec()
            .unwrap()
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        assert_eq!(got, vec![10.0, 11.0, 20.0, 21.0, 10.0, 11.0, 20.0, 21.0]);
    }
}
