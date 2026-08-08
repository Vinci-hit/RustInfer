use crate::domain::dtype::Dtype;
use crate::domain::exec::{ExecDevice, ExecScope, StepCtx};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;

/// Device-resident metadata for token rows arranged in expert-major order.
///
/// `source_tokens[row]` and `route_weights[row]` describe how a grouped expert
/// output row will eventually be combined back into the token stream.
pub struct MoeRoutePlan<D: LlmBackend> {
    num_tokens: usize,
    top_k: usize,
    num_experts: usize,
    source_tokens: Tensor<i32, D>,
    route_weights: Tensor<f32, D>,
    expert_offsets: Tensor<i32, D>,
}

impl<D: LlmBackend> MoeRoutePlan<D> {
    pub fn allocate(
        num_tokens: usize,
        top_k: usize,
        num_experts: usize,
        device: &D,
    ) -> OpResult<Self> {
        validate_route_dimensions(num_tokens, top_k, num_experts)?;
        let route_count = num_tokens.checked_mul(top_k).ok_or_else(|| {
            OpError::Shape("MoeRoutePlan token route count overflows usize".into())
        })?;
        if route_count > i32::MAX as usize {
            return Err(OpError::Shape(format!(
                "MoeRoutePlan route count {} exceeds i32 offsets",
                route_count
            )));
        }
        let offsets_len = num_experts.checked_add(1).ok_or_else(|| {
            OpError::Shape("MoeRoutePlan expert offset count overflows usize".into())
        })?;

        Ok(Self {
            num_tokens,
            top_k,
            num_experts,
            source_tokens: Tensor::zeros([route_count], device)?,
            route_weights: Tensor::zeros([route_count], device)?,
            expert_offsets: Tensor::zeros([offsets_len], device)?,
        })
    }

    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn route_count(&self) -> usize {
        self.source_tokens.numel()
    }

    pub fn source_tokens(&self) -> &Tensor<i32, D> {
        &self.source_tokens
    }

    pub fn route_weights(&self) -> &Tensor<f32, D> {
        &self.route_weights
    }

    pub fn expert_offsets(&self) -> &Tensor<i32, D> {
        &self.expert_offsets
    }
}

/// Stable token permutation for the routes selected by [`super::MoeRouter`].
///
/// This component only produces expert-major rows and a [`MoeRoutePlan`]. It
/// does not execute experts, communicate across ranks, or combine outputs.
pub struct MoeTokenPermuter {
    num_experts: usize,
    top_k: usize,
}

impl MoeTokenPermuter {
    pub fn new(num_experts: usize, top_k: usize) -> OpResult<Self> {
        validate_route_dimensions(1, top_k, num_experts)?;
        Ok(Self { num_experts, top_k })
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn allocate_plan<D: LlmBackend>(
        &self,
        num_tokens: usize,
        device: &D,
    ) -> OpResult<MoeRoutePlan<D>> {
        MoeRoutePlan::allocate(num_tokens, self.top_k, self.num_experts, device)
    }

    pub fn forward<T: Dtype, D: LlmBackend>(
        &self,
        input: &Tensor<T, D>,
        expert_ids: &Tensor<i32, D>,
        expert_weights: &Tensor<f32, D>,
        permuted_input: &mut Tensor<T, D>,
        plan: &mut MoeRoutePlan<D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let input_shape = input.shape().as_slice();
        if input_shape.len() != 2 || input_shape.contains(&0) {
            return Err(OpError::Shape(format!(
                "MoeTokenPermuter input must be non-empty [tokens,hidden], got {:?}",
                input_shape
            )));
        }
        let num_tokens = input_shape[0];
        let hidden = input_shape[1];
        let route_shape = [num_tokens, self.top_k];
        if expert_ids.shape().as_slice() != route_shape {
            return Err(OpError::Shape(format!(
                "MoeTokenPermuter expert_ids must be {:?}, got {:?}",
                route_shape,
                expert_ids.shape().as_slice()
            )));
        }
        if expert_weights.shape().as_slice() != route_shape {
            return Err(OpError::Shape(format!(
                "MoeTokenPermuter expert_weights must be {:?}, got {:?}",
                route_shape,
                expert_weights.shape().as_slice()
            )));
        }
        if (plan.num_tokens, plan.top_k, plan.num_experts)
            != (num_tokens, self.top_k, self.num_experts)
        {
            return Err(OpError::Shape(format!(
                "MoeTokenPermuter plan is tokens/top_k/experts={}/{}/{}, expected {}/{}/{}",
                plan.num_tokens,
                plan.top_k,
                plan.num_experts,
                num_tokens,
                self.top_k,
                self.num_experts
            )));
        }
        let route_count = num_tokens.checked_mul(self.top_k).ok_or_else(|| {
            OpError::Shape("MoeTokenPermuter token route count overflows usize".into())
        })?;
        if permuted_input.shape().as_slice() != [route_count, hidden] {
            return Err(OpError::Shape(format!(
                "MoeTokenPermuter output must be [{},{}], got {:?}",
                route_count,
                hidden,
                permuted_input.shape().as_slice()
            )));
        }

        validate_tensor(input, "input", ctx)?;
        validate_tensor(expert_ids, "expert_ids", ctx)?;
        validate_tensor(expert_weights, "expert_weights", ctx)?;
        validate_tensor(permuted_input, "permuted_input", ctx)?;
        validate_tensor(&plan.source_tokens, "source_tokens", ctx)?;
        validate_tensor(&plan.route_weights, "route_weights", ctx)?;
        validate_tensor(&plan.expert_offsets, "expert_offsets", ctx)?;

        D::moe_permute_tokens(
            ctx,
            input,
            expert_ids,
            expert_weights,
            permuted_input,
            &mut plan.source_tokens,
            &mut plan.route_weights,
            &mut plan.expert_offsets,
        )
    }
}

fn validate_route_dimensions(num_tokens: usize, top_k: usize, num_experts: usize) -> OpResult<()> {
    if num_tokens == 0 {
        return Err(OpError::Shape(
            "MoE token permutation requires at least one token".into(),
        ));
    }
    if num_experts == 0 {
        return Err(OpError::Shape(
            "MoE token permutation requires at least one expert".into(),
        ));
    }
    if num_experts > i32::MAX as usize {
        return Err(OpError::Shape(format!(
            "MoE token permutation expert count {} exceeds i32 offsets",
            num_experts
        )));
    }
    if top_k == 0 || top_k > num_experts {
        return Err(OpError::Shape(format!(
            "MoE token permutation top_k {} must be in 1..={}",
            top_k, num_experts
        )));
    }
    Ok(())
}

fn validate_tensor<T: Dtype, D: LlmBackend>(
    tensor: &Tensor<T, D>,
    name: &str,
    ctx: &StepCtx<'_, D>,
) -> OpResult<()> {
    if !tensor.is_contiguous() {
        return Err(OpError::Shape(format!(
            "MoeTokenPermuter {} must be contiguous, got {:?}",
            name,
            tensor.shape().as_slice()
        )));
    }
    let tensor_device = <D as ExecDevice>::device_id(tensor.device());
    let scope_device = <D as ExecDevice>::device_id(ctx.scope().device());
    if tensor_device != scope_device {
        return Err(OpError::Shape(format!(
            "MoeTokenPermuter {} belongs to device {}, but scope uses device {}",
            name, tensor_device.0, scope_device.0
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::exec::{HostScope, StepCtx};
    use crate::domain::plan::{BatchKind, BatchPlan};
    use crate::infrastructure::cpu::Cpu;

    fn decode_plan(tokens: usize) -> BatchPlan {
        BatchPlan {
            kind: BatchKind::DecodeOnly,
            num_tokens: tokens,
            batch: tokens,
            q_lens: vec![1; tokens],
            kv_lens: vec![1; tokens],
            seq_positions: vec![0; tokens],
            rope_positions: vec![0; tokens],
            max_blocks_per_seq: 1,
            block_size: 1,
            total_q_tiles: 0,
        }
    }

    #[test]
    fn route_plan_exposes_exact_route_capacity() {
        let permuter = MoeTokenPermuter::new(4, 2).unwrap();
        let plan = permuter.allocate_plan(3, &Cpu).unwrap();

        assert_eq!(permuter.num_experts(), 4);
        assert_eq!(permuter.top_k(), 2);
        assert_eq!(plan.num_tokens(), 3);
        assert_eq!(plan.top_k(), 2);
        assert_eq!(plan.num_experts(), 4);
        assert_eq!(plan.route_count(), 6);
        assert_eq!(plan.expert_offsets().shape().as_slice(), [5]);
    }

    #[test]
    fn permuter_rejects_invalid_top_k() {
        let err = MoeTokenPermuter::new(4, 5).err().unwrap();
        assert!(err.to_string().contains("top_k 5"));
    }

    #[test]
    fn cpu_token_permutation_is_explicitly_unsupported() {
        let cpu = Cpu;
        let permuter = MoeTokenPermuter::new(4, 2).unwrap();
        let input = Tensor::from_host_slice(&[1.0f32; 2 * 3], [2, 3], &cpu).unwrap();
        let ids = Tensor::from_host_slice(&[0i32, 1, 2, 3], [2, 2], &cpu).unwrap();
        let weights = Tensor::from_host_slice(&[0.6f32, 0.4, 0.7, 0.3], [2, 2], &cpu).unwrap();
        let mut output = Tensor::<f32, Cpu>::zeros([4, 3], &cpu).unwrap();
        let mut route_plan = permuter.allocate_plan(2, &cpu).unwrap();
        let scope = HostScope::new(cpu);
        let batch_plan = decode_plan(2);
        let ctx = StepCtx::new(&scope, &batch_plan);

        let err = permuter
            .forward(&input, &ids, &weights, &mut output, &mut route_plan, &ctx)
            .unwrap_err();
        assert!(matches!(
            err,
            OpError::Unsupported {
                backend: "cpu",
                op: "moe_permute_tokens"
            }
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_permuter_builds_stable_expert_major_rows() {
        use crate::infrastructure::cuda::{Cuda, CudaScope};
        use half::bf16;

        let cuda = Cuda::new(0).unwrap();
        let permuter = MoeTokenPermuter::new(4, 2).unwrap();
        let input = Tensor::from_host_slice(
            &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0].map(bf16::from_f32),
            [3, 2],
            &cuda,
        )
        .unwrap();
        let ids = Tensor::from_host_slice(&[2i32, 0, 1, 2, 0, 1], [3, 2], &cuda).unwrap();
        let weights =
            Tensor::from_host_slice(&[0.6f32, 0.4, 0.7, 0.3, 0.8, 0.2], [3, 2], &cuda).unwrap();
        let mut output = Tensor::<bf16, Cuda>::zeros([6, 2], &cuda).unwrap();
        let mut route_plan = permuter.allocate_plan(3, &cuda).unwrap();
        let scope = CudaScope::new(cuda);
        let batch_plan = decode_plan(3);
        let ctx = StepCtx::new(&scope, &batch_plan);

        permuter
            .forward(&input, &ids, &weights, &mut output, &mut route_plan, &ctx)
            .unwrap();

        assert_eq!(
            route_plan.expert_offsets().to_host_vec().unwrap(),
            vec![0, 2, 4, 6, 6]
        );
        assert_eq!(
            route_plan.source_tokens().to_host_vec().unwrap(),
            vec![0, 2, 1, 2, 0, 1]
        );
        assert_eq!(
            route_plan.route_weights().to_host_vec().unwrap(),
            vec![0.4, 0.8, 0.7, 0.2, 0.6, 0.3]
        );
        let got = output
            .to_host_vec()
            .unwrap()
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        assert_eq!(
            got,
            vec![
                10.0, 11.0, 30.0, 31.0, 20.0, 21.0, 30.0, 31.0, 10.0, 11.0, 20.0, 21.0
            ]
        );
    }
}
