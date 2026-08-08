use crate::components::moe_permute::MoeRoutePlan;
use crate::domain::dtype::Dtype;
use crate::domain::exec::{ExecDevice, ExecScope, StepCtx};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;

/// FP32 accumulation storage for weighted route combination.
pub struct MoeCombineScratch<D: LlmBackend> {
    num_tokens: usize,
    hidden_features: usize,
    accumulator: Tensor<f32, D>,
}

impl<D: LlmBackend> MoeCombineScratch<D> {
    fn allocate(num_tokens: usize, hidden_features: usize, device: &D) -> OpResult<Self> {
        if num_tokens == 0 || hidden_features == 0 {
            return Err(OpError::Shape(format!(
                "MoeCombineScratch requires nonzero tokens/hidden, got {}/{}",
                num_tokens, hidden_features
            )));
        }
        Ok(Self {
            num_tokens,
            hidden_features,
            accumulator: Tensor::zeros([num_tokens, hidden_features], device)?,
        })
    }

    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    pub fn hidden_features(&self) -> usize {
        self.hidden_features
    }
}

/// Combine weighted expert-major route rows back into token-major order.
#[derive(Debug, Default, Clone, Copy)]
pub struct MoeCombiner;

impl MoeCombiner {
    pub fn new() -> Self {
        Self
    }

    pub fn allocate_scratch<D: LlmBackend>(
        &self,
        num_tokens: usize,
        hidden_features: usize,
        device: &D,
    ) -> OpResult<MoeCombineScratch<D>> {
        MoeCombineScratch::allocate(num_tokens, hidden_features, device)
    }

    pub fn forward<T: Dtype, D: LlmBackend>(
        &self,
        expert_output: &Tensor<T, D>,
        plan: &MoeRoutePlan<D>,
        output: &mut Tensor<T, D>,
        scratch: &mut MoeCombineScratch<D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let expert_shape = expert_output.shape().as_slice();
        if expert_shape.len() != 2 || expert_shape.contains(&0) {
            return Err(OpError::Shape(format!(
                "MoeCombiner expert_output must be non-empty [routes,hidden], got {:?}",
                expert_shape
            )));
        }
        let route_count = expert_shape[0];
        let hidden_features = expert_shape[1];
        if route_count != plan.route_count() {
            return Err(OpError::Shape(format!(
                "MoeCombiner expert rows {} != plan routes {}",
                route_count,
                plan.route_count()
            )));
        }
        let output_shape = [plan.num_tokens(), hidden_features];
        if output.shape().as_slice() != output_shape {
            return Err(OpError::Shape(format!(
                "MoeCombiner output must be {:?}, got {:?}",
                output_shape,
                output.shape().as_slice()
            )));
        }
        if (scratch.num_tokens, scratch.hidden_features) != (plan.num_tokens(), hidden_features) {
            return Err(OpError::Shape(format!(
                "MoeCombiner scratch tokens/hidden={}/{}, expected {}/{}",
                scratch.num_tokens,
                scratch.hidden_features,
                plan.num_tokens(),
                hidden_features
            )));
        }

        validate_tensor(expert_output, "expert_output", ctx)?;
        validate_tensor(plan.source_tokens(), "source_tokens", ctx)?;
        validate_tensor(plan.route_weights(), "route_weights", ctx)?;
        validate_tensor(output, "output", ctx)?;
        validate_tensor(&scratch.accumulator, "accumulator", ctx)?;

        D::moe_combine(
            ctx,
            expert_output,
            plan.source_tokens(),
            plan.route_weights(),
            output,
            &mut scratch.accumulator,
        )
    }
}

fn validate_tensor<T: Dtype, D: LlmBackend>(
    tensor: &Tensor<T, D>,
    name: &str,
    ctx: &StepCtx<'_, D>,
) -> OpResult<()> {
    if !tensor.is_contiguous() {
        return Err(OpError::Shape(format!(
            "MoeCombiner {} must be contiguous, got {:?}",
            name,
            tensor.shape().as_slice()
        )));
    }
    let tensor_device = <D as ExecDevice>::device_id(tensor.device());
    let scope_device = <D as ExecDevice>::device_id(ctx.scope().device());
    if tensor_device != scope_device {
        return Err(OpError::Shape(format!(
            "MoeCombiner {} belongs to device {}, but scope uses device {}",
            name, tensor_device.0, scope_device.0
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::MoeTokenPermuter;
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
    fn combine_scratch_exposes_token_geometry() {
        let scratch = MoeCombiner::new().allocate_scratch(3, 8, &Cpu).unwrap();
        assert_eq!(scratch.num_tokens(), 3);
        assert_eq!(scratch.hidden_features(), 8);
    }

    #[test]
    fn cpu_combine_is_explicitly_unsupported() {
        let cpu = Cpu;
        let permuter = MoeTokenPermuter::new(2, 2).unwrap();
        let route_plan = permuter.allocate_plan(2, &cpu).unwrap();
        let expert_output = Tensor::from_host_slice(&[1.0f32; 4 * 3], [4, 3], &cpu).unwrap();
        let mut output = Tensor::<f32, Cpu>::zeros([2, 3], &cpu).unwrap();
        let combiner = MoeCombiner::new();
        let mut scratch = combiner.allocate_scratch(2, 3, &cpu).unwrap();
        let scope = HostScope::new(cpu);
        let batch_plan = decode_plan(2);
        let ctx = StepCtx::new(&scope, &batch_plan);

        let err = combiner
            .forward(&expert_output, &route_plan, &mut output, &mut scratch, &ctx)
            .unwrap_err();
        assert!(matches!(
            err,
            OpError::Unsupported {
                backend: "cpu",
                op: "moe_combine"
            }
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_combiner_consumes_permuted_route_plan() {
        use crate::infrastructure::cuda::{Cuda, CudaScope};
        use half::bf16;

        let cuda = Cuda::new(0).unwrap();
        let permuter = MoeTokenPermuter::new(4, 2).unwrap();
        let original =
            Tensor::from_host_slice(&[0.0f32; 3 * 2].map(bf16::from_f32), [3, 2], &cuda).unwrap();
        let ids = Tensor::from_host_slice(&[2i32, 0, 1, 2, 0, 1], [3, 2], &cuda).unwrap();
        let weights =
            Tensor::from_host_slice(&[0.75f32, 0.25, 0.75, 0.25, 0.5, 0.5], [3, 2], &cuda).unwrap();
        let mut ignored_permuted = Tensor::<bf16, Cuda>::zeros([6, 2], &cuda).unwrap();
        let mut route_plan = permuter.allocate_plan(3, &cuda).unwrap();
        let scope = CudaScope::new(cuda.clone());
        let batch_plan = decode_plan(3);
        let ctx = StepCtx::new(&scope, &batch_plan);
        permuter
            .forward(
                &original,
                &ids,
                &weights,
                &mut ignored_permuted,
                &mut route_plan,
                &ctx,
            )
            .unwrap();

        let expert_output = Tensor::from_host_slice(
            &[
                4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0,
            ]
            .map(bf16::from_f32),
            [6, 2],
            &cuda,
        )
        .unwrap();
        let mut output = Tensor::<bf16, Cuda>::zeros([3, 2], &cuda).unwrap();
        let combiner = MoeCombiner::new();
        let mut scratch = combiner.allocate_scratch(3, 2, &cuda).unwrap();

        combiner
            .forward(&expert_output, &route_plan, &mut output, &mut scratch, &ctx)
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
