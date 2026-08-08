use crate::components::moe_combine::{MoeCombineScratch, MoeCombiner};
use crate::components::moe_experts::{MoeExpertScratch, MoeExperts};
use crate::components::moe_permute::{MoeRoutePlan, MoeTokenPermuter};
use crate::components::moe_router::MoeRouter;
use crate::domain::dtype::Dtype;
use crate::domain::exec::{DeviceId, ExecDevice, ExecScope, StepCtx};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;

/// Address-stable storage for one fixed-size local MoE invocation.
///
/// Capacity is exact: a scratch allocated for `num_tokens` cannot be reused for
/// a different token count. All route metadata and intermediates stay on the
/// device supplied to [`MoeLocalPipeline::allocate_scratch`].
pub struct MoeLocalScratch<T: Dtype, D: LlmBackend> {
    device_id: DeviceId,
    num_tokens: usize,
    route_count: usize,
    num_experts: usize,
    top_k: usize,
    hidden_features: usize,
    router_logits: Tensor<T, D>,
    expert_ids: Tensor<i32, D>,
    expert_weights: Tensor<f32, D>,
    permuted_input: Tensor<T, D>,
    route_plan: MoeRoutePlan<D>,
    expert_output: Tensor<T, D>,
    expert_scratch: MoeExpertScratch<T, D>,
    combine_scratch: MoeCombineScratch<D>,
}

impl<T: Dtype, D: LlmBackend> MoeLocalScratch<T, D> {
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    pub fn route_count(&self) -> usize {
        self.route_count
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn hidden_features(&self) -> usize {
        self.hidden_features
    }
}

/// Single-device routed MoE pipeline without norm, residual, or shared expert.
///
/// The pipeline composes the already-independent stages in this exact order:
/// router -> stable token permutation -> routed expert MLP -> weighted combine.
/// Every expert must be resident on the same device. Expert placement,
/// communication, quantization, and model-layer integration are deliberately
/// outside this type.
pub struct MoeLocalPipeline<T: Dtype, D: LlmBackend> {
    router: MoeRouter<T, D>,
    permuter: MoeTokenPermuter,
    experts: MoeExperts<T, D>,
    combiner: MoeCombiner,
    device_id: DeviceId,
}

impl<T: Dtype, D: LlmBackend> MoeLocalPipeline<T, D> {
    pub fn new(router: MoeRouter<T, D>, experts: MoeExperts<T, D>) -> OpResult<Self> {
        if router.num_experts() != experts.num_experts() {
            return Err(OpError::Shape(format!(
                "MoeLocalPipeline router experts {} != routed experts {}",
                router.num_experts(),
                experts.num_experts()
            )));
        }
        if router.input_features() != experts.hidden_features() {
            return Err(OpError::Shape(format!(
                "MoeLocalPipeline router input {} != expert hidden {}",
                router.input_features(),
                experts.hidden_features()
            )));
        }
        let router_tp = router.gate.parallelism().tp();
        if router_tp.size != 1 {
            return Err(OpError::Kernel(format!(
                "MoeLocalPipeline requires TP1 until distributed routing is implemented, got rank {}/{}",
                router_tp.rank, router_tp.size
            )));
        }

        let router_weight = router.gate.weight.as_dense().ok_or_else(|| {
            OpError::Kernel("MoeLocalPipeline requires a dense router weight".into())
        })?;
        let device_id = <D as ExecDevice>::device_id(router_weight.device());
        if let Some(router_bias) = &router.gate.bias {
            let bias_device = <D as ExecDevice>::device_id(router_bias.device());
            if bias_device != device_id {
                return Err(OpError::Shape(format!(
                    "MoeLocalPipeline router bias belongs to device {}, but router weight belongs to device {}",
                    bias_device.0, device_id.0
                )));
            }
        }
        for (name, weight) in [
            ("expert gate/up", experts.gate_up().weight()),
            ("expert down", experts.down().weight()),
        ] {
            let weight_device = <D as ExecDevice>::device_id(weight.device());
            if weight_device != device_id {
                return Err(OpError::Shape(format!(
                    "MoeLocalPipeline {name} weight belongs to device {}, but router belongs to device {}",
                    weight_device.0, device_id.0
                )));
            }
        }

        let permuter = MoeTokenPermuter::new(router.num_experts(), router.top_k())?;
        Ok(Self {
            router,
            permuter,
            experts,
            combiner: MoeCombiner::new(),
            device_id,
        })
    }

    pub fn router(&self) -> &MoeRouter<T, D> {
        &self.router
    }

    pub fn experts(&self) -> &MoeExperts<T, D> {
        &self.experts
    }

    pub fn num_experts(&self) -> usize {
        self.router.num_experts()
    }

    pub fn top_k(&self) -> usize {
        self.router.top_k()
    }

    pub fn hidden_features(&self) -> usize {
        self.experts.hidden_features()
    }

    pub(crate) fn device_id(&self) -> DeviceId {
        self.device_id
    }

    pub fn allocate_scratch(
        &self,
        num_tokens: usize,
        device: &D,
    ) -> OpResult<MoeLocalScratch<T, D>> {
        let scratch_device = <D as ExecDevice>::device_id(device);
        if scratch_device != self.device_id {
            return Err(OpError::Shape(format!(
                "MoeLocalPipeline scratch device {} != weight device {}",
                scratch_device.0, self.device_id.0
            )));
        }

        let route_plan = self.permuter.allocate_plan(num_tokens, device)?;
        let route_count = route_plan.route_count();
        let hidden_features = self.hidden_features();
        let num_experts = self.num_experts();
        let top_k = self.top_k();

        Ok(MoeLocalScratch {
            device_id: scratch_device,
            num_tokens,
            route_count,
            num_experts,
            top_k,
            hidden_features,
            router_logits: Tensor::zeros([num_tokens, num_experts], device)?,
            expert_ids: Tensor::zeros([num_tokens, top_k], device)?,
            expert_weights: Tensor::zeros([num_tokens, top_k], device)?,
            permuted_input: Tensor::zeros([route_count, hidden_features], device)?,
            route_plan,
            expert_output: Tensor::zeros([route_count, hidden_features], device)?,
            expert_scratch: self.experts.allocate_scratch(route_count, device)?,
            combine_scratch: self
                .combiner
                .allocate_scratch(num_tokens, hidden_features, device)?,
        })
    }

    /// Run the local routed experts and write their weighted token-major sum.
    pub fn forward(
        &self,
        input: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        scratch: &mut MoeLocalScratch<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let input_shape = input.shape().as_slice();
        if input_shape.len() != 2 || input_shape[0] == 0 || input_shape[1] != self.hidden_features()
        {
            return Err(OpError::Shape(format!(
                "MoeLocalPipeline input must be non-empty [tokens,{}], got {:?}",
                self.hidden_features(),
                input_shape
            )));
        }
        if output.shape().as_slice() != input_shape {
            return Err(OpError::Shape(format!(
                "MoeLocalPipeline output must be {:?}, got {:?}",
                input_shape,
                output.shape().as_slice()
            )));
        }

        let num_tokens = input_shape[0];
        let expected = (
            num_tokens,
            num_tokens.checked_mul(self.top_k()).ok_or_else(|| {
                OpError::Shape("MoeLocalPipeline route count overflows usize".into())
            })?,
            self.num_experts(),
            self.top_k(),
            self.hidden_features(),
        );
        let actual = (
            scratch.num_tokens,
            scratch.route_count,
            scratch.num_experts,
            scratch.top_k,
            scratch.hidden_features,
        );
        if actual != expected {
            return Err(OpError::Shape(format!(
                "MoeLocalPipeline scratch tokens/routes/experts/top_k/hidden={}/{}/{}/{}/{}, expected {}/{}/{}/{}/{}",
                actual.0,
                actual.1,
                actual.2,
                actual.3,
                actual.4,
                expected.0,
                expected.1,
                expected.2,
                expected.3,
                expected.4
            )));
        }
        if ctx.plan().num_tokens != num_tokens {
            return Err(OpError::Shape(format!(
                "MoeLocalPipeline input tokens {} != execution plan tokens {}",
                num_tokens,
                ctx.plan().num_tokens
            )));
        }

        let scope_device = <D as ExecDevice>::device_id(ctx.scope().device());
        if scope_device != self.device_id || scratch.device_id != self.device_id {
            return Err(OpError::Shape(format!(
                "MoeLocalPipeline weight/scratch/scope devices={}/{}/{}, expected one local device",
                self.device_id.0, scratch.device_id.0, scope_device.0
            )));
        }
        validate_tensor(input, "input", self.device_id)?;
        validate_tensor(output, "output", self.device_id)?;

        self.router.forward(
            input,
            &mut scratch.router_logits,
            &mut scratch.expert_ids,
            &mut scratch.expert_weights,
            ctx,
        )?;
        self.permuter.forward(
            input,
            &scratch.expert_ids,
            &scratch.expert_weights,
            &mut scratch.permuted_input,
            &mut scratch.route_plan,
            ctx,
        )?;
        self.experts.forward_routed(
            &scratch.permuted_input,
            &mut scratch.expert_output,
            scratch.route_plan.expert_offsets(),
            &mut scratch.expert_scratch,
            ctx,
        )?;
        self.combiner.forward(
            &scratch.expert_output,
            &scratch.route_plan,
            output,
            &mut scratch.combine_scratch,
            ctx,
        )
    }
}

fn validate_tensor<T: Dtype, D: LlmBackend>(
    tensor: &Tensor<T, D>,
    name: &str,
    expected_device: DeviceId,
) -> OpResult<()> {
    if !tensor.is_contiguous() {
        return Err(OpError::Shape(format!(
            "MoeLocalPipeline {name} must be contiguous, got {:?}",
            tensor.shape().as_slice()
        )));
    }
    let tensor_device = <D as ExecDevice>::device_id(tensor.device());
    if tensor_device != expected_device {
        return Err(OpError::Shape(format!(
            "MoeLocalPipeline {name} belongs to device {}, expected device {}",
            tensor_device.0, expected_device.0
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{ExpertLinear, Linear};
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

    fn cpu_pipeline() -> MoeLocalPipeline<f32, Cpu> {
        let router = MoeRouter::new(
            Linear::new(
                Tensor::from_host_slice(&[1.0f32, 0.0, 0.0, 1.0], [2, 2], &Cpu).unwrap(),
                None,
            ),
            1,
            true,
        )
        .unwrap();
        let gate_up = ExpertLinear::new_dense(
            Tensor::from_host_slice(&[1.0f32; 2 * 2 * 2], [2, 2, 2], &Cpu).unwrap(),
        )
        .unwrap();
        let down = ExpertLinear::new_dense(
            Tensor::from_host_slice(&[1.0f32; 2 * 2], [2, 2, 1], &Cpu).unwrap(),
        )
        .unwrap();
        MoeLocalPipeline::new(router, MoeExperts::new(gate_up, down).unwrap()).unwrap()
    }

    #[test]
    fn local_pipeline_exposes_fixed_scratch_geometry() {
        let pipeline = cpu_pipeline();
        let scratch = pipeline.allocate_scratch(3, &Cpu).unwrap();

        assert_eq!(pipeline.num_experts(), 2);
        assert_eq!(pipeline.top_k(), 1);
        assert_eq!(pipeline.hidden_features(), 2);
        assert_eq!(scratch.num_tokens(), 3);
        assert_eq!(scratch.route_count(), 3);
        assert_eq!(scratch.num_experts(), 2);
        assert_eq!(scratch.top_k(), 1);
        assert_eq!(scratch.hidden_features(), 2);
    }

    #[test]
    fn local_pipeline_rejects_router_expert_geometry_mismatch() {
        let router = MoeRouter::new(
            Linear::new(
                Tensor::from_host_slice(&[0.0f32; 3 * 2], [3, 2], &Cpu).unwrap(),
                None,
            ),
            1,
            true,
        )
        .unwrap();
        let gate_up = ExpertLinear::new_dense(
            Tensor::from_host_slice(&[0.0f32; 2 * 2 * 2], [2, 2, 2], &Cpu).unwrap(),
        )
        .unwrap();
        let down = ExpertLinear::new_dense(
            Tensor::from_host_slice(&[0.0f32; 2 * 2], [2, 2, 1], &Cpu).unwrap(),
        )
        .unwrap();

        let err = MoeLocalPipeline::new(router, MoeExperts::new(gate_up, down).unwrap())
            .err()
            .unwrap();
        assert!(err.to_string().contains("router experts 3"));
    }

    #[test]
    fn cpu_local_pipeline_is_explicitly_unsupported() {
        let cpu = Cpu;
        let pipeline = cpu_pipeline();
        let input = Tensor::from_host_slice(&[1.0f32, 2.0, 2.0, 1.0], [2, 2], &cpu).unwrap();
        let mut output = Tensor::<f32, Cpu>::zeros([2, 2], &cpu).unwrap();
        let mut scratch = pipeline.allocate_scratch(2, &cpu).unwrap();
        let scope = HostScope::new(cpu);
        let plan = decode_plan(2);
        let ctx = StepCtx::new(&scope, &plan);

        let err = pipeline
            .forward(&input, &mut output, &mut scratch, &ctx)
            .unwrap_err();
        assert!(matches!(
            err,
            OpError::Unsupported {
                backend: "cpu",
                op: "moe_route_topk"
            }
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_local_pipeline_routes_runs_experts_and_combines() {
        use crate::infrastructure::cuda::{Cuda, CudaScope};
        use half::bf16;

        const EXPERTS: usize = 3;
        const HIDDEN: usize = 2;
        const INTERMEDIATE: usize = 8;
        let cuda = Cuda::new(0).unwrap();
        let bf16s = |values: &[f32]| {
            values
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>()
        };

        let router = MoeRouter::new(
            Linear::new(
                Tensor::from_host_slice(
                    &bf16s(&[1.0, 0.0, 0.0, 1.0, -1.0, -1.0]),
                    [EXPERTS, HIDDEN],
                    &cuda,
                )
                .unwrap(),
                None,
            ),
            2,
            true,
        )
        .unwrap();

        let mut gate_up_host = vec![0.0f32; EXPERTS * 2 * INTERMEDIATE * HIDDEN];
        let gate_up_index = |expert: usize, output: usize, input: usize| {
            (expert * 2 * INTERMEDIATE + output) * HIDDEN + input
        };
        for expert in 0..2 {
            gate_up_host[gate_up_index(expert, 0, 0)] = 1.0;
            gate_up_host[gate_up_index(expert, INTERMEDIATE, 1)] = 1.0;
        }

        let mut down_host = vec![0.0f32; EXPERTS * HIDDEN * INTERMEDIATE];
        let down_index = |expert: usize, output: usize, input: usize| {
            (expert * HIDDEN + output) * INTERMEDIATE + input
        };
        down_host[down_index(0, 0, 0)] = 1.0;
        down_host[down_index(1, 1, 0)] = 1.0;

        let gate_up = ExpertLinear::new_dense(
            Tensor::from_host_slice(
                &bf16s(&gate_up_host),
                [EXPERTS, 2 * INTERMEDIATE, HIDDEN],
                &cuda,
            )
            .unwrap(),
        )
        .unwrap();
        let down = ExpertLinear::new_dense(
            Tensor::from_host_slice(&bf16s(&down_host), [EXPERTS, HIDDEN, INTERMEDIATE], &cuda)
                .unwrap(),
        )
        .unwrap();
        let pipeline =
            MoeLocalPipeline::new(router, MoeExperts::new(gate_up, down).unwrap()).unwrap();

        let input =
            Tensor::from_host_slice(&bf16s(&[1.0, 2.0, 2.0, 1.0]), [2, HIDDEN], &cuda).unwrap();
        let mut output =
            Tensor::from_host_slice(&bf16s(&[99.0; 2 * HIDDEN]), [2, HIDDEN], &cuda).unwrap();
        let mut scratch = pipeline.allocate_scratch(2, &cuda).unwrap();
        let scope = CudaScope::new(cuda.clone());
        let plan = decode_plan(2);
        let ctx = StepCtx::new(&scope, &plan);

        let high = 1.0f32 / (1.0 + (-1.0f32).exp());
        let low = 1.0 - high;
        let activation =
            |gate: f32, up: f32| bf16::from_f32(gate / (1.0 + (-gate).exp()) * up).to_f32();
        let token_0 = activation(1.0, 2.0);
        let token_1 = activation(2.0, 1.0);
        let expected = [
            bf16::from_f32(low * token_0).to_f32(),
            bf16::from_f32(high * token_0).to_f32(),
            bf16::from_f32(high * token_1).to_f32(),
            bf16::from_f32(low * token_1).to_f32(),
        ];

        for _ in 0..2 {
            pipeline
                .forward(&input, &mut output, &mut scratch, &ctx)
                .unwrap();

            assert_eq!(
                scratch.route_plan.expert_offsets().to_host_vec().unwrap(),
                vec![0, 2, 4, 4]
            );
            assert_eq!(
                scratch.route_plan.source_tokens().to_host_vec().unwrap(),
                vec![0, 1, 0, 1]
            );
            let got = output
                .to_host_vec()
                .unwrap()
                .into_iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            for (actual, expected) in got.iter().zip(expected) {
                assert!((actual - expected).abs() <= 0.02, "{actual} != {expected}");
            }
        }
    }
}
