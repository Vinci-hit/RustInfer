use crate::components::linear::ExpertLinear;
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;

/// Address-stable intermediates for one routed expert MLP invocation.
///
/// Capacity is measured in routes (`tokens * top_k`), not source tokens.
pub struct MoeExpertScratch<T: Dtype, D: LlmBackend> {
    route_count: usize,
    gate_up_features: usize,
    intermediate_features: usize,
    gate_up: Tensor<T, D>,
    activated: Tensor<T, D>,
}

impl<T: Dtype, D: LlmBackend> MoeExpertScratch<T, D> {
    fn allocate(
        route_count: usize,
        gate_up_features: usize,
        intermediate_features: usize,
        device: &D,
    ) -> OpResult<Self> {
        if route_count == 0 {
            return Err(OpError::Shape(
                "MoeExpertScratch requires at least one route".into(),
            ));
        }
        Ok(Self {
            route_count,
            gate_up_features,
            intermediate_features,
            gate_up: Tensor::zeros([route_count, gate_up_features], device)?,
            activated: Tensor::zeros([route_count, intermediate_features], device)?,
        })
    }

    pub fn route_count(&self) -> usize {
        self.route_count
    }

    pub fn gate_up_features(&self) -> usize {
        self.gate_up_features
    }

    pub fn intermediate_features(&self) -> usize {
        self.intermediate_features
    }
}

/// Dense routed experts operating entirely in expert-major order.
///
/// Computes `down(swiglu(gate_up(input)))` for rows delimited by one shared
/// expert-offset table. Routing, permutation, communication, weighting and
/// combine remain outside this component.
pub struct MoeExperts<T: Dtype, D: LlmBackend> {
    gate_up: ExpertLinear<T, D>,
    down: ExpertLinear<T, D>,
    num_experts: usize,
    hidden_features: usize,
    intermediate_features: usize,
}

impl<T: Dtype, D: LlmBackend> MoeExperts<T, D> {
    pub fn new(gate_up: ExpertLinear<T, D>, down: ExpertLinear<T, D>) -> OpResult<Self> {
        if gate_up.num_experts() != down.num_experts() {
            return Err(OpError::Shape(format!(
                "MoeExperts gate/up experts {} != down experts {}",
                gate_up.num_experts(),
                down.num_experts()
            )));
        }
        let gate_up_features = gate_up.out_features();
        if !gate_up_features.is_multiple_of(2) {
            return Err(OpError::Shape(format!(
                "MoeExperts gate/up output {} must be 2 * intermediate",
                gate_up_features
            )));
        }
        let intermediate_features = gate_up_features / 2;
        let hidden_features = gate_up.in_features();
        if down.in_features() != intermediate_features || down.out_features() != hidden_features {
            return Err(OpError::Shape(format!(
                "MoeExperts down must be [experts,{},{}], got [experts,{},{}]",
                hidden_features,
                intermediate_features,
                down.out_features(),
                down.in_features()
            )));
        }
        Ok(Self {
            num_experts: gate_up.num_experts(),
            gate_up,
            down,
            hidden_features,
            intermediate_features,
        })
    }

    pub fn gate_up(&self) -> &ExpertLinear<T, D> {
        &self.gate_up
    }

    pub fn down(&self) -> &ExpertLinear<T, D> {
        &self.down
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn hidden_features(&self) -> usize {
        self.hidden_features
    }

    pub fn intermediate_features(&self) -> usize {
        self.intermediate_features
    }

    pub fn allocate_scratch(
        &self,
        route_count: usize,
        device: &D,
    ) -> OpResult<MoeExpertScratch<T, D>> {
        MoeExpertScratch::allocate(
            route_count,
            self.gate_up.out_features(),
            self.intermediate_features,
            device,
        )
    }

    /// Execute all locally resident experts without changing route order.
    pub fn forward_routed(
        &self,
        input: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        expert_offsets: &Tensor<i32, D>,
        scratch: &mut MoeExpertScratch<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let input_shape = input.shape().as_slice();
        if input_shape.len() != 2 || input_shape[0] == 0 || input_shape[1] != self.hidden_features {
            return Err(OpError::Shape(format!(
                "MoeExperts input must be non-empty [routes,{}], got {:?}",
                self.hidden_features, input_shape
            )));
        }
        let route_count = input_shape[0];
        if output.shape().as_slice() != [route_count, self.hidden_features] {
            return Err(OpError::Shape(format!(
                "MoeExperts output must be [{},{}], got {:?}",
                route_count,
                self.hidden_features,
                output.shape().as_slice()
            )));
        }
        if expert_offsets.shape().as_slice() != [self.num_experts + 1] {
            return Err(OpError::Shape(format!(
                "MoeExperts offsets must be [{}], got {:?}",
                self.num_experts + 1,
                expert_offsets.shape().as_slice()
            )));
        }
        if (
            scratch.route_count,
            scratch.gate_up_features,
            scratch.intermediate_features,
        ) != (
            route_count,
            self.gate_up.out_features(),
            self.intermediate_features,
        ) {
            return Err(OpError::Shape(format!(
                "MoeExperts scratch routes/gate_up/inter={}/{}/{}, expected {}/{}/{}",
                scratch.route_count,
                scratch.gate_up_features,
                scratch.intermediate_features,
                route_count,
                self.gate_up.out_features(),
                self.intermediate_features
            )));
        }

        self.gate_up
            .forward_routed(input, &mut scratch.gate_up, expert_offsets, ctx)?;
        D::swiglu_packed(
            ctx,
            &scratch.gate_up,
            &mut scratch.activated,
            route_count,
            self.intermediate_features,
        )?;
        self.down
            .forward_routed(&scratch.activated, output, expert_offsets, ctx)
    }
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
    fn experts_expose_validated_mlp_geometry() {
        let gate_up = ExpertLinear::new_dense(
            Tensor::from_host_slice(&[0.0f32; 2 * 6 * 4], [2, 6, 4], &Cpu).unwrap(),
        )
        .unwrap();
        let down = ExpertLinear::new_dense(
            Tensor::from_host_slice(&[0.0f32; 2 * 4 * 3], [2, 4, 3], &Cpu).unwrap(),
        )
        .unwrap();
        let experts = MoeExperts::new(gate_up, down).unwrap();
        let scratch = experts.allocate_scratch(5, &Cpu).unwrap();

        assert_eq!(experts.num_experts(), 2);
        assert_eq!(experts.hidden_features(), 4);
        assert_eq!(experts.intermediate_features(), 3);
        assert_eq!(scratch.route_count(), 5);
        assert_eq!(scratch.gate_up_features(), 6);
        assert_eq!(scratch.intermediate_features(), 3);
    }

    #[test]
    fn experts_reject_mismatched_down_projection() {
        let gate_up = ExpertLinear::new_dense(
            Tensor::from_host_slice(&[0.0f32; 2 * 6 * 4], [2, 6, 4], &Cpu).unwrap(),
        )
        .unwrap();
        let down = ExpertLinear::new_dense(
            Tensor::from_host_slice(&[0.0f32; 2 * 4 * 4], [2, 4, 4], &Cpu).unwrap(),
        )
        .unwrap();

        let err = MoeExperts::new(gate_up, down).err().unwrap();
        assert!(err.to_string().contains("down must be"));
    }

    #[test]
    fn cpu_routed_experts_are_explicitly_unsupported() {
        let cpu = Cpu;
        let gate_up = ExpertLinear::new_dense(
            Tensor::from_host_slice(&[1.0f32; 2 * 2 * 2], [2, 2, 2], &cpu).unwrap(),
        )
        .unwrap();
        let down = ExpertLinear::new_dense(
            Tensor::from_host_slice(&[1.0f32; 2 * 2], [2, 2, 1], &cpu).unwrap(),
        )
        .unwrap();
        let experts = MoeExperts::new(gate_up, down).unwrap();
        let input = Tensor::from_host_slice(&[1.0f32; 2 * 2], [2, 2], &cpu).unwrap();
        let offsets = Tensor::from_host_slice(&[0i32, 1, 2], [3], &cpu).unwrap();
        let mut output = Tensor::<f32, Cpu>::zeros([2, 2], &cpu).unwrap();
        let mut scratch = experts.allocate_scratch(2, &cpu).unwrap();
        let scope = HostScope::new(cpu);
        let batch_plan = decode_plan(2);
        let ctx = StepCtx::new(&scope, &batch_plan);

        let err = experts
            .forward_routed(&input, &mut output, &offsets, &mut scratch, &ctx)
            .unwrap_err();
        assert!(matches!(
            err,
            OpError::Unsupported {
                backend: "cpu",
                op: "grouped_expert_gemm"
            }
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_routed_experts_run_gate_swiglu_and_down() {
        use crate::infrastructure::cuda::{Cuda, CudaScope};
        use half::bf16;

        const EXPERTS: usize = 3;
        const HIDDEN: usize = 2;
        const INTER: usize = 8;
        let cuda = Cuda::new(0).unwrap();

        let mut gate_up_host = vec![0.0f32; EXPERTS * 2 * INTER * HIDDEN];
        let gate_up_index = |expert: usize, output: usize, input: usize| {
            (expert * 2 * INTER + output) * HIDDEN + input
        };
        gate_up_host[gate_up_index(0, 0, 0)] = 1.0;
        gate_up_host[gate_up_index(0, INTER, 1)] = 1.0;
        gate_up_host[gate_up_index(2, 0, 1)] = 1.0;
        gate_up_host[gate_up_index(2, INTER, 0)] = 1.0;

        let mut down_host = vec![0.0f32; EXPERTS * HIDDEN * INTER];
        let down_index =
            |expert: usize, output: usize, input: usize| (expert * HIDDEN + output) * INTER + input;
        down_host[down_index(0, 0, 0)] = 1.0;
        down_host[down_index(0, 1, 0)] = 2.0;
        down_host[down_index(2, 0, 0)] = 1.0;
        down_host[down_index(2, 1, 0)] = -1.0;

        let bf16s = |values: &[f32]| {
            values
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>()
        };
        let gate_up = ExpertLinear::new_dense(
            Tensor::from_host_slice(&bf16s(&gate_up_host), [EXPERTS, 2 * INTER, HIDDEN], &cuda)
                .unwrap(),
        )
        .unwrap();
        let down = ExpertLinear::new_dense(
            Tensor::from_host_slice(&bf16s(&down_host), [EXPERTS, HIDDEN, INTER], &cuda).unwrap(),
        )
        .unwrap();
        let experts = MoeExperts::new(gate_up, down).unwrap();
        let input = Tensor::from_host_slice(
            &bf16s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            [4, HIDDEN],
            &cuda,
        )
        .unwrap();
        let offsets = Tensor::from_host_slice(&[0i32, 2, 2, 4], [4], &cuda).unwrap();
        let mut output = Tensor::<bf16, Cuda>::zeros([4, HIDDEN], &cuda).unwrap();
        let mut scratch = experts.allocate_scratch(4, &cuda).unwrap();
        let scope = CudaScope::new(cuda);
        let batch_plan = decode_plan(4);
        let ctx = StepCtx::new(&scope, &batch_plan);

        experts
            .forward_routed(&input, &mut output, &offsets, &mut scratch, &ctx)
            .unwrap();

        let activation =
            |gate: f32, up: f32| bf16::from_f32(gate / (1.0 + (-gate).exp()) * up).to_f32();
        let expert_0_row_0 = activation(1.0, 2.0);
        let expert_0_row_1 = activation(3.0, 4.0);
        let expert_2_row_0 = activation(6.0, 5.0);
        let expert_2_row_1 = activation(8.0, 7.0);
        let expected = [
            expert_0_row_0,
            bf16::from_f32(2.0 * expert_0_row_0).to_f32(),
            expert_0_row_1,
            bf16::from_f32(2.0 * expert_0_row_1).to_f32(),
            expert_2_row_0,
            bf16::from_f32(-expert_2_row_0).to_f32(),
            expert_2_row_1,
            bf16::from_f32(-expert_2_row_1).to_f32(),
        ];
        let got = output
            .to_host_vec()
            .unwrap()
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        for (actual, expected) in got.iter().zip(expected) {
            assert!((actual - expected).abs() <= 0.25, "{actual} != {expected}");
        }
    }
}
