use crate::components::linear::Linear;
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;

/// Dense gate plus top-k selection for one sparse MoE layer.
///
/// This component stops at `[token, route]` expert IDs and weights. Token
/// permutation, expert placement/communication, expert Linear execution and
/// combine belong to later stages of the MoE layer.
pub struct MoeRouter<T: Dtype, D: LlmBackend> {
    pub gate: Linear<T, D>,
    num_experts: usize,
    input_features: usize,
    top_k: usize,
    renormalize: bool,
}

impl<T: Dtype, D: LlmBackend> MoeRouter<T, D> {
    pub fn new(gate: Linear<T, D>, top_k: usize, renormalize: bool) -> OpResult<Self> {
        let (num_experts, input_features) = {
            let weight = gate.weight.as_dense().ok_or_else(|| {
                OpError::Kernel("MoeRouter currently requires a dense gate Linear".into())
            })?;
            let shape = weight.shape().as_slice();
            if shape.len() != 2 || shape.contains(&0) {
                return Err(OpError::Shape(format!(
                    "MoeRouter gate weight must be non-empty [experts,in], got {:?}",
                    shape
                )));
            }
            (shape[0], shape[1])
        };
        if top_k == 0 || top_k > num_experts {
            return Err(OpError::Shape(format!(
                "MoeRouter top_k {} must be in 1..={}",
                top_k, num_experts
            )));
        }
        Ok(Self {
            gate,
            num_experts,
            input_features,
            top_k,
            renormalize,
        })
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn input_features(&self) -> usize {
        self.input_features
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn renormalize(&self) -> bool {
        self.renormalize
    }

    pub fn forward(
        &self,
        input: &Tensor<T, D>,
        logits: &mut Tensor<T, D>,
        expert_ids: &mut Tensor<i32, D>,
        expert_weights: &mut Tensor<f32, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let input_shape = input.shape().as_slice();
        if input_shape.len() != 2 || input_shape[0] == 0 || input_shape[1] != self.input_features {
            return Err(OpError::Shape(format!(
                "MoeRouter input must be non-empty [tokens,{}], got {:?}",
                self.input_features, input_shape
            )));
        }
        let tokens = input_shape[0];
        let logits_shape = logits.shape().as_slice();
        if logits_shape != [tokens, self.num_experts] {
            return Err(OpError::Shape(format!(
                "MoeRouter logits must be [{},{}], got {:?}",
                tokens, self.num_experts, logits_shape
            )));
        }
        let route_shape = [tokens, self.top_k];
        if expert_ids.shape().as_slice() != route_shape {
            return Err(OpError::Shape(format!(
                "MoeRouter expert_ids must be {:?}, got {:?}",
                route_shape,
                expert_ids.shape().as_slice()
            )));
        }
        if expert_weights.shape().as_slice() != route_shape {
            return Err(OpError::Shape(format!(
                "MoeRouter expert_weights must be {:?}, got {:?}",
                route_shape,
                expert_weights.shape().as_slice()
            )));
        }

        self.gate.forward(input, logits, ctx)?;
        D::moe_route_topk(
            ctx,
            logits,
            expert_ids,
            expert_weights,
            self.top_k,
            self.renormalize,
        )
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
    fn router_exposes_gate_contract() {
        let gate = Linear::new(
            Tensor::from_host_slice(&[0.0f32; 4 * 3], [4, 3], &Cpu).unwrap(),
            None,
        );
        let router = MoeRouter::new(gate, 2, true).unwrap();

        assert_eq!(router.num_experts(), 4);
        assert_eq!(router.input_features(), 3);
        assert_eq!(router.top_k(), 2);
        assert!(router.renormalize());
    }

    #[test]
    fn router_rejects_invalid_top_k() {
        let gate = Linear::new(
            Tensor::from_host_slice(&[0.0f32; 4 * 3], [4, 3], &Cpu).unwrap(),
            None,
        );
        let err = MoeRouter::new(gate, 5, true).err().unwrap();

        assert!(err.to_string().contains("top_k 5"));
    }

    #[test]
    fn cpu_router_is_explicitly_unsupported() {
        let cpu = Cpu;
        let gate = Linear::new(
            Tensor::from_host_slice(&[1.0f32; 4 * 3], [4, 3], &cpu).unwrap(),
            None,
        );
        let router = MoeRouter::new(gate, 2, true).unwrap();
        let input = Tensor::from_host_slice(&[1.0f32; 2 * 3], [2, 3], &cpu).unwrap();
        let mut logits = Tensor::from_host_slice(&[0.0f32; 2 * 4], [2, 4], &cpu).unwrap();
        let mut ids = Tensor::from_host_slice(&[0i32; 2 * 2], [2, 2], &cpu).unwrap();
        let mut weights = Tensor::from_host_slice(&[0.0f32; 2 * 2], [2, 2], &cpu).unwrap();
        let scope = HostScope::new(cpu);
        let plan = decode_plan(2);
        let ctx = StepCtx::new(&scope, &plan);

        let err = router
            .forward(&input, &mut logits, &mut ids, &mut weights, &ctx)
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
    fn cuda_router_runs_gate_linear_then_top_k() {
        use crate::infrastructure::cuda::{Cuda, CudaScope};
        use half::bf16;

        let cuda = Cuda::new(0).unwrap();
        let gate_weight = [
            1.0, 0.0, // expert 0
            0.0, 1.0, // expert 1
            1.0, 1.0, // expert 2
            -1.0, -1.0, // expert 3
        ]
        .map(bf16::from_f32);
        let gate = Linear::new(
            Tensor::from_host_slice(&gate_weight, [4, 2], &cuda).unwrap(),
            None,
        );
        let router = MoeRouter::new(gate, 2, true).unwrap();
        let input =
            Tensor::from_host_slice(&[1.0, 2.0, 2.0, 1.0].map(bf16::from_f32), [2, 2], &cuda)
                .unwrap();
        let mut logits = Tensor::<bf16, Cuda>::zeros([2, 4], &cuda).unwrap();
        let mut ids = Tensor::<i32, Cuda>::zeros([2, 2], &cuda).unwrap();
        let mut weights = Tensor::<f32, Cuda>::zeros([2, 2], &cuda).unwrap();
        let scope = CudaScope::new(cuda);
        let plan = decode_plan(2);
        let ctx = StepCtx::new(&scope, &plan);

        router
            .forward(&input, &mut logits, &mut ids, &mut weights, &ctx)
            .unwrap();

        assert_eq!(ids.to_host_vec().unwrap(), vec![2, 1, 2, 0]);
        let got = weights.to_host_vec().unwrap();
        let expected = [0.7310586, 0.2689414, 0.7310586, 0.2689414];
        for (actual, expected) in got.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }
}
