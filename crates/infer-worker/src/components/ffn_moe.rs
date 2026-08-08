use std::rc::Rc;

use crate::components::ffn_dense::DenseFfn;
use crate::components::moe_local::MoeLocalPipeline;
use crate::components::norm::RmsNorm;
use crate::domain::component::{Component, Hidden, StageKind};
use crate::domain::dtype::Dtype;
use crate::domain::exec::{DeviceId, ExecDevice, ExecScope, StepCtx};
use crate::domain::forward_scratch::ForwardScratch;
use crate::domain::kv::KvView;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::Shape;

/// Pre-norm, single-device sparse MoE FFN sublayer.
///
/// The incoming deferred residual is fused with RMSNorm exactly like
/// [`DenseFfn`]. The local routed result becomes the next deferred residual.
/// Shared experts, quantized experts, and distributed routing remain disabled.
pub struct MoeFfn<T: Dtype, D: LlmBackend> {
    pub post_attention_layernorm: RmsNorm<T, D>,
    pub routed: MoeLocalPipeline<T, D>,
    pub shared: Option<DenseFfn<T, D>>,
    /// Shared dense-forward scratch supplies only `[tokens, hidden]` norm/output
    /// views. Route-sized MoE scratch is allocated per invocation for now.
    pub scratch: Option<Rc<ForwardScratch<T, D>>>,
}

impl<T: Dtype, D: LlmBackend> MoeFfn<T, D> {
    pub fn new(
        post_attention_layernorm: RmsNorm<T, D>,
        routed: MoeLocalPipeline<T, D>,
    ) -> OpResult<Self> {
        let hidden_features = routed.hidden_features();
        if post_attention_layernorm.weight.shape().as_slice() != [hidden_features] {
            return Err(OpError::Shape(format!(
                "MoeFfn norm weight must be [{}], got {:?}",
                hidden_features,
                post_attention_layernorm.weight.shape().as_slice()
            )));
        }
        let norm_device = <D as ExecDevice>::device_id(post_attention_layernorm.weight.device());
        if norm_device != routed.device_id() {
            return Err(OpError::Shape(format!(
                "MoeFfn norm belongs to device {}, but routed weights belong to device {}",
                norm_device.0,
                routed.device_id().0
            )));
        }
        Ok(Self {
            post_attention_layernorm,
            routed,
            shared: None,
            scratch: None,
        })
    }
}

impl<T: Dtype, D: LlmBackend> Component<T, D> for MoeFfn<T, D> {
    fn kind(&self) -> StageKind {
        StageKind::Ffn
    }

    fn run(
        &self,
        hidden: &mut Hidden<T, D>,
        kv: Option<&mut KvView<'_, T, D>>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let _ = kv;
        let backend = hidden.stream.device().name();
        if !D::local_moe_available::<T>() {
            return Err(OpError::unsupported(backend, "moe_ffn"));
        }
        if self.shared.is_some() {
            return Err(OpError::unsupported(backend, "moe_ffn.shared_expert"));
        }

        let stream_shape = hidden.stream.shape().as_slice();
        if stream_shape.len() != 2
            || stream_shape[0] == 0
            || stream_shape[1] != self.routed.hidden_features()
        {
            return Err(OpError::Shape(format!(
                "MoeFfn residual must be non-empty [tokens,{}], got {:?}",
                self.routed.hidden_features(),
                stream_shape
            )));
        }
        let num_tokens = stream_shape[0];
        let hidden_features = stream_shape[1];
        if ctx.plan().num_tokens != num_tokens {
            return Err(OpError::Shape(format!(
                "MoeFfn residual tokens {} != execution plan tokens {}",
                num_tokens,
                ctx.plan().num_tokens
            )));
        }
        if self.post_attention_layernorm.weight.shape().as_slice() != [hidden_features] {
            return Err(OpError::Shape(format!(
                "MoeFfn norm weight must be [{}], got {:?}",
                hidden_features,
                self.post_attention_layernorm.weight.shape().as_slice()
            )));
        }

        let device_id = <D as ExecDevice>::device_id(hidden.stream.device());
        let scope_device = <D as ExecDevice>::device_id(ctx.scope().device());
        if scope_device != device_id {
            return Err(OpError::Shape(format!(
                "MoeFfn residual belongs to device {}, but scope uses device {}",
                device_id.0, scope_device.0
            )));
        }
        validate_tensor(
            &hidden.stream,
            "residual",
            &[num_tokens, hidden_features],
            device_id,
        )?;
        validate_tensor(
            &self.post_attention_layernorm.weight,
            "norm weight",
            &[hidden_features],
            device_id,
        )?;
        if let Some(delta) = &hidden.pending {
            validate_tensor(
                delta,
                "incoming residual delta",
                &[num_tokens, hidden_features],
                device_id,
            )?;
        }

        let device = hidden.stream.device().clone();
        let dense_scratch = self.scratch.as_deref().filter(|s| s.fits(num_tokens));
        let mut normed = match dense_scratch {
            Some(scratch) => scratch.normed(num_tokens),
            None => D::alloc_tensor(Shape::from_slice(&[num_tokens, hidden_features]), &device)?,
        };
        let mut ffn_out = match dense_scratch {
            Some(scratch) => scratch.ffn_out(num_tokens),
            None => D::alloc_tensor(Shape::from_slice(&[num_tokens, hidden_features]), &device)?,
        };
        validate_tensor(
            &normed,
            "norm scratch",
            &[num_tokens, hidden_features],
            device_id,
        )?;
        validate_tensor(
            &ffn_out,
            "output scratch",
            &[num_tokens, hidden_features],
            device_id,
        )?;
        let mut routed_scratch = self.routed.allocate_scratch(num_tokens, &device)?;

        match hidden.pending.take() {
            Some(delta) => D::fused_add_rmsnorm(
                ctx,
                &mut normed,
                &mut hidden.stream,
                &delta,
                &self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
            )?,
            None => self
                .post_attention_layernorm
                .forward(&hidden.stream, &mut normed, ctx)?,
        }
        self.routed
            .forward(&normed, &mut ffn_out, &mut routed_scratch, ctx)?;
        hidden.pending = Some(ffn_out);
        Ok(())
    }
}

fn validate_tensor<T: Dtype, D: LlmBackend>(
    tensor: &Tensor<T, D>,
    name: &str,
    expected_shape: &[usize],
    expected_device: DeviceId,
) -> OpResult<()> {
    if tensor.shape().as_slice() != expected_shape {
        return Err(OpError::Shape(format!(
            "MoeFfn {name} must be {:?}, got {:?}",
            expected_shape,
            tensor.shape().as_slice()
        )));
    }
    if !tensor.is_contiguous() {
        return Err(OpError::Shape(format!(
            "MoeFfn {name} must be contiguous, got {:?}",
            tensor.shape().as_slice()
        )));
    }
    let tensor_device = <D as ExecDevice>::device_id(tensor.device());
    if tensor_device != expected_device {
        return Err(OpError::Shape(format!(
            "MoeFfn {name} belongs to device {}, expected device {}",
            tensor_device.0, expected_device.0
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{ExpertLinear, Linear, MoeExperts};
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

    fn cpu_ffn() -> MoeFfn<f32, Cpu> {
        let router = crate::components::MoeRouter::new(
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
        let routed =
            MoeLocalPipeline::new(router, MoeExperts::new(gate_up, down).unwrap()).unwrap();
        MoeFfn::new(
            RmsNorm {
                weight: Tensor::from_host_slice(&[1.0f32, 1.0], [2], &Cpu).unwrap(),
                eps: 0.0,
            },
            routed,
        )
        .unwrap()
    }

    #[test]
    fn cpu_moe_ffn_fails_before_consuming_pending_residual() {
        let cpu = Cpu;
        let ffn = cpu_ffn();
        let mut hidden = Hidden {
            stream: Tensor::from_host_slice(&[0.5f32, 0.25], [1, 2], &cpu).unwrap(),
            pending: Some(Tensor::from_host_slice(&[0.5f32, 0.75], [1, 2], &cpu).unwrap()),
        };
        let scope = HostScope::new(cpu);
        let plan = decode_plan(1);
        let ctx = StepCtx::new(&scope, &plan);

        let err = ffn.run(&mut hidden, None, &ctx).unwrap_err();

        assert!(matches!(
            err,
            OpError::Unsupported {
                backend: "cpu",
                op: "moe_ffn"
            }
        ));
        assert_eq!(hidden.stream.to_host_vec().unwrap(), vec![0.5, 0.25]);
        assert_eq!(
            hidden.pending.as_ref().unwrap().to_host_vec().unwrap(),
            vec![0.5, 0.75]
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_moe_ffn_fuses_incoming_residual_and_defers_routed_output() {
        use crate::domain::model::ModelDims;
        use crate::infrastructure::cuda::{Cuda, CudaScope};
        use half::bf16;

        const EXPERTS: usize = 2;
        const HIDDEN: usize = 8;
        const INTERMEDIATE: usize = 8;
        let cuda = Cuda::new(0).unwrap();
        let bf16s = |values: &[f32]| {
            values
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>()
        };

        let mut router_host = vec![0.0f32; EXPERTS * HIDDEN];
        router_host[0] = 1.0;
        let router = crate::components::MoeRouter::new(
            Linear::new(
                Tensor::from_host_slice(&bf16s(&router_host), [EXPERTS, HIDDEN], &cuda).unwrap(),
                None,
            ),
            1,
            true,
        )
        .unwrap();

        let mut gate_up_host = vec![0.0f32; EXPERTS * 2 * INTERMEDIATE * HIDDEN];
        let gate_up_index = |expert: usize, output: usize, input: usize| {
            (expert * 2 * INTERMEDIATE + output) * HIDDEN + input
        };
        gate_up_host[gate_up_index(0, 0, 0)] = 1.0;
        gate_up_host[gate_up_index(0, INTERMEDIATE, 1)] = 1.0;

        let mut down_host = vec![0.0f32; EXPERTS * HIDDEN * INTERMEDIATE];
        let down_index = |expert: usize, output: usize, input: usize| {
            (expert * HIDDEN + output) * INTERMEDIATE + input
        };
        down_host[down_index(0, 0, 0)] = 1.0;
        down_host[down_index(0, 1, 0)] = 2.0;

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
        let routed =
            MoeLocalPipeline::new(router, MoeExperts::new(gate_up, down).unwrap()).unwrap();
        let mut ffn = MoeFfn::new(
            RmsNorm {
                weight: Tensor::from_host_slice(&bf16s(&[1.0; HIDDEN]), [HIDDEN], &cuda).unwrap(),
                eps: 0.0,
            },
            routed,
        )
        .unwrap();
        let forward_scratch = ForwardScratch::new(
            &cuda,
            ModelDims {
                dim: HIDDEN,
                q_dim: HIDDEN,
                kv_dim: HIDDEN,
                qkv_dim: 3 * HIDDEN,
                intermediate_size: INTERMEDIATE,
                vocab_size: 4,
                head_num: 1,
                head_dim: HIDDEN,
                kv_head_num: 1,
                ..ModelDims::default()
            },
            1,
            1,
        )
        .unwrap();
        let mut incoming_delta = forward_scratch.o_out(1);
        incoming_delta
            .copy_from(
                &Tensor::from_host_slice(&bf16s(&[0.5; HIDDEN]), [1, HIDDEN], &cuda).unwrap(),
            )
            .unwrap();
        ffn.scratch = Some(forward_scratch);

        let mut hidden = Hidden {
            stream: Tensor::from_host_slice(&bf16s(&[0.5; HIDDEN]), [1, HIDDEN], &cuda).unwrap(),
            pending: Some(incoming_delta),
        };
        let scope = CudaScope::new(cuda.clone());
        let plan = decode_plan(1);
        let ctx = StepCtx::new(&scope, &plan);

        ffn.run(&mut hidden, None, &ctx).unwrap();

        let residual = hidden
            .stream
            .to_host_vec()
            .unwrap()
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        assert_eq!(residual, vec![1.0; HIDDEN]);

        let activation = bf16::from_f32(1.0 / (1.0 + (-1.0f32).exp())).to_f32();
        let mut expected = vec![0.0; HIDDEN];
        expected[0] = activation;
        expected[1] = bf16::from_f32(2.0 * activation).to_f32();
        let pending = hidden
            .pending
            .as_ref()
            .unwrap()
            .to_host_vec()
            .unwrap()
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        for (actual, expected) in pending.iter().zip(expected) {
            assert!((actual - expected).abs() <= 0.02, "{actual} != {expected}");
        }
    }
}
