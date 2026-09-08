use std::rc::Rc;

use super::linear::Linear;
use super::norm::RmsNorm;
use crate::domain::cache::{LinearDims, LinearLayerStateView};
use crate::domain::component::Hidden;
use crate::domain::dtype::{DTypeId, Dtype};
use crate::domain::exec::{ExecScope, RankPair, StepCtx};
use crate::domain::gdn_scratch::GdnScratch;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::Shape;

/// The input norm is supplied by the model builder. Checkpoint-specific norm
/// conventions are separate from the recurrent attention and its gated output norm.
pub struct GdnWeights<T: Dtype, D: LlmBackend> {
    pub input_layernorm: RmsNorm<T, D>,
    pub in_proj_qkv: Linear<T, D>,
    pub in_proj_a: Linear<T, D>,
    pub in_proj_b: Linear<T, D>,
    pub in_proj_z: Linear<T, D>,
    pub conv1d: Tensor<T, D>,
    pub a_log: Tensor<f32, D>,
    pub dt_bias: Tensor<T, D>,
    pub norm_weight: Tensor<f32, D>,
    pub norm_eps: f32,
    pub out_proj: Linear<T, D>,
}

/// Stateful token attention over an ordinary causal ragged tape. Weights and
/// scratch belong to the component; all persistent sequence state is borrowed.
pub struct GatedDeltaNet<T: Dtype, D: LlmBackend> {
    weights: GdnWeights<T, D>,
    dims: LinearDims,
    dim: usize,
    scratch: Option<Rc<GdnScratch<T, D>>>,
}

impl<T: Dtype, D: LlmBackend> GatedDeltaNet<T, D> {
    pub fn new(weights: GdnWeights<T, D>, dims: LinearDims) -> OpResult<Self> {
        dims.validate()?;
        // Match the current CUDA delta kernel before any convolution can
        // update persistent state. Keep the component contract backend-neutral.
        if dims.key_head_dim > 1024 {
            return Err(OpError::Shape(format!(
                "GDN key_head_dim {} exceeds supported limit 1024",
                dims.key_head_dim
            )));
        }
        if !matches!(T::ID, DTypeId::F32 | DTypeId::F16 | DTypeId::BF16) {
            return Err(OpError::unsupported("GatedDeltaNet", "activation dtype"));
        }
        let dim = weights.input_layernorm.weight.numel();
        if dim == 0
            || weights.input_layernorm.weight.shape().as_slice() != [dim]
            || !weights.input_layernorm.eps.is_finite()
            || weights.input_layernorm.eps < 0.0
            || !weights.norm_eps.is_finite()
            || weights.norm_eps < 0.0
        {
            return Err(OpError::Shape(
                "GDN input/norm dimensions or epsilon are invalid".into(),
            ));
        }
        for (name, linear, rows, cols) in [
            ("qkv", &weights.in_proj_qkv, dims.conv_dim(), dim),
            ("a", &weights.in_proj_a, dims.num_value_heads, dim),
            ("b", &weights.in_proj_b, dims.num_value_heads, dim),
            ("z", &weights.in_proj_z, dims.value_dim(), dim),
            ("out", &weights.out_proj, dim, dims.value_dim()),
        ] {
            let weight = linear
                .weight
                .as_dense()
                .ok_or_else(|| OpError::unsupported("GatedDeltaNet", "quantized weights"))?;
            if weight.shape().as_slice() != [rows, cols]
                || !weight.is_contiguous()
                || linear
                    .bias
                    .as_ref()
                    .is_some_and(|b| b.shape().as_slice() != [rows] || !b.is_contiguous())
                || linear.parallelism().tp() != (RankPair { rank: 0, size: 1 })
            {
                return Err(OpError::Shape(format!("invalid GDN {name} projection")));
            }
        }
        for (valid, name) in [
            (
                weights.conv1d.shape().as_slice() == [dims.conv_dim(), 1, dims.conv_kernel_dim]
                    && weights.conv1d.is_contiguous(),
                "conv1d",
            ),
            (
                weights.a_log.shape().as_slice() == [dims.num_value_heads]
                    && weights.a_log.is_contiguous(),
                "A_log",
            ),
            (
                weights.dt_bias.shape().as_slice() == [dims.num_value_heads]
                    && weights.dt_bias.is_contiguous(),
                "dt_bias",
            ),
            (
                weights.norm_weight.shape().as_slice() == [dims.value_head_dim]
                    && weights.norm_weight.is_contiguous(),
                "norm",
            ),
            (weights.input_layernorm.weight.is_contiguous(), "input norm"),
        ] {
            if !valid {
                return Err(OpError::Shape(format!("invalid GDN {name} weight")));
            }
        }
        Ok(Self {
            weights,
            dims,
            dim,
            scratch: None,
        })
    }

    pub fn dims(&self) -> LinearDims {
        self.dims
    }
    pub fn hidden_dim(&self) -> usize {
        self.dim
    }

    pub fn install_scratch(&mut self, scratch: Rc<GdnScratch<T, D>>) -> OpResult<()> {
        scratch.validate(self.dim, self.dims, 0)?;
        self.scratch = Some(scratch);
        Ok(())
    }

    pub(crate) fn validate_execution(
        &self,
        hidden: &Hidden<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let topology = ctx.scope().topology();
        if topology.tp.size != 1 || topology.pp.size != 1 {
            return Err(OpError::unsupported("GatedDeltaNet", "parallel execution"));
        }
        if hidden.stream.shape().as_slice() != [ctx.plan().num_tokens, self.dim]
            || !hidden.stream.is_contiguous()
            || hidden.pending.as_ref().is_some_and(|p| {
                p.shape().as_slice() != hidden.stream.shape().as_slice() || !p.is_contiguous()
            })
        {
            return Err(OpError::Shape("GDN hidden/pending shape mismatch".into()));
        }
        if let Some(scratch) = &self.scratch {
            scratch.validate(self.dim, self.dims, hidden.num_tokens())?;
        }
        Ok(())
    }

    /// Mutates the supplied states in place. A failed forward is not a state
    /// transaction: callers must rebuild affected states before retrying it.
    pub fn run(
        &self,
        hidden: &mut Hidden<T, D>,
        mut state: LinearLayerStateView<'_, T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        self.validate_execution(hidden, ctx)?;
        state.validate(self.dims, ctx.plan())?;
        let tokens = hidden.num_tokens();
        let fallback;
        let scratch = if let Some(scratch) = &self.scratch {
            scratch
        } else {
            fallback = GdnScratch::new(hidden.stream.device(), self.dim, self.dims, tokens)?;
            &fallback
        };
        let mut buf = scratch.buffers(tokens)?;
        let w = &self.weights;
        match hidden.pending.take() {
            Some(delta) => {
                w.input_layernorm
                    .add_forward(&mut hidden.stream, &delta, &mut buf.normed, ctx)?
            }
            None => w
                .input_layernorm
                .forward(&hidden.stream, &mut buf.normed, ctx)?,
        }
        w.in_proj_qkv.forward(&buf.normed, &mut buf.qkv, ctx)?;
        w.in_proj_a.forward(&buf.normed, &mut buf.a, ctx)?;
        w.in_proj_b.forward(&buf.normed, &mut buf.b, ctx)?;
        w.in_proj_z.forward(&buf.normed, &mut buf.z, ctx)?;

        let (conv_state, ssm_state, slots, cu) = state.parts();
        D::causal_conv1d_silu(
            ctx.scope(),
            &buf.qkv,
            &w.conv1d,
            conv_state,
            slots,
            cu,
            &mut buf.conv,
        )?;
        let key = self.dims.key_dim();
        let value = self.dims.value_dim();
        // Unlike full attention, GDN has q/k of the same width and a wider v.
        // Contiguous destinations also serve the CPU reference implementation.
        D::split_cols(
            ctx.scope(),
            &buf.conv,
            &mut buf.q,
            tokens,
            self.dims.conv_dim(),
            0,
            key,
        )?;
        D::split_cols(
            ctx.scope(),
            &buf.conv,
            &mut buf.k,
            tokens,
            self.dims.conv_dim(),
            key,
            key,
        )?;
        D::split_cols(
            ctx.scope(),
            &buf.conv,
            &mut buf.v,
            tokens,
            self.dims.conv_dim(),
            2 * key,
            value,
        )?;
        D::gated_delta_rule(
            ctx.scope(),
            &buf.q,
            &buf.k,
            &buf.v,
            &buf.a,
            &buf.b,
            &w.a_log,
            &w.dt_bias,
            ssm_state,
            slots,
            cu,
            &mut buf.core,
        )?;
        let heads =
            Shape::from_slice(&[tokens, self.dims.num_value_heads, self.dims.value_head_dim]);
        let core = buf.core.view_contiguous(heads)?;
        let z = buf.z.view_contiguous(heads)?;
        let mut gated = buf.gated.view_contiguous(heads)?;
        D::gated_rmsnorm(
            ctx.scope(),
            &core,
            &z,
            &w.norm_weight,
            &mut gated,
            w.norm_eps,
        )?;
        w.out_proj.forward(&buf.gated, &mut buf.out, ctx)?;
        hidden.pending = Some(buf.out);
        Ok(())
    }
}
