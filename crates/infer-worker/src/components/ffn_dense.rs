use std::rc::Rc;

use crate::components::linear::Linear;
use crate::components::norm::RmsNorm;
use crate::domain::component::{Component, Hidden, StageKind};
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::forward_scratch::ForwardScratch;
use crate::domain::kv::KvView;
use crate::domain::ports::OpResult;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::tensor::Tensor;
use crate::domain::types::Shape;

/// Pre-norm dense SwiGLU FFN sublayer.
///
/// Reads the residual, normalizes out-of-place into private scratch, runs the
/// gate/up → SwiGLU → down projection, and adds the result back into the
/// residual. Owns its post-attention norm. Swapping this for `MoeFfn` (same
/// `Component` contract) is the entire dense↔MoE model change.
///
/// Each projection is a [`Linear`], which transparently carries either a
/// full-precision weight or an int4 group-quantized (`pack-quantized`) weight
/// — so this one struct backs both bf16 and int4-MLP models with no change to
/// the FFN dataflow.
pub struct DenseFfn<T: Dtype, D: LlmBackend> {
    pub post_attention_layernorm: RmsNorm<T, D>,
    pub gate_up_proj: Linear<T, D>,
    pub down_proj: Linear<T, D>,
    /// Shared, address-stable per-layer forward scratch (installed by
    /// `Runtime::new`). `None` → fall back to per-call device allocation.
    pub scratch: Option<Rc<ForwardScratch<T, D>>>,
}

impl<T: Dtype, D: LlmBackend> DenseFfn<T, D> {
    /// Compute `down(swiglu(gate_up(input)))` into `out` (no residual add). Used
    /// by `MoeFfn` for the shared-expert branch, which already holds the
    /// normalized input and manages the residual itself.
    pub fn project(
        &self,
        input: &Tensor<T, D>,
        out: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let num_tokens = input.shape().as_slice()[0];
        let gate_cols = self.gate_up_proj.out_features();
        let inter = gate_cols / 2;
        let dev = input.device().clone();
        // Reuse address-stable scratch when its geometry matches this FFN's
        // fused gate/up width; otherwise allocate (pooled). The `fits_ffn`
        // guard lets an MoE shared expert with a different intermediate size
        // safely fall back instead of aliasing the wrong-width buffer.
        let scratch = self
            .scratch
            .as_deref()
            .filter(|s| s.fits_ffn(num_tokens, gate_cols));
        let mut gate_up = match scratch {
            Some(s) => s.gate_up(num_tokens),
            None => D::alloc_tensor(Shape::from_slice(&[num_tokens, gate_cols]), &dev)?,
        };
        let mut swiglu = match scratch {
            Some(s) => s.swiglu(num_tokens),
            None => D::alloc_tensor(Shape::from_slice(&[num_tokens, inter]), &dev)?,
        };
        self.gate_up_proj.forward(input, &mut gate_up, ctx)?;
        D::swiglu_packed(ctx, &gate_up, &mut swiglu, num_tokens, inter)?;
        self.down_proj.forward(&swiglu, out, ctx)
    }
}

impl<T: Dtype, D: LlmBackend> Component<T, D> for DenseFfn<T, D> {
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
        let num_tokens = hidden.num_tokens();
        let dim = hidden.stream.shape().as_slice()[1];
        let dev = hidden.stream.device().clone();
        let scratch = self.scratch.as_deref().filter(|s| s.fits(num_tokens));
        let mut normed = match scratch {
            Some(s) => s.normed(num_tokens),
            None => D::alloc_tensor(Shape::from_slice(&[num_tokens, dim]), &dev)?,
        };
        let mut ffn_out = match scratch {
            Some(s) => s.ffn_out(num_tokens),
            None => D::alloc_tensor(Shape::from_slice(&[num_tokens, dim]), &dev)?,
        };
        // Pre-FFN norm, fusing the attention sublayer's deferred residual add
        // when present (stream += delta; normed = rmsnorm(stream)); else plain.
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
        self.project(&normed, &mut ffn_out, ctx)?;
        // Defer the residual add: stash `ffn_out` for the next sublayer's
        // pre-norm to fuse. `decode_layers` flushes the final leftover.
        hidden.pending = Some(ffn_out);
        Ok(())
    }
}
