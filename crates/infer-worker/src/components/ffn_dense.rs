use crate::components::linear::Linear;
use crate::components::norm::RmsNorm;
use crate::domain::component::{Component, Hidden, StageKind};
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
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
pub struct DenseFfn<T: Dtype, D: LlmBackend> {
    pub post_attention_layernorm: RmsNorm<T, D>,
    pub gate_up_proj: Linear<T, D>,
    pub down_proj: Linear<T, D>,
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
        let gate_cols = self.gate_up_proj.weight.shape().as_slice()[0];
        let inter = gate_cols / 2;
        let dev = input.device().clone();
        let mut gate_up = D::alloc_tensor(Shape::from_slice(&[num_tokens, gate_cols]), &dev)?;
        let mut swiglu = D::alloc_tensor(Shape::from_slice(&[num_tokens, inter]), &dev)?;
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
        let mut normed = D::alloc_tensor(Shape::from_slice(&[num_tokens, dim]), &dev)?;
        let mut ffn_out = D::alloc_tensor(Shape::from_slice(&[num_tokens, dim]), &dev)?;
        // Pre-FFN norm — residual stays intact in `hidden.stream`.
        self.post_attention_layernorm
            .forward(&hidden.stream, &mut normed, ctx)?;
        self.project(&normed, &mut ffn_out, ctx)?;
        // Residual update: hidden.stream += FFN output.
        D::add_inplace(ctx.scope(), &mut hidden.stream, &ffn_out)
    }
}
