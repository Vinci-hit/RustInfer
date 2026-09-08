use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::ports::OpResult;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::tensor::Tensor;

/// RMSNorm weights. Held by the sublayer that consumes it — `FullAttention` (input
/// norm), the FFN (post-attention norm), and the model (final norm) — so
/// "each layer owns its own norms" (inv 7).
pub struct RmsNorm<T: Dtype, D: LlmBackend> {
    pub weight: Tensor<T, D>,
    pub eps: f32,
    pub zero_centered: bool,
}

impl<T: Dtype, D: LlmBackend> RmsNorm<T, D> {
    /// Out-of-place RMSNorm: `output = rmsnorm(input)`. Leaves `input` (the
    /// residual stream) untouched so the residual survives the normalization —
    /// the pre-norm transformer invariant the sublayers rely on.
    pub fn forward(
        &self,
        input: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        if self.zero_centered {
            D::rmsnorm_zero_centered(ctx.scope(), input, &self.weight, output, self.eps)
        } else {
            D::rmsnorm(ctx.scope(), input, &self.weight, output, self.eps)
        }
    }
    pub fn add_forward(
        &self,
        residual: &mut Tensor<T, D>,
        delta: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        if self.zero_centered {
            D::add_inplace(ctx.scope(), residual, delta)?;
            self.forward(residual, output, ctx)
        } else {
            D::fused_add_rmsnorm(ctx, output, residual, delta, &self.weight, self.eps)
        }
    }
}
