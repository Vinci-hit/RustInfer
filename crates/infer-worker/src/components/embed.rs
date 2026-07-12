use crate::domain::component::Hidden;
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::ports::OpResult;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::tensor::Tensor;

/// Token embedding table. Initializes the residual stream (`hidden.stream`)
/// from input token ids. Not a `Component` — embedding runs once at the model
/// boundary (`DecoderModel::embed`), not inside the per-layer stage list.
pub struct Embed<T: Dtype, D: LlmBackend> {
    pub table: Tensor<T, D>,
}

impl<T: Dtype, D: LlmBackend> Embed<T, D> {
    pub fn forward(
        &self,
        input_ids: &Tensor<i32, D>,
        hidden: &mut Hidden<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        D::embedding(ctx.scope(), &self.table, input_ids, &mut hidden.stream)
    }
}
