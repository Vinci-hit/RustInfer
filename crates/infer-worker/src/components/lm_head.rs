use crate::components::linear::Linear;
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::ports::OpResult;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::tensor::Tensor;

pub struct LmHead<T: Dtype, D: LlmBackend> {
    pub proj: Linear<T, D>,
}

impl<T: Dtype, D: LlmBackend> LmHead<T, D> {
    pub fn forward(
        &self,
        hidden: &Tensor<T, D>,
        logits: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        self.proj.forward(hidden, logits, ctx)
    }
}
