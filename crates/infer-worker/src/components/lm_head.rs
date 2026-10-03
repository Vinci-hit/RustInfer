use crate::components::linear::Linear;
use infer_core::dtype::Dtype;
use infer_core::exec::StepCtx;
use infer_core::ports::OpResult;
use infer_core::ports::backend::LlmBackend;
use infer_core::tensor::Tensor;

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
