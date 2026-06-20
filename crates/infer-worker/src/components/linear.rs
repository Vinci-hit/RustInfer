use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::ports::OpResult;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::tensor::Tensor;

pub struct Linear<T: Dtype, D: LlmBackend> {
    pub weight: Tensor<T, D>,
    pub bias: Option<Tensor<T, D>>,
}

impl<T: Dtype, D: LlmBackend> Linear<T, D> {
    pub fn forward(
        &self,
        input: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        D::matmul(ctx.scope(), input, &self.weight, output)?;
        if let Some(bias) = &self.bias {
            D::broadcast_add_inplace(ctx.scope(), output, bias)?;
        }
        Ok(())
    }
}
