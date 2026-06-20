use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::ports::OpResult;
use crate::domain::ports::math_ops::MathOps;
use crate::domain::tensor::Tensor;

pub trait DiffusionOps: MathOps {
    fn conv2d<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: Option<&Tensor<T, Self>>,
        output: &mut Tensor<T, Self>,
        stride: usize,
        padding: usize,
    ) -> OpResult<()>;

    fn groupnorm<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_groups: usize,
        eps: f32,
    ) -> OpResult<()>;

    fn groupnorm_silu<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_groups: usize,
        eps: f32,
    ) -> OpResult<()>;

    fn layernorm<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()>;

    fn upsample_nearest_2x<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        input: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    fn apply_rope_interleaved<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        x: &mut Tensor<T, Self>,
        cos: &Tensor<f32, Self>,
        sin: &Tensor<f32, Self>,
        head_dim: usize,
    ) -> OpResult<()>;

    fn tanh_inplace<T: Dtype>(ctx: &StepCtx<'_, Self>, x: &mut Tensor<T, Self>) -> OpResult<()>;
}
