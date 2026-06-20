use std::marker::PhantomData;

use crate::domain::dtype::Dtype;
use crate::domain::dtype::quant::QuantScheme;
use crate::domain::exec::ExecScope;
use crate::domain::ports::OpResult;
use crate::domain::ports::math_ops::MathOps;
use crate::domain::tensor::Tensor;

pub struct QuantLinear<A: Dtype, W: Dtype, O: Dtype, D: MathOps> {
    pub weight: Tensor<W, D>,
    pub scales: Tensor<A, D>,
    pub zeros: Option<Tensor<W, D>>,
    pub scheme: QuantScheme,
    pub _out: PhantomData<O>,
}

impl<A: Dtype, W: Dtype, O: Dtype, D: MathOps> QuantLinear<A, W, O, D> {
    pub fn forward(
        &self,
        scope: &D::Scope,
        input: &Tensor<A, D>,
        output: &mut Tensor<O, D>,
    ) -> OpResult<()> {
        let _guard = scope.enter();
        D::matmul_quant(
            scope,
            input,
            &self.weight,
            output,
            &self.scales,
            self.zeros.as_ref(),
            &self.scheme,
        )
    }
}
