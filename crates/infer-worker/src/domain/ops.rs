//! Op-level free functions (thin wrappers that dispatch through OpBackend).

use super::types::Dtype;
use super::tensor::Tensor;
use super::ports::{OpBackend, OpResult};

pub fn add<T: Dtype, D: OpBackend>(a: &Tensor<T, D>, b: &Tensor<T, D>, dst: &mut Tensor<T, D>) -> OpResult<()> { D::add(a, b, dst) }
pub fn add_inplace<T: Dtype, D: OpBackend>(dst: &mut Tensor<T, D>, src: &Tensor<T, D>) -> OpResult<()> { D::add_inplace(dst, src) }
pub fn rmsnorm<T: Dtype, D: OpBackend>(input: &Tensor<T, D>, weight: &Tensor<T, D>, output: &mut Tensor<T, D>, eps: f32) -> OpResult<()> { D::rmsnorm(input, weight, output, eps) }
pub fn matmul<T: Dtype, D: OpBackend>(input: &Tensor<T, D>, weight: &Tensor<T, D>, output: &mut Tensor<T, D>) -> OpResult<()> { D::matmul(input, weight, output) }
pub fn silu_inplace<T: Dtype, D: OpBackend>(x: &mut Tensor<T, D>) -> OpResult<()> { D::silu_inplace(x) }
pub fn softmax<T: Dtype, D: OpBackend>(input: &Tensor<T, D>, output: &mut Tensor<T, D>) -> OpResult<()> { D::softmax(input, output) }
pub fn embedding<T: Dtype, D: OpBackend>(table: &Tensor<T, D>, indices: &Tensor<i32, D>, output: &mut Tensor<T, D>) -> OpResult<()> { D::embedding(table, indices, output) }
