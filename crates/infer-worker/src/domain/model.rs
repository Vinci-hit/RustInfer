//! Model trait — pure computation interface.

use super::types::Dtype;
use super::tensor::Tensor;
use super::ports::{OpBackend, OpResult};

/// Context passed by the Application layer to the model during forward.
/// The model borrows everything — it never owns runtime resources.
pub struct ForwardContext<'a, T: Dtype, D: OpBackend> {
    pub k_caches: &'a mut [Tensor<T, D>],
    pub v_caches: &'a mut [Tensor<T, D>],
    pub positions: &'a [i32],
    pub seq_lens: &'a [usize],
}

/// The core LLM model trait — pure computation, no resource ownership.
pub trait LlmModel<T: Dtype, D: OpBackend> {
    fn forward(&self, input_ids: &Tensor<i32, D>, ctx: &mut ForwardContext<'_, T, D>) -> OpResult<Tensor<T, D>>;
    fn num_layers(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn dim(&self) -> usize;
    fn kv_dim(&self) -> usize;
}
