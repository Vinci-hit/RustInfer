//! Model trait — pure computation interface.

use super::batch::{BatchPlan, PagedKvPool};
use super::forward_workspace::ForwardWorkspace;
use super::ports::{MemoryPort, OpResult};
use super::tensor::Tensor;
use super::types::Dtype;

/// Context passed by the application layer to the model during forward.
/// The model borrows everything — it never owns runtime resources.
///
/// Carries the per-step `BatchPlan`, the worker-owned paged KV pool, and a
/// scratch `ForwardWorkspace` whose pre-allocated tensors the model uses
/// instead of calling `D::alloc_tensor` per call. Address stability of the
/// workspace is required for CUDA Graph capture.
pub struct ForwardContext<'a, T: Dtype, D: MemoryPort> {
    pub kv_pool: &'a mut PagedKvPool<T, D>,
    pub plan: &'a BatchPlan<D>,
    pub workspace: &'a mut ForwardWorkspace<T, D>,
}

/// The core LLM model trait — pure computation, no resource ownership.
pub trait LlmModel<T: Dtype, D: MemoryPort> {
    fn forward(&self, input_ids: &Tensor<i32, D>, ctx: &mut ForwardContext<'_, T, D>) -> OpResult<Tensor<T, D>>;
    fn num_layers(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn dim(&self) -> usize;
    fn kv_dim(&self) -> usize;
    /// Q projection output dim = `head_num * head_dim`. May differ from
    /// `dim()` (e.g. Qwen3-4B has `dim=2560`, `q_dim=4096`).
    fn q_dim(&self) -> usize;
    fn head_num(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn kv_head_num(&self) -> usize;
    fn intermediate_size(&self) -> usize;
}
