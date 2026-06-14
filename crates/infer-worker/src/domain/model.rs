//! Model trait — pure computation interface.

use super::batch::{BatchPlan, PagedKvPool};
use super::ports::{MemoryPort, OpResult};
use super::tensor::Tensor;
use super::types::Dtype;

/// Minimal scratch-buffer surface a decoder model needs during forward.
///
/// The concrete workspace is an application/runtime concern; domain only
/// specifies the views a model may request.
pub trait LlmForwardWorkspace<T: Dtype, D: MemoryPort> {
    fn x_view(&self, n: usize) -> Tensor<T, D>;
    fn h_view(&self, n: usize) -> Tensor<T, D>;
    fn qkv_view(&self, n: usize) -> Tensor<T, D>;
    fn attn_out_view(&self, n: usize) -> Tensor<T, D>;
    fn gate_up_view(&self, n: usize) -> Tensor<T, D>;
    fn gate_view(&self, n: usize) -> Tensor<T, D>;
    fn ffn_view(&self, n: usize) -> Tensor<T, D>;
    fn o_out_view(&self, n: usize) -> Tensor<T, D>;
    fn logits_view(&self, n: usize) -> Tensor<T, D>;
    fn flash_decode_workspace(&mut self) -> &mut Tensor<f32, D>;
}

/// Context passed by the application layer to the model during forward.
/// The model borrows everything — it never owns runtime resources.
///
/// Carries the per-step `BatchPlan`, the worker-owned paged KV pool, and a
/// scratch workspace whose pre-allocated tensors the model uses instead of
/// calling `D::alloc_tensor` per call.
pub struct ForwardContext<'a, T: Dtype, D: MemoryPort, W: LlmForwardWorkspace<T, D>> {
    pub kv_pool: &'a mut PagedKvPool<T, D>,
    pub plan: &'a BatchPlan<D>,
    pub workspace: &'a mut W,
}

/// The core LLM model trait — pure computation, no resource ownership.
pub trait LlmModel<T: Dtype, D: MemoryPort, W: LlmForwardWorkspace<T, D>> {
    fn forward(
        &self,
        input_ids: &Tensor<i32, D>,
        ctx: &mut ForwardContext<'_, T, D, W>,
    ) -> OpResult<Tensor<T, D>>;
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
