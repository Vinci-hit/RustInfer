//! Computational contract for hidden-conditioned draft heads. Cache ownership,
//! feature-layer selection and speculative acceptance live outside the head.
use super::{
    cache::{CacheLayout, ModelCacheView},
    dtype::Dtype,
    exec::StepCtx,
    model::ModelDims,
    ports::{OpResult, backend::LlmBackend},
    tensor::Tensor,
};

pub trait ConditionedDraft<T: Dtype, D: LlmBackend> {
    fn dims(&self) -> ModelDims;
    fn device(&self) -> &D;
    fn cache_layout(&self) -> &CacheLayout;
    fn feature_width(&self) -> usize;
    fn logits_vocab_size(&self) -> usize {
        self.dims().vocab_size
    }
    /// Absolute target IDs for a compact draft vocabulary; absent for identity.
    fn token_map(&self) -> Option<&Tensor<i32, D>> {
        None
    }
    fn prepare(&mut self, capacity: usize, batch: usize) -> OpResult<()>;
    fn project_features_into(
        &self,
        features: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()>;
    /// Output is the head's recurrent conditioning representation, which may
    /// be normalized (MTP) or unnormalized (EAGLE3).
    fn forward_hidden_into(
        &self,
        next_tokens: &Tensor<i32, D>,
        conditioning: &Tensor<T, D>,
        cache: &mut ModelCacheView<'_, T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()>;
    fn project_logits_into(
        &self,
        hidden: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()>;
}

/// A single block diffusion pass. Confirmed target features populate context
/// K/V separately from the temporary masked block; no autoregressive shift.
pub trait BlockDraft<T: Dtype, D: LlmBackend> {
    fn dims(&self) -> ModelDims;
    fn device(&self) -> &D;
    fn cache_layout(&self) -> &CacheLayout;
    fn feature_width(&self) -> usize;
    fn block_size(&self) -> usize;
    fn mask_token_id(&self) -> i32;
    fn prepare(&mut self, capacity: usize) -> OpResult<()>;
    fn cache_features(
        &self,
        features: &Tensor<T, D>,
        cache: &mut ModelCacheView<'_, T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()>;
    /// Full-visibility [anchor, masks...] -> logits for mask positions only.
    fn forward_block(
        &self,
        ids: &Tensor<i32, D>,
        cache: &mut ModelCacheView<'_, T, D>,
        logits: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()>;
}
