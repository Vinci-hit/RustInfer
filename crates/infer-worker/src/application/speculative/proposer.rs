//! Sequence-local proposal policy. Target verification and cache commits are
//! shared by autoregressive and block-parallel draft models.
use crate::domain::{
    dtype::Dtype,
    model::ModelDims,
    ports::{OpResult, backend::LlmBackend},
    tensor::Tensor,
};

pub trait DraftProposer<T: Dtype, D: LlmBackend> {
    fn dims(&self) -> ModelDims;
    fn feature_width(&self) -> usize;
    /// Number of target tokens whose features have been observed.
    fn context_len(&self) -> usize;
    /// Physical draft history can differ from the observed target length.
    fn committed_len(&self) -> usize;
    fn prepare_metrics(&self, scope: &D::Scope) -> OpResult<()>;
    fn reset(&mut self);
    fn observe_with_input(
        &mut self,
        ids: &[i32],
        start: usize,
        hidden: &Tensor<T, D>,
        scope: &D::Scope,
        device_input: Option<&Tensor<i32, D>>,
    ) -> OpResult<()>;
    fn observe(
        &mut self,
        ids: &[i32],
        start: usize,
        hidden: &Tensor<T, D>,
        scope: &D::Scope,
    ) -> OpResult<()> {
        self.observe_with_input(ids, start, hidden, scope, None)
    }
    /// Returns drafts and the matching [anchor, drafts...] device tape, valid
    /// until the next proposer operation. Drafting never commits history.
    fn draft_with_device(
        &mut self,
        pending: i32,
        count: usize,
        scope: &D::Scope,
    ) -> OpResult<(Vec<i32>, Tensor<i32, D>)>;
}
