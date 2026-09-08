use std::rc::Rc;

use super::{FullAttention, GatedDeltaNet};
use crate::domain::cache::{LayerCacheSpec, LayerCacheView};
use crate::domain::component::{Component, Hidden};
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::forward_scratch::ForwardScratch;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};

#[allow(clippy::large_enum_variant)]
pub enum Attention<T: Dtype, D: LlmBackend> {
    Full(FullAttention<T, D>),
    Linear(GatedDeltaNet<T, D>),
}

impl<T: Dtype, D: LlmBackend> Attention<T, D> {
    pub(crate) fn cache_spec(&self) -> LayerCacheSpec {
        match self {
            Self::Full(attention) => LayerCacheSpec::Full {
                kv_dim: attention.kv_head_num * attention.head_dim,
            },
            Self::Linear(gdn) => LayerCacheSpec::Linear(gdn.dims()),
        }
    }

    pub(crate) fn install_scratch(&mut self, scratch: Rc<ForwardScratch<T, D>>) {
        if let Self::Full(attention) = self {
            attention.scratch = Some(scratch);
        }
    }

    pub fn run(
        &self,
        hidden: &mut Hidden<T, D>,
        cache: LayerCacheView<'_, T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        match (self, cache) {
            (Self::Full(attention), LayerCacheView::Full(mut kv)) => {
                attention.run(hidden, Some(&mut kv), ctx)
            }
            (Self::Linear(gdn), LayerCacheView::Linear(state)) => gdn.run(hidden, state, ctx),
            _ => Err(OpError::Shape(
                "attention and layer cache types do not match".into(),
            )),
        }
    }
}
