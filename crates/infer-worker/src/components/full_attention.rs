use super::{RmsNorm, attention_core::AttentionCore};
use crate::domain::component::{Component, Hidden, StageKind};
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::forward_scratch::ForwardScratch;
use crate::domain::kv::KvView;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::types::Shape;
use std::rc::Rc;

/// Pre-norm attention with deferred residual addition. The projection core is
/// also usable by components with a different input-normalization contract.
pub struct FullAttention<T: Dtype, D: LlmBackend> {
    pub input_layernorm: RmsNorm<T, D>,
    pub core: AttentionCore<T, D>,
    pub scratch: Option<Rc<ForwardScratch<T, D>>>,
}

impl<T: Dtype, D: LlmBackend> Component<T, D> for FullAttention<T, D> {
    fn kind(&self) -> StageKind {
        StageKind::Attention
    }
    fn run(
        &self,
        hidden: &mut Hidden<T, D>,
        kv: Option<&mut KvView<'_, T, D>>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let kv = kv.ok_or_else(|| OpError::Shape("FullAttention::run: missing KV view".into()))?;
        let n = hidden.num_tokens();
        let dim = hidden.stream.shape()[1];
        let scratch = self.scratch.as_deref().filter(|s| s.fits(n));
        let mut normed = match scratch {
            Some(s) => s.normed(n),
            None => D::alloc_tensor(Shape::from_slice(&[n, dim]), hidden.stream.device())?,
        };
        let mut out = match scratch {
            Some(s) => s.o_out(n),
            None => D::alloc_tensor(Shape::from_slice(&[n, dim]), hidden.stream.device())?,
        };
        match hidden.pending.take() {
            Some(delta) => {
                self.input_layernorm
                    .add_forward(&mut hidden.stream, &delta, &mut normed, ctx)?
            }
            None => self
                .input_layernorm
                .forward(&hidden.stream, &mut normed, ctx)?,
        }
        self.core
            .project_into(&normed, kv, &mut out, scratch, ctx)?;
        hidden.pending = Some(out);
        Ok(())
    }
}
