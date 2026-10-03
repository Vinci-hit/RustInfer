use crate::components::attention::Attention;
use crate::domain::cache::LayerCacheView;
use infer_core::component::{Component, Hidden};
use infer_core::dtype::Dtype;
use infer_core::exec::StepCtx;
use infer_core::ports::OpResult;
use infer_core::ports::backend::LlmBackend;

/// One transformer decoder layer: a pre-norm attention sublayer followed by a
/// pre-norm FFN sublayer, both operating on the carried residual stream. Each
/// sublayer owns its own input norm (inv 7), so the block is just their
/// composition. `F` is the FFN — `DenseFfn` or `MoeFfn` (the dense↔MoE swap).
pub struct DecoderBlock<T: Dtype, D: LlmBackend, F: Component<T, D>> {
    pub attention: Attention<T, D>,
    pub ffn: F,
}

impl<T: Dtype, D: LlmBackend, F: Component<T, D>> DecoderBlock<T, D, F> {
    pub fn run(
        &self,
        hidden: &mut Hidden<T, D>,
        cache: LayerCacheView<'_, T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        self.attention.run(hidden, cache, ctx)?;
        self.ffn.run(hidden, None, ctx)
    }
}
