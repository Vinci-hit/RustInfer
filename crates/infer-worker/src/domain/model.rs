//! Decoder model trait — the sliceable embed / decode_layers / finalize
//! contract every LLM the runtime drives implements.

use super::cache::{CacheLayout, ModelCacheView};
use super::component::{Hidden, LayerRange, StageKind};
use super::dtype::Dtype as V2Dtype;
use super::ports::OpResult;
use super::ports::backend::LlmBackend;
use super::tensor::Tensor;

#[derive(Debug, Clone, Copy, Default)]
pub struct ModelDims {
    pub dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub qkv_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_num: usize,
    pub head_dim: usize,
    pub kv_head_num: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub num_shared_experts: usize,
}

impl ModelDims {
    pub fn validate(&self) -> OpResult<()> {
        if self.head_num > 0 && self.head_dim > 0 && self.q_dim != self.head_num * self.head_dim {
            return Err(crate::domain::ports::OpError::Shape(format!(
                "q_dim={} does not equal head_num*head_dim={}",
                self.q_dim,
                self.head_num * self.head_dim
            )));
        }
        if self.kv_head_num > 0
            && self.head_dim > 0
            && self.kv_dim != self.kv_head_num * self.head_dim
        {
            return Err(crate::domain::ports::OpError::Shape(format!(
                "kv_dim={} does not equal kv_head_num*head_dim={}",
                self.kv_dim,
                self.kv_head_num * self.head_dim
            )));
        }
        Ok(())
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }
}

pub struct Logits<T: V2Dtype, D: LlmBackend>(pub Tensor<T, D>);

pub enum SampleRows<'a> {
    All,
    LastPerSeq,
    Explicit(&'a [i32]),
}

pub trait DecoderModel<T: V2Dtype, D: LlmBackend> {
    /// (rotary dimensions, theta, interleaved T/H/W frequency counts).
    fn multimodal_rope(&self) -> Option<(usize, f64, [usize; 3])> {
        None
    }

    fn encode_image(
        &self,
        _image: &infer_protocol::multimodal::ImageInput,
        _scope: &D::Scope,
    ) -> OpResult<Tensor<T, D>> {
        Err(crate::domain::ports::OpError::unsupported(
            "model",
            "image inputs",
        ))
    }

    fn dims(&self) -> ModelDims;
    fn cache_layout(&self) -> &CacheLayout;
    fn stages(&self) -> &[StageKind];

    /// Install the shared, address-stable per-layer forward scratch into the
    /// model's sublayers. Called once by `Runtime::new`. Default: no-op (for
    /// models that don't carry component scratch, e.g. speculative wrappers).
    fn install_scratch(
        &mut self,
        _scratch: std::rc::Rc<crate::domain::forward_scratch::ForwardScratch<T, D>>,
    ) {
    }

    fn install_gdn_scratch(
        &mut self,
        _scratch: std::rc::Rc<crate::domain::gdn_scratch::GdnScratch<T, D>>,
    ) -> OpResult<()> {
        Ok(())
    }

    fn embed(
        &self,
        input_ids: &Tensor<i32, D>,
        hidden: &mut Hidden<T, D>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<()>;

    fn decode_layers(
        &self,
        range: LayerRange,
        hidden: &mut Hidden<T, D>,
        cache: &mut ModelCacheView<'_, T, D>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<()>;

    fn finalize(
        &self,
        hidden: &Hidden<T, D>,
        rows: SampleRows<'_>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<Logits<T, D>>;

    fn forward(
        &self,
        input_ids: &Tensor<i32, D>,
        hidden: &mut Hidden<T, D>,
        cache: &mut ModelCacheView<'_, T, D>,
        rows: SampleRows<'_>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<Logits<T, D>> {
        self.embed(input_ids, hidden, ctx)?;
        self.decode_layers(LayerRange::all(self.dims().num_layers), hidden, cache, ctx)?;
        self.finalize(hidden, rows, ctx)
    }
}

/// Optional target-model readout used by hidden-conditioned draft heads.
/// Outputs belong to the caller and must not alias model forward scratch.
/// Implementations use static dispatch; ordinary decoders need not expose it.
pub trait DecoderReadout<T: V2Dtype, D: LlmBackend>: DecoderModel<T, D> {
    /// Apply the final model norm to a fully materialized residual stream.
    fn normalize_hidden_into(
        &self,
        hidden: &Hidden<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<()>;
    /// Project already normalized hidden states through the shared LM head.
    fn project_logits_into(
        &self,
        normalized: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &crate::domain::exec::StepCtx<'_, D>,
    ) -> OpResult<()>;
}
