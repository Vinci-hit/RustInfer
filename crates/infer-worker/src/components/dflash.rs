//! DFlash's context projection and block diffusion decoder. Uses the common
//! attention/FFN components; cache ownership and acceptance live in the proposer.
use super::{DenseFfn, Embed, FullAttention, Linear, LmHead, RmsNorm};
use crate::domain::{
    cache::{CacheLayout, LayerCacheSpec, LayerCacheView, ModelCacheView},
    component::{Component, Hidden, LayerRange},
    draft::BlockDraft,
    dtype::Dtype,
    exec::StepCtx,
    forward_scratch::ForwardScratch,
    model::ModelDims,
    ports::{OpError, OpResult, backend::LlmBackend},
    tensor::Tensor,
};
use std::rc::Rc;

pub struct DFlashLayer<T: Dtype, D: LlmBackend> {
    pub attention: FullAttention<T, D>,
    pub ffn: DenseFfn<T, D>,
}

pub struct DFlashWeights<T: Dtype, D: LlmBackend> {
    pub embedding: Embed<T, D>,
    pub feature_projection: Linear<T, D>,
    pub hidden_norm: RmsNorm<T, D>,
    pub layers: Vec<DFlashLayer<T, D>>,
    pub output_norm: RmsNorm<T, D>,
    pub lm_head: LmHead<T, D>,
}

pub struct DFlashDraftHead<T: Dtype, D: LlmBackend> {
    weights: DFlashWeights<T, D>,
    dims: ModelDims,
    feature_width: usize,
    block_size: usize,
    mask_token_id: i32,
    layout: CacheLayout,
    scratch: Option<DFlashScratch<T, D>>,
}

struct DFlashScratch<T: Dtype, D: LlmBackend> {
    projected: Tensor<T, D>,
    context: Tensor<T, D>,
    hidden: Tensor<T, D>,
    normalized: Tensor<T, D>,
    forward: Rc<ForwardScratch<T, D>>,
    capacity: usize,
}

impl<T: Dtype, D: LlmBackend> DFlashDraftHead<T, D> {
    pub fn new(
        weights: DFlashWeights<T, D>,
        dims: ModelDims,
        feature_count: usize,
        block_size: usize,
        mask_token_id: i32,
    ) -> OpResult<Self> {
        dims.validate()?;
        let feature_width = dims
            .dim
            .checked_mul(feature_count)
            .ok_or_else(|| OpError::Shape("DFlash feature width overflow".into()))?;
        if feature_count == 0
            || block_size < 2
            || mask_token_id < 0
            || mask_token_id as usize >= dims.vocab_size
            || dims.num_layers == 0
            || weights.layers.len() != dims.num_layers
            || weights.feature_projection.out_features() != dims.dim
            || weights.lm_head.proj.out_features() != dims.vocab_size
        {
            return Err(OpError::Shape("invalid DFlash geometry".into()));
        }
        Ok(Self {
            weights,
            dims,
            feature_width,
            block_size,
            mask_token_id,
            layout: CacheLayout::new((0..dims.num_layers).map(|_| LayerCacheSpec::Full {
                kv_dim: dims.kv_dim,
            }))?,
            scratch: None,
        })
    }
    fn scratch(&self, n: usize) -> OpResult<&DFlashScratch<T, D>> {
        self.scratch
            .as_ref()
            .filter(|s| n > 0 && n <= s.capacity)
            .ok_or_else(|| OpError::Shape("DFlash workspace not prepared or exceeded".into()))
    }
}

impl<T: Dtype, D: LlmBackend> BlockDraft<T, D> for DFlashDraftHead<T, D> {
    fn dims(&self) -> ModelDims {
        self.dims
    }
    fn device(&self) -> &D {
        self.weights.embedding.table.device()
    }
    fn cache_layout(&self) -> &CacheLayout {
        &self.layout
    }
    fn feature_width(&self) -> usize {
        self.feature_width
    }
    fn block_size(&self) -> usize {
        self.block_size
    }
    fn mask_token_id(&self) -> i32 {
        self.mask_token_id
    }
    fn prepare(&mut self, capacity: usize) -> OpResult<()> {
        if capacity < 2 {
            return Err(OpError::Shape(
                "DFlash requires at least two workspace rows".into(),
            ));
        }
        let alloc = || Tensor::zeros([capacity, self.dims.dim], self.device());
        let forward = ForwardScratch::new(
            self.device(),
            ModelDims {
                vocab_size: 0,
                ..self.dims
            },
            capacity,
            1,
        )?;
        self.scratch = Some(DFlashScratch {
            projected: alloc()?,
            context: alloc()?,
            hidden: alloc()?,
            normalized: alloc()?,
            forward: forward.clone(),
            capacity,
        });
        for layer in &mut self.weights.layers {
            layer.attention.scratch = Some(forward.clone());
            layer.ffn.scratch = Some(forward.clone());
        }
        Ok(())
    }
    fn cache_features(
        &self,
        features: &Tensor<T, D>,
        cache: &mut ModelCacheView<'_, T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let n = ctx.plan().num_tokens;
        let scratch = self.scratch(n)?;
        if ctx.plan().batch != 1 || features.shape().as_slice() != [n, self.feature_width] {
            return Err(OpError::Shape(
                "DFlash target feature shape mismatch".into(),
            ));
        }
        cache.validate(
            &self.layout,
            LayerRange::all(self.dims.num_layers),
            ctx.plan(),
        )?;
        let mut projected = scratch.projected.narrow(0, 0, n)?;
        let mut context = scratch.context.narrow(0, 0, n)?;
        self.weights
            .feature_projection
            .forward(features, &mut projected, ctx)?;
        self.weights
            .hidden_norm
            .forward(&projected, &mut context, ctx)?;
        // Every layer projects the same target features with its own weights.
        // This path never runs a draft attention/MLP over confirmed history.
        for (layer, binding) in self.weights.layers.iter().zip(self.layout.layers()) {
            let LayerCacheView::Full(mut kv) = cache.layer(*binding)? else {
                return Err(OpError::Shape("DFlash requires full-attention KV".into()));
            };
            layer
                .attention
                .core
                .cache_context(&context, &mut kv, &scratch.forward, ctx)?;
        }
        Ok(())
    }
    fn forward_block(
        &self,
        ids: &Tensor<i32, D>,
        cache: &mut ModelCacheView<'_, T, D>,
        logits: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let n = ctx.plan().num_tokens;
        let scratch = self.scratch(n)?;
        if ctx.plan().batch != 1
            || ctx.plan().attention_is_causal()?
            || !(2..=self.block_size).contains(&n)
            || ids.shape().as_slice() != [n]
            || logits.shape().as_slice() != [n - 1, self.dims.vocab_size]
        {
            return Err(OpError::Shape(
                "DFlash requires a full-visibility masked block".into(),
            ));
        }
        cache.validate(
            &self.layout,
            LayerRange::all(self.dims.num_layers),
            ctx.plan(),
        )?;
        let mut hidden = Hidden {
            stream: scratch.hidden.narrow(0, 0, n)?,
            pending: None,
        };
        self.weights.embedding.forward(ids, &mut hidden, ctx)?;
        for (layer, binding) in self.weights.layers.iter().zip(self.layout.layers()) {
            let LayerCacheView::Full(mut kv) = cache.layer(*binding)? else {
                return Err(OpError::Shape("DFlash requires full-attention KV".into()));
            };
            layer.attention.run(&mut hidden, Some(&mut kv), ctx)?;
            layer.ffn.run(&mut hidden, None, ctx)?;
        }
        let mut normalized = scratch.normalized.narrow(0, 0, n)?;
        match hidden.pending.take() {
            Some(delta) => self.weights.output_norm.add_forward(
                &mut hidden.stream,
                &delta,
                &mut normalized,
                ctx,
            )?,
            None => self
                .weights
                .output_norm
                .forward(&hidden.stream, &mut normalized, ctx)?,
        }
        // Position zero is the known anchor, not a predicted token.
        self.weights
            .lm_head
            .proj
            .forward(&normalized.narrow(0, 1, n - 1)?, logits, ctx)
    }
}
