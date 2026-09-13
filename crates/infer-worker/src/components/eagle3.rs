//! EAGLE3's feature fusion and conditioned decoder composition. No target
//! model, sampler, sequence history or KV allocation is owned here.
use super::{DenseFfn, Embed, Linear, LmHead, RmsNorm, attention_core::AttentionCore};
use crate::domain::{
    cache::{CacheLayout, LayerCacheSpec, LayerCacheView, ModelCacheView},
    component::{Component, Hidden, LayerRange},
    draft::ConditionedDraft,
    dtype::Dtype,
    exec::StepCtx,
    forward_scratch::ForwardScratch,
    model::ModelDims,
    ports::{OpError, OpResult, backend::LlmBackend},
    tensor::Tensor,
};
use std::rc::Rc;

pub struct Eagle3Weights<T: Dtype, D: LlmBackend> {
    pub embedding: Embed<T, D>,
    pub feature_projection: Linear<T, D>,
    pub embedding_norm: RmsNorm<T, D>,
    pub hidden_norm: RmsNorm<T, D>,
    pub attention: AttentionCore<T, D>,
    pub ffn: DenseFfn<T, D>,
    pub output_norm: RmsNorm<T, D>,
    pub lm_head: LmHead<T, D>,
}

pub struct Eagle3DraftHead<T: Dtype, D: LlmBackend> {
    weights: Eagle3Weights<T, D>,
    dims: ModelDims,
    feature_width: usize,
    token_map: Option<Tensor<i32, D>>,
    layout: CacheLayout,
    scratch: Option<Eagle3Scratch<T, D>>,
}

struct Eagle3Scratch<T: Dtype, D: LlmBackend> {
    embeddings: Tensor<T, D>,
    norm_embedding: Tensor<T, D>,
    norm_hidden: Tensor<T, D>,
    concat: Tensor<T, D>,
    output_norm: Tensor<T, D>,
    forward: Rc<ForwardScratch<T, D>>,
    capacity: usize,
}

impl<T: Dtype, D: LlmBackend> Eagle3DraftHead<T, D> {
    pub fn new(
        weights: Eagle3Weights<T, D>,
        dims: ModelDims,
        feature_count: usize,
    ) -> OpResult<Self> {
        Self::with_token_map(weights, dims, feature_count, None)
    }

    pub fn with_token_map(
        weights: Eagle3Weights<T, D>,
        dims: ModelDims,
        feature_count: usize,
        token_map: Option<Tensor<i32, D>>,
    ) -> OpResult<Self> {
        dims.validate()?;
        let feature_width = dims
            .dim
            .checked_mul(feature_count)
            .ok_or_else(|| OpError::Shape("feature width overflow".into()))?;
        if dims.dim == 0 || dims.vocab_size == 0 || dims.num_layers != 1 || feature_count == 0 {
            return Err(OpError::Shape("invalid EAGLE3 head geometry".into()));
        }
        let output_vocab = weights.lm_head.proj.out_features();
        if output_vocab == 0
            || output_vocab > dims.vocab_size
            || token_map
                .as_ref()
                .map_or(output_vocab != dims.vocab_size, |m| {
                    m.shape().as_slice() != [output_vocab] || !m.is_contiguous()
                })
        {
            return Err(OpError::Shape(
                "EAGLE3 output vocabulary and token map mismatch".into(),
            ));
        }
        Ok(Self {
            token_map,
            weights,
            dims,
            feature_width,
            layout: CacheLayout::new([LayerCacheSpec::Full {
                kv_dim: dims.kv_dim,
            }])?,
            scratch: None,
        })
    }
    fn scratch(&self, n: usize) -> OpResult<&Eagle3Scratch<T, D>> {
        self.scratch
            .as_ref()
            .filter(|s| n > 0 && n <= s.capacity)
            .ok_or_else(|| OpError::Shape("EAGLE3 workspace not prepared or exceeded".into()))
    }
}

impl<T: Dtype, D: LlmBackend> ConditionedDraft<T, D> for Eagle3DraftHead<T, D> {
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
    fn logits_vocab_size(&self) -> usize {
        self.weights.lm_head.proj.out_features()
    }
    fn token_map(&self) -> Option<&Tensor<i32, D>> {
        self.token_map.as_ref()
    }
    fn prepare(&mut self, capacity: usize, batch: usize) -> OpResult<()> {
        if capacity == 0 || batch != 1 {
            return Err(OpError::Shape(
                "EAGLE3 requires positive capacity and batch=1".into(),
            ));
        }
        let alloc = |width| Tensor::zeros([capacity, width], self.device());
        let forward = ForwardScratch::new(
            self.device(),
            ModelDims {
                vocab_size: 0,
                ..self.dims
            },
            capacity,
            1,
        )?;
        let scratch = Eagle3Scratch {
            embeddings: alloc(self.dims.dim)?,
            norm_embedding: alloc(self.dims.dim)?,
            norm_hidden: alloc(self.dims.dim)?,
            concat: alloc(2 * self.dims.dim)?,
            output_norm: alloc(self.dims.dim)?,
            forward: forward.clone(),
            capacity,
        };
        self.weights.ffn.scratch = Some(forward);
        self.scratch = Some(scratch);
        Ok(())
    }
    fn project_features_into(
        &self,
        features: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let n = ctx.plan().num_tokens;
        self.scratch(n)?;
        if features.shape().as_slice() != [n, self.feature_width]
            || output.shape().as_slice() != [n, self.dims.dim]
        {
            return Err(OpError::Shape("EAGLE3 feature shape mismatch".into()));
        }
        self.weights
            .feature_projection
            .forward(features, output, ctx)
    }
    fn forward_hidden_into(
        &self,
        next_tokens: &Tensor<i32, D>,
        conditioning: &Tensor<T, D>,
        cache: &mut ModelCacheView<'_, T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let n = ctx.plan().num_tokens;
        let scratch = self.scratch(n)?;
        if ctx.plan().batch != 1
            || next_tokens.shape().as_slice() != [n]
            || conditioning.shape().as_slice() != [n, self.dims.dim]
            || output.shape().as_slice() != [n, self.dims.dim]
        {
            return Err(OpError::Shape("EAGLE3 input/output shape mismatch".into()));
        }
        cache.validate(&self.layout, LayerRange::all(1), ctx.plan())?;
        let LayerCacheView::Full(mut kv) = cache.layer(self.layout.layers()[0])? else {
            return Err(OpError::Shape("EAGLE3 needs full-attention KV".into()));
        };
        let mut embedded = Hidden {
            stream: scratch.embeddings.narrow(0, 0, n)?,
            pending: None,
        };
        self.weights
            .embedding
            .forward(next_tokens, &mut embedded, ctx)?;
        let mut e = scratch.norm_embedding.narrow(0, 0, n)?;
        let mut h = scratch.norm_hidden.narrow(0, 0, n)?;
        self.weights
            .embedding_norm
            .forward(&embedded.stream, &mut e, ctx)?;
        self.weights
            .hidden_norm
            .forward(conditioning, &mut h, ctx)?;
        let mut cat = scratch.concat.narrow(0, 0, n)?;
        D::concat_cols(ctx.scope(), &e, &h, &mut cat)?;
        let mut attention_out = scratch.forward.o_out(n);
        self.weights.attention.project_into(
            &cat,
            &mut kv,
            &mut attention_out,
            Some(&scratch.forward),
            ctx,
        )?;
        D::copy_tensor(ctx.scope(), conditioning, output)?;
        let mut hidden = Hidden {
            stream: output.clone(),
            pending: Some(attention_out),
        };
        self.weights.ffn.run(&mut hidden, None, ctx)?;
        if let Some(delta) = hidden.pending.take() {
            D::add_inplace(ctx.scope(), &mut hidden.stream, &delta)?;
        }
        Ok(())
    }
    fn project_logits_into(
        &self,
        hidden: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let n = ctx.plan().num_tokens;
        let mut normalized = self.scratch(n)?.output_norm.narrow(0, 0, n)?;
        self.weights
            .output_norm
            .forward(hidden, &mut normalized, ctx)?;
        self.weights.lm_head.forward(&normalized, output, ctx)
    }
}
