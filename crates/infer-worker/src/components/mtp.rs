//! Hidden-conditioned draft head. Model families supply weights and a decoder.
//! KV state belongs to the caller; all mutable work buffers are head-local.
use crate::components::{Linear, RmsNorm};
use crate::domain::cache::{CacheLayout, ModelCacheView};
use crate::domain::component::{Hidden, LayerRange};
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::forward_scratch::ForwardScratch;
use crate::domain::model::{DecoderReadout, ModelDims};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;

/// Each row pairs target final-normalized h_i with token_(i+1).
/// The execution plan and KV write position must refer to i, not i+1.
pub struct MtpInput<'a, T: Dtype, D: LlmBackend> {
    pub next_token_ids: &'a Tensor<i32, D>,
    pub target_hidden: &'a Tensor<T, D>,
}

/// The decoder implementation is selected at compile time. This component
/// depends only on its domain contract, never on a concrete model family.
pub struct MtpHead<T: Dtype, D: LlmBackend, M: DecoderReadout<T, D>> {
    embedding_norm: RmsNorm<T, D>,
    hidden_norm: RmsNorm<T, D>,
    fc: Linear<T, D>,
    decoder: M,
    scratch: Option<MtpScratch<T, D>>,
}

struct MtpScratch<T: Dtype, D: LlmBackend> {
    embeddings: Tensor<T, D>,
    norm_embedding: Tensor<T, D>,
    norm_hidden: Tensor<T, D>,
    concatenated: Tensor<T, D>,
    projected: Tensor<T, D>,
    capacity: usize,
    max_batch: usize,
}

impl<T: Dtype, D: LlmBackend, M: DecoderReadout<T, D>> MtpHead<T, D, M> {
    pub(crate) fn new(
        embedding_norm: RmsNorm<T, D>,
        hidden_norm: RmsNorm<T, D>,
        fc: Linear<T, D>,
        decoder: M,
    ) -> Self {
        Self {
            embedding_norm,
            hidden_norm,
            fc,
            decoder,
            scratch: None,
        }
    }

    pub fn dims(&self) -> ModelDims {
        self.decoder.dims()
    }
    pub fn device(&self) -> &D {
        self.embedding_norm.weight.device()
    }
    pub fn cache_layout(&self) -> &CacheLayout {
        self.decoder.cache_layout()
    }

    /// Allocate a separate workspace once, before execution or graph capture.
    pub fn prepare(&mut self, max_tokens: usize, max_batch: usize) -> OpResult<()> {
        if max_tokens == 0 || max_batch == 0 || max_batch > max_tokens {
            return Err(OpError::Shape("invalid MTP workspace capacity".into()));
        }
        let dims = self.dims();
        let device = self.embedding_norm.weight.device();
        let alloc = |cols| Tensor::zeros([max_tokens, cols], device);
        let scratch = MtpScratch {
            embeddings: alloc(dims.dim)?,
            norm_embedding: alloc(dims.dim)?,
            norm_hidden: alloc(dims.dim)?,
            concatenated: alloc(2 * dims.dim)?,
            projected: alloc(dims.dim)?,
            capacity: max_tokens,
            max_batch,
        };
        // Projection uses caller-owned logits: no full-vocabulary scratch needed.
        let forward = ForwardScratch::new(
            device,
            ModelDims {
                vocab_size: 0,
                ..dims
            },
            max_tokens,
            max_batch,
        )?;
        self.decoder.install_scratch(forward);
        self.scratch = Some(scratch);
        Ok(())
    }

    pub fn forward_hidden_into(
        &self,
        input: MtpInput<'_, T, D>,
        cache: &mut ModelCacheView<'_, T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let n = ctx.plan().num_tokens;
        let dim = self.dims().dim;
        let scratch = self
            .scratch
            .as_ref()
            .ok_or_else(|| OpError::Shape("MTP workspace not prepared".into()))?;
        if n == 0
            || n > scratch.capacity
            || ctx.plan().batch > scratch.max_batch
            || input.next_token_ids.shape().as_slice() != [n]
            || input.target_hidden.shape().as_slice() != [n, dim]
            || output.shape().as_slice() != [n, dim]
            || !input.next_token_ids.is_contiguous()
            || !input.target_hidden.is_contiguous()
            || !output.is_contiguous()
        {
            return Err(OpError::Shape(
                "MTP input/output shape or capacity mismatch".into(),
            ));
        }
        cache.validate(
            self.cache_layout(),
            LayerRange::all(self.dims().num_layers),
            ctx.plan(),
        )?;
        let mut embedded = Hidden {
            stream: scratch.embeddings.narrow(0, 0, n)?,
            pending: None,
        };
        self.decoder
            .embed(input.next_token_ids, &mut embedded, ctx)?;
        let mut e = scratch.norm_embedding.narrow(0, 0, n)?;
        let mut h = scratch.norm_hidden.narrow(0, 0, n)?;
        self.embedding_norm.forward(&embedded.stream, &mut e, ctx)?;
        self.hidden_norm.forward(input.target_hidden, &mut h, ctx)?;
        let mut cat = scratch.concatenated.narrow(0, 0, n)?;
        D::concat_cols(ctx.scope(), &e, &h, &mut cat)?;
        let mut hidden = Hidden {
            stream: scratch.projected.narrow(0, 0, n)?,
            pending: None,
        };
        self.fc.forward(&cat, &mut hidden.stream, ctx)?;
        self.decoder.decode_layers(
            LayerRange::all(self.dims().num_layers),
            &mut hidden,
            cache,
            ctx,
        )?;
        self.decoder.normalize_hidden_into(&hidden, output, ctx)
    }

    pub fn project_logits_into(
        &self,
        normalized: &Tensor<T, D>,
        output: &mut Tensor<T, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        self.decoder.project_logits_into(normalized, output, ctx)
    }
}
