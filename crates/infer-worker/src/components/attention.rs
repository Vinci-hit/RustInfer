use std::rc::Rc;

use crate::components::linear::Linear;
use crate::components::norm::RmsNorm;
use crate::domain::component::{Component, Hidden, StageKind};
use crate::domain::forward_scratch::ForwardScratch;
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::kv::KvView;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::Shape;

/// Pre-norm self-attention sublayer.
///
/// Reads the residual (`hidden.stream`), normalizes it out-of-place into private
/// scratch (so the residual survives), runs paged attention, and adds the
/// projection back into the residual. Owns its input norm and — for Qwen3-style
/// models — the per-head Q/K norms applied before RoPE.
pub struct Attention<T: Dtype, D: LlmBackend> {
    pub input_layernorm: RmsNorm<T, D>,
    pub qkv_proj: Linear<T, D>,
    pub o_proj: Linear<T, D>,
    /// Qwen3 per-head Q/K RMSNorm (applied before RoPE). `None` for Llama3.
    pub q_norm: Option<RmsNorm<T, D>>,
    pub k_norm: Option<RmsNorm<T, D>>,
    pub sin: Tensor<T, D>,
    pub cos: Tensor<T, D>,
    pub head_num: usize,
    pub kv_head_num: usize,
    pub head_dim: usize,
    pub scale: f32,
    /// Shared, address-stable per-layer forward scratch (installed by
    /// `Runtime::new`). `None` → fall back to per-call device allocation.
    pub scratch: Option<Rc<ForwardScratch<T, D>>>,
}

impl<T: Dtype, D: LlmBackend> Component<T, D> for Attention<T, D> {
    fn kind(&self) -> StageKind {
        StageKind::Attention
    }

    fn run(
        &self,
        hidden: &mut Hidden<T, D>,
        kv: Option<&mut KvView<'_, T, D>>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let kv = kv.ok_or_else(|| OpError::Shape("Attention::run: missing KV view".into()))?;
        let num_tokens = hidden.num_tokens();
        let q_dim = self.head_num * self.head_dim;
        let kv_dim = self.kv_head_num * self.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let dim = hidden.stream.shape().as_slice()[1];
        let dev = hidden.stream.device().clone();

        // Per-layer scratch: reuse the address-stable workspace when installed
        // (zero alloc, zero memset, CUDA-graph friendly); otherwise fall back
        // to the device allocator (pooled). See `ForwardScratch`.
        let scratch = self.scratch.as_deref().filter(|s| s.fits(num_tokens));
        let mut normed = match scratch {
            Some(s) => s.normed(num_tokens),
            None => D::alloc_tensor(Shape::from_slice(&[num_tokens, dim]), &dev)?,
        };
        let mut qkv = match scratch {
            Some(s) => s.qkv(num_tokens),
            None => D::alloc_tensor(Shape::from_slice(&[num_tokens, qkv_dim]), &dev)?,
        };
        let mut attn_out = match scratch {
            Some(s) => s.attn_out(num_tokens),
            None => D::alloc_tensor(Shape::from_slice(&[num_tokens, q_dim]), &dev)?,
        };
        let mut o_out = match scratch {
            Some(s) => s.o_out(num_tokens),
            None => D::alloc_tensor(Shape::from_slice(&[num_tokens, dim]), &dev)?,
        };

        // Pre-attention norm. If the previous sublayer left a deferred residual
        // delta, fuse its add into this norm (one kernel: stream += delta;
        // normed = rmsnorm(stream)); else plain norm. Residual stays in
        // `hidden.stream` for the post-attention add.
        match hidden.pending.take() {
            Some(delta) => D::fused_add_rmsnorm(
                ctx,
                &mut normed,
                &mut hidden.stream,
                &delta,
                &self.input_layernorm.weight,
                self.input_layernorm.eps,
            )?,
            None => self.input_layernorm.forward(&hidden.stream, &mut normed, ctx)?,
        }
        self.qkv_proj.forward(&normed, &mut qkv, ctx)?;
        // Q/K/V: zero-copy column views of `qkv` on CUDA (its kernels honor row
        // strides → no copy, no per-layer alloc); contiguous copies on backends
        // that require them (CPU reference). See `D::qkv_split`.
        let (mut q, mut k, v) = D::qkv_split(ctx, &qkv, num_tokens, q_dim, kv_dim)?;

        match (&self.q_norm, &self.k_norm) {
            (Some(qn), Some(kn)) => {
                // Qwen3: fused Q/K-norm + RoPE + paged scatter.
                let mut layer = kv.layer_mut(0);
                D::qkv_norm_rope_scatter(
                    ctx,
                    &mut q,
                    &mut k,
                    &v,
                    Some(&qn.weight),
                    Some(&kn.weight),
                    qn.eps,
                    kn.eps,
                    &self.sin,
                    &self.cos,
                    &layer.index.rope_positions,
                    &mut layer,
                    self.head_num,
                    self.kv_head_num,
                    self.head_dim,
                    kv_dim,
                )?;
            }
            (None, None) => {
                // Llama3: RoPE then paged scatter.
                D::rope_inplace(
                    ctx.scope(),
                    &mut q,
                    &mut k,
                    &self.sin,
                    &self.cos,
                    &kv.index.rope_positions,
                    self.head_num,
                    self.kv_head_num,
                    self.head_dim,
                )?;
                let mut layer = kv.layer_mut(0);
                D::scatter_kv_paged(ctx, &k, &v, &mut layer, kv_dim)?;
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err(OpError::Shape(
                    "Attention::run: q_norm and k_norm must both be present or both absent".into(),
                ));
            }
        }

        // Flash-attention decode workspace: prefer the preallocated buffer in
        // `ForwardScratch` (address-stable across all layers and across CUDA
        // graph capture/replay, zero per-layer alloc+memset). Fall back to
        // backend self-allocation only when scratch is absent (CPU reference;
        // tests). Each layer takes a fresh full-buffer view — layers run
        // serially on one stream so the kernel's stream-ordered reads/writes
        // do not race.
        let mut flash_ws = self.scratch.as_deref().map(|s| s.flash_workspace_mut());
        D::attention_paged(
            ctx,
            &q,
            kv,
            &mut attn_out,
            self.head_num,
            self.kv_head_num,
            self.head_dim,
            self.scale,
            flash_ws.as_mut(),
        )?;
        self.o_proj.forward(&attn_out, &mut o_out, ctx)?;
        // Defer the residual add: stash `o_out`; the next sublayer's pre-norm
        // fuses it (add + norm in one kernel). `decode_layers` flushes any
        // leftover delta before finalize.
        hidden.pending = Some(o_out);
        Ok(())
    }
}
