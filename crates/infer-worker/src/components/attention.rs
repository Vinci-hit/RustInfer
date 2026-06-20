use crate::components::linear::Linear;
use crate::components::norm::RmsNorm;
use crate::domain::component::{Component, Hidden, StageKind};
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

        let mut normed = D::alloc_tensor(Shape::from_slice(&[num_tokens, dim]), &dev)?;
        let mut qkv = D::alloc_tensor(Shape::from_slice(&[num_tokens, qkv_dim]), &dev)?;
        let mut q = D::alloc_tensor(Shape::from_slice(&[num_tokens, q_dim]), &dev)?;
        let mut k = D::alloc_tensor(Shape::from_slice(&[num_tokens, kv_dim]), &dev)?;
        let mut v = D::alloc_tensor(Shape::from_slice(&[num_tokens, kv_dim]), &dev)?;
        let mut attn_out = D::alloc_tensor(Shape::from_slice(&[num_tokens, q_dim]), &dev)?;
        let mut o_out = D::alloc_tensor(Shape::from_slice(&[num_tokens, dim]), &dev)?;

        // Pre-attention norm — residual stays intact in `hidden.stream`.
        self.input_layernorm.forward(&hidden.stream, &mut normed, ctx)?;
        self.qkv_proj.forward(&normed, &mut qkv, ctx)?;
        D::split_qkv(ctx, &qkv, &mut q, &mut k, &mut v, num_tokens, q_dim, kv_dim)?;

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

        D::attention_paged(
            ctx,
            &q,
            kv,
            &mut attn_out,
            self.head_num,
            self.kv_head_num,
            self.head_dim,
            self.scale,
        )?;
        self.o_proj.forward(&attn_out, &mut o_out, ctx)?;
        // Residual update: hidden.stream += attention output.
        D::add_inplace(ctx.scope(), &mut hidden.stream, &o_out)
    }
}
