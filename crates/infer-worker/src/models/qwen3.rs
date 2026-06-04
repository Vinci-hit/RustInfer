//! Qwen3 model — same as Llama3 + QK-norm before RoPE.

use crate::domain::ports::{LlmOps, OpResult};
use crate::domain::types::{Dtype, Shape, Strides};
use crate::domain::tensor::Tensor;
use crate::domain::model::{LlmModel, ForwardContext};
use super::layers::{Linear, RMSNorm, Embedding};

pub struct Qwen3Layer<T: Dtype, D: LlmOps> {
    pub input_layernorm: RMSNorm<T, D>,
    pub post_attention_layernorm: RMSNorm<T, D>,
    pub qkv_proj: Linear<T, D>,
    pub o_proj: Linear<T, D>,
    /// Fused [2*intermediate_size, dim] = vstack(gate_proj, up_proj).
    pub gate_up_proj: Linear<T, D>,
    pub down_proj: Linear<T, D>,
    pub q_norm: Option<RMSNorm<T, D>>,
    pub k_norm: Option<RMSNorm<T, D>>,
}

pub struct Qwen3Model<T: Dtype, D: LlmOps> {
    pub embed_tokens: Embedding<T, D>,
    pub layers: Vec<Qwen3Layer<T, D>>,
    pub norm: RMSNorm<T, D>,
    pub lm_head: Linear<T, D>,
    pub sin_cache: Tensor<T, D>,
    pub cos_cache: Tensor<T, D>,
    pub head_num: usize,
    pub kv_head_num: usize,
    pub head_dim: usize,
    pub dim: usize,
    pub kv_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
}

impl<T: Dtype, D: LlmOps> LlmModel<T, D> for Qwen3Model<T, D> {
    fn forward(
        &self,
        input_ids: &Tensor<i32, D>,
        ctx: &mut ForwardContext<'_, T, D>,
    ) -> OpResult<Tensor<T, D>> {
        let num_tokens = input_ids.numel();
        let q_dim = self.head_num * self.head_dim;
        let kv_dim = self.kv_head_num * self.head_dim;
        let plan = ctx.plan;

        // Workspace views (Arc-cloned, address-stable).
        let mut x        = ctx.workspace.x_view(num_tokens);
        let mut h        = ctx.workspace.h_view(num_tokens);
        let mut qkv_buf  = ctx.workspace.qkv_view(num_tokens);
        let mut attn_out = ctx.workspace.attn_out_view(num_tokens);
        let mut gate_up  = ctx.workspace.gate_up_view(num_tokens);
        let mut gate_buf = ctx.workspace.gate_view(num_tokens);
        let mut ffn_out  = ctx.workspace.ffn_view(num_tokens);
        let mut o_out    = ctx.workspace.o_out_view(num_tokens);
        let logits       = ctx.workspace.logits_view(num_tokens);

        // ── 1. Embedding ──
        self.embed_tokens.forward(input_ids, &mut x)?;

        // ── 2. First input norm ──
        self.layers[0].input_layernorm.forward(&x, &mut h)?;

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];

            // ── QKV projection ──
            layer.qkv_proj.forward(&h, &mut qkv_buf)?;

            // Zero-copy QKV split via strided views (saves 3 split_cols launches).
            let qkv_cols = q_dim + 2 * kv_dim;
            let mut q = qkv_buf.narrow(1, 0, q_dim)?;
            let mut k = qkv_buf.narrow(1, q_dim, kv_dim)?;
            let v       = qkv_buf.narrow(1, q_dim + kv_dim, kv_dim)?;

            // ── QK-norm (Qwen3 specific) ──
            //
            // RMSNorm kernel natively accepts a strided 3D
            // `[T, head_num, head_size]` view; we rewrite the strided
            // `[T, q_dim]` Q into `[T, head_num, head_size]` via raw_view
            // (strides `[qkv_cols, head_size, 1]`). No copy.
            if let Some(ref qn) = layer.q_norm {
                let mut q3 = q.view_raw(
                    Shape::from_slice(&[num_tokens, self.head_num, self.head_dim]),
                    Strides::from_slice(&[qkv_cols, self.head_dim, 1]),
                    q.offset_elems(),
                    false,
                );
                qn.forward_inplace(&mut q3)?;
            }
            if let Some(ref kn) = layer.k_norm {
                let mut k3 = k.view_raw(
                    Shape::from_slice(&[num_tokens, self.kv_head_num, self.head_dim]),
                    Strides::from_slice(&[qkv_cols, self.head_dim, 1]),
                    k.offset_elems(),
                    false,
                );
                kn.forward_inplace(&mut k3)?;
            }

            // ── RoPE ──
            D::rope_inplace(
                &mut q, &mut k,
                &self.sin_cache, &self.cos_cache,
                &plan.rope_positions,
                self.head_num, self.kv_head_num, self.head_dim,
            )?;

            // ── KV scatter (paged) ──
            {
                let layer_kv = &mut ctx.kv_pool.layers[layer_idx];
                D::scatter_kv_paged(
                    &k, &v,
                    &mut layer_kv.k, &mut layer_kv.v,
                    &plan.block_tables, &plan.seq_positions,
                    &plan.cu_q_lens, &plan.seq_lens_step,
                    plan.max_blocks_per_seq, plan.block_size, kv_dim,
                )?;
            }

            // ── Attention (paged) ──
            {
                let layer_kv = &ctx.kv_pool.layers[layer_idx];
                let k_pool = &layer_kv.k;
                let v_pool = &layer_kv.v;
                let scratch = ctx.workspace.flash_decode_workspace();
                D::attention_paged(
                    &q, k_pool, v_pool,
                    &mut attn_out, plan,
                    scratch,
                    self.head_num, self.kv_head_num, self.head_dim,
                    1.0 / (self.head_dim as f32).sqrt(),
                )?;
            }

            // ── O proj + residual (fused) ──
            layer.o_proj.forward(&attn_out, &mut o_out)?;
            D::fused_add_rmsnorm(
                &mut h, &mut x, &o_out,
                &layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps,
            )?;

            // ── MLP (fused gate_up + swiglu_packed) ──
            layer.gate_up_proj.forward(&h, &mut gate_up)?;
            D::swiglu_packed(&gate_up, &mut gate_buf, num_tokens, self.intermediate_size)?;
            layer.down_proj.forward(&gate_buf, &mut ffn_out)?;

            // ── Residual + next norm (fused) ──
            let next_norm_weight = if layer_idx + 1 < self.layers.len() {
                &self.layers[layer_idx + 1].input_layernorm.weight
            } else {
                &self.norm.weight
            };
            let next_eps = if layer_idx + 1 < self.layers.len() {
                self.layers[layer_idx + 1].input_layernorm.eps
            } else {
                self.norm.eps
            };
            D::fused_add_rmsnorm(&mut h, &mut x, &ffn_out, next_norm_weight, next_eps)?;
        }

        // ── 4. LM head ──
        let mut logits_mut = logits;
        self.lm_head.forward(&h, &mut logits_mut)?;
        Ok(logits_mut)
    }

    fn num_layers(&self) -> usize { self.layers.len() }
    fn vocab_size(&self) -> usize { self.vocab_size }
    fn dim(&self) -> usize { self.dim }
    fn kv_dim(&self) -> usize { self.kv_dim }
    fn q_dim(&self) -> usize { self.head_num * self.head_dim }
    fn head_num(&self) -> usize { self.head_num }
    fn head_dim(&self) -> usize { self.head_dim }
    fn kv_head_num(&self) -> usize { self.kv_head_num }
    fn intermediate_size(&self) -> usize { self.intermediate_size }
}
