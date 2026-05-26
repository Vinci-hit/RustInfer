//! Qwen3 model — same as Llama3 + QK-norm before RoPE.

use std::marker::PhantomData;
use crate::domain::ports::{OpBackend, OpResult};
use crate::domain::types::{Dtype, Shape};
use crate::domain::tensor::Tensor;
use crate::domain::model::{LlmModel, ForwardContext};
use super::layers::{Linear, RMSNorm, Embedding};
use super::llama3::{alloc_i32, alloc_seq_starts};

pub struct Qwen3Layer<T: Dtype, D: OpBackend> {
    pub input_layernorm: RMSNorm<T, D>,
    pub post_attention_layernorm: RMSNorm<T, D>,
    pub qkv_proj: Linear<T, D>,
    pub o_proj: Linear<T, D>,
    pub gate_proj: Linear<T, D>,
    pub up_proj: Linear<T, D>,
    pub down_proj: Linear<T, D>,
    pub q_norm: Option<RMSNorm<T, D>>,
    pub k_norm: Option<RMSNorm<T, D>>,
}

pub struct Qwen3Model<T: Dtype, D: OpBackend> {
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

impl<T: Dtype, D: OpBackend> LlmModel<T, D> for Qwen3Model<T, D> {
    fn forward(
        &self,
        input_ids: &Tensor<i32, D>,
        ctx: &mut ForwardContext<'_, T, D>,
    ) -> OpResult<Tensor<T, D>> {
        let num_tokens = input_ids.numel();
        let q_dim = self.head_num * self.head_dim;
        let kv_dim = self.kv_head_num * self.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let dev = input_ids.device();

        // ── 1. Embedding ──
        let mut x = D::alloc_tensor::<T>(Shape::from_slice(&[num_tokens, self.dim]), dev)?;
        self.embed_tokens.forward(input_ids, &mut x)?;

        // ── 2. First input norm ──
        let mut h = D::alloc_tensor::<T>(Shape::from_slice(&[num_tokens, self.dim]), dev)?;
        self.layers[0].input_layernorm.forward(&x, &mut h)?;

        // ── 3. Buffers ──
        let mut qkv_buf = D::alloc_tensor::<T>(Shape::from_slice(&[num_tokens, qkv_dim]), dev)?;
        let mut attn_out = D::alloc_tensor::<T>(Shape::from_slice(&[num_tokens, self.dim]), dev)?;
        let mut gate_buf = D::alloc_tensor::<T>(Shape::from_slice(&[num_tokens, self.intermediate_size]), dev)?;
        let mut up_buf = D::alloc_tensor::<T>(Shape::from_slice(&[num_tokens, self.intermediate_size]), dev)?;
        let mut ffn_out = D::alloc_tensor::<T>(Shape::from_slice(&[num_tokens, self.dim]), dev)?;
        let positions = alloc_i32::<D>(ctx.positions, dev)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // ── QKV projection ──
            layer.qkv_proj.forward(&h, &mut qkv_buf)?;

            let mut q = D::alloc_tensor::<T>(Shape::from_slice(&[num_tokens, q_dim]), dev)?;
            let mut k = D::alloc_tensor::<T>(Shape::from_slice(&[num_tokens, kv_dim]), dev)?;
            let mut v = D::alloc_tensor::<T>(Shape::from_slice(&[num_tokens, kv_dim]), dev)?;
            D::split_qkv(&qkv_buf, &mut q, &mut k, &mut v, num_tokens, q_dim, kv_dim)?;

            // ── QK-norm (Qwen3 specific) ──
            if let Some(ref qn) = layer.q_norm {
                let mut q_reshaped = Tensor {
                    shape: Shape::from_slice(&[num_tokens * self.head_num, self.head_dim]),
                    strides: Shape::from_slice(&[num_tokens * self.head_num, self.head_dim]).contiguous_strides(),
                    offset_elems: q.offset_elems,
                    numel: q.numel,
                    is_contiguous: true,
                    storage_ptr: q.storage_ptr,
                    storage_len: q.storage_len,
                    device: dev.clone(),
                    _marker: PhantomData,
                };
                qn.forward_inplace(&mut q_reshaped)?;
            }
            if let Some(ref kn) = layer.k_norm {
                let mut k_reshaped = Tensor {
                    shape: Shape::from_slice(&[num_tokens * self.kv_head_num, self.head_dim]),
                    strides: Shape::from_slice(&[num_tokens * self.kv_head_num, self.head_dim]).contiguous_strides(),
                    offset_elems: k.offset_elems,
                    numel: k.numel,
                    is_contiguous: true,
                    storage_ptr: k.storage_ptr,
                    storage_len: k.storage_len,
                    device: dev.clone(),
                    _marker: PhantomData,
                };
                kn.forward_inplace(&mut k_reshaped)?;
            }

            // ── RoPE ──
            D::rope_inplace(
                &mut q, &mut k,
                &self.sin_cache, &self.cos_cache,
                &positions,
                self.head_num, self.kv_head_num, self.head_dim,
            )?;

            // ── KV scatter ──
            D::scatter_kv(&k, &v, &mut ctx.k_caches[layer_idx], &mut ctx.v_caches[layer_idx], &positions, kv_dim)?;

            // ── Attention ──
            let seq_starts = alloc_seq_starts::<D>(ctx.seq_lens, dev)?;
            D::attention(
                &q, &ctx.k_caches[layer_idx], &ctx.v_caches[layer_idx],
                &mut attn_out, &seq_starts,
                self.head_num, self.kv_head_num, self.head_dim,
                1.0 / (self.head_dim as f32).sqrt(),
            )?;

            // ── O proj + residual (fused) ──
            let mut o_out = D::alloc_tensor::<T>(Shape::from_slice(&[num_tokens, self.dim]), dev)?;
            layer.o_proj.forward(&attn_out, &mut o_out)?;
            D::fused_add_rmsnorm(
                &mut h, &mut x, &o_out,
                &layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps,
            )?;

            // ── MLP (SwiGLU) ──
            layer.gate_proj.forward(&h, &mut gate_buf)?;
            layer.up_proj.forward(&h, &mut up_buf)?;
            D::silu_inplace(&mut gate_buf)?;
            D::swiglu_inplace(&mut gate_buf, &up_buf)?;
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
        let mut logits = D::alloc_tensor::<T>(Shape::from_slice(&[num_tokens, self.vocab_size]), dev)?;
        self.lm_head.forward(&h, &mut logits)?;
        Ok(logits)
    }

    fn num_layers(&self) -> usize { self.layers.len() }
    fn vocab_size(&self) -> usize { self.vocab_size }
    fn dim(&self) -> usize { self.dim }
    fn kv_dim(&self) -> usize { self.kv_dim }
}
