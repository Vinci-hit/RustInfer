//! Llama3 model implementation — full forward with batched / ragged attention.
//!
//! Allocation policy: zero `D::alloc_tensor` per call. All intermediates
//! come from `ctx.workspace` views (Arc-cloned over a long-lived storage)
//! so addresses are stable across `forward` calls — required for CUDA
//! Graph capture.

use super::layers::{Embedding, Linear, RMSNorm};
use crate::domain::model::{ForwardContext, LlmForwardWorkspace, LlmModel};
use crate::domain::ports::{LlmOps, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{Dtype, Shape};

pub struct Llama3Layer<T: Dtype, D: LlmOps> {
    pub input_layernorm: RMSNorm<T, D>,
    pub post_attention_layernorm: RMSNorm<T, D>,
    pub qkv_proj: Linear<T, D>,
    pub o_proj: Linear<T, D>,
    /// Fused [2*intermediate_size, dim] = vstack(gate_proj, up_proj).
    pub gate_up_proj: Linear<T, D>,
    pub down_proj: Linear<T, D>,
}

pub struct Llama3Model<T: Dtype, D: LlmOps> {
    pub embed_tokens: Embedding<T, D>,
    pub layers: Vec<Llama3Layer<T, D>>,
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

impl<T, D, W> LlmModel<T, D, W> for Llama3Model<T, D>
where
    T: Dtype,
    D: LlmOps,
    W: LlmForwardWorkspace<T, D>,
{
    fn forward(
        &self,
        input_ids: &Tensor<i32, D>,
        ctx: &mut ForwardContext<'_, T, D, W>,
    ) -> OpResult<Tensor<T, D>> {
        let num_tokens = input_ids.numel();
        let kv_dim = self.kv_head_num * self.head_dim;
        let q_dim = self.head_num * self.head_dim;
        let plan = ctx.plan;

        let debug = std::env::var("RUSTINFER_DEBUG_LAYERS").is_ok();
        let dump = |tag: &str, t: &Tensor<T, D>| -> OpResult<()> {
            if !debug {
                return Ok(());
            }
            let host = t.to_host_vec()?;
            let row0_first8: Vec<f32> = host
                .iter()
                .take(8)
                .map(|v| {
                    let b = unsafe {
                        std::slice::from_raw_parts(v as *const T as *const u8, T::SIZE_BYTES)
                    };
                    match T::DATA_TYPE {
                        crate::domain::types::DataType::F32 => {
                            f32::from_le_bytes(b.try_into().unwrap())
                        }
                        crate::domain::types::DataType::BF16 => {
                            half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32()
                        }
                        crate::domain::types::DataType::F16 => {
                            half::f16::from_le_bytes(b.try_into().unwrap()).to_f32()
                        }
                        _ => f32::NAN,
                    }
                })
                .collect();
            eprintln!(
                "[fwd] {:>30}  shape={:?}  row0[0..8]={:?}",
                tag,
                t.shape().as_slice(),
                row0_first8
            );
            Ok(())
        };

        // Workspace views (Arc-cloned, address-stable).
        let mut x = ctx.workspace.x_view(num_tokens);
        let mut h = ctx.workspace.h_view(num_tokens);
        let mut qkv_buf = ctx.workspace.qkv_view(num_tokens);
        let mut attn_out = ctx.workspace.attn_out_view(num_tokens);
        let mut gate_up = ctx.workspace.gate_up_view(num_tokens);
        let mut gate_buf = ctx.workspace.gate_view(num_tokens);
        let mut ffn_out = ctx.workspace.ffn_view(num_tokens);
        let mut o_out = ctx.workspace.o_out_view(num_tokens);
        let logits = ctx.workspace.logits_view(num_tokens);

        // ── 1. Embedding ──
        self.embed_tokens.forward(input_ids, &mut x)?;
        dump("after_embed", &x)?;

        // ── 2. First input norm ──
        self.layers[0].input_layernorm.forward(&x, &mut h)?;
        dump("after_input_norm[0]", &h)?;

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let trace = debug && layer_idx == 0;

            // ── Attention ──
            layer.qkv_proj.forward(&h, &mut qkv_buf)?;
            if trace {
                dump("L0 qkv_proj_out", &qkv_buf)?;
            }

            // Zero-copy QKV split: 3 strided views into qkv_buf.
            // q = qkv_buf[:, 0..q_dim]
            // k = qkv_buf[:, q_dim..q_dim+kv_dim]
            // v = qkv_buf[:, q_dim+kv_dim..q_dim+2*kv_dim]
            // Replaces 3 split_cols kernel launches with O(1) view_raw.
            let mut q = qkv_buf.narrow(1, 0, q_dim)?;
            let mut k = qkv_buf.narrow(1, q_dim, kv_dim)?;
            let v = qkv_buf.narrow(1, q_dim + kv_dim, kv_dim)?;

            // RoPE on Q and K (positions provided by the plan).
            D::rope_inplace(
                &mut q,
                &mut k,
                &self.sin_cache,
                &self.cos_cache,
                &plan.rope_positions,
                self.head_num,
                self.kv_head_num,
                self.head_dim,
            )?;

            // Scatter K, V into the paged KV pool for this layer.
            {
                let layer_kv = &mut ctx.kv_pool.layers[layer_idx];
                D::scatter_kv_paged(
                    &k,
                    &v,
                    &mut layer_kv.k,
                    &mut layer_kv.v,
                    &plan.block_tables,
                    &plan.seq_positions,
                    &plan.cu_q_lens,
                    &plan.seq_lens_step,
                    plan.max_blocks_per_seq,
                    plan.block_size,
                    kv_dim,
                )?;
            }

            // Attention over the paged pool. Disjoint-field borrows: kv_pool
            // is borrowed immutably for the K/V refs; workspace is borrowed
            // mutably for its flash-decode scratch. Rust's split-borrow
            // analysis allows these simultaneously.
            {
                let layer_kv = &ctx.kv_pool.layers[layer_idx];
                let k_pool = &layer_kv.k;
                let v_pool = &layer_kv.v;
                let scratch = ctx.workspace.flash_decode_workspace();
                D::attention_paged(
                    &q,
                    k_pool,
                    v_pool,
                    &mut attn_out,
                    plan,
                    scratch,
                    self.head_num,
                    self.kv_head_num,
                    self.head_dim,
                    1.0 / (self.head_dim as f32).sqrt(),
                )?;
            }
            if trace {
                dump("L0 attn_out", &attn_out)?;
            }

            // O projection
            layer.o_proj.forward(&attn_out, &mut o_out)?;
            if trace {
                dump("L0 o_proj_out", &o_out)?;
            }

            // Residual + post-attention norm (fused).
            D::fused_add_rmsnorm(
                &mut h,
                &mut x,
                &o_out,
                &layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.eps,
            )?;

            // ── MLP (fused gate_up + swiglu_packed) ──
            // gate_up_proj: [num_tokens, dim] → [num_tokens, 2*intermediate]
            // swiglu_packed: [num_tokens, 2*intermediate] → [num_tokens, intermediate]
            // (one GEMV + one fused activation, vs gate/up split + swiglu).
            layer.gate_up_proj.forward(&h, &mut gate_up)?;
            D::swiglu_packed(&gate_up, &mut gate_buf, num_tokens, self.intermediate_size)?;
            layer.down_proj.forward(&gate_buf, &mut ffn_out)?;
            if trace {
                dump("L0 ffn_out", &ffn_out)?;
            }

            // Residual + next norm (fused).
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

        // ── 4. LM head → logits ──
        let mut logits_mut = logits;
        self.lm_head.forward(&h, &mut logits_mut)?;
        Ok(logits_mut)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn kv_dim(&self) -> usize {
        self.kv_dim
    }
    fn q_dim(&self) -> usize {
        self.head_num * self.head_dim
    }
    fn head_num(&self) -> usize {
        self.head_num
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn kv_head_num(&self) -> usize {
        self.kv_head_num
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
}

// ─── Helpers (kept for backward compat with other modules / tests) ─────────

/// Allocate a zeroed tensor via OpBackend.
pub fn alloc<T: Dtype, D: LlmOps>(rows: usize, cols: usize, dev: &D) -> OpResult<Tensor<T, D>> {
    D::alloc_tensor(Shape::from_slice(&[rows, cols]), dev)
}

/// Create a device tensor from host i32 positions (single H2D upload).
pub fn alloc_i32<D: LlmOps>(positions: &[i32], dev: &D) -> OpResult<Tensor<i32, D>> {
    Tensor::<i32, D>::from_host_slice(positions, Shape::from_slice(&[positions.len()]), dev)
}

/// Build seq_starts [batch+1] prefix sum from seq_lens, then upload.
pub fn alloc_seq_starts<D: LlmOps>(seq_lens: &[usize], dev: &D) -> OpResult<Tensor<i32, D>> {
    let mut starts = Vec::with_capacity(seq_lens.len() + 1);
    starts.push(0i32);
    let mut acc = 0i32;
    for &len in seq_lens {
        acc += len as i32;
        starts.push(acc);
    }
    alloc_i32::<D>(&starts, dev)
}
