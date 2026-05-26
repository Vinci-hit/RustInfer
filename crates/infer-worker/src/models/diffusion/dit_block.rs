//! DiTBlock — single transformer block for Z-Image (S3-DiT).
//!
//! Each block performs:
//!   1. AdaLN modulation (optional): MLP(t_embed) → (gate_attn, gate_ffn, scale_attn, ...)
//!   2. Attention: QKV proj → split heads → RMSNorm Q/K → SDPA → O proj
//!   3. FFN: gate/up proj → SwiGLU → down proj
//!   4. Residual connections with scale gates

use crate::domain::ports::{OpBackend, OpResult};
use crate::domain::types::{Dtype, Shape};
use crate::domain::tensor::Tensor;
use crate::models::layers::{Linear, RMSNorm};

/// A single DiT transformer block.
pub struct DiTBlock<T: Dtype, D: OpBackend> {
    // ─── Attention ───
    pub to_qkv: Linear<T, D>,    // [3*dim, dim] fused QKV projection
    pub to_out: Linear<T, D>,    // [dim, dim] output projection
    pub norm_q: RMSNorm<T, D>,   // per-head Q norm
    pub norm_k: RMSNorm<T, D>,   // per-head K norm

    // ─── FFN (SwiGLU) ───
    pub w1: Linear<T, D>,        // gate projection [intermediate, dim]
    pub w3: Linear<T, D>,        // up projection [intermediate, dim]
    pub w2: Linear<T, D>,        // down projection [dim, intermediate]

    // ─── AdaLN modulation ───
    /// If present, maps t_embed [1, adaln_dim] → [1, 6*dim] for 6 modulation scalars.
    pub adaln: Option<Linear<T, D>>,

    // ─── Norms (input) ───
    pub attention_norm: RMSNorm<T, D>,
    pub ffn_norm: RMSNorm<T, D>,

    // ─── Geometry ───
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl<T: Dtype, D: OpBackend> DiTBlock<T, D> {
    /// Forward pass through the DiT block.
    ///
    /// - `x`: input hidden states [seq_len, dim]
    /// - `t_embed`: timestep conditioning [1, adaln_dim] (None if no modulation)
    /// - workspace: pre-allocated buffers from DitState
    ///
    /// Returns: residual-updated x (in-place via workspace swap).
    pub fn forward(
        &self,
        x: &Tensor<T, D>,
        _t_embed: Option<&Tensor<T, D>>,
        q_buf: &mut Tensor<T, D>,
        k_buf: &mut Tensor<T, D>,
        v_buf: &mut Tensor<T, D>,
        attn_out: &mut Tensor<T, D>,
        gate_buf: &mut Tensor<T, D>,
        up_buf: &mut Tensor<T, D>,
        ffn_out: &mut Tensor<T, D>,
        output: &mut Tensor<T, D>,
    ) -> OpResult<()> {
        let seq_len = x.numel() / self.dim;
        let dev = x.device();
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;

        // ─── 1. Attention norm ───
        let mut normed = D::alloc_tensor::<T>(Shape::from_slice(&[seq_len, self.dim]), dev)?;
        self.attention_norm.forward(x, &mut normed)?;

        // ─── 2. QKV projection ───
        let qkv_dim = q_dim + 2 * kv_dim;
        let mut qkv = D::alloc_tensor::<T>(Shape::from_slice(&[seq_len, qkv_dim]), dev)?;
        self.to_qkv.forward(&normed, &mut qkv)?;

        // Split Q/K/V
        D::split_qkv(&qkv, q_buf, k_buf, v_buf, seq_len, q_dim, kv_dim)?;

        // ─── 3. QK norm (per-head) ───
        self.norm_q.forward_inplace(q_buf)?;
        self.norm_k.forward_inplace(k_buf)?;

        // ─── 4. SDPA ───
        D::sdpa(
            q_buf, k_buf, v_buf, attn_out,
            self.n_heads, self.n_kv_heads, self.head_dim,
            1.0 / (self.head_dim as f32).sqrt(),
        )?;

        // ─── 5. O proj ───
        self.to_out.forward(attn_out, &mut normed)?;

        // ─── 6. Residual + FFN norm ───
        D::add_inplace(output, &normed)?;
        self.ffn_norm.forward(output, &mut normed)?;

        // ─── 7. FFN (SwiGLU) ───
        self.w1.forward(&normed, gate_buf)?;
        self.w3.forward(&normed, up_buf)?;
        D::silu_inplace(gate_buf)?;
        D::swiglu_inplace(gate_buf, up_buf)?;
        self.w2.forward(gate_buf, ffn_out)?;

        // ─── 8. Final residual ───
        D::add_inplace(output, ffn_out)?;

        Ok(())
    }
}
