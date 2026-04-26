//! Single DiT transformer block for Z-Image.
//!
//! ```text
//! modulation=True:
//!   scale_msa, gate_msa, scale_mlp, gate_mlp = adaLN(c).chunk(4)
//!   gate_*.tanh(); scale_* = 1 + scale_*
//!   attn_out = Attention(norm1(x) * scale_msa, cos, sin)
//!   x = x + gate_msa * norm2(attn_out)
//!   ffn_out = FFN(norm1(x) * scale_mlp)  # FFN = w2(silu(w1(x)) * w3(x))
//!   x = x + gate_mlp * norm2(ffn_out)
//!
//! modulation=False (context_refiner):
//!   attn_out = Attention(norm1(x), cos, sin)
//!   x = x + norm2(attn_out)
//!   x = x + norm2(FFN(norm1(x)))
//! ```

use crate::OpConfig;
use crate::base::error::{Error, Result};

use crate::model::diffusion::buffer::DiffBufferType as BT;
use crate::model::diffusion::z_image::state::DitState;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::sdpa::scaled_dot_product_attention;
use crate::op::tensor_utils::permute_nd;
use crate::tensor::Tensor;

/// DiT transformer block (Z-Image style).
pub struct DiTBlock {
    pub attention_norm1: RMSNorm,
    pub attention_norm2: RMSNorm,
    pub ffn_norm1: RMSNorm,
    pub ffn_norm2: RMSNorm,

    pub to_q: Matmul,
    pub to_k: Matmul,
    pub to_v: Matmul,
    pub to_out: Matmul,

    pub norm_q: RMSNorm,
    pub norm_k: RMSNorm,

    pub w1: Matmul, // gate
    pub w3: Matmul, // up
    pub w2: Matmul, // down

    /// Linear(adaln_embed_dim=256 → 4*dim), only present when modulation=True
    pub adaln_modulation: Option<Matmul>,

    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub modulation: bool,
}

impl DiTBlock {
    pub fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        self.attention_norm1.to_cuda(device_id)?;
        self.attention_norm2.to_cuda(device_id)?;
        self.ffn_norm1.to_cuda(device_id)?;
        self.ffn_norm2.to_cuda(device_id)?;
        self.to_q.to_cuda(device_id)?;
        self.to_k.to_cuda(device_id)?;
        self.to_v.to_cuda(device_id)?;
        self.to_out.to_cuda(device_id)?;
        self.norm_q.to_cuda(device_id)?;
        self.norm_k.to_cuda(device_id)?;
        self.w1.to_cuda(device_id)?;
        self.w3.to_cuda(device_id)?;
        self.w2.to_cuda(device_id)?;
        if let Some(ref mut m) = self.adaln_modulation {
            m.to_cuda(device_id)?;
        }
        Ok(())
    }

    /// Forward pass, writing the residual output into `dst` without
    /// allocating any intermediate tensors.
    ///
    /// - `x`: `[S, dim]` input activations (read-only).
    /// - `cos`, `sin`: `[S, head_dim/2]` precomputed 3D RoPE tables.
    /// - `adaln_c`: `[ADALN_EMBED_DIM=256]` conditioning (required when
    ///   `self.modulation == true`; ignored otherwise).
    /// - `state`: transformer workspace. Every per-block tensor is
    ///   sliced from its `Blk*` slot; no `Tensor::new` happens here.
    /// - `dst`: `[S, dim]` output buffer. Must be a distinct state slot
    ///   from the one backing `x` (caller ping-pongs between two slots).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln_c: Option<&Tensor>,
        state: &mut DitState,
        dst: &mut Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        let seq = x.shape()[0];
        let dim = self.dim;

        // ── Modulation params (if modulation=True) ──
        //
        // The four [1, dim] chunks of BlkModOut are non-overlapping
        // slices of the same buffer, so each can be mutated in place
        // without clobbering the others — no cloning required.
        let (scale_msa, gate_msa, scale_mlp, gate_mlp) = if self.modulation {
            let adaln = self.adaln_modulation.as_ref().ok_or_else(|| {
                Error::InternalError("adaln_modulation is None but modulation=True".into())
            })?;
            let c = adaln_c.ok_or_else(|| {
                Error::InvalidArgument("adaln_c required when modulation=True".into())
            })?;

            let c_2d = c.view(&[1, c.shape()[0]])?;
            let mut mod_out = state.slice_mut(BT::BlkModOut, &[1, 4 * dim])?;
            adaln.forward(&c_2d, &mut mod_out, cuda_config)?;

            let mut scale_msa = mod_out.slice(&[0, 0], &[1, dim])?;
            let mut gate_msa = mod_out.slice(&[0, dim], &[1, dim])?;
            let mut scale_mlp = mod_out.slice(&[0, 2 * dim], &[1, dim])?;
            let mut gate_mlp = mod_out.slice(&[0, 3 * dim], &[1, dim])?;

            scale_msa += 1.0_f32;
            scale_mlp += 1.0_f32;
            gate_msa.tanh()?;
            gate_mlp.tanh()?;

            (Some(scale_msa), Some(gate_msa), Some(scale_mlp), Some(gate_mlp))
        } else {
            (None, None, None, None)
        };

        // ═══════════════════════ Attention block ═══════════════════════
        let mut norm1_x = state.slice_mut(BT::BlkNorm1X, &[seq, dim])?;
        self.attention_norm1.forward(x, &mut norm1_x, cuda_config)?;
        if let Some(ref s) = scale_msa {
            norm1_x.mul_row(s)?;
        }

        // QKV projections
        let mut q = state.slice_mut(BT::BlkQ, &[seq, dim])?;
        let mut k = state.slice_mut(BT::BlkK, &[seq, dim])?;
        let mut v = state.slice_mut(BT::BlkV, &[seq, dim])?;
        self.to_q.forward(&norm1_x, &mut q, cuda_config)?;
        self.to_k.forward(&norm1_x, &mut k, cuda_config)?;
        self.to_v.forward(&norm1_x, &mut v, cuda_config)?;

        // Per-head QK-norm + RoPE (all in place on BlkQ / BlkK).
        self.per_head_rmsnorm(&self.norm_q, &mut q, BT::BlkQNormIn, BT::BlkQNormOut, state, cuda_config)?;
        self.per_head_rmsnorm(&self.norm_k, &mut k, BT::BlkKNormIn, BT::BlkKNormOut, state, cuda_config)?;
        {
            let mut q_3d = q.view(&[seq, self.n_heads, self.head_dim])?;
            let mut k_3d = k.view(&[seq, self.n_heads, self.head_dim])?;
            q_3d.rope_interleaved(cos, sin, self.head_dim)?;
            k_3d.rope_interleaved(cos, sin, self.head_dim)?;
        }

        // Reshape [S, H, D] → [1, H, S, D] into BlkQHsd / BlkKHsd / BlkVHsd.
        let mut q_sdpa = self.to_bhsd(&q, BT::BlkQHsd, state)?;
        let mut k_sdpa = self.to_bhsd(&k, BT::BlkKHsd, state)?;
        let v_sdpa = self.to_bhsd(&v, BT::BlkVHsd, state)?;

        // SDPA → BlkAttnSdpa
        let mut attn_out_sdpa = state.slice_mut(
            BT::BlkAttnSdpa, &[1, self.n_heads, seq, self.head_dim],
        )?;
        scaled_dot_product_attention(&q_sdpa, &k_sdpa, &v_sdpa, &mut attn_out_sdpa, cuda_config)?;
        // Release the SDPA mutable slices so subsequent `state` borrows
        // don't trip the checker when they slice sibling buffers.
        drop(q_sdpa);
        drop(k_sdpa);

        // [1, H, S, D] → [S, H*D] via permute + view → BlkAttnFlat
        let mut attn_flat = state.slice_mut(BT::BlkAttnFlat, &[seq, dim])?;
        {
            let attn_hsd = attn_out_sdpa.view(&[self.n_heads, seq, self.head_dim])?;
            let attn_shd = permute_nd(&attn_hsd, &[1, 0, 2])?;
            attn_flat.copy_from(&attn_shd.view(&[seq, dim])?)?;
        }

        // to_out projection + norm2(attn) + gate
        let mut to_out_result = state.slice_mut(BT::BlkToOut, &[seq, dim])?;
        self.to_out.forward(&attn_flat, &mut to_out_result, cuda_config)?;

        let mut norm2_attn = state.slice_mut(BT::BlkNorm2Attn, &[seq, dim])?;
        self.attention_norm2.forward(&to_out_result, &mut norm2_attn, cuda_config)?;
        if let Some(ref g) = gate_msa {
            norm2_attn.mul_row(g)?;
        }

        // Residual: dst = x + norm2_attn
        dst.copy_from(x)?;
        *dst += &norm2_attn;

        // ══════════════════════════ FFN block ══════════════════════════
        let mut norm1_ffn = state.slice_mut(BT::BlkNorm1Ffn, &[seq, dim])?;
        self.ffn_norm1.forward(dst, &mut norm1_ffn, cuda_config)?;
        if let Some(ref s) = scale_mlp {
            norm1_ffn.mul_row(s)?;
        }

        let hidden_dim = self.w1.weight.shape()[0];
        let mut w1_out = state.slice_mut(BT::BlkW1Out, &[seq, hidden_dim])?;
        let mut w3_out = state.slice_mut(BT::BlkW3Out, &[seq, hidden_dim])?;
        self.w1.forward(&norm1_ffn, &mut w1_out, cuda_config)?;
        self.w3.forward(&norm1_ffn, &mut w3_out, cuda_config)?;

        // SwiGLU: silu(w1) * w3
        w1_out.silu()?;
        w1_out *= &w3_out;

        let mut ffn_out = state.slice_mut(BT::BlkFfnOut, &[seq, dim])?;
        self.w2.forward(&w1_out, &mut ffn_out, cuda_config)?;

        let mut norm2_ffn = state.slice_mut(BT::BlkNorm2Ffn, &[seq, dim])?;
        self.ffn_norm2.forward(&ffn_out, &mut norm2_ffn, cuda_config)?;
        if let Some(ref g) = gate_mlp {
            norm2_ffn.mul_row(g)?;
        }

        // Residual: dst += norm2_ffn
        *dst += &norm2_ffn;

        Ok(())
    }

    /// Apply per-head RMSNorm in place on `hd`, a `[S, dim]` state slice
    /// viewed as `[S*H, D]`.
    ///
    /// RMSNorm needs a source tensor distinct from its destination, so
    /// we shuttle the data through the `in_ty` / `out_ty` scratch slots
    /// and then copy the result back on top of `hd`.
    fn per_head_rmsnorm(
        &self,
        norm: &RMSNorm,
        hd: &mut Tensor,
        in_ty: BT,
        out_ty: BT,
        state: &mut DitState,
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        debug_assert_eq!(hd.shape().len(), 2);
        debug_assert_eq!(hd.shape()[1], self.dim);
        let seq = hd.shape()[0];
        let flat_shape = [seq * self.n_heads, self.head_dim];

        let hd_flat = hd.view(&flat_shape)?;
        let mut norm_in = state.slice_mut(in_ty, &flat_shape)?;
        norm_in.copy_from(&hd_flat)?;
        let mut norm_out = state.slice_mut(out_ty, &flat_shape)?;
        norm.forward(&norm_in, &mut norm_out, cuda_config)?;

        let mut hd_flat_mut = hd.view(&flat_shape)?;
        hd_flat_mut.copy_from(&norm_out)?;
        Ok(())
    }

    /// Materialize `[S, H, D]` as `[1, H, S, D]` into `dst_ty`.
    ///
    /// SDPA expects BHSD layout; we permute on a temporary (view-only)
    /// tensor and then copy the contiguous result into the state slot.
    fn to_bhsd(
        &self,
        src: &Tensor,
        dst_ty: BT,
        state: &mut DitState,
    ) -> Result<Tensor> {
        let seq = src.shape()[0];
        let shd = src.view(&[seq, self.n_heads, self.head_dim])?;
        let hsd = permute_nd(&shd, &[1, 0, 2])?;
        let bhsd_view = hsd.view(&[1, self.n_heads, seq, self.head_dim])?;

        let mut dst = state.slice_mut(dst_ty, &[1, self.n_heads, seq, self.head_dim])?;
        dst.copy_from(&bhsd_view)?;
        Ok(dst)
    }
}

// ───────────────────────── Tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use crate::base::error::Result;

    /// Build a minimal DiTBlock with random weights for testing.
    fn make_dit_block(dim: usize, n_heads: usize, hidden_dim: usize, modulation: bool, device: DeviceType) -> Result<DiTBlock> {
        let dtype = DataType::F32;
        let head_dim = dim / n_heads;

        let adaln_embed_dim = 256;

        fn randn_matmul(in_f: usize, out_f: usize, has_bias: bool, dtype: DataType, device: DeviceType, seed: u64) -> Result<Matmul> {
            let w = Tensor::randn(&[out_f, in_f], dtype, device, Some(seed))?;
            let b = if has_bias { Some(Tensor::randn(&[out_f], dtype, device, Some(seed + 1))?) } else { None };
            Ok(Matmul::from(w, b))
        }

        fn make_rmsnorm(dim: usize, dtype: DataType, device: DeviceType) -> Result<RMSNorm> {
            let mut w = Tensor::new(&[dim], dtype, device)?;
            // 初始化为 1.0（标准 RMSNorm 初始权重）
            if device == DeviceType::Cpu {
                let data = w.as_f32_mut()?.as_slice_mut()?;
                for v in data.iter_mut() { *v = 1.0; }
            }
            Ok(RMSNorm::from(w, 1e-5))
        }

        Ok(DiTBlock {
            attention_norm1: make_rmsnorm(dim, dtype, device)?,
            attention_norm2: make_rmsnorm(dim, dtype, device)?,
            ffn_norm1: make_rmsnorm(dim, dtype, device)?,
            ffn_norm2: make_rmsnorm(dim, dtype, device)?,
            to_q: randn_matmul(dim, dim, true, dtype, device, 10)?,
            to_k: randn_matmul(dim, dim, true, dtype, device, 20)?,
            to_v: randn_matmul(dim, dim, true, dtype, device, 30)?,
            to_out: randn_matmul(dim, dim, true, dtype, device, 40)?,
            norm_q: make_rmsnorm(head_dim, dtype, device)?,
            norm_k: make_rmsnorm(head_dim, dtype, device)?,
            w1: randn_matmul(dim, hidden_dim, true, dtype, device, 50)?,
            w3: randn_matmul(dim, hidden_dim, true, dtype, device, 60)?,
            w2: randn_matmul(hidden_dim, dim, true, dtype, device, 70)?,
            adaln_modulation: if modulation {
                Some(randn_matmul(adaln_embed_dim, 4 * dim, true, dtype, device, 80)?)
            } else {
                None
            },
            dim,
            n_heads,
            head_dim,
            modulation,
        })
    }

    fn assert_finite(t: &Tensor) {
        let cpu = if t.device() != DeviceType::Cpu { t.to_cpu().unwrap() } else { t.clone() };
        let data = cpu.as_f32().unwrap().as_slice().unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {}: {}", i, v);
        }
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        let mut max_diff = 0.0f32;
        let mut max_idx = 0;
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            if diff > max_diff { max_diff = diff; max_idx = i; }
        }
        assert!(max_diff < tol,
            "max diff {} at index {} (cpu={} gpu={}), tol={}",
            max_diff, max_idx, a[max_idx], b[max_idx], tol);
    }

    /// Build a minimally sized [`DitState`] big enough to run a single
    /// DiTBlock with the given `(seq, dim, n_heads, hidden_dim)`.
    fn make_state(
        dim: usize, n_heads: usize, hidden_dim: usize,
        dtype: DataType, device: DeviceType,
    ) -> Result<DitState> {
        use crate::model::diffusion::z_image::state::{DitShapeSpec, ZImageCapacity};
        let head_dim = dim / n_heads;
        // Small capacity so seq-wise slots cover at least 32 tokens
        // (SEQ_MULTI_OF rounds 1 patch up to 32).
        let capacity = ZImageCapacity {
            max_height: 64,
            max_width: 64,
            max_cap_len: 8,
        };
        let spec = DitShapeSpec {
            device,
            dtype,
            dim,
            n_heads,
            head_dim,
            hidden_dim,
            cap_feat_dim: 16,
            patch_size: 1,
            f_patch_size: 1,
            patch_in_dim: 16,
            final_out_dim: 16,
            capacity,
        };
        DitState::new(spec)
    }

    // ── CPU basic tests (no modulation) ──

    #[test]
    fn test_dit_block_cpu_no_mod() -> Result<()> {
        let (dim, heads, hidden) = (64, 4, 128);
        let block = make_dit_block(dim, heads, hidden, false, DeviceType::Cpu)?;
        let seq = 8;
        let head_dim = dim / heads;

        let x = Tensor::randn(&[seq, dim], DataType::F32, DeviceType::Cpu, Some(42))?;
        let cos = Tensor::randn(&[seq, head_dim / 2], DataType::F32, DeviceType::Cpu, Some(43))?;
        let sin = Tensor::randn(&[seq, head_dim / 2], DataType::F32, DeviceType::Cpu, Some(44))?;

        let mut state = make_state(dim, heads, hidden, DataType::F32, DeviceType::Cpu)?;
        let mut out = Tensor::new(&[seq, dim], DataType::F32, DeviceType::Cpu)?;
        block.forward(&x, &cos, &sin, None, &mut state, &mut out, None)?;
        assert_eq!(out.shape(), &[seq, dim]);
        assert_finite(&out);
        Ok(())
    }

    // ── CPU basic tests (with modulation) ──

    #[test]
    fn test_dit_block_cpu_with_mod() -> Result<()> {
        let (dim, heads, hidden) = (64, 4, 128);
        let block = make_dit_block(dim, heads, hidden, true, DeviceType::Cpu)?;
        let seq = 8;
        let head_dim = dim / heads;

        let x = Tensor::randn(&[seq, dim], DataType::F32, DeviceType::Cpu, Some(42))?;
        let cos = Tensor::randn(&[seq, head_dim / 2], DataType::F32, DeviceType::Cpu, Some(43))?;
        let sin = Tensor::randn(&[seq, head_dim / 2], DataType::F32, DeviceType::Cpu, Some(44))?;
        let adaln_c = Tensor::randn(&[256], DataType::F32, DeviceType::Cpu, Some(45))?;

        let mut state = make_state(dim, heads, hidden, DataType::F32, DeviceType::Cpu)?;
        let mut out = Tensor::new(&[seq, dim], DataType::F32, DeviceType::Cpu)?;
        block.forward(&x, &cos, &sin, Some(&adaln_c), &mut state, &mut out, None)?;
        assert_eq!(out.shape(), &[seq, dim]);
        assert_finite(&out);
        Ok(())
    }

    // ── CPU vs CUDA (no modulation) ──

    #[test]
    #[cfg(feature = "cuda")]
    fn test_dit_block_cpu_vs_cuda_no_mod() -> Result<()> {
        let (dim, heads, hidden) = (64, 4, 128);
        let seq = 8;
        let head_dim = dim / heads;

        let x = Tensor::randn(&[seq, dim], DataType::F32, DeviceType::Cpu, Some(42))?;
        let cos = Tensor::randn(&[seq, head_dim / 2], DataType::F32, DeviceType::Cpu, Some(43))?;
        let sin = Tensor::randn(&[seq, head_dim / 2], DataType::F32, DeviceType::Cpu, Some(44))?;

        // CPU
        let block_cpu = make_dit_block(dim, heads, hidden, false, DeviceType::Cpu)?;
        let mut state_cpu = make_state(dim, heads, hidden, DataType::F32, DeviceType::Cpu)?;
        let mut out_cpu = Tensor::new(&[seq, dim], DataType::F32, DeviceType::Cpu)?;
        block_cpu.forward(&x, &cos, &sin, None, &mut state_cpu, &mut out_cpu, None)?;

        // CUDA (same weights via same seeds)
        let mut block_gpu = make_dit_block(dim, heads, hidden, false, DeviceType::Cpu)?;
        block_gpu.to_cuda(0)?;
        let x_gpu = x.to_cuda(0)?;
        let cos_gpu = cos.to_cuda(0)?;
        let sin_gpu = sin.to_cuda(0)?;
        let cuda_config = crate::cuda::CudaConfig::new()?;
        let mut state_gpu = make_state(dim, heads, hidden, DataType::F32, DeviceType::Cuda(0))?;
        let mut out_gpu = Tensor::new(&[seq, dim], DataType::F32, DeviceType::Cuda(0))?;
        block_gpu.forward(&x_gpu, &cos_gpu, &sin_gpu, None, &mut state_gpu, &mut out_gpu, Some(&cuda_config))?;
        let out_gpu_cpu = out_gpu.to_cpu()?;

        let a = out_cpu.as_f32()?.as_slice()?;
        let b = out_gpu_cpu.as_f32()?.as_slice()?;
        assert_close(a, b, 0.1);
        Ok(())
    }

    // ── CPU vs CUDA (with modulation) ──

    #[test]
    #[cfg(feature = "cuda")]
    fn test_dit_block_cpu_vs_cuda_with_mod() -> Result<()> {
        let (dim, heads, hidden) = (64, 4, 128);
        let seq = 8;
        let head_dim = dim / heads;

        let x = Tensor::randn(&[seq, dim], DataType::F32, DeviceType::Cpu, Some(42))?;
        let cos = Tensor::randn(&[seq, head_dim / 2], DataType::F32, DeviceType::Cpu, Some(43))?;
        let sin = Tensor::randn(&[seq, head_dim / 2], DataType::F32, DeviceType::Cpu, Some(44))?;
        let adaln_c = Tensor::randn(&[256], DataType::F32, DeviceType::Cpu, Some(45))?;

        // CPU
        let block_cpu = make_dit_block(dim, heads, hidden, true, DeviceType::Cpu)?;
        let mut state_cpu = make_state(dim, heads, hidden, DataType::F32, DeviceType::Cpu)?;
        let mut out_cpu = Tensor::new(&[seq, dim], DataType::F32, DeviceType::Cpu)?;
        block_cpu.forward(&x, &cos, &sin, Some(&adaln_c), &mut state_cpu, &mut out_cpu, None)?;

        // CUDA
        let mut block_gpu = make_dit_block(dim, heads, hidden, true, DeviceType::Cpu)?;
        block_gpu.to_cuda(0)?;
        let x_gpu = x.to_cuda(0)?;
        let cos_gpu = cos.to_cuda(0)?;
        let sin_gpu = sin.to_cuda(0)?;
        let adaln_c_gpu = adaln_c.to_cuda(0)?;
        let cuda_config = crate::cuda::CudaConfig::new()?;
        let mut state_gpu = make_state(dim, heads, hidden, DataType::F32, DeviceType::Cuda(0))?;
        let mut out_gpu = Tensor::new(&[seq, dim], DataType::F32, DeviceType::Cuda(0))?;
        block_gpu.forward(&x_gpu, &cos_gpu, &sin_gpu, Some(&adaln_c_gpu), &mut state_gpu, &mut out_gpu, Some(&cuda_config))?;
        let out_gpu_cpu = out_gpu.to_cpu()?;

        let a = out_cpu.as_f32()?.as_slice()?;
        let b = out_gpu_cpu.as_f32()?.as_slice()?;
        assert_close(a, b, 0.15);
        Ok(())
    }
}
