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
use crate::op::sdpa::dit_sdpa;
use crate::tensor::Tensor;

/// DiT transformer block (Z-Image style).
pub struct DiTBlock {
    pub attention_norm1: RMSNorm,
    pub attention_norm2: RMSNorm,
    pub ffn_norm1: RMSNorm,
    pub ffn_norm2: RMSNorm,

    /// Fused `[to_q; to_k; to_v]` projection — weight shape `[3*dim, dim]`,
    /// bias-less (Z-Image convention).
    ///
    /// A single GEMM writes `[seq, 3*dim]` into `BlkQkvOut`. Three
    /// independent `split_cols` kernel launches then copy the three
    /// `[seq, dim]` column slabs into the contiguous `BlkQ` / `BlkK` /
    /// `BlkV` buffers — mirroring the LLM-side pattern used in
    /// `Llama3::forward_prefill` and `Qwen3::forward_prefill`.
    pub to_qkv: Matmul,
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
        self.to_qkv.to_cuda(device_id)?;
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

        // ── Fused QKV projection + column split ──
        //
        // 1. One GEMM of `[seq, dim] @ [3*dim, dim]^T → [seq, 3*dim]`
        //    writes Q / K / V as three contiguous column slabs in
        //    `BlkQkvOut` (cols `[0..dim)`=Q, `[dim..2*dim)`=K,
        //    `[2*dim..3*dim)`=V).
        // 2. Three independent `split_cols` launches copy each slab into
        //    the `[seq, dim]` `BlkQ` / `BlkK` / `BlkV` buffers — same
        //    pattern used by `Llama3::forward_prefill`.
        //
        // `q` / `k` / `v` are owned views sharing the underlying `Arc<Buffer>`
        // of the `Blk*` slots, so they don't borrow `state` — grabbing
        // `BlkQkvOut` via `state.slice_mut` in the inner block is free
        // to re-borrow.
        let mut q = state.slice_mut(BT::BlkQ, &[seq, dim])?;
        let mut k = state.slice_mut(BT::BlkK, &[seq, dim])?;
        let mut v = state.slice_mut(BT::BlkV, &[seq, dim])?;
        {
            let mut qkv_out = state.slice_mut(BT::BlkQkvOut, &[seq, 3 * dim])?;
            self.to_qkv.forward(&norm1_x, &mut qkv_out, cuda_config)?;

            #[cfg(feature = "cuda")]
            let stream = crate::cuda::CudaConfig::resolve_stream(cuda_config);
            #[cfg(feature = "cuda")]
            {
                crate::op::split_cols::split_cols_tensor(&qkv_out, &mut q, seq, 3 * dim, 0,         dim, stream)?;
                crate::op::split_cols::split_cols_tensor(&qkv_out, &mut k, seq, 3 * dim, dim,       dim, stream)?;
                crate::op::split_cols::split_cols_tensor(&qkv_out, &mut v, seq, 3 * dim, 2 * dim,   dim, stream)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                crate::op::split_cols::split_cols_tensor(&qkv_out, &mut q, seq, 3 * dim, 0,         dim)?;
                crate::op::split_cols::split_cols_tensor(&qkv_out, &mut k, seq, 3 * dim, dim,       dim)?;
                crate::op::split_cols::split_cols_tensor(&qkv_out, &mut v, seq, 3 * dim, 2 * dim,   dim)?;
            }
        }

        // Per-head QK-norm + RoPE (all in place on Q / K slabs).
        self.per_head_rmsnorm(&self.norm_q, &mut q, cuda_config)?;
        self.per_head_rmsnorm(&self.norm_k, &mut k, cuda_config)?;
        {
            let mut q_3d = q.view(&[seq, self.n_heads, self.head_dim])?;
            let mut k_3d = k.view(&[seq, self.n_heads, self.head_dim])?;
            q_3d.rope_interleaved(cos, sin, self.head_dim)?;
            k_3d.rope_interleaved(cos, sin, self.head_dim)?;
        }

        // ── Self-attention in native SHD layout ──
        //
        // Q/K/V already live in the `BlkQ` / `BlkK` / `BlkV` slabs as
        // `[seq, dim]`, which is byte-equivalent to
        // `[seq, n_heads, head_dim]` (SHD). `dit_sdpa` takes SHD directly
        // and writes the result into `BlkAttnFlat` also viewed as SHD —
        // no permute/copy before or after attention. On the
        // `(BF16, head_dim=128, seq%128==0)` fast path this collapses the
        // entire attention into a single CUTLASS flash-attention launch.
        let q_shd = q.view(&[seq, self.n_heads, self.head_dim])?;
        let k_shd = k.view(&[seq, self.n_heads, self.head_dim])?;
        let v_shd = v.view(&[seq, self.n_heads, self.head_dim])?;
        let attn_flat = state.slice_mut(BT::BlkAttnFlat, &[seq, dim])?;
        {
            let mut attn_shd = attn_flat.view(&[seq, self.n_heads, self.head_dim])?;
            dit_sdpa(
                &q_shd, &k_shd, &v_shd, &mut attn_shd,
                self.n_heads, self.head_dim, cuda_config,
            )?;
        }

        // to_out projection + norm2(attn) + gate
        let mut to_out_result = state.slice_mut(BT::BlkToOut, &[seq, dim])?;
        self.to_out.forward(&attn_flat, &mut to_out_result, cuda_config)?;

        let mut norm2_attn = state.slice_mut(BT::BlkNorm2Attn, &[seq, dim])?;
        self.attention_norm2.forward(&to_out_result, &mut norm2_attn, cuda_config)?;
        if let Some(ref g) = gate_msa {
            norm2_attn.mul_row(g)?;
        }

        // Residual + next-norm fusion (FFN side):
        //
        //   dst         = x + norm2_attn          (residual add)
        //   norm1_ffn   = ffn_norm1(dst)          (RMSNorm)
        //
        // On CUDA these two ops fuse into a single kernel
        // (`fused_add_rmsnorm`): it reads `dst` (post-copy-from-x) plus
        // `norm2_attn`, writes back the summed residual into `dst`, and
        // simultaneously emits `rmsnorm(dst, ffn_norm1.weight)` into
        // `norm1_ffn`. Saves one `add_inplace` + one `rmsnorm` launch
        // per block × 2 norm types across the 34 DiT blocks.
        dst.copy_from_on_current_stream(x)?;
        let mut norm1_ffn = state.slice_mut(BT::BlkNorm1Ffn, &[seq, dim])?;
        // fused_add_rmsnorm currently ships only BF16/F16 kernels — fall
        // back to the two-op sequence on CPU or when running CUDA tests
        // in F32 (the production diffusion pipeline is always BF16).
        #[cfg(feature = "cuda")]
        let use_fused = dst.device().is_cuda()
            && matches!(dst.dtype(), crate::base::DataType::BF16 | crate::base::DataType::F16);
        #[cfg(not(feature = "cuda"))]
        let use_fused = false;
        if use_fused {
            #[cfg(feature = "cuda")]
            crate::op::fused_add_rmsnorm::fused_add_rmsnorm(
                &mut norm1_ffn,
                dst,
                &norm2_attn,
                &self.ffn_norm1.weight,
                self.ffn_norm1.eps(),
                cuda_config,
            )?;
        } else {
            *dst += &norm2_attn;
            self.ffn_norm1.forward(dst, &mut norm1_ffn, cuda_config)?;
        }

        // ══════════════════════════ FFN block ══════════════════════════
        if let Some(ref s) = scale_mlp {
            norm1_ffn.mul_row(s)?;
        }

        let hidden_dim = self.w1.weight.shape()[0];
        let mut w1_out = state.slice_mut(BT::BlkW1Out, &[seq, hidden_dim])?;
        let mut w3_out = state.slice_mut(BT::BlkW3Out, &[seq, hidden_dim])?;
        self.w1.forward(&norm1_ffn, &mut w1_out, cuda_config)?;
        self.w3.forward(&norm1_ffn, &mut w3_out, cuda_config)?;

        // SwiGLU: w1_out = silu(w1_out) * w3_out (single fused kernel).
        // Saves one silu_inplace + one ewise_mul launch (plus the
        // intermediate DRAM round-trip of w1_out) versus the two-op
        // sequence `w1_out.silu(); w1_out *= &w3_out;`.
        crate::op::swiglu::SwiGLU::new()
            .forward(&w3_out, &mut w1_out, cuda_config)?;

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
    /// Uses [`RMSNorm::forward_inplace`]: `hd = rmsnorm(hd, weight, eps)`
    /// directly — no scratch buffer, no surrounding copies.
    fn per_head_rmsnorm(
        &self,
        norm: &RMSNorm,
        hd: &mut Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        debug_assert_eq!(hd.shape().len(), 2);
        debug_assert_eq!(hd.shape()[1], self.dim);
        let seq = hd.shape()[0];
        let flat_shape = [seq * self.n_heads, self.head_dim];

        let mut hd_flat = hd.view(&flat_shape)?;
        norm.forward_inplace(&mut hd_flat, cuda_config)?;
        Ok(())
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
            // Fused QKV weight: `[3*dim, dim]`, mirrors what
            // `load_fused_qkv_linear` produces from the three separate
            // checkpoint tensors. Deterministic seed so CPU and CUDA
            // instances built with the same seed match bit-for-bit.
            to_qkv: randn_matmul(dim, 3 * dim, false, dtype, device, 10)?,
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
