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

use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};

use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::sdpa::scaled_dot_product_attention;
use crate::op::scalar;
use crate::op::tensor_utils::{clone_tensor, materialize, apply_rope_interleaved_dev, ewise_mul_inplace, permute_nd};
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

    /// Forward pass.
    ///
    /// - `x`: [S, dim]  (single-batch; outer loop handles batching)
    /// - `cos`, `sin`: [S, head_dim/2] — precomputed 3D RoPE
    /// - `adaln_c`: [adaln_embed_dim=256] or None (only used when modulation=True)
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        adaln_c: Option<&Tensor>,
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<Tensor> {
        let device = x.device();
        let dtype = x.dtype();
        let seq = x.shape()[0];
        let dim = self.dim;

        // ---- Compute modulation params (if modulation=True) ----
        let (scale_msa, gate_msa, scale_mlp, gate_mlp) = if self.modulation {
            let adaln = self.adaln_modulation.as_ref()
                .ok_or_else(|| Error::InternalError("adaln_modulation is None but modulation=True".into()))?;
            let c = adaln_c.ok_or_else(|| Error::InvalidArgument("adaln_c required when modulation=True".into()))?;

            // c: [256] → reshape to [1, 256], run Linear → [1, 4*dim]
            let c_2d = c.view(&[1, c.shape()[0]])?;
            let mut mod_out = Tensor::new(&[1, 4 * dim], dtype, device)?;
            adaln.forward(&c_2d, &mut mod_out, cuda_config)?;

            // Split into 4 chunks of size [1, dim]
            let scale_msa_raw = mod_out.slice(&[0, 0], &[1, dim])?;
            let gate_msa_raw = mod_out.slice(&[0, dim], &[1, dim])?;
            let scale_mlp_raw = mod_out.slice(&[0, 2 * dim], &[1, dim])?;
            let gate_mlp_raw = mod_out.slice(&[0, 3 * dim], &[1, dim])?;

            // scale = 1 + scale (allocate fresh dst, no redundant clone)
            let mut scale_msa = Tensor::new(&[1, dim], dtype, device)?;
            scalar::scalar_add(&scale_msa_raw, &mut scale_msa, 1.0)?;
            let mut scale_mlp = Tensor::new(&[1, dim], dtype, device)?;
            scalar::scalar_add(&scale_mlp_raw, &mut scale_mlp, 1.0)?;

            // gate = tanh(gate)
            let mut gate_msa = clone_tensor(&gate_msa_raw)?;
            scalar::tanh_inplace(&mut gate_msa)?;
            let mut gate_mlp = clone_tensor(&gate_mlp_raw)?;
            scalar::tanh_inplace(&mut gate_mlp)?;

            (Some(scale_msa), Some(gate_msa), Some(scale_mlp), Some(gate_mlp))
        } else {
            (None, None, None, None)
        };

        // ========== Attention block ==========
        let mut norm1_x = Tensor::new(&[seq, dim], dtype, device)?;
        self.attention_norm1.forward(x, &mut norm1_x, cuda_config)?;

        // Modulate: norm1_x *= scale_msa (broadcast [1,D] over [S,D])
        if let Some(ref s) = scale_msa {
            broadcast_mul_rowvec(&mut norm1_x, s)?;
        }

        // QKV projections: [S, dim]
        let mut q = Tensor::new(&[seq, dim], dtype, device)?;
        let mut k = Tensor::new(&[seq, dim], dtype, device)?;
        let mut v = Tensor::new(&[seq, dim], dtype, device)?;
        self.to_q.forward(&norm1_x, &mut q, cuda_config)?;
        self.to_k.forward(&norm1_x, &mut k, cuda_config)?;
        self.to_v.forward(&norm1_x, &mut v, cuda_config)?;

        // Reshape to [S, n_heads, head_dim] and apply per-head QK-norm
        let q = q.view(&[seq, self.n_heads, self.head_dim])?;
        let k = k.view(&[seq, self.n_heads, self.head_dim])?;
        let mut q = apply_rmsnorm_per_head(
            &self.norm_q, &q, seq, self.n_heads, self.head_dim, device, dtype, cuda_config,
        )?;
        let mut k = apply_rmsnorm_per_head(
            &self.norm_k, &k, seq, self.n_heads, self.head_dim, device, dtype, cuda_config,
        )?;

        // Apply 3D RoPE
        apply_rope_interleaved_dev(&mut q, cos, sin, self.head_dim)?;
        apply_rope_interleaved_dev(&mut k, cos, sin, self.head_dim)?;

        // sdpa expects [B, H, S, D]: reshape [S, H, D] → [1, H, S, D]
        let q_sdpa = reshape_shd_to_bhsd(&q, seq, self.n_heads, self.head_dim)?;
        let k_sdpa = reshape_shd_to_bhsd(&k, seq, self.n_heads, self.head_dim)?;
        let v_shd = v.view(&[seq, self.n_heads, self.head_dim])?;
        let v_hsd = permute_nd(&v_shd, &[1, 0, 2])?;
        let v_sdpa = v_hsd.view(&[1, self.n_heads, seq, self.head_dim])?;
        let v_sdpa = materialize(&v_sdpa)?;

        let mut attn_out_sdpa = Tensor::new(&[1, self.n_heads, seq, self.head_dim], dtype, device)?;
        scaled_dot_product_attention(&q_sdpa, &k_sdpa, &v_sdpa, &mut attn_out_sdpa, cuda_config)?;

        // [1, H, S, D] → [S, H*D]
        let attn_hsd = attn_out_sdpa.view(&[self.n_heads, seq, self.head_dim])?;
        let attn_shd = permute_nd(&attn_hsd, &[1, 0, 2])?;
        let attn_out = attn_shd.view(&[seq, dim])?;
        let attn_out = materialize(&attn_out)?;

        // to_out projection
        let mut to_out_result = Tensor::new(&[seq, dim], dtype, device)?;
        self.to_out.forward(&attn_out, &mut to_out_result, cuda_config)?;

        // norm2(attn)
        let mut norm2_attn = Tensor::new(&[seq, dim], dtype, device)?;
        self.attention_norm2.forward(&to_out_result, &mut norm2_attn, cuda_config)?;

        // gate_msa modulation
        if let Some(ref g) = gate_msa {
            broadcast_mul_rowvec(&mut norm2_attn, g)?;
        }

        // Residual: x = x + norm2_attn
        let mut x = clone_tensor(x)?;
        tensor_add_inplace(&mut x, &norm2_attn)?;

        // ========== FFN block ==========
        let mut norm1_ffn = Tensor::new(&[seq, dim], dtype, device)?;
        self.ffn_norm1.forward(&x, &mut norm1_ffn, cuda_config)?;

        if let Some(ref s) = scale_mlp {
            broadcast_mul_rowvec(&mut norm1_ffn, s)?;
        }

        // w1(x), w3(x) → [S, hidden_dim]
        let hidden_dim = self.w1.weight.shape()[0];
        let mut w1_out = Tensor::new(&[seq, hidden_dim], dtype, device)?;
        let mut w3_out = Tensor::new(&[seq, hidden_dim], dtype, device)?;
        self.w1.forward(&norm1_ffn, &mut w1_out, cuda_config)?;
        self.w3.forward(&norm1_ffn, &mut w3_out, cuda_config)?;

        // SwiGLU: silu(w1) * w3
        scalar::silu_inplace(&mut w1_out)?;
        ewise_mul_inplace(&mut w1_out, &w3_out)?;

        // w2 down-projection
        let mut ffn_out = Tensor::new(&[seq, dim], dtype, device)?;
        self.w2.forward(&w1_out, &mut ffn_out, cuda_config)?;

        // norm2(ffn)
        let mut norm2_ffn = Tensor::new(&[seq, dim], dtype, device)?;
        self.ffn_norm2.forward(&ffn_out, &mut norm2_ffn, cuda_config)?;

        // gate_mlp modulation
        if let Some(ref g) = gate_mlp {
            broadcast_mul_rowvec(&mut norm2_ffn, g)?;
        }

        // Residual: x = x + norm2_ffn
        tensor_add_inplace(&mut x, &norm2_ffn)?;

        Ok(x)
    }
}

// ───────────────────────── Helpers ─────────────────────────

/// In-place add: `a[i] += b[i]`
fn tensor_add_inplace(a: &mut Tensor, b: &Tensor) -> Result<()> {
    use crate::op::add_inplace::AddInplace;
    AddInplace::new().forward(b, a, None)
}

/// Broadcast-multiply: `x[s, d] *= v[0, d]` for all s (v shape [1, D]).
fn broadcast_mul_rowvec(x: &mut Tensor, v: &Tensor) -> Result<()> {
    use crate::op::broadcast_mul::broadcast_mul;
    let s = x.shape()[0];
    let d = x.shape()[1];
    let mut out = Tensor::new(&[s, d], x.dtype(), x.device())?;
    broadcast_mul(x, v, &mut out)?;
    x.copy_from(&out)?;
    Ok(())
}

/// Apply per-head RMSNorm over [S, H, D] by reshaping to [S*H, D].
#[allow(clippy::too_many_arguments)]
fn apply_rmsnorm_per_head(
    norm: &RMSNorm,
    x: &Tensor,           // [S, H, D]
    seq: usize, h: usize, d: usize,
    device: DeviceType,
    dtype: DataType,
    cuda_config: Option<&crate::OpConfig>,
) -> Result<Tensor> {
    let x_flat = x.view(&[seq * h, d])?;
    let x_flat_mat = materialize(&x_flat)?;
    let mut out = Tensor::new(&[seq * h, d], dtype, device)?;
    norm.forward(&x_flat_mat, &mut out, cuda_config)?;
    let out_shd = out.view(&[seq, h, d])?;
    materialize(&out_shd)
}

/// Reshape [S, H, D] → [1, H, S, D] (contiguous) using a CUDA-native permute.
fn reshape_shd_to_bhsd(
    x: &Tensor, seq: usize, h: usize, d: usize,
) -> Result<Tensor> {
    let permuted = permute_nd(x, &[1, 0, 2])?;
    permuted.view(&[1, h, seq, d]).and_then(|v| materialize(&v))
}

// ───────────────────────── Tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
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

        let out = block.forward(&x, &cos, &sin, None, None)?;
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

        let out = block.forward(&x, &cos, &sin, Some(&adaln_c), None)?;
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
        let out_cpu = block_cpu.forward(&x, &cos, &sin, None, None)?;

        // CUDA (same weights via same seeds)
        let mut block_gpu = make_dit_block(dim, heads, hidden, false, DeviceType::Cpu)?;
        block_gpu.to_cuda(0)?;
        let x_gpu = x.to_cuda(0)?;
        let cos_gpu = cos.to_cuda(0)?;
        let sin_gpu = sin.to_cuda(0)?;
        let cuda_config = crate::cuda::CudaConfig::new()?;
        let out_gpu = block_gpu.forward(&x_gpu, &cos_gpu, &sin_gpu, None, Some(&cuda_config))?;
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
        let out_cpu = block_cpu.forward(&x, &cos, &sin, Some(&adaln_c), None)?;

        // CUDA
        let mut block_gpu = make_dit_block(dim, heads, hidden, true, DeviceType::Cpu)?;
        block_gpu.to_cuda(0)?;
        let x_gpu = x.to_cuda(0)?;
        let cos_gpu = cos.to_cuda(0)?;
        let sin_gpu = sin.to_cuda(0)?;
        let adaln_c_gpu = adaln_c.to_cuda(0)?;
        let cuda_config = crate::cuda::CudaConfig::new()?;
        let out_gpu = block_gpu.forward(&x_gpu, &cos_gpu, &sin_gpu, Some(&adaln_c_gpu), Some(&cuda_config))?;
        let out_gpu_cpu = out_gpu.to_cpu()?;

        let a = out_cpu.as_f32()?.as_slice()?;
        let b = out_gpu_cpu.as_f32()?.as_slice()?;
        assert_close(a, b, 0.15);
        Ok(())
    }
}
