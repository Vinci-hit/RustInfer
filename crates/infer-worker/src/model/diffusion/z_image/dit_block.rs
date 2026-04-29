//! Single DiT transformer block for Z-Image.

use crate::OpConfig;
use crate::base::error::{Error, Result};

use crate::model::diffusion::buffer::DiffBufferType as BT;
use crate::model::diffusion::z_image::state::DitState;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::sdpa::dit_sdpa;
use crate::tensor::Tensor;

pub struct DiTBlock {
    pub attention_norm1: RMSNorm,
    pub attention_norm2: RMSNorm,
    pub ffn_norm1: RMSNorm,
    pub ffn_norm2: RMSNorm,

    /// Fused `[to_q; to_k; to_v]` — weight shape `[3*dim, dim]`, no bias.
    pub to_qkv: Matmul,
    pub to_out: Matmul,

    pub norm_q: RMSNorm,
    pub norm_k: RMSNorm,

    pub w1: Matmul,
    pub w3: Matmul,
    pub w2: Matmul,

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

        // ── Modulation ──
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

        // ── Attention ──
        let mut norm1_x = state.slice_mut(BT::BlkNorm1X, &[seq, dim])?;
        self.attention_norm1.forward(x, &mut norm1_x, cuda_config)?;
        if let Some(ref s) = scale_msa {
            norm1_x.mul_row(s)?;
        }

        let mut q = state.slice_mut(BT::BlkQ, &[seq, dim])?;
        let mut k = state.slice_mut(BT::BlkK, &[seq, dim])?;
        let mut v = state.slice_mut(BT::BlkV, &[seq, dim])?;
        {
            let mut qkv_out = state.slice_mut(BT::BlkQkvOut, &[seq, 3 * dim])?;
            self.to_qkv.forward(&norm1_x, &mut qkv_out, cuda_config)?;
        }

        // Fused QKV split + per-head RMSNorm + RoPE.
        {
            let qkv_out = state.slice(BT::BlkQkvOut, &[seq, 3 * dim])?;
            crate::op::kernels::cuda::zimage_fused_qkv_split_head_rmsnorm_rope_interleaved_bf16(
                &qkv_out,
                &self.norm_q.weight,
                &self.norm_k.weight,
                cos, sin,
                &mut q, &mut k, &mut v,
                seq, self.n_heads, self.head_dim,
                self.norm_q.eps(),
                cuda_config,
            )?;
        }

        // ── Self-attention (SHD layout) ──
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

        let mut to_out_result = state.slice_mut(BT::BlkToOut, &[seq, dim])?;
        self.to_out.forward(&attn_flat, &mut to_out_result, cuda_config)?;

        let mut norm2_attn = state.slice_mut(BT::BlkNorm2Attn, &[seq, dim])?;
        self.attention_norm2.forward(&to_out_result, &mut norm2_attn, cuda_config)?;
        if let Some(ref g) = gate_msa {
            norm2_attn.mul_row(g)?;
        }

        // ── Residual + fused_add_rmsnorm ──
        dst.copy_from_on_current_stream(x)?;
        let mut norm1_ffn = state.slice_mut(BT::BlkNorm1Ffn, &[seq, dim])?;
        crate::op::fused_add_rmsnorm::fused_add_rmsnorm(
            &mut norm1_ffn, dst, &norm2_attn,
            &self.ffn_norm1.weight, self.ffn_norm1.eps(),
            cuda_config,
        )?;

        // ── FFN (SwiGLU) ──
        if let Some(ref s) = scale_mlp {
            norm1_ffn.mul_row(s)?;
        }

        let hidden_dim = self.w1.weight.shape()[0];
        let mut w1_out = state.slice_mut(BT::BlkW1Out, &[seq, hidden_dim])?;
        let mut w3_out = state.slice_mut(BT::BlkW3Out, &[seq, hidden_dim])?;
        self.w1.forward(&norm1_ffn, &mut w1_out, cuda_config)?;
        self.w3.forward(&norm1_ffn, &mut w3_out, cuda_config)?;

        crate::op::swiglu::SwiGLU::new()
            .forward(&w3_out, &mut w1_out, cuda_config)?;

        let mut ffn_out = state.slice_mut(BT::BlkFfnOut, &[seq, dim])?;
        self.w2.forward(&w1_out, &mut ffn_out, cuda_config)?;

        let mut norm2_ffn = state.slice_mut(BT::BlkNorm2Ffn, &[seq, dim])?;
        self.ffn_norm2.forward(&ffn_out, &mut norm2_ffn, cuda_config)?;
        if let Some(ref g) = gate_mlp {
            norm2_ffn.mul_row(g)?;
        }

        *dst += &norm2_ffn;
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use crate::base::error::Result;

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
            dim, n_heads, head_dim, modulation,
        })
    }

    fn assert_finite(t: &Tensor) {
        let cpu = if t.device() != DeviceType::Cpu { t.to_cpu().unwrap() } else { t.clone() };
        let data = cpu.as_f32().unwrap().as_slice().unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {}: {}", i, v);
        }
    }

    fn make_state(
        dim: usize, n_heads: usize, hidden_dim: usize,
        dtype: DataType, device: DeviceType,
    ) -> Result<DitState> {
        use crate::model::diffusion::z_image::state::{DitShapeSpec, ZImageCapacity};
        let head_dim = dim / n_heads;
        let capacity = ZImageCapacity { max_height: 64, max_width: 64, max_cap_len: 8 };
        let spec = DitShapeSpec {
            device, dtype, dim, n_heads, head_dim, hidden_dim,
            cap_feat_dim: 16, patch_size: 1, f_patch_size: 1,
            patch_in_dim: 16, final_out_dim: 16, capacity,
        };
        DitState::new(spec)
    }

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
}
