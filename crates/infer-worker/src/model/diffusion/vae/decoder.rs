//! Flux AutoencoderKL Decoder.

use std::path::Path;

use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
use crate::OpConfig;

use crate::model::diffusion::diffusers_loader::DiffusersLoader;
use crate::op::conv2d::conv2d;
use crate::op::groupnorm::groupnorm;
use crate::op::matmul::Matmul;
use crate::op::sdpa::scaled_dot_product_attention;
use crate::op::tensor_utils::{clone_tensor as tu_clone, materialize as tu_materialize, permute_nd};
use crate::op::upsample::upsample_nearest_2x;
use crate::tensor::Tensor;

const NORM_GROUPS: usize = 32;
const EPS: f32 = 1e-6;

// ───────────────────────── Config ─────────────────────────

#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub latent_channels: usize,     // 16
    pub out_channels: usize,        // 3
    pub block_out_channels: Vec<usize>, // [128, 256, 512, 512]
    pub layers_per_block: usize,    // 2 (but decoder uses layers_per_block + 1 = 3)
    pub scaling_factor: f32,        // 0.3611
    pub shift_factor: f32,          // 0.1159
}

impl VaeConfig {
    pub fn from_json<P: AsRef<Path>>(config_path: P) -> Result<Self> {
        let file = std::fs::File::open(config_path)?;
        let v: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| Error::InvalidArgument(format!("Failed to parse VAE config: {}", e)))?;
        let block_out_channels: Vec<usize> = v["block_out_channels"].as_array()
            .ok_or_else(|| Error::InvalidArgument("missing block_out_channels".into()))?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect();
        Ok(Self {
            latent_channels: v["latent_channels"].as_u64().unwrap_or(16) as usize,
            out_channels: v["out_channels"].as_u64().unwrap_or(3) as usize,
            block_out_channels,
            layers_per_block: v["layers_per_block"].as_u64().unwrap_or(2) as usize,
            scaling_factor: v["scaling_factor"].as_f64().unwrap_or(0.3611) as f32,
            shift_factor: v["shift_factor"].as_f64().unwrap_or(0.1159) as f32,
        })
    }
}

// ───────────────────────── ResnetBlock ─────────────────────────

pub struct ResnetBlock {
    pub norm1_w: Tensor, pub norm1_b: Tensor,
    pub conv1_w: Tensor, pub conv1_b: Tensor,
    pub norm2_w: Tensor, pub norm2_b: Tensor,
    pub conv2_w: Tensor, pub conv2_b: Tensor,
    /// 1×1 conv if in_ch != out_ch
    pub shortcut_w: Option<Tensor>,
    pub shortcut_b: Option<Tensor>,
    pub in_ch: usize,
    pub out_ch: usize,
}

impl ResnetBlock {
    fn forward(
        &self,
        x: &Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<Tensor> {
        let shape = x.shape();
        let (b, _c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        // norm1 + silu (fused on CUDA for BF16/F32)
        let mut h1 = Tensor::new(&[b, self.in_ch, h, w], x.dtype(), x.device())?;
        crate::op::groupnorm::groupnorm_silu(x, &self.norm1_w, &self.norm1_b, &mut h1, NORM_GROUPS, EPS)?;

        // conv1
        let mut h2 = Tensor::new(&[b, self.out_ch, h, w], x.dtype(), x.device())?;
        conv2d(&h1, &self.conv1_w, Some(&self.conv1_b), &mut h2, 1, 1, cuda_config)?;

        // norm2 + silu (fused)
        let mut h3 = Tensor::new(&[b, self.out_ch, h, w], x.dtype(), x.device())?;
        crate::op::groupnorm::groupnorm_silu(&h2, &self.norm2_w, &self.norm2_b, &mut h3, NORM_GROUPS, EPS)?;

        // conv2 (fused with residual add via beta=1 when possible)
        let mut h4 = Tensor::new(&[b, self.out_ch, h, w], x.dtype(), x.device())?;
        conv2d(&h3, &self.conv2_w, Some(&self.conv2_b), &mut h4, 1, 1, cuda_config)?;

        // shortcut (if needed) + residual
        let residual = if let (Some(sw), Some(sb)) = (&self.shortcut_w, &self.shortcut_b) {
            let mut sc = Tensor::new(&[b, self.out_ch, h, w], x.dtype(), x.device())?;
            conv2d(x, sw, Some(sb), &mut sc, 1, 0, cuda_config)?;
            sc
        } else {
            clone_tensor(x)?
        };

        // h4 += residual
        h4 += &residual;
        Ok(h4)
    }

    fn to_cuda(&mut self, id: i32) -> Result<()> {
        self.norm1_w = self.norm1_w.to_cuda(id)?;
        self.norm1_b = self.norm1_b.to_cuda(id)?;
        self.conv1_w = self.conv1_w.to_cuda(id)?;
        self.conv1_b = self.conv1_b.to_cuda(id)?;
        self.norm2_w = self.norm2_w.to_cuda(id)?;
        self.norm2_b = self.norm2_b.to_cuda(id)?;
        self.conv2_w = self.conv2_w.to_cuda(id)?;
        self.conv2_b = self.conv2_b.to_cuda(id)?;
        if let Some(w) = self.shortcut_w.take() { self.shortcut_w = Some(w.to_cuda(id)?); }
        if let Some(b) = self.shortcut_b.take() { self.shortcut_b = Some(b.to_cuda(id)?); }
        Ok(())
    }
}

// ───────────────────────── AttnBlock ─────────────────────────
//
// Standard diffusers VAE AttnBlock:
//   input [B, C, H, W]
//   group_norm([C]) → [B, C, H, W]
//   reshape to [B, N=H*W, C]
//   to_q, to_k, to_v: Linear(C → C, bias) applied to [B, N, C] → [B, N, C]
//   reshape to [B, 1, N, C] for sdpa (single head, head_dim=C)
//   sdpa → [B, 1, N, C]
//   reshape → [B, N, C] → to_out.0: Linear(C, C, bias)
//   reshape back → [B, C, H, W]
//   return input + result

pub struct AttnBlock {
    pub group_norm_w: Tensor, pub group_norm_b: Tensor,
    pub to_q: Matmul,
    pub to_k: Matmul,
    pub to_v: Matmul,
    pub to_out: Matmul,
    pub channels: usize,
}

impl AttnBlock {
    fn forward(
        &self,
        x: &Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<Tensor> {
        let shape = x.shape();
        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let n = h * w;

        // group_norm
        let mut normed = Tensor::new(&[b, c, h, w], x.dtype(), x.device())?;
        groupnorm(x, &self.group_norm_w, &self.group_norm_b, &mut normed, NORM_GROUPS, EPS)?;

        // [B, C, H, W] → [B, N, C]: permute(0, 2, 3, 1) → reshape(B, N, C)
        let permuted = permute_nd(&normed, &[0, 2, 3, 1])?; // [B, H, W, C] contiguous
        let bnc_view = permuted.view(&[b * n, c])?;
        let bnc = tu_materialize(&bnc_view)?;

        // q/k/v projections: [B*N, C] @ [C, C]^T → [B*N, C]
        let mut q = Tensor::new(&[b * n, c], x.dtype(), x.device())?;
        let mut k = Tensor::new(&[b * n, c], x.dtype(), x.device())?;
        let mut v = Tensor::new(&[b * n, c], x.dtype(), x.device())?;
        {
            self.to_q.forward(&bnc, &mut q, cuda_config)?;
            self.to_k.forward(&bnc, &mut k, cuda_config)?;
            self.to_v.forward(&bnc, &mut v, cuda_config)?;
        }

        // Reshape to [B, 1, N, C] for sdpa
        let q4 = materialize(&q.view(&[b, 1, n, c])?)?;
        let k4 = materialize(&k.view(&[b, 1, n, c])?)?;
        let v4 = materialize(&v.view(&[b, 1, n, c])?)?;
        let mut attn_out = Tensor::new(&[b, 1, n, c], x.dtype(), x.device())?;
        scaled_dot_product_attention(&q4, &k4, &v4, &mut attn_out, cuda_config)?;

        // [B, 1, N, C] → [B*N, C]
        let attn_flat = materialize(&attn_out.view(&[b * n, c])?)?;

        // to_out
        let mut proj = Tensor::new(&[b * n, c], x.dtype(), x.device())?;
        self.to_out.forward(&attn_flat, &mut proj, cuda_config)?;

        // Reshape [B*N, C] → [B, H, W, C] → permute → [B, C, H, W]
        let bhwc = proj.view(&[b, h, w, c])?;
        let bhwc_mat = tu_materialize(&bhwc)?;
        let bchw = permute_nd(&bhwc_mat, &[0, 3, 1, 2])?;

        // Add residual
        let mut out = clone_tensor(x)?;
        out += &bchw;
        Ok(out)
    }

    fn to_cuda(&mut self, id: i32) -> Result<()> {
        self.group_norm_w = self.group_norm_w.to_cuda(id)?;
        self.group_norm_b = self.group_norm_b.to_cuda(id)?;
        self.to_q.to_cuda(id)?;
        self.to_k.to_cuda(id)?;
        self.to_v.to_cuda(id)?;
        self.to_out.to_cuda(id)?;
        Ok(())
    }
}

// ───────────────────────── UpBlock ─────────────────────────

pub struct UpBlock {
    pub resnets: Vec<ResnetBlock>,
    pub upsampler: Option<(Tensor, Tensor)>, // conv weight, bias for 3x3 conv after nearest-2x
    pub out_ch: usize,
}

impl UpBlock {
    fn forward(
        &self,
        mut x: Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<Tensor> {
        for r in &self.resnets {
            { x = r.forward(&x, cuda_config)?; }
        }
        if let Some((w, bias)) = &self.upsampler {
            let shape = x.shape();
            let (b, c, h, width) = (shape[0], shape[1], shape[2], shape[3]);
            let mut up = Tensor::new(&[b, c, 2 * h, 2 * width], x.dtype(), x.device())?;
            upsample_nearest_2x(&x, &mut up)?;

            let mut conv_out = Tensor::new(&[b, self.out_ch, 2 * h, 2 * width], x.dtype(), x.device())?;
            conv2d(&up, w, Some(bias), &mut conv_out, 1, 1, cuda_config)?;
            x = conv_out;
        }
        Ok(x)
    }

    fn to_cuda(&mut self, id: i32) -> Result<()> {
        for r in &mut self.resnets { r.to_cuda(id)?; }
        if let Some((w, b)) = self.upsampler.take() {
            self.upsampler = Some((w.to_cuda(id)?, b.to_cuda(id)?));
        }
        Ok(())
    }
}

// ───────────────────────── VaeDecoder ─────────────────────────

pub struct VaeDecoder {
    pub config: VaeConfig,
    pub device: DeviceType,

    pub conv_in_w: Tensor, pub conv_in_b: Tensor,
    pub mid_resnet_0: ResnetBlock,
    pub mid_attn: AttnBlock,
    pub mid_resnet_1: ResnetBlock,
    pub up_blocks: Vec<UpBlock>,
    pub conv_norm_out_w: Tensor, pub conv_norm_out_b: Tensor,
    pub conv_out_w: Tensor, pub conv_out_b: Tensor,
}

impl VaeDecoder {
    pub fn from_pretrained<P: AsRef<Path>>(
        vae_dir: P,
        device: DeviceType,
    ) -> Result<Self> {
        let dir = vae_dir.as_ref();
        let config = VaeConfig::from_json(dir.join("config.json"))?;
        let loader = DiffusersLoader::load(dir)?;

        let boc = &config.block_out_channels;  // [128, 256, 512, 512]
        // decoder up_blocks are reversed: ch=[512, 512, 256, 128]
        // mid_block channels = boc[-1] = 512
        let mid_ch = boc[boc.len() - 1];

        // conv_in: (16 → 512, 3x3, pad=1)
        let conv_in_w = load_tensor(&loader, "decoder.conv_in.weight", device)?;
        let conv_in_b = load_tensor(&loader, "decoder.conv_in.bias", device)?;

        // mid block
        let mid_resnet_0 = load_resnet(&loader, "decoder.mid_block.resnets.0", mid_ch, mid_ch, device)?;
        let mid_attn = load_attn(&loader, "decoder.mid_block.attentions.0", mid_ch, device)?;
        let mid_resnet_1 = load_resnet(&loader, "decoder.mid_block.resnets.1", mid_ch, mid_ch, device)?;

        // up_blocks (4 blocks, reversed channel order from boc)
        // Diffusers decoder up_blocks[i] corresponds to reverse order:
        //   up_blocks[0]: in=boc[-1]=512, out=boc[-1]=512, has upsample
        //   up_blocks[1]: in=boc[-1]=512, out=boc[-2]=512, has upsample
        //   up_blocks[2]: in=boc[-2]=512, out=boc[-3]=256, has upsample
        //   up_blocks[3]: in=boc[-3]=256, out=boc[-4]=128, NO upsample (last)
        let n_blocks = boc.len();
        let mut up_blocks = Vec::with_capacity(n_blocks);
        let mut prev_out = mid_ch;
        for i in 0..n_blocks {
            // Output channels:
            //   diffusers: reversed_block_out_channels = boc reversed = [512, 512, 256, 128]
            //   up_blocks[i] has out_ch = reversed[i]
            let out_ch_block = boc[n_blocks - 1 - i];
            // But actual resnet structure: resnets[0].in = prev_out, resnets[0].out = out_ch_block
            //                              resnets[1..].in/out = out_ch_block
            let num_resnets = config.layers_per_block + 1; // 3

            let mut resnets = Vec::with_capacity(num_resnets);
            for r in 0..num_resnets {
                let in_c = if r == 0 { prev_out } else { out_ch_block };
                let prefix = format!("decoder.up_blocks.{}.resnets.{}", i, r);
                resnets.push(load_resnet(&loader, &prefix, in_c, out_ch_block, device)?);
            }

            // Upsampler: present in up_blocks[0..n_blocks-1]
            let upsampler = if i < n_blocks - 1 {
                let w = load_tensor(&loader, &format!("decoder.up_blocks.{}.upsamplers.0.conv.weight", i), device)?;
                let b = load_tensor(&loader, &format!("decoder.up_blocks.{}.upsamplers.0.conv.bias", i), device)?;
                Some((w, b))
            } else {
                None
            };

            up_blocks.push(UpBlock {
                resnets,
                upsampler,
                out_ch: out_ch_block,
            });
            prev_out = out_ch_block;
        }

        // Final
        let conv_norm_out_w = load_tensor(&loader, "decoder.conv_norm_out.weight", device)?;
        let conv_norm_out_b = load_tensor(&loader, "decoder.conv_norm_out.bias", device)?;
        let conv_out_w = load_tensor(&loader, "decoder.conv_out.weight", device)?;
        let conv_out_b = load_tensor(&loader, "decoder.conv_out.bias", device)?;

        Ok(Self {
            config,
            device,
            conv_in_w, conv_in_b,
            mid_resnet_0, mid_attn, mid_resnet_1,
            up_blocks,
            conv_norm_out_w, conv_norm_out_b,
            conv_out_w, conv_out_b,
        })
    }

    /// Decode latents: [B, 16, H, W] → [B, 3, H*8, W*8]
    ///
    /// latents 的 dtype 必须与权重一致（CPU: F32, CUDA: BF16）。
    pub fn decode(
        &self,
        latents: &Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<Tensor> {
        let shape = latents.shape();
        let (b, _c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let mid_ch = self.mid_resnet_0.out_ch;
        let dtype = latents.dtype();

        // conv_in: 16 → mid_ch
        let mut x = Tensor::new(&[b, mid_ch, h, w], dtype, latents.device())?;
        conv2d(latents, &self.conv_in_w, Some(&self.conv_in_b), &mut x, 1, 1, cuda_config)?;

        // mid block
        x = self.mid_resnet_0.forward(&x, cuda_config)?;
        x = self.mid_attn.forward(&x, cuda_config)?;
        x = self.mid_resnet_1.forward(&x, cuda_config)?;

        // up_blocks
        for up in &self.up_blocks {
            x = up.forward(x, cuda_config)?;
        }

        // conv_norm_out + silu + conv_out (fused gn+silu)
        let out_shape: Vec<usize> = x.shape().to_vec();
        let mut h2 = Tensor::new(out_shape.as_slice(), dtype, x.device())?;
        crate::op::groupnorm::groupnorm_silu(&x, &self.conv_norm_out_w, &self.conv_norm_out_b, &mut h2, NORM_GROUPS, EPS)?;

        let mut out = Tensor::new(&[out_shape[0], self.config.out_channels, out_shape[2], out_shape[3]],
            dtype, x.device())?;
        conv2d(&h2, &self.conv_out_w, Some(&self.conv_out_b), &mut out, 1, 1, cuda_config)?;

        Ok(out)
    }

    pub fn to_cuda(&mut self, id: i32) -> Result<()> {
        self.conv_in_w = self.conv_in_w.to_cuda(id)?;
        self.conv_in_b = self.conv_in_b.to_cuda(id)?;
        self.mid_resnet_0.to_cuda(id)?;
        self.mid_attn.to_cuda(id)?;
        self.mid_resnet_1.to_cuda(id)?;
        for up in &mut self.up_blocks { up.to_cuda(id)?; }
        self.conv_norm_out_w = self.conv_norm_out_w.to_cuda(id)?;
        self.conv_norm_out_b = self.conv_norm_out_b.to_cuda(id)?;
        self.conv_out_w = self.conv_out_w.to_cuda(id)?;
        self.conv_out_b = self.conv_out_b.to_cuda(id)?;
        self.device = DeviceType::Cuda(id);
        Ok(())
    }
}

// ───────────────────────── Loading helpers ─────────────────────────

fn load_tensor(loader: &DiffusersLoader, name: &str, device: DeviceType) -> Result<Tensor> {
    let view = loader.get_tensor(name)?;
    let t = Tensor::from_view_on_cpu(&view)?;
    let t = if device.is_cpu() && t.dtype() != DataType::F32 {
        t.to_dtype(DataType::F32)?
    } else { t };
    t.to_device(device)
}

fn load_resnet(
    loader: &DiffusersLoader,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    device: DeviceType,
) -> Result<ResnetBlock> {
    let norm1_w = load_tensor(loader, &format!("{}.norm1.weight", prefix), device)?;
    let norm1_b = load_tensor(loader, &format!("{}.norm1.bias", prefix), device)?;
    let conv1_w = load_tensor(loader, &format!("{}.conv1.weight", prefix), device)?;
    let conv1_b = load_tensor(loader, &format!("{}.conv1.bias", prefix), device)?;
    let norm2_w = load_tensor(loader, &format!("{}.norm2.weight", prefix), device)?;
    let norm2_b = load_tensor(loader, &format!("{}.norm2.bias", prefix), device)?;
    let conv2_w = load_tensor(loader, &format!("{}.conv2.weight", prefix), device)?;
    let conv2_b = load_tensor(loader, &format!("{}.conv2.bias", prefix), device)?;

    // shortcut only exists if in_ch != out_ch
    let (shortcut_w, shortcut_b) = if in_ch != out_ch {
        let w_name = format!("{}.conv_shortcut.weight", prefix);
        if loader.has_tensor(&w_name) {
            let sw = load_tensor(loader, &w_name, device)?;
            let sb = load_tensor(loader, &format!("{}.conv_shortcut.bias", prefix), device)?;
            (Some(sw), Some(sb))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    Ok(ResnetBlock {
        norm1_w, norm1_b, conv1_w, conv1_b,
        norm2_w, norm2_b, conv2_w, conv2_b,
        shortcut_w, shortcut_b,
        in_ch, out_ch,
    })
}

fn load_attn(
    loader: &DiffusersLoader,
    prefix: &str,
    ch: usize,
    device: DeviceType,
) -> Result<AttnBlock> {
    let group_norm_w = load_tensor(loader, &format!("{}.group_norm.weight", prefix), device)?;
    let group_norm_b = load_tensor(loader, &format!("{}.group_norm.bias", prefix), device)?;

    let to_q_w = load_tensor(loader, &format!("{}.to_q.weight", prefix), device)?;
    let to_q_b = load_tensor(loader, &format!("{}.to_q.bias", prefix), device)?;
    let to_k_w = load_tensor(loader, &format!("{}.to_k.weight", prefix), device)?;
    let to_k_b = load_tensor(loader, &format!("{}.to_k.bias", prefix), device)?;
    let to_v_w = load_tensor(loader, &format!("{}.to_v.weight", prefix), device)?;
    let to_v_b = load_tensor(loader, &format!("{}.to_v.bias", prefix), device)?;
    let to_out_w = load_tensor(loader, &format!("{}.to_out.0.weight", prefix), device)?;
    let to_out_b = load_tensor(loader, &format!("{}.to_out.0.bias", prefix), device)?;

    let to_q = Matmul::from(to_q_w, Some(to_q_b));
    let to_k = Matmul::from(to_k_w, Some(to_k_b));
    let to_v = Matmul::from(to_v_w, Some(to_v_b));
    let to_out = Matmul::from(to_out_w, Some(to_out_b));

    Ok(AttnBlock {
        group_norm_w, group_norm_b,
        to_q, to_k, to_v, to_out,
        channels: ch,
    })
}

// ───────────────────────── Tensor helpers ─────────────────────────

fn clone_tensor(t: &Tensor) -> Result<Tensor> { tu_clone(t) }
fn materialize(t: &Tensor) -> Result<Tensor> { tu_materialize(t) }

// ───────────────────────── Tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use crate::base::error::Result;
    use crate::tensor::Tensor;

    // ────────── Helper: create a ResnetBlock with random weights ──────────

    fn make_resnet(in_ch: usize, out_ch: usize, device: DeviceType) -> Result<ResnetBlock> {
        let dtype = DataType::F32;
        let seed_base = 100u64;
        let norm1_w = Tensor::randn(&[in_ch], dtype, device, Some(seed_base))?;
        let norm1_b = Tensor::randn(&[in_ch], dtype, device, Some(seed_base + 1))?;
        let conv1_w = Tensor::randn(&[out_ch, in_ch, 3, 3], dtype, device, Some(seed_base + 2))?;
        let conv1_b = Tensor::randn(&[out_ch], dtype, device, Some(seed_base + 3))?;
        let norm2_w = Tensor::randn(&[out_ch], dtype, device, Some(seed_base + 4))?;
        let norm2_b = Tensor::randn(&[out_ch], dtype, device, Some(seed_base + 5))?;
        let conv2_w = Tensor::randn(&[out_ch, out_ch, 3, 3], dtype, device, Some(seed_base + 6))?;
        let conv2_b = Tensor::randn(&[out_ch], dtype, device, Some(seed_base + 7))?;
        let (shortcut_w, shortcut_b) = if in_ch != out_ch {
            let sw = Tensor::randn(&[out_ch, in_ch, 1, 1], dtype, device, Some(seed_base + 8))?;
            let sb = Tensor::randn(&[out_ch], dtype, device, Some(seed_base + 9))?;
            (Some(sw), Some(sb))
        } else {
            (None, None)
        };
        Ok(ResnetBlock {
            norm1_w, norm1_b, conv1_w, conv1_b,
            norm2_w, norm2_b, conv2_w, conv2_b,
            shortcut_w, shortcut_b,
            in_ch, out_ch,
        })
    }

    fn make_attn(ch: usize, device: DeviceType) -> Result<AttnBlock> {
        let dtype = DataType::F32;
        let seed_base = 200u64;
        let group_norm_w = Tensor::randn(&[ch], dtype, device, Some(seed_base))?;
        let group_norm_b = Tensor::randn(&[ch], dtype, device, Some(seed_base + 1))?;
        let to_q = Matmul::from(
            Tensor::randn(&[ch, ch], dtype, device, Some(seed_base + 2))?,
            Some(Tensor::randn(&[ch], dtype, device, Some(seed_base + 3))?),
        );
        let to_k = Matmul::from(
            Tensor::randn(&[ch, ch], dtype, device, Some(seed_base + 4))?,
            Some(Tensor::randn(&[ch], dtype, device, Some(seed_base + 5))?),
        );
        let to_v = Matmul::from(
            Tensor::randn(&[ch, ch], dtype, device, Some(seed_base + 6))?,
            Some(Tensor::randn(&[ch], dtype, device, Some(seed_base + 7))?),
        );
        let to_out = Matmul::from(
            Tensor::randn(&[ch, ch], dtype, device, Some(seed_base + 8))?,
            Some(Tensor::randn(&[ch], dtype, device, Some(seed_base + 9))?),
        );
        Ok(AttnBlock { group_norm_w, group_norm_b, to_q, to_k, to_v, to_out, channels: ch })
    }

    fn assert_finite(t: &Tensor) {
        let cpu = t.to_device(DeviceType::Cpu).unwrap();
        let cpu = if cpu.dtype() != DataType::F32 { cpu.to_dtype(DataType::F32).unwrap() } else { cpu };
        let data = cpu.as_f32().unwrap().as_slice().unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {}: {}", i, v);
        }
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "mismatch at {}: cpu={} gpu={}, diff={}",
                i, x, y, (x - y).abs()
            );
        }
    }

    // ────────── ResnetBlock tests ──────────

    #[test]
    fn test_resnet_block_cpu_same_ch() -> Result<()> {
        let ch = 32;
        let resnet = make_resnet(ch, ch, DeviceType::Cpu)?;
        let x = Tensor::randn(&[1, ch, 4, 4], DataType::F32, DeviceType::Cpu, Some(42))?;

        let out = resnet.forward(&x, None)?;

        assert_eq!(out.shape(), &[1, ch, 4, 4]);
        assert_finite(&out);
        Ok(())
    }

    #[test]
    fn test_resnet_block_cpu_shortcut() -> Result<()> {
        let (in_ch, out_ch) = (32, 64);
        let resnet = make_resnet(in_ch, out_ch, DeviceType::Cpu)?;
        let x = Tensor::randn(&[1, in_ch, 4, 4], DataType::F32, DeviceType::Cpu, Some(42))?;

        let out = resnet.forward(&x, None)?;

        assert_eq!(out.shape(), &[1, out_ch, 4, 4]);
        assert_finite(&out);
        Ok(())
    }

    #[test]
    fn test_resnet_block_cpu_vs_cuda() -> Result<()> {
        let ch = 32;
        let resnet_cpu = make_resnet(ch, ch, DeviceType::Cpu)?;
        let x_cpu = Tensor::randn(&[1, ch, 4, 4], DataType::F32, DeviceType::Cpu, Some(42))?;
        let out_cpu = resnet_cpu.forward(&x_cpu, None)?;

        // Build CUDA resnet with same weights
        let mut resnet_gpu = make_resnet(ch, ch, DeviceType::Cpu)?;
        resnet_gpu.to_cuda(0)?;
        let x_gpu = x_cpu.to_cuda(0)?;
        let cuda_config = crate::cuda::CudaConfig::new()?;
        let out_gpu = resnet_gpu.forward(&x_gpu, Some(&cuda_config))?;
        let out_gpu_cpu = out_gpu.to_cpu()?;

        let a = out_cpu.as_f32()?.as_slice()?;
        let b = out_gpu_cpu.as_f32()?.as_slice()?;
        assert_close(a, b, 0.05);
        Ok(())
    }

    // ────────── AttnBlock tests ──────────

    #[test]
    fn test_attn_block_cpu() -> Result<()> {
        let ch = 32;
        let attn = make_attn(ch, DeviceType::Cpu)?;
        let x = Tensor::randn(&[1, ch, 4, 4], DataType::F32, DeviceType::Cpu, Some(42))?;

        let out = attn.forward(&x, None)?;

        assert_eq!(out.shape(), &[1, ch, 4, 4]);
        assert_finite(&out);
        Ok(())
    }

    #[test]
    fn test_attn_block_cpu_vs_cuda() -> Result<()> {
        let ch = 32;
        let attn_cpu = make_attn(ch, DeviceType::Cpu)?;
        let x_cpu = Tensor::randn(&[1, ch, 4, 4], DataType::F32, DeviceType::Cpu, Some(42))?;
        let out_cpu = attn_cpu.forward(&x_cpu, None)?;

        let mut attn_gpu = make_attn(ch, DeviceType::Cpu)?;
        attn_gpu.to_cuda(0)?;
        let x_gpu = x_cpu.to_cuda(0)?;
        let cuda_config = crate::cuda::CudaConfig::new()?;
        let out_gpu = attn_gpu.forward(&x_gpu, Some(&cuda_config))?;
        let out_gpu_cpu = out_gpu.to_cpu()?;

        let a = out_cpu.as_f32()?.as_slice()?;
        let b = out_gpu_cpu.as_f32()?.as_slice()?;
        assert_close(a, b, 0.1);
        Ok(())
    }

    // ────────── UpBlock tests ──────────

    #[test]
    fn test_upblock_cpu_no_upsample() -> Result<()> {
        let ch = 32;
        let resnet = make_resnet(ch, ch, DeviceType::Cpu)?;
        let up = UpBlock { resnets: vec![resnet], upsampler: None, out_ch: ch };
        let x = Tensor::randn(&[1, ch, 4, 4], DataType::F32, DeviceType::Cpu, Some(42))?;

        let out = up.forward(x, None)?;

        assert_eq!(out.shape(), &[1, ch, 4, 4]);
        assert_finite(&out);
        Ok(())
    }

    #[test]
    fn test_upblock_cpu_with_upsample() -> Result<()> {
        let ch = 32;
        let resnet = make_resnet(ch, ch, DeviceType::Cpu)?;
        let up_w = Tensor::randn(&[ch, ch, 3, 3], DataType::F32, DeviceType::Cpu, Some(50))?;
        let up_b = Tensor::randn(&[ch], DataType::F32, DeviceType::Cpu, Some(51))?;
        let up = UpBlock { resnets: vec![resnet], upsampler: Some((up_w, up_b)), out_ch: ch };
        let x = Tensor::randn(&[1, ch, 4, 4], DataType::F32, DeviceType::Cpu, Some(42))?;

        let out = up.forward(x, None)?;

        // Upsample 2x: 4→8
        assert_eq!(out.shape(), &[1, ch, 8, 8]);
        assert_finite(&out);
        Ok(())
    }

    // ────────── VaeDecoder mini decode (random weights) ──────────

    /// Build a minimal VaeDecoder with random weights for testing.
    /// Uses tiny channels to keep it fast: block_out_channels=[32,32,32,32].
    fn make_mini_decoder(device: DeviceType) -> Result<VaeDecoder> {
        let dtype = DataType::F32;
        let latent_ch = 4;
        let mid_ch = 32;
        let out_ch_img = 3;

        let config = VaeConfig {
            latent_channels: latent_ch,
            out_channels: out_ch_img,
            block_out_channels: vec![32, 32, 32, 32],
            layers_per_block: 0, // 0+1=1 resnet per block → fast
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
        };

        let conv_in_w = Tensor::randn(&[mid_ch, latent_ch, 3, 3], dtype, device, Some(1))?;
        let conv_in_b = Tensor::randn(&[mid_ch], dtype, device, Some(2))?;
        let mid_resnet_0 = make_resnet(mid_ch, mid_ch, device)?;
        let mid_attn = make_attn(mid_ch, device)?;
        let mid_resnet_1 = make_resnet(mid_ch, mid_ch, device)?;

        // 4 up_blocks: all 32→32, first 3 have upsampler
        let mut up_blocks = Vec::new();
        for i in 0..4 {
            let resnet = make_resnet(mid_ch, mid_ch, device)?;
            let upsampler = if i < 3 {
                let w = Tensor::randn(&[mid_ch, mid_ch, 3, 3], dtype, device, Some(300 + i as u64))?;
                let b = Tensor::randn(&[mid_ch], dtype, device, Some(310 + i as u64))?;
                Some((w, b))
            } else {
                None
            };
            up_blocks.push(UpBlock { resnets: vec![resnet], upsampler, out_ch: mid_ch });
        }

        let conv_norm_out_w = Tensor::randn(&[mid_ch], dtype, device, Some(400))?;
        let conv_norm_out_b = Tensor::randn(&[mid_ch], dtype, device, Some(401))?;
        let conv_out_w = Tensor::randn(&[out_ch_img, mid_ch, 3, 3], dtype, device, Some(402))?;
        let conv_out_b = Tensor::randn(&[out_ch_img], dtype, device, Some(403))?;

        Ok(VaeDecoder {
            config,
            device,
            conv_in_w, conv_in_b,
            mid_resnet_0, mid_attn, mid_resnet_1,
            up_blocks,
            conv_norm_out_w, conv_norm_out_b,
            conv_out_w, conv_out_b,
        })
    }

    #[test]
    fn test_vae_decode_cpu_shape() -> Result<()> {
        let decoder = make_mini_decoder(DeviceType::Cpu)?;
        let latents = Tensor::randn(&[1, 4, 2, 2], DataType::F32, DeviceType::Cpu, Some(42))?;

        let out = decoder.decode(&latents, None)?;

        // 3 upsamplers → 2*2^3 = 16
        assert_eq!(out.shape(), &[1, 3, 16, 16]);
        assert_finite(&out);
        Ok(())
    }

    #[test]
    fn test_vae_decode_cpu_vs_cuda() -> Result<()> {
        let decoder_cpu = make_mini_decoder(DeviceType::Cpu)?;
        let latents = Tensor::randn(&[1, 4, 2, 2], DataType::F32, DeviceType::Cpu, Some(42))?;
        let out_cpu = decoder_cpu.decode(&latents, None)?;

        let mut decoder_gpu = make_mini_decoder(DeviceType::Cpu)?;
        decoder_gpu.to_cuda(0)?;
        let latents_gpu = latents.to_cuda(0)?;
        let cuda_config = crate::cuda::CudaConfig::new()?;
        let out_gpu = decoder_gpu.decode(&latents_gpu, Some(&cuda_config))?;
        let out_gpu_cpu = out_gpu.to_device(DeviceType::Cpu)?;

        let a = out_cpu.as_f32()?.as_slice()?;
        let b = out_gpu_cpu.as_f32()?.as_slice()?;
        assert_eq!(a.len(), b.len());
        // VAE decoder is a deep network so tolerance needs to be reasonable
        assert_close(a, b, 0.5);
        Ok(())
    }
}
