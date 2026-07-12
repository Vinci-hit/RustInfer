//! VAE decoder for Z-Image (AutoencoderKL, Flux variant).
//!
//! Architecture: `latents [B, 16, h, w]` →
//!   conv_in (16→512)
//!   mid: ResnetBlock + AttnBlock + ResnetBlock
//!   up_blocks[0..4]: 3 ResnetBlocks + (optional) Upsample2x + Conv
//!   conv_norm_out + SiLU + conv_out (→ 3 channels)
//! → `image [B, 3, h*8, w*8]`
//!
//! All sub-modules carry their own weights; `forward` allocates the
//! intermediate tensors (decoder volume scales by 4× per up-block, so
//! reusable scratches don't help much — eager allocation matches the
//! reference implementation).

use crate::domain::ports::{CoreOps, DiffusionOps, OpBackend, OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{Dtype, Shape};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::io::SafetensorsReader;
use crate::models::layers::Linear;
use crate::models::loader::WeightLoader;

const NORM_GROUPS: usize = 32;
const EPS: f32 = 1e-6;

#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub latent_channels: usize,
    pub out_channels: usize,
    /// `[128, 256, 512, 512]` for Flux-style.
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub scaling_factor: f32,
    pub shift_factor: f32,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            latent_channels: 16,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }
}

impl VaeConfig {
    pub fn from_json<P: AsRef<std::path::Path>>(path: P) -> OpResult<Self> {
        let s = std::fs::read_to_string(&path)
            .map_err(|e| OpError::Kernel(format!("vae config: {}", e)))?;
        let v: serde_json::Value = serde_json::from_str(&s)
            .map_err(|e| OpError::Kernel(format!("vae config parse: {}", e)))?;
        let block_out_channels: Vec<usize> = v["block_out_channels"]
            .as_array()
            .ok_or_else(|| OpError::Kernel("vae config: missing block_out_channels".into()))?
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

pub struct ResnetBlock<T: Dtype, D: OpBackend> {
    pub norm1_w: Tensor<T, D>,
    pub norm1_b: Tensor<T, D>,
    pub conv1_w: Tensor<T, D>,
    pub conv1_b: Tensor<T, D>,
    pub norm2_w: Tensor<T, D>,
    pub norm2_b: Tensor<T, D>,
    pub conv2_w: Tensor<T, D>,
    pub conv2_b: Tensor<T, D>,
    pub shortcut_w: Option<Tensor<T, D>>,
    pub shortcut_b: Option<Tensor<T, D>>,
    pub in_ch: usize,
    pub out_ch: usize,
}

impl<T: Dtype> ResnetBlock<T, Cuda> {
    fn forward(&self, x: &Tensor<T, Cuda>, dev: &Cuda) -> OpResult<Tensor<T, Cuda>> {
        let s = x.shape().as_slice();
        let (b, _c, h, w) = (s[0], s[1], s[2], s[3]);
        // norm1+silu
        let mut h1: Tensor<T, Cuda> = Tensor::zeros([b, self.in_ch, h, w], dev)?;
        Cuda::groupnorm_silu(x, &self.norm1_w, &self.norm1_b, &mut h1, NORM_GROUPS, EPS)?;
        // conv1
        let mut h2: Tensor<T, Cuda> = Tensor::zeros([b, self.out_ch, h, w], dev)?;
        Cuda::conv2d(&h1, &self.conv1_w, Some(&self.conv1_b), &mut h2, 1, 1)?;
        // norm2+silu
        let mut h3: Tensor<T, Cuda> = Tensor::zeros([b, self.out_ch, h, w], dev)?;
        Cuda::groupnorm_silu(&h2, &self.norm2_w, &self.norm2_b, &mut h3, NORM_GROUPS, EPS)?;
        // conv2
        let mut h4: Tensor<T, Cuda> = Tensor::zeros([b, self.out_ch, h, w], dev)?;
        Cuda::conv2d(&h3, &self.conv2_w, Some(&self.conv2_b), &mut h4, 1, 1)?;
        // residual
        if let (Some(sw), Some(sb)) = (&self.shortcut_w, &self.shortcut_b) {
            let mut sc: Tensor<T, Cuda> = Tensor::zeros([b, self.out_ch, h, w], dev)?;
            Cuda::conv2d(x, sw, Some(sb), &mut sc, 1, 0)?;
            Cuda::add_inplace(&mut h4, &sc)?;
        } else {
            Cuda::add_inplace(&mut h4, x)?;
        }
        Ok(h4)
    }
}

// ───────────────────────── AttnBlock (mid-block self-attention) ──

pub struct VaeAttnBlock<T: Dtype, D: OpBackend> {
    pub group_norm_w: Tensor<T, D>,
    pub group_norm_b: Tensor<T, D>,
    pub to_q: Linear<T, D>,
    pub to_k: Linear<T, D>,
    pub to_v: Linear<T, D>,
    pub to_out: Linear<T, D>,
    pub channels: usize,
}

impl<T: Dtype> VaeAttnBlock<T, Cuda> {
    fn forward(&self, x: &Tensor<T, Cuda>, dev: &Cuda) -> OpResult<Tensor<T, Cuda>> {
        let s = x.shape().as_slice();
        let (b, c, h, w) = (s[0], s[1], s[2], s[3]);
        let n = h * w;
        // GroupNorm.
        let mut normed: Tensor<T, Cuda> = Tensor::zeros([b, c, h, w], dev)?;
        Cuda::groupnorm(
            x,
            &self.group_norm_w,
            &self.group_norm_b,
            &mut normed,
            NORM_GROUPS,
            EPS,
        )?;

        // [B, C, H, W] → [B*N, C]: permute(0,2,3,1) then flatten last two.
        // We do this on-host via D2H → permute-by-index → H2D, since the new
        // tensor layer has no general permute kernel. The volume here is at
        // most ~16k tokens × 512 ch (mid-block), small enough.
        let bnc = permute_bchw_to_bnc::<T>(&normed, b, c, h, w, dev)?;

        // Q, K, V: [B*N, C] @ [C, C]^T (Linear) → [B*N, C].
        let mut q: Tensor<T, Cuda> = Tensor::zeros([b * n, c], dev)?;
        let mut k: Tensor<T, Cuda> = Tensor::zeros([b * n, c], dev)?;
        let mut v: Tensor<T, Cuda> = Tensor::zeros([b * n, c], dev)?;
        self.to_q.forward(&bnc, &mut q)?;
        self.to_k.forward(&bnc, &mut k)?;
        self.to_v.forward(&bnc, &mut v)?;

        // SDPA single-head: reshape to [N, 1, C] (B is folded into seq).
        // Our SDPA expects [seq, n_heads, head_dim] (SHD).
        let q3 = q.view_raw(
            Shape::from_slice(&[b * n, 1, c]),
            Shape::from_slice(&[c, c, 1]).contiguous_strides(),
            q.offset_elems(),
            true,
        );
        let k3 = k.view_raw(
            Shape::from_slice(&[b * n, 1, c]),
            Shape::from_slice(&[c, c, 1]).contiguous_strides(),
            k.offset_elems(),
            true,
        );
        let v3 = v.view_raw(
            Shape::from_slice(&[b * n, 1, c]),
            Shape::from_slice(&[c, c, 1]).contiguous_strides(),
            v.offset_elems(),
            true,
        );
        let mut attn: Tensor<T, Cuda> = Tensor::zeros([b * n, 1, c], dev)?;
        let scale = 1.0 / (c as f32).sqrt();
        Cuda::sdpa(&q3, &k3, &v3, &mut attn, 1, 1, c, scale)?;
        let attn_2d = attn.view_raw(
            Shape::from_slice(&[b * n, c]),
            Shape::from_slice(&[c, 1]).contiguous_strides(),
            attn.offset_elems(),
            true,
        );

        // to_out projection.
        let mut proj: Tensor<T, Cuda> = Tensor::zeros([b * n, c], dev)?;
        self.to_out.forward(&attn_2d, &mut proj)?;

        // Reshape [B*N, C] → [B, C, H, W] (inverse permute).
        let bchw = permute_bnc_to_bchw::<T>(&proj, b, c, h, w, dev)?;

        // Residual.
        let mut out: Tensor<T, Cuda> = Tensor::zeros([b, c, h, w], dev)?;
        out.copy_from(x)?;
        Cuda::add_inplace(&mut out, &bchw)?;
        Ok(out)
    }
}

/// Reshape `[B, C, H, W]` → `[B*N, C]` (`N = H*W`) where the inner-most
/// axis is `C`. Goes through host memory (D2H → permute → H2D).
fn permute_bchw_to_bnc<T: Dtype>(
    x: &Tensor<T, Cuda>,
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    dev: &Cuda,
) -> OpResult<Tensor<T, Cuda>> {
    let n = h * w;
    let host = x.to_host_vec()?;
    let mut out: Vec<T> = Vec::with_capacity(b * n * c);
    unsafe {
        out.set_len(b * n * c);
    }
    // src indexed as src[bi, ci, hi, wi] = src[(((bi*c+ci)*h+hi)*w+wi)]
    // dst indexed as dst[bi*n + (hi*w + wi), ci]
    for bi in 0..b {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    let src_idx = ((bi * c + ci) * h + hi) * w + wi;
                    let dst_idx = (bi * n + hi * w + wi) * c + ci;
                    out[dst_idx] = host[src_idx];
                }
            }
        }
    }
    Tensor::from_host_slice(&out, [b * n, c], dev)
}

/// Inverse: `[B*N, C]` → `[B, C, H, W]`.
fn permute_bnc_to_bchw<T: Dtype>(
    x: &Tensor<T, Cuda>,
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    dev: &Cuda,
) -> OpResult<Tensor<T, Cuda>> {
    let n = h * w;
    let host = x.to_host_vec()?;
    let mut out: Vec<T> = Vec::with_capacity(b * c * h * w);
    unsafe {
        out.set_len(b * c * h * w);
    }
    for bi in 0..b {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src_idx = (bi * n + hi * w + wi) * c + ci;
                    let dst_idx = ((bi * c + ci) * h + hi) * w + wi;
                    out[dst_idx] = host[src_idx];
                }
            }
        }
    }
    Tensor::from_host_slice(&out, [b, c, h, w], dev)
}

// ───────────────────────── UpBlock ─────────────────────────

pub struct UpBlock<T: Dtype, D: OpBackend> {
    pub resnets: Vec<ResnetBlock<T, D>>,
    pub upsampler: Option<(Tensor<T, D>, Tensor<T, D>)>, // (3x3 conv weight, bias)
    pub out_ch: usize,
}

impl<T: Dtype> UpBlock<T, Cuda> {
    fn forward(&self, mut x: Tensor<T, Cuda>, dev: &Cuda) -> OpResult<Tensor<T, Cuda>> {
        for r in &self.resnets {
            x = r.forward(&x, dev)?;
        }
        if let Some((w, b)) = &self.upsampler {
            let s = x.shape().as_slice();
            let (bn, c, hi, wi) = (s[0], s[1], s[2], s[3]);
            let mut up: Tensor<T, Cuda> = Tensor::zeros([bn, c, 2 * hi, 2 * wi], dev)?;
            Cuda::upsample_nearest_2x(&x, &mut up)?;
            let mut conv_out: Tensor<T, Cuda> =
                Tensor::zeros([bn, self.out_ch, 2 * hi, 2 * wi], dev)?;
            Cuda::conv2d(&up, w, Some(b), &mut conv_out, 1, 1)?;
            x = conv_out;
        }
        Ok(x)
    }
}

// ───────────────────────── VaeDecoder ─────────────────────────

pub struct VaeDecoder<T: Dtype, D: OpBackend> {
    pub config: VaeConfig,
    pub conv_in_w: Tensor<T, D>,
    pub conv_in_b: Tensor<T, D>,
    pub mid_resnet_0: ResnetBlock<T, D>,
    pub mid_attn: VaeAttnBlock<T, D>,
    pub mid_resnet_1: ResnetBlock<T, D>,
    pub up_blocks: Vec<UpBlock<T, D>>,
    pub conv_norm_out_w: Tensor<T, D>,
    pub conv_norm_out_b: Tensor<T, D>,
    pub conv_out_w: Tensor<T, D>,
    pub conv_out_b: Tensor<T, D>,
}

impl<T: Dtype> VaeDecoder<T, Cuda> {
    /// Load VAE decoder from a diffusers `vae/` directory.
    /// Expected: `config.json`, single `diffusion_pytorch_model.safetensors`.
    pub fn from_pretrained<P: AsRef<std::path::Path>>(vae_dir: P, device: &Cuda) -> OpResult<Self> {
        let dir = vae_dir.as_ref();
        let config = VaeConfig::from_json(dir.join("config.json"))?;
        let reader =
            SafetensorsReader::open(dir).map_err(|e| OpError::Kernel(format!("vae: {}", e)))?;
        let loader = WeightLoader::new(&reader);

        let boc = &config.block_out_channels;
        let mid_ch = boc[boc.len() - 1];

        // conv_in
        let conv_in_w = loader.load_tensor::<T, Cuda>("decoder.conv_in.weight", device)?;
        let conv_in_b = loader.load_tensor::<T, Cuda>("decoder.conv_in.bias", device)?;

        // mid block: resnet_0 → attn → resnet_1
        let mid_resnet_0 = load_resnet::<T>(
            &loader,
            "decoder.mid_block.resnets.0",
            mid_ch,
            mid_ch,
            device,
        )?;
        let mid_attn = load_attn::<T>(&loader, "decoder.mid_block.attentions.0", mid_ch, device)?;
        let mid_resnet_1 = load_resnet::<T>(
            &loader,
            "decoder.mid_block.resnets.1",
            mid_ch,
            mid_ch,
            device,
        )?;

        // up_blocks: reversed channel order [512, 512, 256, 128].
        let n_blocks = boc.len();
        let mut up_blocks = Vec::with_capacity(n_blocks);
        let mut prev_out = mid_ch;
        for i in 0..n_blocks {
            let out_ch_block = boc[n_blocks - 1 - i];
            let num_resnets = config.layers_per_block + 1; // 3
            let mut resnets = Vec::with_capacity(num_resnets);
            for r in 0..num_resnets {
                let in_c = if r == 0 { prev_out } else { out_ch_block };
                let prefix = format!("decoder.up_blocks.{}.resnets.{}", i, r);
                resnets.push(load_resnet::<T>(
                    &loader,
                    &prefix,
                    in_c,
                    out_ch_block,
                    device,
                )?);
            }
            // Upsampler: present in up_blocks[0..n_blocks-1].
            let upsampler = if i < n_blocks - 1 {
                let w = loader.load_tensor::<T, Cuda>(
                    &format!("decoder.up_blocks.{}.upsamplers.0.conv.weight", i),
                    device,
                )?;
                let b = loader.load_tensor::<T, Cuda>(
                    &format!("decoder.up_blocks.{}.upsamplers.0.conv.bias", i),
                    device,
                )?;
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

        // Final norm + conv.
        let conv_norm_out_w =
            loader.load_tensor::<T, Cuda>("decoder.conv_norm_out.weight", device)?;
        let conv_norm_out_b =
            loader.load_tensor::<T, Cuda>("decoder.conv_norm_out.bias", device)?;
        let conv_out_w = loader.load_tensor::<T, Cuda>("decoder.conv_out.weight", device)?;
        let conv_out_b = loader.load_tensor::<T, Cuda>("decoder.conv_out.bias", device)?;

        Ok(Self {
            config,
            conv_in_w,
            conv_in_b,
            mid_resnet_0,
            mid_attn,
            mid_resnet_1,
            up_blocks,
            conv_norm_out_w,
            conv_norm_out_b,
            conv_out_w,
            conv_out_b,
        })
    }

    /// Decode `latents [B, 16, h, w]` → `image [B, 3, h*8, w*8]`.
    pub fn decode(&self, latents: &Tensor<T, Cuda>, dev: &Cuda) -> OpResult<Tensor<T, Cuda>> {
        let s = latents.shape().as_slice();
        let (b, _c, h, w) = (s[0], s[1], s[2], s[3]);
        let mid_ch = self.mid_resnet_0.out_ch;
        // conv_in.
        let mut x: Tensor<T, Cuda> = Tensor::zeros([b, mid_ch, h, w], dev)?;
        Cuda::conv2d(
            latents,
            &self.conv_in_w,
            Some(&self.conv_in_b),
            &mut x,
            1,
            1,
        )?;
        // mid: resnet → attn → resnet.
        x = self.mid_resnet_0.forward(&x, dev)?;
        x = self.mid_attn.forward(&x, dev)?;
        x = self.mid_resnet_1.forward(&x, dev)?;
        // up blocks.
        for up in &self.up_blocks {
            x = up.forward(x, dev)?;
        }
        // norm + silu + conv_out.
        let s2 = x.shape().as_slice().to_vec();
        let (bn, ch_out, hh, ww) = (s2[0], s2[1], s2[2], s2[3]);
        let mut h2: Tensor<T, Cuda> = Tensor::zeros([bn, ch_out, hh, ww], dev)?;
        Cuda::groupnorm_silu(
            &x,
            &self.conv_norm_out_w,
            &self.conv_norm_out_b,
            &mut h2,
            NORM_GROUPS,
            EPS,
        )?;
        let mut out: Tensor<T, Cuda> = Tensor::zeros([bn, self.config.out_channels, hh, ww], dev)?;
        Cuda::conv2d(
            &h2,
            &self.conv_out_w,
            Some(&self.conv_out_b),
            &mut out,
            1,
            1,
        )?;
        Ok(out)
    }
}

/// Load a ResnetBlock from `{prefix}.norm{1,2}.{weight,bias}`,
/// `{prefix}.conv{1,2}.{weight,bias}`, optional `{prefix}.conv_shortcut`.
fn load_resnet<T: Dtype>(
    loader: &WeightLoader,
    prefix: &str,
    in_ch: usize,
    out_ch: usize,
    device: &Cuda,
) -> OpResult<ResnetBlock<T, Cuda>> {
    let norm1_w = loader.load_tensor::<T, Cuda>(&format!("{}.norm1.weight", prefix), device)?;
    let norm1_b = loader.load_tensor::<T, Cuda>(&format!("{}.norm1.bias", prefix), device)?;
    let conv1_w = loader.load_tensor::<T, Cuda>(&format!("{}.conv1.weight", prefix), device)?;
    let conv1_b = loader.load_tensor::<T, Cuda>(&format!("{}.conv1.bias", prefix), device)?;
    let norm2_w = loader.load_tensor::<T, Cuda>(&format!("{}.norm2.weight", prefix), device)?;
    let norm2_b = loader.load_tensor::<T, Cuda>(&format!("{}.norm2.bias", prefix), device)?;
    let conv2_w = loader.load_tensor::<T, Cuda>(&format!("{}.conv2.weight", prefix), device)?;
    let conv2_b = loader.load_tensor::<T, Cuda>(&format!("{}.conv2.bias", prefix), device)?;
    let (shortcut_w, shortcut_b) = if in_ch != out_ch {
        let w_name = format!("{}.conv_shortcut.weight", prefix);
        if loader.has_tensor(&w_name) {
            let sw = loader.load_tensor::<T, Cuda>(&w_name, device)?;
            let sb =
                loader.load_tensor::<T, Cuda>(&format!("{}.conv_shortcut.bias", prefix), device)?;
            (Some(sw), Some(sb))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };
    Ok(ResnetBlock {
        norm1_w,
        norm1_b,
        conv1_w,
        conv1_b,
        norm2_w,
        norm2_b,
        conv2_w,
        conv2_b,
        shortcut_w,
        shortcut_b,
        in_ch,
        out_ch,
    })
}

fn load_attn<T: Dtype>(
    loader: &WeightLoader,
    prefix: &str,
    ch: usize,
    device: &Cuda,
) -> OpResult<VaeAttnBlock<T, Cuda>> {
    let group_norm_w =
        loader.load_tensor::<T, Cuda>(&format!("{}.group_norm.weight", prefix), device)?;
    let group_norm_b =
        loader.load_tensor::<T, Cuda>(&format!("{}.group_norm.bias", prefix), device)?;
    let to_q = loader.load_linear::<T, Cuda>(
        &format!("{}.to_q.weight", prefix),
        Some(&format!("{}.to_q.bias", prefix)),
        device,
    )?;
    let to_k = loader.load_linear::<T, Cuda>(
        &format!("{}.to_k.weight", prefix),
        Some(&format!("{}.to_k.bias", prefix)),
        device,
    )?;
    let to_v = loader.load_linear::<T, Cuda>(
        &format!("{}.to_v.weight", prefix),
        Some(&format!("{}.to_v.bias", prefix)),
        device,
    )?;
    let to_out = loader.load_linear::<T, Cuda>(
        &format!("{}.to_out.0.weight", prefix),
        Some(&format!("{}.to_out.0.bias", prefix)),
        device,
    )?;
    Ok(VaeAttnBlock {
        group_norm_w,
        group_norm_b,
        to_q,
        to_k,
        to_v,
        to_out,
        channels: ch,
    })
}

/// Load a ResnetBlock from `{prefix}.norm{1,2}.{weight,bias}`,
/// `{prefix}.conv{1,2}.{weight,bias}`, optional `{prefix}.conv_shortcut`.

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_norm_f32(c: usize, cuda: &Cuda) -> (Tensor<f32, Cuda>, Tensor<f32, Cuda>) {
        let w: Tensor<f32, Cuda> = Tensor::from_host_slice(&vec![1.0_f32; c], [c], cuda).unwrap();
        let b: Tensor<f32, Cuda> = Tensor::from_host_slice(&vec![0.0_f32; c], [c], cuda).unwrap();
        (w, b)
    }

    fn rand_conv_f32(
        out: usize,
        in_: usize,
        k: usize,
        cuda: &Cuda,
        seed: u64,
    ) -> (Tensor<f32, Cuda>, Tensor<f32, Cuda>) {
        let w: Tensor<f32, Cuda> = Tensor::randn([out, in_, k, k], cuda, Some(seed)).unwrap();
        let b: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&vec![0.0_f32; out], [out], cuda).unwrap();
        (w, b)
    }

    fn rand_linear_f32(out: usize, in_: usize, cuda: &Cuda, seed: u64) -> Linear<f32, Cuda> {
        let w: Tensor<f32, Cuda> = Tensor::randn([out, in_], cuda, Some(seed)).unwrap();
        let b: Tensor<f32, Cuda> =
            Tensor::from_host_slice(&vec![0.0_f32; out], [out], cuda).unwrap();
        Linear::new(w, Some(b))
    }

    #[test]
    fn permute_bchw_bnc_roundtrip() {
        let cuda = Cuda::new(0).unwrap();
        let (b, c, h, w) = (1, 4, 3, 5);
        let n = h * w;
        let host: Vec<f32> = (0..b * c * h * w).map(|i| i as f32).collect();
        let x: Tensor<f32, Cuda> = Tensor::from_host_slice(&host, [b, c, h, w], &cuda).unwrap();
        let bnc = permute_bchw_to_bnc::<f32>(&x, b, c, h, w, &cuda).unwrap();
        assert_eq!(bnc.shape().as_slice(), &[b * n, c]);
        let back = permute_bnc_to_bchw::<f32>(&bnc, b, c, h, w, &cuda).unwrap();
        assert_eq!(back.shape().as_slice(), &[b, c, h, w]);
        let back_v = back.to_host_vec().unwrap();
        for (a, b_) in host.iter().zip(back_v.iter()) {
            assert_eq!(a, b_);
        }
    }

    #[test]
    fn resnet_block_shape_smoke() {
        let cuda = Cuda::new(0).unwrap();
        let in_ch = 32;
        let out_ch = 32;
        let (b, h, w) = (1, 8, 8);
        let block: ResnetBlock<f32, Cuda> = {
            let (n1w, n1b) = unit_norm_f32(in_ch, &cuda);
            let (n2w, n2b) = unit_norm_f32(out_ch, &cuda);
            let (c1w, c1b) = rand_conv_f32(out_ch, in_ch, 3, &cuda, 1);
            let (c2w, c2b) = rand_conv_f32(out_ch, out_ch, 3, &cuda, 2);
            ResnetBlock {
                norm1_w: n1w,
                norm1_b: n1b,
                conv1_w: c1w,
                conv1_b: c1b,
                norm2_w: n2w,
                norm2_b: n2b,
                conv2_w: c2w,
                conv2_b: c2b,
                shortcut_w: None,
                shortcut_b: None,
                in_ch,
                out_ch,
            }
        };
        let x: Tensor<f32, Cuda> = Tensor::randn([b, in_ch, h, w], &cuda, Some(7)).unwrap();
        let out = block.forward(&x, &cuda).unwrap();
        assert_eq!(out.shape().as_slice(), &[b, out_ch, h, w]);
    }

    #[test]
    fn upblock_2x_doubles_spatial() {
        let cuda = Cuda::new(0).unwrap();
        let in_ch = 32;
        let (b, h, w) = (1, 4, 4);
        // One resnet (in=in_ch, out=in_ch) + 3x3 upsample conv.
        let resnet = {
            let (n1w, n1b) = unit_norm_f32(in_ch, &cuda);
            let (n2w, n2b) = unit_norm_f32(in_ch, &cuda);
            let (c1w, c1b) = rand_conv_f32(in_ch, in_ch, 3, &cuda, 10);
            let (c2w, c2b) = rand_conv_f32(in_ch, in_ch, 3, &cuda, 11);
            ResnetBlock {
                norm1_w: n1w,
                norm1_b: n1b,
                conv1_w: c1w,
                conv1_b: c1b,
                norm2_w: n2w,
                norm2_b: n2b,
                conv2_w: c2w,
                conv2_b: c2b,
                shortcut_w: None,
                shortcut_b: None,
                in_ch,
                out_ch: in_ch,
            }
        };
        let (uw, ub) = rand_conv_f32(in_ch, in_ch, 3, &cuda, 20);
        let up = UpBlock {
            resnets: vec![resnet],
            upsampler: Some((uw, ub)),
            out_ch: in_ch,
        };
        let x: Tensor<f32, Cuda> = Tensor::randn([b, in_ch, h, w], &cuda, Some(15)).unwrap();
        let out = up.forward(x, &cuda).unwrap();
        assert_eq!(out.shape().as_slice(), &[b, in_ch, h * 2, w * 2]);
    }

    #[test]
    fn vae_attn_block_shape_smoke() {
        let cuda = Cuda::new(0).unwrap();
        let c = 32; // divisible by NORM_GROUPS
        let (b, h, w) = (1, 4, 4);
        let (gw, gb) = unit_norm_f32(c, &cuda);
        let attn: VaeAttnBlock<f32, Cuda> = VaeAttnBlock {
            group_norm_w: gw,
            group_norm_b: gb,
            to_q: rand_linear_f32(c, c, &cuda, 30),
            to_k: rand_linear_f32(c, c, &cuda, 31),
            to_v: rand_linear_f32(c, c, &cuda, 32),
            to_out: rand_linear_f32(c, c, &cuda, 33),
            channels: c,
        };
        let x: Tensor<f32, Cuda> = Tensor::randn([b, c, h, w], &cuda, Some(99)).unwrap();
        let out = attn.forward(&x, &cuda).unwrap();
        assert_eq!(out.shape().as_slice(), &[b, c, h, w]);
    }
}
