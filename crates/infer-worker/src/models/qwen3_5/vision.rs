//! Qwen3.5 ViT: non-causal attention, 2D rotary positions and patch merger.
use super::*;
use crate::domain::exec::ExecScope;
use infer_protocol::multimodal::{ImageInput, PATCH_WIDTH};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub in_channels: usize,
    pub hidden_act: String,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
}

struct PositionEmbeddings<T: Dtype, D: LlmBackend> {
    positions: Tensor<T, D>,
    sin: Tensor<f32, D>,
    cos: Tensor<f32, D>,
}

struct Projection<T: Dtype, D: LlmBackend> {
    weight: Tensor<T, D>,
    bias: Tensor<T, D>,
}

impl<T: Dtype, D: LlmBackend> Projection<T, D> {
    fn forward(&self, x: &Tensor<T, D>, scope: &D::Scope) -> OpResult<Tensor<T, D>> {
        let mut out = Tensor::zeros([x.shape()[0], self.weight.shape()[0]], x.device())?;
        D::linear(scope, x, &self.weight, &self.bias, &mut out)?;
        Ok(out)
    }
}

struct LayerNorm<T: Dtype, D: LlmBackend> {
    weight: Tensor<T, D>,
    bias: Tensor<T, D>,
}
impl<T: Dtype, D: LlmBackend> LayerNorm<T, D> {
    fn forward(&self, x: &Tensor<T, D>, scope: &D::Scope) -> OpResult<Tensor<T, D>> {
        let mut out = Tensor::zeros(*x.shape(), x.device())?;
        D::layer_norm(scope, x, &self.weight, &self.bias, &mut out, 1e-6)?;
        Ok(out)
    }
}

struct VisionBlock<T: Dtype, D: LlmBackend> {
    norm1: LayerNorm<T, D>,
    norm2: LayerNorm<T, D>,
    qkv: Projection<T, D>,
    proj: Projection<T, D>,
    fc1: Projection<T, D>,
    fc2: Projection<T, D>,
}

pub struct VisionEncoder<T: Dtype, D: LlmBackend> {
    pub config: VisionConfig,
    patch: Projection<T, D>,
    positions: Vec<T>,
    blocks: Vec<VisionBlock<T, D>>,
    norm: LayerNorm<T, D>,
    fc1: Projection<T, D>,
    fc2: Projection<T, D>,
}

impl<T: Dtype, D: OpBackend + LlmBackend> VisionEncoder<T, D> {
    pub fn load(loader: &WeightLoader<'_>, config: VisionConfig, device: &D) -> OpResult<Self> {
        let c = &config;
        let side = (c.num_position_embeddings as f64).sqrt() as usize;
        if c.patch_size != 16
            || c.temporal_patch_size != 2
            || c.spatial_merge_size != 2
            || c.in_channels != 3
            || c.hidden_act != "gelu_pytorch_tanh"
            || !c.deepstack_visual_indexes.is_empty()
            || c.depth == 0
            || c.num_heads == 0
            || !c.hidden_size.is_multiple_of(c.num_heads)
            || !(c.hidden_size / c.num_heads).is_multiple_of(4)
            || side * side != c.num_position_embeddings
            || side < 2
        {
            return Err(OpError::Shape(
                "unsupported Qwen3.5 vision configuration".into(),
            ));
        }
        let root = "model.visual";
        let projection = |name: &str, rows: usize, cols: usize| -> OpResult<Projection<T, D>> {
            Ok(Projection {
                weight: load_shaped(loader, &format!("{name}.weight"), &[rows, cols], device)?,
                bias: load_shaped(loader, &format!("{name}.bias"), &[rows], device)?,
            })
        };
        let norm = |name: &str| -> OpResult<LayerNorm<T, D>> {
            Ok(LayerNorm {
                weight: load_shaped(loader, &format!("{name}.weight"), &[c.hidden_size], device)?,
                bias: load_shaped(loader, &format!("{name}.bias"), &[c.hidden_size], device)?,
            })
        };
        let patch_weight = load_shaped(
            loader,
            &format!("{root}.patch_embed.proj.weight"),
            &[c.hidden_size, 3, 2, 16, 16],
            device,
        )?;
        let patch = Projection {
            weight: patch_weight.view_contiguous(crate::domain::types::Shape::from_slice(&[
                c.hidden_size,
                PATCH_WIDTH,
            ]))?,
            bias: load_shaped(
                loader,
                &format!("{root}.patch_embed.proj.bias"),
                &[c.hidden_size],
                device,
            )?,
        };
        let positions = load_shaped::<T, D>(
            loader,
            &format!("{root}.pos_embed.weight"),
            &[c.num_position_embeddings, c.hidden_size],
            device,
        )?
        .to_host_vec()?;
        let mut blocks = Vec::with_capacity(c.depth);
        for i in 0..c.depth {
            let p = format!("{root}.blocks.{i}");
            blocks.push(VisionBlock {
                norm1: norm(&format!("{p}.norm1"))?,
                norm2: norm(&format!("{p}.norm2"))?,
                qkv: projection(&format!("{p}.attn.qkv"), 3 * c.hidden_size, c.hidden_size)?,
                proj: projection(&format!("{p}.attn.proj"), c.hidden_size, c.hidden_size)?,
                fc1: projection(
                    &format!("{p}.mlp.linear_fc1"),
                    c.intermediate_size,
                    c.hidden_size,
                )?,
                fc2: projection(
                    &format!("{p}.mlp.linear_fc2"),
                    c.hidden_size,
                    c.intermediate_size,
                )?,
            });
        }
        let merger_norm = norm(&format!("{root}.merger.norm"))?;
        let fc1 = projection(
            &format!("{root}.merger.linear_fc1"),
            4 * c.hidden_size,
            4 * c.hidden_size,
        )?;
        let fc2 = projection(
            &format!("{root}.merger.linear_fc2"),
            c.out_hidden_size,
            4 * c.hidden_size,
        )?;
        Ok(Self {
            config,
            patch,
            positions,
            blocks,
            norm: merger_norm,
            fc1,
            fc2,
        })
    }
}

impl<T: Dtype, D: LlmBackend> VisionEncoder<T, D> {
    pub fn forward(&self, image: &ImageInput, scope: &D::Scope) -> OpResult<Tensor<T, D>> {
        self.forward_with_trace(image, scope, |_, _| Ok(()))
    }

    pub fn forward_with_trace(
        &self,
        image: &ImageInput,
        scope: &D::Scope,
        mut trace: impl FnMut(&str, &Tensor<T, D>) -> OpResult<()>,
    ) -> OpResult<Tensor<T, D>> {
        image.validate().map_err(OpError::Shape)?;
        let _guard = scope.enter();
        let c = &self.config;
        let [_, h, w] = image.grid_thw;
        let n = h as usize * w as usize;
        let values: Vec<T> = image
            .patches
            .chunks_exact(2)
            .map(|b| {
                T::write_f64(
                    half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32() as f64,
                )
            })
            .collect();
        let input = Tensor::from_host_slice(&values, [n, PATCH_WIDTH], scope.device())?;
        // The checkpoint's Conv3d patch projection rounds its convolution
        // output before adding bias; ViT Linear layers use fused affine sums.
        let mut x = Tensor::zeros([n, c.hidden_size], scope.device())?;
        D::matmul(scope, &input, &self.patch.weight, &mut x)?;
        D::broadcast_add_inplace(scope, &mut x, &self.patch.bias)?;
        trace("patch", &x)?;
        let PositionEmbeddings {
            positions,
            sin,
            cos,
        } = self.position_embeddings(h as usize, w as usize, scope)?;
        D::add_inplace(scope, &mut x, &positions)?;
        trace("position", &x)?;
        let hd = c.hidden_size / c.num_heads;
        for (i, block) in self.blocks.iter().enumerate() {
            let norm = block.norm1.forward(&x, scope)?;
            let qkv = block.qkv.forward(&norm, scope)?;
            let mut split = Vec::with_capacity(3);
            for j in 0..3 {
                let mut out = Tensor::zeros([n, c.hidden_size], scope.device())?;
                D::split_cols(
                    scope,
                    &qkv,
                    &mut out,
                    n,
                    3 * c.hidden_size,
                    j * c.hidden_size,
                    c.hidden_size,
                )?;
                if j < 2 {
                    D::rope_with_angles(scope, &mut out, &sin, &cos, hd)?;
                }
                split.push(
                    out.view_contiguous(crate::domain::types::Shape::from_slice(&[
                        n,
                        c.num_heads,
                        hd,
                    ]))?,
                );
            }
            let mut attn = Tensor::zeros([n, c.num_heads, hd], scope.device())?;
            D::sdpa(
                scope,
                &split[0],
                &split[1],
                &split[2],
                &mut attn,
                None,
                c.num_heads,
                c.num_heads,
                hd,
                (hd as f32).sqrt().recip(),
            )?;
            let attn =
                attn.view_contiguous(crate::domain::types::Shape::from_slice(&[n, c.hidden_size]))?;
            let out = block.proj.forward(&attn, scope)?;
            D::add_inplace(scope, &mut x, &out)?;
            let norm = block.norm2.forward(&x, scope)?;
            let mut inter = block.fc1.forward(&norm, scope)?;
            D::gelu_inplace(scope, &mut inter, true)?;
            let out = block.fc2.forward(&inter, scope)?;
            D::add_inplace(scope, &mut x, &out)?;
            trace(&format!("block{i}"), &x)?;
        }
        let norm = self.norm.forward(&x, scope)?;
        let merged = norm.view_contiguous(crate::domain::types::Shape::from_slice(&[
            n / 4,
            4 * c.hidden_size,
        ]))?;
        let mut inter = self.fc1.forward(&merged, scope)?;
        D::gelu_inplace(scope, &mut inter, false)?;
        let out = self.fc2.forward(&inter, scope)?;
        trace("merger", &out)?;
        Ok(out)
    }

    fn position_embeddings(
        &self,
        h: usize,
        w: usize,
        scope: &D::Scope,
    ) -> OpResult<PositionEmbeddings<T, D>> {
        let c = &self.config;
        let side = (c.num_position_embeddings as f64).sqrt() as usize;
        let quarter = c.hidden_size / c.num_heads / 4;
        let mut positions = Vec::with_capacity(h * w * c.hidden_size);
        let mut sin = Vec::with_capacity(h * w * quarter * 2);
        let mut cos = Vec::with_capacity(h * w * quarter * 2);
        // Processor and learned positions both group neighboring 2x2 patches.
        for by in 0..h / 2 {
            for bx in 0..w / 2 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let y = by * 2 + dy;
                        let x = bx * 2 + dx;
                        let fy = y as f32 * (side - 1) as f32 / (h - 1) as f32;
                        let fx = x as f32 * (side - 1) as f32 / (w - 1) as f32;
                        let y0 = fy.floor() as usize;
                        let x0 = fx.floor() as usize;
                        let y1 = (y0 + 1).min(side - 1);
                        let x1 = (x0 + 1).min(side - 1);
                        let wy = fy - y0 as f32;
                        let wx = fx - x0 as f32;
                        let corners = [
                            (y0 * side + x0, (1.0 - wy) * (1.0 - wx)),
                            (y0 * side + x1, (1.0 - wy) * wx),
                            (y1 * side + x0, wy * (1.0 - wx)),
                            (y1 * side + x1, wy * wx),
                        ];
                        for k in 0..c.hidden_size {
                            let v = corners
                                .iter()
                                .map(|&(idx, weight)| {
                                    T::read_f64(&self.positions[idx * c.hidden_size + k]) as f32
                                        * weight
                                })
                                .sum::<f32>();
                            positions.push(T::write_f64(v as f64));
                        }
                        for axis in [y, x] {
                            for j in 0..quarter {
                                let inv = 1.0f32 / 10000.0f32.powf(j as f32 / quarter as f32);
                                let angle = axis as f32 * inv;
                                sin.push(angle.sin());
                                cos.push(angle.cos());
                            }
                        }
                    }
                }
            }
        }
        Ok(PositionEmbeddings {
            positions: Tensor::from_host_slice(&positions, [h * w, c.hidden_size], scope.device())?,
            sin: Tensor::from_host_slice(&sin, [h * w, quarter * 2], scope.device())?,
            cos: Tensor::from_host_slice(&cos, [h * w, quarter * 2], scope.device())?,
        })
    }
}
