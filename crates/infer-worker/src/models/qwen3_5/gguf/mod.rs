//! Single-file qwen35 GGUF main-decoder loader and offline text inference.
//! Validate the entire manifest before allocating device weights. Quantized
//! matrices stay encoded, including separately quantized Q/K/V and gate/up.
mod config;
pub mod probe;
pub mod text;
mod weights;

use super::Qwen3_5Model;
use crate::components::{
    Attention, DecoderBlock, DenseFfn, Embed, FullAttention, GatedDeltaNet, GdnWeights, Linear,
    LmHead, RmsNorm,
};
use crate::domain::{cache::LinearDims, model::ModelDims};
use crate::infrastructure::io::gguf::{GgmlType, GgufReader};
use crate::models::decoder::Decoder;
use infer_core::dtype::DTypeId;
use infer_core::{
    dtype::Dtype,
    ports::{OpError, OpResult, backend::LlmBackend},
    tensor::Tensor,
};
use std::collections::{BTreeMap, HashSet};
use weights::{Transform, Weights};

#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    /// Runtime RoPE capacity, bounded by checkpoint context; no KV allocation.
    pub context_length: usize,
}
impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            context_length: 4096,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub dims: ModelDims,
    pub linear: LinearDims,
    pub layer_is_full: Vec<bool>,
    pub context_length: usize,
    pub trained_context_length: usize,
    pub rotary_dim: usize,
    pub rope_theta: f64,
    pub mrope_sections: [usize; 3],
    pub rms_norm_eps: f32,
    pub mtp_layers: usize,
}

#[derive(Clone, Debug, Default)]
pub struct LoadReport {
    pub loaded_tensors: usize,
    /// Encoded source payload (no file headers/alignment or generated RoPE).
    pub source_weight_bytes: u64,
    pub format_counts: BTreeMap<String, usize>,
    pub skipped_mtp_tensors: Vec<String>,
    pub skipped_mtp_bytes: u64,
    pub tied_output: bool,
}

/// A validated loading plan borrowing the mapping. Loaded models own their
/// storage and remain usable after this loader and the reader are dropped.
pub struct Qwen35GgufLoader<'a> {
    reader: &'a GgufReader,
    config: ModelConfig,
    report: LoadReport,
}

impl<'a> Qwen35GgufLoader<'a> {
    pub fn new(reader: &'a GgufReader, options: LoadOptions) -> OpResult<Self> {
        let config = config::parse(reader, options)?;
        let report = validate_manifest(reader, &config)?;
        Ok(Self {
            reader,
            config,
            report,
        })
    }
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn report(&self) -> &LoadReport {
        &self.report
    }

    /// Estimated persistent device weight + RoPE bytes for T; excludes backend
    /// workspace, allocator overhead, activations, KV and recurrent state.
    pub fn device_weight_bytes<T: Dtype>(&self) -> OpResult<u64> {
        check_dtype::<T>()?;
        let mut bytes = 0u64;
        let skipped: HashSet<_> = self
            .report
            .skipped_mtp_tensors
            .iter()
            .map(String::as_str)
            .collect();
        for t in self
            .reader
            .tensors()
            .iter()
            .filter(|t| !skipped.contains(t.name()))
        {
            let n = if t.ggml_type().block_quant_format().is_some() {
                t.byte_len()
            } else {
                let size = if t.name().ends_with(".ssm_a") || t.name().ends_with(".ssm_norm.weight")
                {
                    4
                } else {
                    T::SIZE_BYTES as u64
                };
                t.dimensions()
                    .iter()
                    .product::<u64>()
                    .checked_mul(size)
                    .ok_or_else(|| OpError::Shape("GGUF device weight size overflow".into()))?
            };
            bytes = bytes
                .checked_add(n)
                .ok_or_else(|| OpError::Shape("GGUF device weight size overflow".into()))?;
        }
        let rope = (self.config.context_length as u64)
            * (self.config.rotary_dim as u64)
            * T::SIZE_BYTES as u64;
        bytes
            .checked_add(rope)
            .ok_or_else(|| OpError::Shape("GGUF device weight size overflow".into()))
    }

    pub fn load<T: Dtype, D: LlmBackend>(&self, device: &D) -> OpResult<Qwen3_5Model<T, D>> {
        check_dtype::<T>()?;
        let cfg = &self.config;
        let d = cfg.dims;
        let w = Weights {
            reader: self.reader,
        };
        let embed = w.embedding(d, device)?;
        let lm_head = if self.report.tied_output {
            embed.shared_linear()?
        } else {
            w.linear(&["output.weight".into()], device)?
        };
        let (sin, cos) = rope::<T, D>(cfg, device)?;
        let mut blocks = Vec::with_capacity(d.num_layers);
        for (i, &full) in cfg.layer_is_full.iter().enumerate() {
            let name = |s: &str| format!("blk.{i}.{s}");
            let proj = |s: &str| w.linear(&[name(s)], device);
            let norm = |s: &str, dim| w.norm(&name(s), dim, cfg.rms_norm_eps, device);
            let input_layernorm = norm("attn_norm.weight", d.dim)?;
            let attention = if full {
                Attention::Full(FullAttention {
                    input_layernorm,
                    scratch: None,
                    core: crate::components::attention_core::AttentionCore {
                        qkv_proj: w.linear(
                            &[
                                name("attn_q.weight"),
                                name("attn_k.weight"),
                                name("attn_v.weight"),
                            ],
                            device,
                        )?,
                        o_proj: proj("attn_output.weight")?,
                        q_norm: Some(norm("attn_q_norm.weight", d.head_dim)?),
                        k_norm: Some(norm("attn_k_norm.weight", d.head_dim)?),
                        sin: sin.clone(),
                        cos: cos.clone(),
                        head_num: d.head_num,
                        kv_head_num: d.kv_head_num,
                        head_dim: d.head_dim,
                        rotary_dim: cfg.rotary_dim,
                        attn_output_gate: true,
                        scale: 1.0 / (d.head_dim as f32).sqrt(),
                    },
                })
            } else {
                let l = cfg.linear;
                Attention::Linear(GatedDeltaNet::new_tiled_value_heads(
                    GdnWeights {
                        input_layernorm,
                        in_proj_qkv: proj("attn_qkv.weight")?,
                        in_proj_z: proj("attn_gate.weight")?,
                        in_proj_a: proj("ssm_alpha.weight")?,
                        in_proj_b: proj("ssm_beta.weight")?,
                        conv1d: w.dense(
                            &name("ssm_conv1d.weight"),
                            &[l.conv_dim(), 1, l.conv_kernel_dim],
                            Transform::Identity,
                            device,
                        )?,
                        a_log: w.dense(
                            &name("ssm_a"),
                            &[l.num_value_heads],
                            Transform::NegativeExp,
                            device,
                        )?,
                        dt_bias: w.dense(
                            &name("ssm_dt.bias"),
                            &[l.num_value_heads],
                            Transform::Identity,
                            device,
                        )?,
                        norm_weight: w.dense(
                            &name("ssm_norm.weight"),
                            &[l.value_head_dim],
                            Transform::Identity,
                            device,
                        )?,
                        norm_eps: cfg.rms_norm_eps,
                        out_proj: proj("ssm_out.weight")?,
                    },
                    l,
                )?)
            };
            blocks.push(DecoderBlock {
                attention,
                ffn: DenseFfn {
                    post_attention_layernorm: norm("post_attention_norm.weight", d.dim)?,
                    gate_up_proj: w
                        .linear(&[name("ffn_gate.weight"), name("ffn_up.weight")], device)?,
                    down_proj: proj("ffn_down.weight")?,
                    scratch: None,
                },
            });
        }
        Ok(Qwen3_5Model {
            eager_ragged: true,
            decoder: Decoder::new(
                embed,
                blocks,
                w.norm("output_norm.weight", d.dim, cfg.rms_norm_eps, device)?,
                LmHead { proj: lm_head },
                d,
            )?,
            vision: None,
            rotary_dim: cfg.rotary_dim,
            rope_theta: cfg.rope_theta,
            mrope_section: cfg.mrope_sections,
        })
    }
}

fn check_dtype<T: Dtype>() -> OpResult<()> {
    if !matches!(T::ID, DTypeId::F32 | DTypeId::F16 | DTypeId::BF16)
        || T::SIZE_BYTES != std::mem::size_of::<T>()
    {
        return Err(OpError::unsupported(
            "GGUF model loader",
            "activation dtype",
        ));
    }
    Ok(())
}

fn rope<T: Dtype, D: LlmBackend>(
    cfg: &ModelConfig,
    device: &D,
) -> OpResult<(Tensor<T, D>, Tensor<T, D>)> {
    let half = cfg.rotary_dim / 2;
    let make = |sin: bool| {
        let mut host = Vec::with_capacity(cfg.context_length * half);
        for p in 0..cfg.context_length {
            for i in 0..half {
                let a = p as f64 / cfg.rope_theta.powf(2.0 * i as f64 / cfg.rotary_dim as f64);
                host.push(T::write_f64(if sin { a.sin() } else { a.cos() }));
            }
        }
        weights::upload(host, &[cfg.context_length, half], device)
    };
    Ok((make(true)?, make(false)?))
}

fn dense_type(t: GgmlType) -> bool {
    matches!(t, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

fn validate_manifest(reader: &GgufReader, cfg: &ModelConfig) -> OpResult<LoadReport> {
    let d = cfg.dims;
    let mut expected = HashSet::new();
    let mut report = LoadReport {
        tied_output: !reader.contains("output.weight"),
        ..LoadReport::default()
    };
    let mut check = |name: String, shape: &[usize], matrix: bool, transform| -> OpResult<()> {
        let v = reader
            .read_view(&name)
            .map_err(|e| OpError::Shape(e.to_string()))?;
        let want: Vec<u64> = shape.iter().rev().map(|&x| x as u64).collect();
        if v.info.dimensions() != want {
            return Err(OpError::Shape(format!(
                "GGUF {name}: expected disk shape {want:?}, got {:?}",
                v.info.dimensions()
            )));
        }
        let t = v.info.ggml_type();
        if matrix && t.block_quant_format().is_some() {
            crate::models::gguf_weights::block_quant_view(v)?;
        } else if dense_type(t) {
            // Inspect the small architectural tensors before device allocation.
            if !matrix {
                weights::values(v, transform)?;
            }
        } else {
            return Err(OpError::Shape(format!(
                "GGUF {name}: unsupported model weight type {}",
                t.name()
            )));
        }
        let info = reader.tensor_info(&name).unwrap();
        report.loaded_tensors += 1;
        report.source_weight_bytes += info.byte_len();
        *report.format_counts.entry(t.name().into()).or_default() += 1;
        expected.insert(name);
        Ok(())
    };
    check(
        "token_embd.weight".into(),
        &[d.vocab_size, d.dim],
        true,
        Transform::Identity,
    )?;
    if reader.contains("output.weight") {
        check(
            "output.weight".into(),
            &[d.vocab_size, d.dim],
            true,
            Transform::Identity,
        )?;
    }
    check(
        "output_norm.weight".into(),
        &[d.dim],
        false,
        Transform::Norm,
    )?;
    for (i, &full) in cfg.layer_is_full.iter().enumerate() {
        let mut c =
            |s: &str, shape: &[usize], matrix, tr| check(format!("blk.{i}.{s}"), shape, matrix, tr);
        for s in ["attn_norm.weight", "post_attention_norm.weight"] {
            c(s, &[d.dim], false, Transform::Norm)?;
        }
        for s in ["ffn_gate.weight", "ffn_up.weight"] {
            c(s, &[d.intermediate_size, d.dim], true, Transform::Identity)?;
        }
        c(
            "ffn_down.weight",
            &[d.dim, d.intermediate_size],
            true,
            Transform::Identity,
        )?;
        if full {
            for (s, n, k) in [
                ("attn_q.weight", 2 * d.q_dim, d.dim),
                ("attn_k.weight", d.kv_dim, d.dim),
                ("attn_v.weight", d.kv_dim, d.dim),
                ("attn_output.weight", d.dim, d.q_dim),
            ] {
                c(s, &[n, k], true, Transform::Identity)?;
            }
            for s in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                c(s, &[d.head_dim], false, Transform::Norm)?;
            }
        } else {
            let l = cfg.linear;
            for (s, n, k) in [
                ("attn_qkv.weight", l.conv_dim(), d.dim),
                ("attn_gate.weight", l.value_dim(), d.dim),
                ("ssm_alpha.weight", l.num_value_heads, d.dim),
                ("ssm_beta.weight", l.num_value_heads, d.dim),
                ("ssm_out.weight", d.dim, l.value_dim()),
            ] {
                c(s, &[n, k], true, Transform::Identity)?;
            }
            c(
                "ssm_conv1d.weight",
                &[l.conv_dim(), l.conv_kernel_dim],
                false,
                Transform::Identity,
            )?;
            c("ssm_a", &[l.num_value_heads], false, Transform::NegativeExp)?;
            c(
                "ssm_dt.bias",
                &[l.num_value_heads],
                false,
                Transform::Identity,
            )?;
            c(
                "ssm_norm.weight",
                &[l.value_head_dim],
                false,
                Transform::Identity,
            )?;
        }
        // The component can concatenate different encoded formats, or dense
        // matrices; a dense/encoded mixture needs a different projection type.
        let groups: &[&[&str]] = if full {
            &[
                &["attn_q.weight", "attn_k.weight", "attn_v.weight"],
                &["ffn_gate.weight", "ffn_up.weight"],
            ]
        } else {
            &[&["ffn_gate.weight", "ffn_up.weight"]]
        };
        for group in groups {
            let quant: Vec<_> = group
                .iter()
                .map(|s| {
                    reader
                        .tensor_info(&format!("blk.{i}.{s}"))
                        .unwrap()
                        .ggml_type()
                        .block_quant_format()
                        .is_some()
                })
                .collect();
            if quant.iter().any(|&q| q != quant[0]) {
                return Err(OpError::Shape(format!(
                    "GGUF blk.{i}: unsupported dense/quantized fusion {group:?}"
                )));
            }
        }
    }
    for info in reader
        .tensors()
        .iter()
        .filter(|t| !expected.contains(t.name()))
    {
        let mtp = info
            .name()
            .strip_prefix("blk.")
            .and_then(|s| s.split_once('.'))
            .and_then(|(s, _)| s.parse::<usize>().ok())
            .is_some_and(|i| i >= d.num_layers && i < d.num_layers + cfg.mtp_layers);
        if !mtp {
            return Err(OpError::Shape(format!(
                "GGUF unexpected tensor {}",
                info.name()
            )));
        }
        report.skipped_mtp_tensors.push(info.name().to_owned());
        report.skipped_mtp_bytes += info.byte_len();
    }
    Ok(report)
}

#[cfg(test)]
mod tests;
