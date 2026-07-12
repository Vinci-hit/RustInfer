//! Qwen3-based text encoder for Z-Image.
//!
//! Standalone Qwen3 implementation tailored for diffusion's caption pipeline:
//! one-shot prefill (no KV cache), single sequence (no batching), runs
//! `n_layers - 1` decoder layers and returns the **second-to-last layer
//! hidden states** (mask-stripped). diffusers `ZImagePipeline.encode_prompt`
//! pulls `hidden_states[-2]` after `output_hidden_states=True`.
//!
//! Architecture per layer (BF16 / Cuda):
//!   x_in  = x_padded
//!   normed = input_layernorm(x_in)
//!   q,k,v = qkv_proj(normed) → split
//!   q,k   = qk_norm(q), qk_norm(k)   (Qwen3 quirk)
//!   q,k   = rope_inplace(q, k, positions, sin, cos)
//!   attn  = sdpa_causal(q, k, v)      (we use the unfused dit-style sdpa
//!                                       with a triangular mask applied via
//!                                       a host-built bias tensor)
//!   o     = o_proj(attn)
//!   x_mid = x_in + o
//!   normed = post_attention_layernorm(x_mid)
//!   gate, up = gate_up_proj(normed) → split
//!   ffn = down_proj(silu(gate) * up)
//!   x_out = x_mid + ffn

use std::path::{Path, PathBuf};

use crate::domain::ports::{CoreOps, DiffusionOps, OpBackend, OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{Dtype, Shape};
use crate::infrastructure::cuda::Cuda;
use crate::infrastructure::io::SafetensorsReader;
use crate::models::layers::{Embedding, Linear, RMSNorm};
use crate::models::loader::WeightLoader;

/// Maximum prompt length the encoder supports.
pub const TEXT_ENCODER_MAX_SEQ_LEN: usize = 512;
/// Qwen3 padding token id (`<|endoftext|>`). Tokens past the prompt should
/// be filled with this id; the encoder drops them after the forward pass.
pub const PAD_TOKEN_ID: i32 = 151643;

#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

impl Qwen3Config {
    pub fn from_json<P: AsRef<Path>>(path: P) -> OpResult<Self> {
        let s = std::fs::read_to_string(&path)
            .map_err(|e| OpError::Kernel(format!("text encoder config: {}", e)))?;
        let v: serde_json::Value = serde_json::from_str(&s)
            .map_err(|e| OpError::Kernel(format!("text encoder config parse: {}", e)))?;
        let dim = v["hidden_size"].as_u64().unwrap_or(2560) as usize;
        let n_heads = v["num_attention_heads"].as_u64().unwrap_or(16) as usize;
        Ok(Self {
            vocab_size: v["vocab_size"].as_u64().unwrap_or(151936) as usize,
            dim,
            n_layers: v["num_hidden_layers"].as_u64().unwrap_or(28) as usize,
            n_heads,
            n_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(n_heads as u64) as usize,
            head_dim: v["head_dim"].as_u64().unwrap_or((dim / n_heads) as u64) as usize,
            intermediate_size: v["intermediate_size"].as_u64().unwrap_or(9728) as usize,
            norm_eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_theta: v["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32,
            max_position_embeddings: v["max_position_embeddings"].as_u64().unwrap_or(40960)
                as usize,
        })
    }
}

pub struct Qwen3TextEncoderLayer<T: Dtype, D: OpBackend> {
    pub input_layernorm: RMSNorm<T, D>,
    pub post_attention_layernorm: RMSNorm<T, D>,
    pub qkv_proj: Linear<T, D>,     // [q_dim+2*kv_dim, dim]
    pub o_proj: Linear<T, D>,       // [dim, dim]
    pub gate_up_proj: Linear<T, D>, // [2*intermediate, dim]
    pub down_proj: Linear<T, D>,    // [dim, intermediate]
    pub q_norm: RMSNorm<T, D>,
    pub k_norm: RMSNorm<T, D>,
}

pub struct Qwen3TextEncoder<T: Dtype, D: OpBackend> {
    pub config: Qwen3Config,
    pub embed_tokens: Embedding<T, D>,
    pub layers: Vec<Qwen3TextEncoderLayer<T, D>>,
    /// `[max_position_embeddings, head_dim/2]` cos/sin caches (T-dtype).
    pub cos_cache: Tensor<T, D>,
    pub sin_cache: Tensor<T, D>,
    /// Number of layers to run before returning hidden states (= n_layers - 1).
    pub output_layer_count: usize,
    /// Path to tokenizer.json, kept for `tokenize` calls.
    pub tokenizer_path: PathBuf,
}

impl<T: Dtype> Qwen3TextEncoder<T, Cuda> {
    /// Load Qwen3 text encoder from a diffusers `text_encoder/` directory.
    ///
    /// Expected files:
    /// - `config.json`
    /// - `model.safetensors.index.json` + sharded `model-XXXXX-of-YYYYY.safetensors`
    /// - the project's `tokenizer/tokenizer.json` is read separately by the
    ///   pipeline (caller passes the path).
    pub fn from_pretrained<P: AsRef<Path>, Q: AsRef<Path>>(
        text_encoder_dir: P,
        tokenizer_path: Q,
        device: &Cuda,
    ) -> OpResult<Self> {
        let dir = text_encoder_dir.as_ref();
        let cfg = Qwen3Config::from_json(dir.join("config.json"))?;
        let reader = SafetensorsReader::open(dir)
            .map_err(|e| OpError::Kernel(format!("text_encoder: {}", e)))?;
        let loader = WeightLoader::new(&reader);

        // embed_tokens: `model.embed_tokens.weight` shape `[vocab, dim]`.
        let embed_w: Tensor<T, Cuda> =
            loader.load_tensor::<T, Cuda>("model.embed_tokens.weight", device)?;
        let embed_tokens = Embedding { table: embed_w };

        // Per-layer weights. Diffusers Qwen3 layout:
        //   model.layers.{i}.input_layernorm.weight
        //   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight (no bias for Qwen3)
        //   model.layers.{i}.self_attn.{q,k}_norm.weight
        //   model.layers.{i}.post_attention_layernorm.weight
        //   model.layers.{i}.mlp.{gate,up,down}_proj.weight
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let dim = cfg.dim;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let prefix = format!("model.layers.{}", i);
            let input_layernorm: RMSNorm<T, Cuda> = loader.load_rmsnorm::<T, Cuda>(
                &format!("{}.input_layernorm.weight", prefix),
                device,
                cfg.norm_eps,
            )?;
            let post_attention_layernorm: RMSNorm<T, Cuda> = loader.load_rmsnorm::<T, Cuda>(
                &format!("{}.post_attention_layernorm.weight", prefix),
                device,
                cfg.norm_eps,
            )?;
            // Build fused QKV [q_dim+2*kv_dim, dim] from individual q/k/v_proj.
            let qkv_proj = loader.load_fused_qkv::<T, Cuda>(&prefix, q_dim, kv_dim, dim, device)?;
            let o_proj = loader.load_linear::<T, Cuda>(
                &format!("{}.self_attn.o_proj.weight", prefix),
                None,
                device,
            )?;
            let q_norm: RMSNorm<T, Cuda> = loader.load_rmsnorm::<T, Cuda>(
                &format!("{}.self_attn.q_norm.weight", prefix),
                device,
                cfg.norm_eps,
            )?;
            let k_norm: RMSNorm<T, Cuda> = loader.load_rmsnorm::<T, Cuda>(
                &format!("{}.self_attn.k_norm.weight", prefix),
                device,
                cfg.norm_eps,
            )?;
            // Fused gate_up [2*inter, dim] from gate/up_proj.
            let gate_up_proj = loader.load_fused_gate_up::<T, Cuda>(&prefix, inter, dim, device)?;
            let down_proj = loader.load_linear::<T, Cuda>(
                &format!("{}.mlp.down_proj.weight", prefix),
                None,
                device,
            )?;
            layers.push(Qwen3TextEncoderLayer {
                input_layernorm,
                post_attention_layernorm,
                qkv_proj,
                o_proj,
                gate_up_proj,
                down_proj,
                q_norm,
                k_norm,
            });
        }

        // RoPE caches: `[max_seq, head_dim]` (interleaved cos+sin? our
        // standard rope kernel reads sin/cos as [max_seq, head_dim/2],
        // dtype T). We compute on host then upload.
        let half = cfg.head_dim / 2;
        let max_pos = cfg.max_position_embeddings.min(2048); // cap for memory
        let sin_host = vec![T::DATA_TYPE; 0]; // placeholder
        let mut cos_host_bytes = vec![0u8; max_pos * half * T::SIZE_BYTES];
        let mut sin_host_bytes = vec![0u8; max_pos * half * T::SIZE_BYTES];
        let theta = cfg.rope_theta as f64;
        let freqs: Vec<f64> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / cfg.head_dim as f64))
            .collect();
        for p in 0..max_pos {
            for i in 0..half {
                let arg = p as f64 * freqs[i];
                let c = arg.cos() as f32;
                let s = arg.sin() as f32;
                let off = (p * half + i) * T::SIZE_BYTES;
                match T::DATA_TYPE {
                    crate::domain::types::DataType::F32 => {
                        cos_host_bytes[off..off + 4].copy_from_slice(&c.to_le_bytes());
                        sin_host_bytes[off..off + 4].copy_from_slice(&s.to_le_bytes());
                    }
                    crate::domain::types::DataType::BF16 => {
                        cos_host_bytes[off..off + 2]
                            .copy_from_slice(&half::bf16::from_f32(c).to_le_bytes());
                        sin_host_bytes[off..off + 2]
                            .copy_from_slice(&half::bf16::from_f32(s).to_le_bytes());
                    }
                    crate::domain::types::DataType::F16 => {
                        cos_host_bytes[off..off + 2]
                            .copy_from_slice(&half::f16::from_f32(c).to_le_bytes());
                        sin_host_bytes[off..off + 2]
                            .copy_from_slice(&half::f16::from_f32(s).to_le_bytes());
                    }
                    other => {
                        return Err(OpError::Kernel(format!(
                            "Qwen3TextEncoder: unsupported dtype {:?}",
                            other,
                        )));
                    }
                }
            }
        }
        let _ = sin_host; // unused
        let cos_cache: Tensor<T, Cuda> =
            Tensor::from_host_bytes(&cos_host_bytes, Shape::from_slice(&[max_pos, half]), device)?;
        let sin_cache: Tensor<T, Cuda> =
            Tensor::from_host_bytes(&sin_host_bytes, Shape::from_slice(&[max_pos, half]), device)?;

        Ok(Self {
            output_layer_count: cfg.n_layers.saturating_sub(1),
            config: cfg,
            embed_tokens,
            layers,
            cos_cache,
            sin_cache,
            tokenizer_path: tokenizer_path.as_ref().to_path_buf(),
        })
    }
}

impl<T: Dtype> Qwen3TextEncoder<T, Cuda> {
    /// One-shot forward: input ids `[seq_len]` (i32, host-built) →
    /// hidden states `[seq_len, dim]` (T, on device) at layer
    /// `output_layer_count`. Caller is responsible for stripping padding
    /// tokens afterwards (use `attention_mask`).
    pub fn forward(
        &self,
        input_ids: &[i32],
        attention_mask: &[i32],
        dev: &Cuda,
    ) -> OpResult<Tensor<T, Cuda>> {
        if input_ids.len() != attention_mask.len() {
            return Err(OpError::Shape(format!(
                "Qwen3TextEncoder::forward: input/mask length mismatch {} vs {}",
                input_ids.len(),
                attention_mask.len(),
            )));
        }
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(OpError::Shape(
                "Qwen3TextEncoder::forward: empty input".into(),
            ));
        }

        let dim = self.config.dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let qkv_cols = q_dim + 2 * kv_dim;
        let inter = self.config.intermediate_size;

        // Upload input ids to device.
        let ids: Tensor<i32, Cuda> = Tensor::from_host_slice(input_ids, [seq_len], dev)?;

        // 1. Embedding → x [seq, dim].
        let mut x: Tensor<T, Cuda> = Tensor::zeros([seq_len, dim], dev)?;
        self.embed_tokens.forward(&ids, &mut x)?;

        // Position ids = [0, 1, 2, ..., seq_len - 1].
        let positions_host: Vec<i32> = (0..seq_len as i32).collect();
        let positions: Tensor<i32, Cuda> =
            Tensor::from_host_slice(&positions_host, [seq_len], dev)?;

        // Build attention mask `[seq, seq]` (T-dtype) with the same semantics
        // as transformers' `_prepare_4d_causal_attention_mask`:
        //   mask[i, j] = 0.0           if j <= i AND attention_mask[j] == 1
        //              = -3.3895e+38   otherwise
        // -3.3895e+38 is bf16's minimum finite value (avoids -inf in softmax
        // overflow paths). We cast to T (f32 / bf16 / f16) when uploading.
        let mask_neg_inf: f32 = -3.3895313892515355e+38_f32;
        let mut mask_host_bytes = vec![0u8; seq_len * seq_len * T::SIZE_BYTES];
        for i in 0..seq_len {
            for j in 0..seq_len {
                // Causal: j > i is masked. Padding: attention_mask[j]==0 is
                // masked. (We mask along the *key* axis only, matching the
                // diffusers/transformers convention.)
                let allow = j <= i && attention_mask[j] != 0;
                let v = if allow { 0.0_f32 } else { mask_neg_inf };
                let off = (i * seq_len + j) * T::SIZE_BYTES;
                match T::DATA_TYPE {
                    crate::domain::types::DataType::F32 => {
                        mask_host_bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
                    }
                    crate::domain::types::DataType::BF16 => {
                        mask_host_bytes[off..off + 2]
                            .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
                    }
                    crate::domain::types::DataType::F16 => {
                        mask_host_bytes[off..off + 2]
                            .copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
                    }
                    other => {
                        return Err(OpError::Kernel(format!(
                            "Qwen3TextEncoder: unsupported dtype {:?}",
                            other,
                        )));
                    }
                }
            }
        }
        let mask_dev: Tensor<T, Cuda> = Tensor::from_host_bytes(
            &mask_host_bytes,
            Shape::from_slice(&[seq_len, seq_len]),
            dev,
        )?;

        // Per-layer scratches.
        let mut h: Tensor<T, Cuda> = Tensor::zeros([seq_len, dim], dev)?;
        let mut qkv: Tensor<T, Cuda> = Tensor::zeros([seq_len, qkv_cols], dev)?;
        let mut q: Tensor<T, Cuda> = Tensor::zeros([seq_len, q_dim], dev)?;
        let mut k: Tensor<T, Cuda> = Tensor::zeros([seq_len, kv_dim], dev)?;
        let mut v: Tensor<T, Cuda> = Tensor::zeros([seq_len, kv_dim], dev)?;
        let attn_out: Tensor<T, Cuda> = Tensor::zeros([seq_len, q_dim], dev)?;
        let mut o_out: Tensor<T, Cuda> = Tensor::zeros([seq_len, dim], dev)?;
        let mut gate_up: Tensor<T, Cuda> = Tensor::zeros([seq_len, 2 * inter], dev)?;
        let mut gate: Tensor<T, Cuda> = Tensor::zeros([seq_len, inter], dev)?;
        let mut ffn_out: Tensor<T, Cuda> = Tensor::zeros([seq_len, dim], dev)?;

        let n_layers_to_run = self.output_layer_count;
        for layer_idx in 0..n_layers_to_run {
            let layer = &self.layers[layer_idx];

            // 2.1 input_layernorm(x) → h
            layer.input_layernorm.forward(&x, &mut h)?;

            // 2.2 qkv_proj(h) → qkv
            layer.qkv_proj.forward(&h, &mut qkv)?;
            Cuda::split_cols(&qkv, &mut q, seq_len, qkv_cols, 0, q_dim)?;
            Cuda::split_cols(&qkv, &mut k, seq_len, qkv_cols, q_dim, kv_dim)?;
            Cuda::split_cols(&qkv, &mut v, seq_len, qkv_cols, q_dim + kv_dim, kv_dim)?;

            // 2.3 q_norm(q) per-head + k_norm(k) per-head.
            // Reshape Q [seq, q_dim] → [seq*n_heads, head_dim].
            let mut q_reshape = q.view_raw(
                Shape::from_slice(&[seq_len * n_heads, head_dim]),
                Shape::from_slice(&[head_dim, 1]).contiguous_strides(),
                q.offset_elems(),
                true,
            );
            layer.q_norm.forward_inplace(&mut q_reshape)?;
            let mut k_reshape = k.view_raw(
                Shape::from_slice(&[seq_len * n_kv_heads, head_dim]),
                Shape::from_slice(&[head_dim, 1]).contiguous_strides(),
                k.offset_elems(),
                true,
            );
            layer.k_norm.forward_inplace(&mut k_reshape)?;

            // 2.4 RoPE (standard, not interleaved). The new OpBackend
            // exposes `rope_inplace` for the standard LLM path.
            Cuda::rope_inplace(
                &mut q,
                &mut k,
                &self.sin_cache,
                &self.cos_cache,
                &positions,
                n_heads,
                n_kv_heads,
                head_dim,
            )?;

            // 2.5 SDPA (GQA-aware): SDPA gathers KV per query head via head/group.
            let q3 = q.view_raw(
                Shape::from_slice(&[seq_len, n_heads, head_dim]),
                Shape::from_slice(&[head_dim * n_heads, head_dim, 1]).contiguous_strides(),
                q.offset_elems(),
                true,
            );
            let k3 = k.view_raw(
                Shape::from_slice(&[seq_len, n_kv_heads, head_dim]),
                Shape::from_slice(&[head_dim * n_kv_heads, head_dim, 1]).contiguous_strides(),
                k.offset_elems(),
                true,
            );
            let v3 = v.view_raw(
                Shape::from_slice(&[seq_len, n_kv_heads, head_dim]),
                Shape::from_slice(&[head_dim * n_kv_heads, head_dim, 1]).contiguous_strides(),
                v.offset_elems(),
                true,
            );
            let mut attn3 = attn_out.view_raw(
                Shape::from_slice(&[seq_len, n_heads, head_dim]),
                Shape::from_slice(&[head_dim * n_heads, head_dim, 1]).contiguous_strides(),
                attn_out.offset_elems(),
                true,
            );
            let scale = 1.0 / (head_dim as f32).sqrt();
            // Qwen3 is a *causal* decoder-only LM (Qwen3Model = Qwen3 stack
            // without the LM head, but still uses causal masking — verified
            // by inspecting `transformers.models.qwen3.modeling_qwen3` where
            // every Qwen3DecoderLayer.self_attn has `is_causal = True`).
            // Z-Image's `ZImagePipeline.encode_prompt` calls this Qwen3Model
            // and pulls `hidden_states[-2]`, so we MUST apply the same mask
            // it would internally build: causal upper-triangular + padding
            // columns set to -inf. Without it, prompt_embeds magnitudes are
            // wrong by 2-3 orders of magnitude (verified empirically — first
            // hidden state shifts from O(8) to O(13000) at layer 10).
            //
            // The mask is built once per forward (cached in `mask_dev`).
            Cuda::sdpa_masked(
                &q3, &k3, &v3, &mut attn3, &mask_dev, n_heads, n_kv_heads, head_dim, scale,
            )?;

            // 2.6 o_proj(attn) → o_out
            layer.o_proj.forward(&attn_out, &mut o_out)?;
            // x += o_out
            Cuda::add_inplace(&mut x, &o_out)?;

            // 2.7 post_attention_layernorm(x) → h
            layer.post_attention_layernorm.forward(&x, &mut h)?;

            // 2.8 MLP: gate_up_proj(h) → [seq, 2*inter]; swiglu_packed → [seq, inter]
            layer.gate_up_proj.forward(&h, &mut gate_up)?;
            Cuda::swiglu_packed(&gate_up, &mut gate, seq_len, inter)?;
            // down_proj
            layer.down_proj.forward(&gate, &mut ffn_out)?;
            // x += ffn_out
            Cuda::add_inplace(&mut x, &ffn_out)?;
        }

        // After running n_layers - 1 layers, `x` *is* hidden_states[-2].
        // (diffusers `output_hidden_states=True` returns the embedding output
        // followed by each layer's output, so `hidden_states[i]` =
        // post-layer-i hidden state. `hidden_states[-2]` is post-(n-2),
        // which equals running n-1 layers.)

        // Strip pad tokens (rows with mask==0).
        let actual_len: usize = attention_mask.iter().filter(|&&m| m == 1).count();
        if actual_len == seq_len {
            return Ok(x);
        }
        // Slice prefix (mask is left-aligned in our pipeline; verify here).
        for i in 0..actual_len {
            if attention_mask[i] != 1 {
                return Err(OpError::Kernel(format!(
                    "Qwen3TextEncoder: attention mask must be left-aligned (mask[{}]=0)",
                    i,
                )));
            }
        }
        // Take rows [0..actual_len].
        Ok(x.view_raw(
            Shape::from_slice(&[actual_len, dim]),
            Shape::from_slice(&[dim, 1]).contiguous_strides(),
            x.offset_elems(),
            true,
        ))
    }
}

/// Build the chat-template-wrapped prompt that diffusers feeds to the
/// Qwen3 tokenizer. Tokenization itself is performed by the worker via
/// `tokenizers` crate (caller's job).
pub fn apply_chat_template(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_format() {
        let s = apply_chat_template("hello");
        assert!(s.starts_with("<|im_start|>user\nhello<|im_end|>"));
        assert!(s.contains("<|im_start|>assistant"));
    }

    #[test]
    fn config_parses_minimum_keys() {
        // Synthetic minimal config (real Qwen3 has many more, but we only
        // read a few fields).
        let dir = tempdir_for_test();
        let path = dir.join("config.json");
        std::fs::write(
            &path,
            r#"{
            "hidden_size": 2560,
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 16,
            "head_dim": 160,
            "intermediate_size": 9728,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 40960,
            "vocab_size": 151936
        }"#,
        )
        .unwrap();
        let cfg = Qwen3Config::from_json(&path).unwrap();
        assert_eq!(cfg.dim, 2560);
        assert_eq!(cfg.n_layers, 28);
        assert_eq!(cfg.n_heads, 16);
        assert_eq!(cfg.n_kv_heads, 16);
        assert_eq!(cfg.head_dim, 160);
        assert_eq!(cfg.intermediate_size, 9728);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1e-3);
    }

    fn tempdir_for_test() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("qwen3_test_{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
