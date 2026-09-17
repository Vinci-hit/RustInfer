use serde::Deserialize;

use crate::domain::ports::{OpError, OpResult};

/// Deliberately bounded, unquantized V4 fixture contract. Production checkpoints
/// must never silently enter this synchronous, host-assisted reference path.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TinyConfig {
    pub model_type: String,
    #[serde(default)]
    pub rustinfer_tiny: bool,
    pub dtype: String,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub partial_rotary_factor: f32,
    pub layer_types: Vec<String>,
    pub mlp_layer_types: Vec<String>,
    pub compress_rates: std::collections::HashMap<String, usize>,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub rms_norm_eps: f32,
    pub routed_scaling_factor: f32,
    pub scoring_func: String,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub attention_dropout: f32,
    pub norm_topk_prob: bool,
    pub swiglu_limit: f32,
    pub num_nextn_predict_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_parameters: serde_json::Value,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

impl TinyConfig {
    pub fn validate(&self) -> OpResult<()> {
        let bad = |s: &str| OpError::Shape(format!("deepseek_v4 tiny: {s}"));
        if self.model_type != "deepseek_v4" || !self.rustinfer_tiny {
            return Err(bad("requires an explicitly marked rustinfer_tiny fixture"));
        }
        for (name, value, max) in [
            ("hidden_size", self.hidden_size, 512),
            ("vocab_size", self.vocab_size, 4096),
            ("num_hidden_layers", self.num_hidden_layers, 6),
            ("num_attention_heads", self.num_attention_heads, 8),
            ("head_dim", self.head_dim, 128),
            ("q_lora_rank", self.q_lora_rank, 256),
            ("o_lora_rank", self.o_lora_rank, 256),
            ("o_groups", self.o_groups, 8),
            ("index_n_heads", self.index_n_heads, 8),
            ("index_head_dim", self.index_head_dim, 128),
            ("index_topk", self.index_topk, 32),
            ("sliding_window", self.sliding_window, 128),
            ("max_position_embeddings", self.max_position_embeddings, 512),
            ("n_routed_experts", self.n_routed_experts, 16),
            ("num_experts_per_tok", self.num_experts_per_tok, 4),
            ("moe_intermediate_size", self.moe_intermediate_size, 256),
            ("hc_sinkhorn_iters", self.hc_sinkhorn_iters, 20),
        ] {
            if value == 0 || value > max {
                return Err(bad(&format!("{name} must be in 1..={max}, got {value}")));
            }
        }
        if self.num_key_value_heads != 1
            || self.n_shared_experts != 1
            || self.hc_mult != 4
            || self.num_experts_per_tok > self.n_routed_experts
            || !self.num_attention_heads.is_multiple_of(self.o_groups)
        {
            return Err(bad(
                "invalid KV heads, shared expert, HC or grouping geometry",
            ));
        }
        if !self.partial_rotary_factor.is_finite()
            || self.partial_rotary_factor <= 0.0
            || self.partial_rotary_factor > 1.0
            || (self.head_dim as f32 * self.partial_rotary_factor).fract() != 0.0
            || self.rotary_dim() == 0
            || !self.rotary_dim().is_multiple_of(2)
            || self.rotary_dim() > self.index_head_dim
        {
            return Err(bad(
                "rotary dimensions must be even and fit both attention and indexer",
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers
            || self.mlp_layer_types.len() != self.num_hidden_layers
            || self.layer_types.iter().any(|s| {
                !matches!(
                    s.as_str(),
                    "sliding_attention"
                        | "compressed_sparse_attention"
                        | "heavily_compressed_attention"
                )
            })
            || self
                .mlp_layer_types
                .iter()
                .any(|s| !matches!(s.as_str(), "hash_moe" | "moe"))
            || self.compress_rates.get("compressed_sparse_attention") != Some(&4)
            || self.compress_rates.get("heavily_compressed_attention") != Some(&128)
        {
            return Err(bad(
                "invalid layer schedule or compression ratios (expected 4/128)",
            ));
        }
        if !matches!(self.dtype.as_str(), "float32" | "bfloat16")
            || self.quantization_config.is_some()
            || self.scoring_func != "sqrtsoftplus"
            || self.hidden_act != "silu"
            || self.attention_bias
            || self.mlp_bias
            || self.attention_dropout != 0.0
            || !self.norm_topk_prob
            || self.num_nextn_predict_layers != 0
            || self.tie_word_embeddings
        {
            return Err(bad(
                "requires unquantized FP32/BF16, sqrtsoftplus, normalized routes, SiLU, no projection bias/dropout, untied head and no MTP",
            ));
        }
        for value in [
            self.hc_eps,
            self.rms_norm_eps,
            self.swiglu_limit,
            self.routed_scaling_factor,
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(bad(
                    "epsilons, activation limit and route scale must be finite and positive",
                ));
            }
        }
        for label in ["main", "compress"] {
            let rp = &self.rope_parameters[label];
            if rp["rope_type"].as_str() != Some("default")
                || rp["rope_theta"]
                    .as_f64()
                    .is_none_or(|x| !x.is_finite() || x <= 0.0)
                || rp["partial_rotary_factor"].as_f64().map(|x| x as f32)
                    != Some(self.partial_rotary_factor)
            {
                return Err(bad(
                    "this short-context fixture requires plain main/compress RoPE",
                ));
            }
        }
        Ok(())
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }

    pub(super) fn ratio(&self, layer: usize) -> usize {
        match self.layer_types[layer].as_str() {
            "compressed_sparse_attention" => 4,
            "heavily_compressed_attention" => 128,
            _ => 0,
        }
    }

    pub(super) fn theta(&self, compressed: bool) -> f32 {
        self.rope_parameters[if compressed { "compress" } else { "main" }]["rope_theta"]
            .as_f64()
            .expect("validated RoPE") as f32
    }
}
