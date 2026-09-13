//! Qwen3 EAGLE3 checkpoint convention. This factory assembles a draft head;
//! it never registers or replaces the target model architecture.
use crate::components::{
    DenseFfn, Embed, Linear, LmHead, RmsNorm,
    attention_core::AttentionCore,
    eagle3::{Eagle3DraftHead, Eagle3Weights},
};
use crate::domain::{
    dtype::Dtype,
    features::FeatureSpec,
    model::ModelDims,
    ports::{OpBackend, OpError, OpResult, backend::LlmBackend},
};
use crate::models::loader::{WeightLoader, compute_rope_cache};

#[derive(Clone, Debug, serde::Deserialize)]
pub struct DeepSpecConfig {
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub draft_num_hidden_layers: usize,
    pub num_target_layers: usize,
    pub target_layer_ids: Vec<usize>,
    pub target_model_name_or_path: String,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: DraftRope,
    #[serde(alias = "torch_dtype")]
    pub dtype: String,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    pub ttt_length: usize,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct DraftRope {
    pub rope_theta: f64,
    pub rope_type: String,
}

impl DeepSpecConfig {
    pub fn feature_spec(&self) -> FeatureSpec {
        FeatureSpec::DecoderLayers(self.target_layer_ids.clone())
    }
    pub fn validate(&self, target: ModelDims, context: usize) -> OpResult<()> {
        if self.architectures != ["Qwen3Eagle3Model"]
            || self.num_hidden_layers != 1
            || self.draft_num_hidden_layers != 1
            || self.target_layer_ids.len() != 5
            || self.target_layer_ids.windows(2).any(|p| p[0] >= p[1])
            || self
                .target_layer_ids
                .iter()
                .any(|&l| l >= self.num_target_layers.saturating_sub(1))
            || self.hidden_size == 0
            || self.hidden_size != target.dim
            || self.vocab_size != target.vocab_size
            || self.num_target_layers != target.num_layers
            || self.head_dim == 0
            || !self.head_dim.is_multiple_of(2)
            || self.num_key_value_heads == 0
            || self.num_attention_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self.intermediate_size == 0
            || self.dtype != "bfloat16"
            || self.hidden_act != "silu"
            || self.attention_bias
            || self.tie_word_embeddings
            || self.rope_parameters.rope_type != "default"
            || !self.rope_parameters.rope_theta.is_finite()
            || self.rope_parameters.rope_theta <= 0.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || context == 0
            || context > self.max_position_embeddings
            || self.ttt_length == 0
        {
            return Err(OpError::Shape(
                "unsupported or incompatible Qwen3 EAGLE3 checkpoint configuration".into(),
            ));
        }
        for width in [
            self.num_attention_heads,
            self.num_key_value_heads,
            self.hidden_size,
            self.intermediate_size,
            self.vocab_size,
        ] {
            if width
                .checked_mul(self.head_dim)
                .and_then(|n| n.checked_mul(self.hidden_size))
                .and_then(|n| n.checked_mul(10))
                .is_none()
            {
                return Err(OpError::Shape(
                    "EAGLE3 checkpoint dimensions overflow".into(),
                ));
            }
        }
        Ok(())
    }
}

pub fn build_draft<T: Dtype, D: OpBackend + LlmBackend>(
    loader: &WeightLoader<'_>,
    cfg: &DeepSpecConfig,
    target: ModelDims,
    context: usize,
    device: &D,
) -> OpResult<Eagle3DraftHead<T, D>> {
    cfg.validate(target, context)?;
    if loader.tensor_parallel().size != 1 {
        return Err(OpError::unsupported(
            "Qwen3 EAGLE3",
            "tensor parallel loading",
        ));
    }
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    // Validate every tensor before uploading weights or constructing a partial head.
    let shapes = vec![
        ("embed_tokens.weight", vec![cfg.vocab_size, h]),
        ("lm_head.weight", vec![cfg.vocab_size, h]),
        ("fc.weight", vec![h, 5 * h]),
        ("norm.weight", vec![h]),
        ("layers.0.hidden_norm.weight", vec![h]),
        ("layers.0.input_layernorm.weight", vec![h]),
        ("layers.0.post_attention_layernorm.weight", vec![h]),
        ("layers.0.self_attn.q_norm.weight", vec![cfg.head_dim]),
        ("layers.0.self_attn.k_norm.weight", vec![cfg.head_dim]),
        ("layers.0.self_attn.q_proj.weight", vec![q, 2 * h]),
        ("layers.0.self_attn.k_proj.weight", vec![kv, 2 * h]),
        ("layers.0.self_attn.v_proj.weight", vec![kv, 2 * h]),
        ("layers.0.self_attn.o_proj.weight", vec![h, q]),
        (
            "layers.0.mlp.gate_proj.weight",
            vec![cfg.intermediate_size, h],
        ),
        (
            "layers.0.mlp.up_proj.weight",
            vec![cfg.intermediate_size, h],
        ),
        (
            "layers.0.mlp.down_proj.weight",
            vec![h, cfg.intermediate_size],
        ),
    ];
    for (name, shape) in &shapes {
        let view = loader
            .read_view(name)
            .map_err(|e| OpError::Shape(format!("EAGLE3 {name}: {e}")))?;
        if view.shape() != shape || view.dtype() != safetensors::Dtype::BF16 {
            return Err(OpError::Shape(format!(
                "EAGLE3 {name}: expected BF16 {shape:?}, got {:?} {:?}",
                view.dtype(),
                view.shape()
            )));
        }
    }
    let tensor = |name| loader.load_tensor::<T, D>(name, device);
    let norm = |name| -> OpResult<RmsNorm<T, D>> {
        Ok(RmsNorm {
            weight: tensor(name)?,
            eps: cfg.rms_norm_eps,
            zero_centered: false,
        })
    };
    let linear = |name| -> OpResult<Linear<T, D>> { Ok(Linear::new(tensor(name)?, None)) };
    let (sin, cos) = compute_rope_cache(
        context,
        cfg.head_dim,
        cfg.rope_parameters.rope_theta,
        None,
        device,
    )?;
    let attention = AttentionCore {
        qkv_proj: loader.load_fused_qkv_with_fp8("layers.0", q, kv, 2 * h, None, device)?,
        o_proj: linear("layers.0.self_attn.o_proj.weight")?,
        q_norm: Some(norm("layers.0.self_attn.q_norm.weight")?),
        k_norm: Some(norm("layers.0.self_attn.k_norm.weight")?),
        sin,
        cos,
        head_num: cfg.num_attention_heads,
        kv_head_num: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        rotary_dim: cfg.head_dim,
        attn_output_gate: false,
    };
    Eagle3DraftHead::new(
        Eagle3Weights {
            embedding: Embed::new(tensor("embed_tokens.weight")?),
            feature_projection: linear("fc.weight")?,
            embedding_norm: norm("layers.0.input_layernorm.weight")?,
            hidden_norm: norm("layers.0.hidden_norm.weight")?,
            attention,
            ffn: DenseFfn {
                post_attention_layernorm: norm("layers.0.post_attention_layernorm.weight")?,
                gate_up_proj: loader.load_fused_gate_up_with_fp8(
                    "layers.0",
                    cfg.intermediate_size,
                    h,
                    None,
                    device,
                )?,
                down_proj: linear("layers.0.mlp.down_proj.weight")?,
                scratch: None,
            },
            output_norm: norm("norm.weight")?,
            lm_head: LmHead {
                proj: linear("lm_head.weight")?,
            },
        },
        ModelDims {
            dim: h,
            q_dim: q,
            kv_dim: kv,
            qkv_dim: q + 2 * kv,
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            head_num: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            kv_head_num: cfg.num_key_value_heads,
            num_layers: 1,
            ..Default::default()
        },
        5,
    )
}
