//! Official DFlashDraftModel checkpoint adapter. Draft weights and RoPE belong
//! to this model; embedding and readout storage are borrowed from the target.
use crate::{
    components::{
        DenseFfn, Embed, FullAttention, Linear, LmHead, RmsNorm,
        attention_core::AttentionCore,
        dflash::{DFlashDraftHead, DFlashLayer, DFlashWeights},
    },
    domain::{
        dtype::Dtype,
        features::FeatureSpec,
        model::ModelDims,
        ports::{OpBackend, OpError, OpResult, backend::LlmBackend},
        tensor::Tensor,
    },
    models::loader::{WeightLoader, compute_rope_cache},
};

#[derive(Clone, Debug, serde::Deserialize)]
pub struct DFlashConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_target_layers: usize,
    pub max_position_embeddings: usize,
    pub block_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    #[serde(alias = "torch_dtype")]
    pub dtype: String,
    pub hidden_act: String,
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    pub layer_types: Vec<String>,
    pub dflash_config: DFlashAuxConfig,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct DFlashAuxConfig {
    pub mask_token_id: i32,
    pub target_layer_ids: Vec<usize>,
}

impl DFlashConfig {
    pub fn parse(raw: &[u8]) -> OpResult<Self> {
        serde_json::from_slice(raw).map_err(|e| OpError::Shape(format!("DFlash config: {e}")))
    }
    pub fn feature_spec(&self, target: ModelDims) -> OpResult<FeatureSpec> {
        let ids = &self.dflash_config.target_layer_ids;
        if ids.is_empty()
            || ids.windows(2).any(|w| w[0] >= w[1])
            || ids
                .iter()
                .any(|&i| i >= target.num_layers.saturating_sub(1))
        {
            return Err(OpError::Shape(
                "invalid DFlash target decoder feature layers".into(),
            ));
        }
        Ok(FeatureSpec::DecoderLayers(ids.clone()))
    }
    pub fn validate(&self, target: ModelDims, context: usize, draft_tokens: usize) -> OpResult<()> {
        self.feature_spec(target)?;
        if self.architectures != ["DFlashDraftModel"]
            || self.model_type != "qwen3"
            || self.hidden_size == 0
            || self.hidden_size != target.dim
            || self.vocab_size != target.vocab_size
            || self.vocab_size > i32::MAX as usize
            || self.num_target_layers != target.num_layers
            || self.num_hidden_layers == 0
            || self.head_dim == 0
            || !self.head_dim.is_multiple_of(2)
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self.intermediate_size == 0
            || self.dtype != "bfloat16"
            || self.hidden_act != "silu"
            || self.attention_bias
            || self.mlp_bias
            || !self.tie_word_embeddings
            || self.rope_scaling.is_some()
            || self.sliding_window.is_some()
            || self.use_sliding_window
            || self.layer_types.len() != self.num_hidden_layers
            || self.layer_types.iter().any(|t| t != "full_attention")
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || context == 0
            || context > self.max_position_embeddings
            || self.block_size < 2
            || draft_tokens == 0
            || draft_tokens >= self.block_size
            || draft_tokens >= context
            || self.dflash_config.mask_token_id < 0
            || self.dflash_config.mask_token_id as usize >= self.vocab_size
        {
            return Err(OpError::Shape(
                "unsupported or incompatible DFlashDraftModel configuration".into(),
            ));
        }
        for width in [
            self.num_attention_heads,
            self.num_key_value_heads,
            self.hidden_size,
            self.intermediate_size,
            self.vocab_size,
            self.dflash_config.target_layer_ids.len(),
        ] {
            if width
                .checked_mul(self.head_dim)
                .and_then(|n| n.checked_mul(self.hidden_size))
                .and_then(|n| n.checked_mul(10))
                .is_none()
            {
                return Err(OpError::Shape(
                    "DFlash checkpoint dimensions overflow".into(),
                ));
            }
        }
        Ok(())
    }
}

pub fn build<T: Dtype, D: OpBackend + LlmBackend>(
    loader: &WeightLoader<'_>,
    cfg: &DFlashConfig,
    target: ModelDims,
    embedding: &Tensor<T, D>,
    lm_head: &Tensor<T, D>,
    context: usize,
    device: &D,
) -> OpResult<DFlashDraftHead<T, D>> {
    cfg.validate(target, context, 1)?;
    if loader.tensor_parallel().size != 1 {
        return Err(OpError::Shape("DFlash loading requires TP1".into()));
    }
    for tensor in [embedding, lm_head] {
        if tensor.shape().as_slice() != [target.vocab_size, target.dim]
            || !tensor.is_contiguous()
            || infer_core::device::Device::device_id(tensor.device())
                != infer_core::device::Device::device_id(device)
        {
            return Err(OpError::Shape(
                "DFlash shared embedding/readout shape/device mismatch".into(),
            ));
        }
    }
    // Recognize the official tensor layout as well as the architecture tag.
    // DSpark/Markov or self-contained checkpoints need a separate adapter.
    if loader.has_tensor("embed_tokens.weight") || loader.has_tensor("lm_head.weight") {
        return Err(OpError::Shape(
            "DFlash adapter expects target-shared embedding and lm_head".into(),
        ));
    }
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    let check = |name: &str, shape: &[usize]| -> OpResult<()> {
        let view = loader.read_view(name).map_err(OpError::Shape)?;
        if view.shape() != shape || view.dtype() != safetensors::Dtype::BF16 {
            return Err(OpError::Shape(format!(
                "DFlash {name}: expected BF16 {shape:?}, got {:?} {:?}",
                view.dtype(),
                view.shape()
            )));
        }
        Ok(())
    };
    check(
        "fc.weight",
        &[h, h * cfg.dflash_config.target_layer_ids.len()],
    )?;
    check("hidden_norm.weight", &[h])?;
    check("norm.weight", &[h])?;
    for i in 0..cfg.num_hidden_layers {
        for (name, shape) in [
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("self_attn.q_proj.weight", vec![q, h]),
            ("self_attn.k_proj.weight", vec![kv, h]),
            ("self_attn.v_proj.weight", vec![kv, h]),
            ("self_attn.o_proj.weight", vec![h, q]),
            ("self_attn.q_norm.weight", vec![cfg.head_dim]),
            ("self_attn.k_norm.weight", vec![cfg.head_dim]),
            ("mlp.gate_proj.weight", vec![cfg.intermediate_size, h]),
            ("mlp.up_proj.weight", vec![cfg.intermediate_size, h]),
            ("mlp.down_proj.weight", vec![h, cfg.intermediate_size]),
        ] {
            check(&format!("layers.{i}.{name}"), &shape)?;
        }
    }
    let tensor = |name: &str| loader.load_tensor::<T, D>(name, device);
    let norm = |name: &str| -> OpResult<RmsNorm<T, D>> {
        Ok(RmsNorm {
            weight: tensor(name)?,
            eps: cfg.rms_norm_eps,
            zero_centered: false,
        })
    };
    let linear = |name: &str| -> OpResult<Linear<T, D>> { Ok(Linear::new(tensor(name)?, None)) };
    let (sin, cos) = compute_rope_cache(context, cfg.head_dim, cfg.rope_theta, None, device)?;
    let layers = (0..cfg.num_hidden_layers)
        .map(|i| {
            let p = format!("layers.{i}");
            Ok(DFlashLayer {
                attention: FullAttention {
                    input_layernorm: norm(&format!("{p}.input_layernorm.weight"))?,
                    core: AttentionCore {
                        qkv_proj: loader.load_fused_qkv_with_fp8(&p, q, kv, h, None, device)?,
                        o_proj: linear(&format!("{p}.self_attn.o_proj.weight"))?,
                        q_norm: Some(norm(&format!("{p}.self_attn.q_norm.weight"))?),
                        k_norm: Some(norm(&format!("{p}.self_attn.k_norm.weight"))?),
                        sin: sin.clone(),
                        cos: cos.clone(),
                        head_num: cfg.num_attention_heads,
                        kv_head_num: cfg.num_key_value_heads,
                        head_dim: cfg.head_dim,
                        rotary_dim: cfg.head_dim,
                        scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                        attn_output_gate: false,
                    },
                    scratch: None,
                },
                ffn: DenseFfn {
                    post_attention_layernorm: norm(&format!(
                        "{p}.post_attention_layernorm.weight"
                    ))?,
                    gate_up_proj: loader.load_fused_gate_up_with_fp8(
                        &p,
                        cfg.intermediate_size,
                        h,
                        None,
                        device,
                    )?,
                    down_proj: linear(&format!("{p}.mlp.down_proj.weight"))?,
                    scratch: None,
                },
            })
        })
        .collect::<OpResult<Vec<_>>>()?;
    DFlashDraftHead::new(
        DFlashWeights {
            embedding: Embed::new(embedding.clone()),
            lm_head: LmHead {
                proj: Linear::new(lm_head.clone(), None),
            },
            feature_projection: linear("fc.weight")?,
            hidden_norm: norm("hidden_norm.weight")?,
            output_norm: norm("norm.weight")?,
            layers,
        },
        ModelDims {
            dim: h,
            q_dim: q,
            kv_dim: kv,
            qkv_dim: q + 2 * kv,
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            head_num: cfg.num_attention_heads,
            kv_head_num: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            num_layers: cfg.num_hidden_layers,
            ..Default::default()
        },
        cfg.dflash_config.target_layer_ids.len(),
        cfg.block_size,
        cfg.dflash_config.mask_token_id,
    )
}
