//! SpecForge's single-layer Llama-style EAGLE3 head: three target features,
//! shared target embedding, and an optional compact output vocabulary.
use crate::{
    components::{
        DenseFfn, Embed, Linear, LmHead, RmsNorm,
        attention_core::AttentionCore,
        eagle3::{Eagle3DraftHead, Eagle3Weights},
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
pub struct SpecForgeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub draft_vocab_size: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    #[serde(alias = "torch_dtype")]
    pub dtype: String,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub fc_norm: bool,
    #[serde(default)]
    pub norm_output: bool,
    #[serde(default)]
    pub target_hidden_size: Option<usize>,
    #[serde(default)]
    pub target_model_name_or_path: Option<String>,
    #[serde(default)]
    pub eagle_config: AuxConfig,
    #[serde(default)]
    pub eagle_aux_hidden_state_layer_ids: Option<Vec<usize>>,
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
pub struct AuxConfig {
    pub eagle_aux_hidden_state_layer_ids: Option<Vec<usize>>,
}

impl SpecForgeConfig {
    pub fn feature_spec(&self, target: ModelDims) -> OpResult<FeatureSpec> {
        if target.num_layers < 6 {
            return Err(OpError::Shape(
                "SpecForge requires at least six target layers".into(),
            ));
        }
        let nested = &self.eagle_config.eagle_aux_hidden_state_layer_ids;
        let direct = &self.eagle_aux_hidden_state_layer_ids;
        if nested.is_some() && direct.is_some() && nested != direct {
            return Err(OpError::Shape(
                "conflicting SpecForge feature layers".into(),
            ));
        }
        // SpecForge captures outputs of these zero-based target decoder blocks.
        let ids = nested
            .as_ref()
            .or(direct.as_ref())
            .cloned()
            .unwrap_or_else(|| vec![1, target.num_layers / 2 - 1, target.num_layers - 4]);
        if ids.len() != 3
            || ids.windows(2).any(|w| w[0] >= w[1])
            || ids.iter().any(|&i| i >= target.num_layers - 1)
        {
            return Err(OpError::Shape(
                "invalid SpecForge target feature layers".into(),
            ));
        }
        Ok(FeatureSpec::DecoderLayers(ids))
    }

    pub fn validate(&self, target: ModelDims, context: usize) -> OpResult<()> {
        self.feature_spec(target)?;
        if self.hidden_size == 0
            || self.hidden_size != target.dim
            || self.vocab_size != target.vocab_size
            || self.draft_vocab_size == 0
            || self.draft_vocab_size > self.vocab_size
            || self.num_hidden_layers != 1
            || self.target_hidden_size.is_some_and(|h| h != target.dim)
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
            || self.tie_word_embeddings
            || self.fc_norm
            || self.norm_output
            || self.rope_scaling.is_some()
            || self.sliding_window.is_some()
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || context == 0
            || context > self.max_position_embeddings
        {
            return Err(OpError::Shape(
                "unsupported or incompatible SpecForge EAGLE3 configuration".into(),
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
                    "SpecForge checkpoint dimensions overflow".into(),
                ));
            }
        }
        Ok(())
    }
}

fn check_tensor(
    loader: &WeightLoader<'_>,
    name: &str,
    shape: &[usize],
    dtype: safetensors::Dtype,
) -> OpResult<()> {
    let t = loader.read_view(name).map_err(OpError::Shape)?;
    if t.shape() != shape || t.dtype() != dtype {
        return Err(OpError::Shape(format!(
            "SpecForge {name}: expected {dtype:?} {shape:?}, got {:?} {:?}",
            t.dtype(),
            t.shape()
        )));
    }
    Ok(())
}

fn token_mapping(loader: &WeightLoader<'_>, cfg: &SpecForgeConfig) -> OpResult<Option<Vec<i32>>> {
    if !loader.has_tensor("d2t")
        && !loader.has_tensor("t2d")
        && cfg.draft_vocab_size == cfg.vocab_size
    {
        return Ok(None);
    }
    check_tensor(
        loader,
        "d2t",
        &[cfg.draft_vocab_size],
        safetensors::Dtype::I64,
    )?;
    check_tensor(loader, "t2d", &[cfg.vocab_size], safetensors::Dtype::BOOL)?;
    let offsets = loader.read_view("d2t").map_err(OpError::Shape)?;
    let mask = loader.read_view("t2d").map_err(OpError::Shape)?;
    let mut seen = vec![false; cfg.vocab_size];
    let mut ids = Vec::with_capacity(cfg.draft_vocab_size);
    for (i, bytes) in offsets.data().chunks_exact(8).enumerate() {
        // Checkpoint entries are offsets, not absolute target token IDs.
        let id = i64::from_le_bytes(bytes.try_into().unwrap())
            .checked_add(i as i64)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v < cfg.vocab_size && v <= i32::MAX as usize)
            .ok_or_else(|| OpError::Shape("SpecForge d2t maps outside target vocabulary".into()))?;
        if std::mem::replace(&mut seen[id], true) {
            return Err(OpError::Shape(
                "SpecForge d2t contains duplicate target IDs".into(),
            ));
        }
        ids.push(id as i32);
    }
    if mask
        .data()
        .iter()
        .zip(seen)
        .any(|(&v, seen)| v > 1 || (v != 0) != seen)
    {
        return Err(OpError::Shape("SpecForge t2d and d2t disagree".into()));
    }
    Ok(Some(ids))
}

pub(super) fn build_draft<T: Dtype, D: OpBackend + LlmBackend>(
    loader: &WeightLoader<'_>,
    cfg: &SpecForgeConfig,
    target: ModelDims,
    target_embedding: &Tensor<T, D>,
    context: usize,
    device: &D,
) -> OpResult<Eagle3DraftHead<T, D>> {
    use infer_core::device::Device;
    cfg.validate(target, context)?;
    if loader.tensor_parallel().size != 1 {
        return Err(OpError::unsupported(
            "SpecForge EAGLE3",
            "tensor parallel loading",
        ));
    }
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    if target_embedding.shape().as_slice() != [target.vocab_size, h]
        || !target_embedding.is_contiguous()
        || Device::device_id(target_embedding.device()) != Device::device_id(device)
    {
        return Err(OpError::Shape(
            "SpecForge target embedding shape/device mismatch".into(),
        ));
    }
    for (name, shape) in [
        ("fc.weight", vec![h, 3 * h]),
        ("lm_head.weight", vec![cfg.draft_vocab_size, h]),
        ("norm.weight", vec![h]),
        ("midlayer.input_layernorm.weight", vec![h]),
        ("midlayer.hidden_norm.weight", vec![h]),
        ("midlayer.post_attention_layernorm.weight", vec![h]),
        ("midlayer.self_attn.q_proj.weight", vec![q, 2 * h]),
        ("midlayer.self_attn.k_proj.weight", vec![kv, 2 * h]),
        ("midlayer.self_attn.v_proj.weight", vec![kv, 2 * h]),
        ("midlayer.self_attn.o_proj.weight", vec![h, q]),
        (
            "midlayer.mlp.gate_proj.weight",
            vec![cfg.intermediate_size, h],
        ),
        (
            "midlayer.mlp.up_proj.weight",
            vec![cfg.intermediate_size, h],
        ),
        (
            "midlayer.mlp.down_proj.weight",
            vec![h, cfg.intermediate_size],
        ),
    ] {
        check_tensor(loader, name, &shape, safetensors::Dtype::BF16)?;
    }
    if loader.has_tensor("midlayer.self_attn.q_norm.weight")
        || loader.has_tensor("midlayer.self_attn.k_norm.weight")
        || loader.has_tensor("fc_norm.0.weight")
    {
        return Err(OpError::Shape(
            "unsupported SpecForge extra normalization weights".into(),
        ));
    }
    let mapping = token_mapping(loader, cfg)?;
    if loader.has_tensor("embed_tokens.weight") {
        check_tensor(
            loader,
            "embed_tokens.weight",
            &[target.vocab_size, h],
            safetensors::Dtype::BF16,
        )?;
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
    let (sin, cos) = compute_rope_cache(context, cfg.head_dim, cfg.rope_theta, None, device)?;
    let embedding = if loader.has_tensor("embed_tokens.weight") {
        tensor("embed_tokens.weight")?
    } else {
        target_embedding.clone()
    };
    let map = mapping
        .map(|ids| Tensor::from_host_slice(&ids, [ids.len()], device))
        .transpose()?;
    Eagle3DraftHead::with_token_map(
        Eagle3Weights {
            embedding: Embed::new(embedding),
            feature_projection: linear("fc.weight")?,
            embedding_norm: norm("midlayer.input_layernorm.weight")?,
            hidden_norm: norm("midlayer.hidden_norm.weight")?,
            attention: AttentionCore {
                qkv_proj: loader.load_fused_qkv_with_fp8("midlayer", q, kv, 2 * h, None, device)?,
                o_proj: linear("midlayer.self_attn.o_proj.weight")?,
                q_norm: None,
                k_norm: None,
                sin,
                cos,
                head_num: cfg.num_attention_heads,
                kv_head_num: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                rotary_dim: cfg.head_dim,
                attn_output_gate: false,
            },
            ffn: DenseFfn {
                post_attention_layernorm: norm("midlayer.post_attention_layernorm.weight")?,
                gate_up_proj: loader.load_fused_gate_up_with_fp8(
                    "midlayer",
                    cfg.intermediate_size,
                    h,
                    None,
                    device,
                )?,
                down_proj: linear("midlayer.mlp.down_proj.weight")?,
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
        3,
        map,
    )
}
