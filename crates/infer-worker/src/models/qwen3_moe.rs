//! Qwen3 MoE model.
//!
//! This file is the ownership boundary for Qwen3 MoE differences: HF identity,
//! MoE config validation, tensor naming, and assembly from explicitly loaded
//! global weights and decoder blocks. The shared `WeightLoader` stays
//! model-agnostic.

use crate::components::{
    Attention, DecoderBlock, Embed, ExpertLinear, Linear, LmHead, MoeExperts, MoeFfn,
    MoeLocalPipeline, MoeRouter, RmsNorm,
};
use crate::domain::dtype::Dtype;
use crate::domain::exec::{DeviceId, ExecDevice};
use crate::domain::model::ModelDims;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpBackend, OpError, OpResult};
use crate::models::decoder::Decoder;
use crate::models::layers::RMSNorm as LayerRmsNorm;
use crate::models::loader::{ExpertLinearLoadSpec, LoadConfig, WeightLoader, compute_rope_cache};

pub const MODEL_TYPE: &str = "qwen3_moe";
pub const HF_MODEL_TYPES: &[&str] = &["qwen3_moe"];
pub const HF_ARCHITECTURES: &[&str] = &["Qwen3MoeForCausalLM"];

pub type Qwen3MoeModel<T, D> = Decoder<T, D, MoeFfn<T, D>>;
pub type Qwen3MoeLayer<T, D> = DecoderBlock<T, D, MoeFfn<T, D>>;

/// Model-level tensors that sit outside the repeated decoder blocks.
pub struct Qwen3MoeGlobalWeights<T: Dtype, D: LlmBackend> {
    pub embed: Embed<T, D>,
    pub norm: RmsNorm<T, D>,
    pub lm_head: LmHead<T, D>,
}

/// Load only the model-level embedding, final norm, and LM head.
///
/// When `lm_head.weight` is absent, the embedding tensor is reused through an
/// O(1) tensor-handle clone, matching the dense decoder's tied-weight path.
/// No decoder block is read by this function.
pub fn load_global_weights<T, D>(
    loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    device: &D,
) -> OpResult<Qwen3MoeGlobalWeights<T, D>>
where
    T: Dtype,
    D: OpBackend + LlmBackend,
{
    validate_model_loading(loader, cfg)?;
    validate_attention_config(cfg)?;

    let embed_name = "model.embed_tokens.weight";
    let norm_name = "model.norm.weight";
    validate_tensor_shape(loader, embed_name, &[cfg.vocab_size, cfg.dim])?;
    validate_tensor_shape(loader, norm_name, &[cfg.dim])?;
    if loader.has_tensor("lm_head.weight") {
        validate_tensor_shape(loader, "lm_head.weight", &[cfg.vocab_size, cfg.dim])?;
    }
    let lm_head_bias = loader.has_tensor("lm_head.bias").then_some("lm_head.bias");
    if let Some(name) = lm_head_bias {
        validate_tensor_shape(loader, name, &[cfg.vocab_size])?;
    }

    let embed = loader.load_vocab_parallel_embedding::<T, D>(
        embed_name,
        cfg.vocab_size,
        cfg.dim,
        device,
    )?;
    let norm =
        component_rmsnorm(loader.load_rmsnorm::<T, D>(norm_name, device, cfg.rms_norm_eps)?);
    let lm_head = if loader.has_tensor("lm_head.weight") {
        loader.load_vocab_parallel_linear::<T, D>(
            "lm_head.weight",
            lm_head_bias,
            cfg.vocab_size,
            cfg.dim,
            device,
        )?
    } else {
        loader.vocab_parallel_linear_from_weight(
            embed.table.clone(),
            lm_head_bias,
            cfg.vocab_size,
            device,
        )?
    };

    let weights = Qwen3MoeGlobalWeights {
        embed,
        norm,
        lm_head: LmHead { proj: lm_head },
    };
    validate_global_geometry(&weights, cfg)?;
    Ok(weights)
}

/// Load the pre-norm Qwen3 self-attention sublayer for one local MoE layer.
///
/// Qwen3 requires both per-head Q/K RMSNorm weights. QKV is fused only after
/// each source tensor has been checked against the model geometry. This stage
/// is intentionally TP1 and unquantized, matching the local MoE FFN path.
pub fn load_layer_attention<T, D>(
    loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    layer_index: usize,
    device: &D,
) -> OpResult<Attention<T, D>>
where
    T: Dtype,
    D: OpBackend + LlmBackend,
{
    validate_layer_loading(loader, cfg, layer_index)?;
    validate_attention_config(cfg)?;

    let prefix = layer_prefix(layer_index);
    let q_dim = checked_attention_width("query", cfg.head_num, cfg.head_dim)?;
    let kv_dim = checked_attention_width("KV", cfg.kv_head_num, cfg.head_dim)?;
    let qkv_dim = q_dim
        .checked_add(
            kv_dim
                .checked_mul(2)
                .ok_or_else(|| OpError::Shape("qwen3_moe: QKV width overflows".into()))?,
        )
        .ok_or_else(|| OpError::Shape("qwen3_moe: QKV width overflows".into()))?;

    let input_norm_name = format!("{prefix}.input_layernorm.weight");
    let q_name = format!("{prefix}.self_attn.q_proj.weight");
    let k_name = format!("{prefix}.self_attn.k_proj.weight");
    let v_name = format!("{prefix}.self_attn.v_proj.weight");
    let o_name = format!("{prefix}.self_attn.o_proj.weight");
    let q_norm_name = format!("{prefix}.self_attn.q_norm.weight");
    let k_norm_name = format!("{prefix}.self_attn.k_norm.weight");

    validate_tensor_shape(loader, &input_norm_name, &[cfg.dim])?;
    validate_tensor_shape(loader, &q_name, &[q_dim, cfg.dim])?;
    validate_tensor_shape(loader, &k_name, &[kv_dim, cfg.dim])?;
    validate_tensor_shape(loader, &v_name, &[kv_dim, cfg.dim])?;
    validate_tensor_shape(loader, &o_name, &[cfg.dim, q_dim])?;
    validate_tensor_shape(loader, &q_norm_name, &[cfg.head_dim])?;
    validate_tensor_shape(loader, &k_norm_name, &[cfg.head_dim])?;

    let input_layernorm = component_rmsnorm(loader.load_rmsnorm::<T, D>(
        &input_norm_name,
        device,
        cfg.rms_norm_eps,
    )?);
    let qkv_proj =
        loader.load_fused_qkv_with_fp8::<T, D>(&prefix, q_dim, kv_dim, cfg.dim, None, device)?;
    let o_proj = loader.load_row_parallel_linear_with_fp8::<T, D>(&o_name, None, None, device)?;
    let q_norm =
        component_rmsnorm(loader.load_rmsnorm::<T, D>(&q_norm_name, device, cfg.rms_norm_eps)?);
    let k_norm =
        component_rmsnorm(loader.load_rmsnorm::<T, D>(&k_norm_name, device, cfg.rms_norm_eps)?);
    let (sin, cos) = compute_rope_cache::<T, D>(
        cfg.seq_len,
        cfg.head_dim,
        cfg.rope_theta,
        cfg.rope_scaling.as_ref(),
        device,
    )?;

    let attention = Attention {
        input_layernorm,
        qkv_proj,
        o_proj,
        q_norm: Some(q_norm),
        k_norm: Some(k_norm),
        sin,
        cos,
        head_num: cfg.head_num,
        kv_head_num: cfg.kv_head_num,
        head_dim: cfg.head_dim,
        scale: 1.0 / (cfg.head_dim as f32).sqrt(),
        scratch: None,
    };
    validate_attention_geometry(&attention, cfg.dim, qkv_dim)?;
    Ok(attention)
}

/// The router and two expert Linear groups loaded for one Qwen3 MoE FFN layer.
/// This intermediate result can be validated independently before it is moved
/// into [`assemble_layer_ffn`].
pub struct Qwen3MoeLayerLinears<T: Dtype, D: LlmBackend> {
    pub router: MoeRouter<T, D>,
    pub gate_up: ExpertLinear<T, D>,
    pub down: ExpertLinear<T, D>,
}

/// Load and pack the linear weights for one Qwen3 MoE layer without building
/// or enabling the MoE layer execution path.
pub fn load_layer_linears<T, D>(
    loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    layer_index: usize,
    device: &D,
) -> OpResult<Qwen3MoeLayerLinears<T, D>>
where
    T: Dtype,
    D: LlmBackend,
{
    validate_layer_loading(loader, cfg, layer_index)?;

    let mlp_prefix = format!("{}.mlp", layer_prefix(layer_index));
    let router_name = format!("{}.gate.weight", mlp_prefix);
    let router_view = loader
        .read_view(&router_name)
        .map_err(|error| OpError::Kernel(format!("tensor '{}': {}", router_name, error)))?;
    if router_view.shape() != [cfg.num_experts, cfg.dim] {
        return Err(OpError::Shape(format!(
            "tensor '{}': expected [{}, {}], got {:?}",
            router_name,
            cfg.num_experts,
            cfg.dim,
            router_view.shape()
        )));
    }
    let router = MoeRouter::new(
        Linear::new(loader.load_tensor::<T, D>(&router_name, device)?, None),
        cfg.experts_per_tok,
        cfg.norm_topk_prob,
    )?;

    let gate_up_spec = expert_linear_spec(
        &mlp_prefix,
        cfg.num_experts,
        &[
            ("gate_proj.weight", cfg.moe_intermediate_size),
            ("up_proj.weight", cfg.moe_intermediate_size),
        ],
        cfg.dim,
    )?;
    let down_spec = expert_linear_spec(
        &mlp_prefix,
        cfg.num_experts,
        &[("down_proj.weight", cfg.dim)],
        cfg.moe_intermediate_size,
    )?;

    Ok(Qwen3MoeLayerLinears {
        router,
        gate_up: loader.load_expert_linear::<T, D>(&gate_up_spec, device)?,
        down: loader.load_expert_linear::<T, D>(&down_spec, device)?,
    })
}

/// Assemble one local routed FFN from already-loaded layer tensors.
pub fn assemble_layer_ffn<T, D>(
    post_attention_layernorm: RmsNorm<T, D>,
    linears: Qwen3MoeLayerLinears<T, D>,
) -> OpResult<MoeFfn<T, D>>
where
    T: Dtype,
    D: LlmBackend,
{
    let experts = MoeExperts::new(linears.gate_up, linears.down)?;
    let routed = MoeLocalPipeline::new(linears.router, experts)?;
    MoeFfn::new(post_attention_layernorm, routed)
}

/// Load and assemble one Qwen3-MoE FFN layer without constructing a decoder.
pub fn load_layer_ffn<T, D>(
    loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    layer_index: usize,
    device: &D,
) -> OpResult<MoeFfn<T, D>>
where
    T: Dtype,
    D: OpBackend + LlmBackend,
{
    validate_layer_loading(loader, cfg, layer_index)?;

    let norm_name = post_attention_norm_name(layer_index);
    let norm_view = loader
        .read_view(&norm_name)
        .map_err(|error| OpError::Kernel(format!("tensor '{}': {}", norm_name, error)))?;
    if norm_view.shape() != [cfg.dim] {
        return Err(OpError::Shape(format!(
            "tensor '{}': expected [{}], got {:?}",
            norm_name,
            cfg.dim,
            norm_view.shape()
        )));
    }

    let layer_norm = loader.load_rmsnorm::<T, D>(&norm_name, device, cfg.rms_norm_eps)?;
    let linears = load_layer_linears::<T, D>(loader, cfg, layer_index, device)?;
    assemble_layer_ffn(
        RmsNorm {
            weight: layer_norm.weight,
            eps: layer_norm.eps,
        },
        linears,
    )
}

/// Join already-loaded attention and routed FFN sublayers into one decoder
/// block after checking their shared hidden width and device.
pub fn assemble_layer_block<T, D>(
    attention: Attention<T, D>,
    ffn: MoeFfn<T, D>,
) -> OpResult<Qwen3MoeLayer<T, D>>
where
    T: Dtype,
    D: LlmBackend,
{
    let hidden_features = ffn.routed.hidden_features();
    let q_dim = checked_attention_width("query", attention.head_num, attention.head_dim)?;
    let kv_dim = checked_attention_width("KV", attention.kv_head_num, attention.head_dim)?;
    let qkv_dim = q_dim
        .checked_add(
            kv_dim
                .checked_mul(2)
                .ok_or_else(|| OpError::Shape("qwen3_moe: QKV width overflows".into()))?,
        )
        .ok_or_else(|| OpError::Shape("qwen3_moe: QKV width overflows".into()))?;
    validate_attention_geometry(&attention, hidden_features, qkv_dim)?;

    let attention_device = <D as ExecDevice>::device_id(attention.input_layernorm.weight.device());
    let ffn_device = <D as ExecDevice>::device_id(ffn.post_attention_layernorm.weight.device());
    if attention_device != ffn_device {
        return Err(OpError::Shape(format!(
            "qwen3_moe layer attention belongs to device {}, but FFN belongs to device {}",
            attention_device.0, ffn_device.0
        )));
    }
    if ffn.shared.is_some() {
        return Err(OpError::unsupported(
            attention.input_layernorm.weight.device().name(),
            "qwen3_moe.shared_expert",
        ));
    }

    Ok(DecoderBlock { attention, ffn })
}

/// Load and assemble one complete Qwen3-MoE decoder block. This does not load
/// embeddings, the final norm, the LM head, or any other decoder layer.
pub fn load_layer_block<T, D>(
    loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    layer_index: usize,
    device: &D,
) -> OpResult<Qwen3MoeLayer<T, D>>
where
    T: Dtype,
    D: OpBackend + LlmBackend,
{
    validate_layer_loading(loader, cfg, layer_index)?;
    validate_attention_config(cfg)?;
    let attention = load_layer_attention::<T, D>(loader, cfg, layer_index, device)?;
    let ffn = load_layer_ffn::<T, D>(loader, cfg, layer_index, device)?;
    assemble_layer_block(attention, ffn)
}

/// Assemble a decoder shell from model-level weights and blocks that have
/// already been loaded explicitly by the caller.
///
/// This is deliberately not a checkpoint builder: it never reads a tensor and
/// never loops over layer indices. Loading remains explicit in [`build`].
pub fn assemble_decoder_shell<T, D>(
    global: Qwen3MoeGlobalWeights<T, D>,
    blocks: Vec<Qwen3MoeLayer<T, D>>,
    cfg: &LoadConfig,
) -> OpResult<Qwen3MoeModel<T, D>>
where
    T: Dtype,
    D: LlmBackend,
{
    validate_config(cfg)?;
    validate_attention_config(cfg)?;
    if blocks.len() != cfg.layer_num {
        return Err(OpError::Shape(format!(
            "qwen3_moe decoder shell requires {} blocks, got {}",
            cfg.layer_num,
            blocks.len()
        )));
    }
    validate_global_geometry(&global, cfg)?;

    let dims = model_dims(cfg)?;
    let global_device = <D as ExecDevice>::device_id(global.embed.table.device());
    let mut checked_blocks = Vec::with_capacity(blocks.len());
    for (layer_index, block) in blocks.into_iter().enumerate() {
        let block = assemble_layer_block(block.attention, block.ffn)?;
        validate_block_config(&block, cfg, layer_index, global_device)?;
        checked_blocks.push(block);
    }

    Ok(Decoder {
        embed: global.embed,
        blocks: checked_blocks,
        norm: global.norm,
        lm_head: global.lm_head,
        dims,
        scratch: None,
    })
}

fn model_dims(cfg: &LoadConfig) -> OpResult<ModelDims> {
    let q_dim = checked_attention_width("query", cfg.head_num, cfg.head_dim)?;
    let kv_dim = checked_attention_width("KV", cfg.kv_head_num, cfg.head_dim)?;
    let qkv_dim = q_dim
        .checked_add(
            kv_dim
                .checked_mul(2)
                .ok_or_else(|| OpError::Shape("qwen3_moe: QKV width overflows".into()))?,
        )
        .ok_or_else(|| OpError::Shape("qwen3_moe: QKV width overflows".into()))?;
    let dims = ModelDims {
        dim: cfg.dim,
        q_dim,
        kv_dim,
        qkv_dim,
        intermediate_size: 0,
        vocab_size: cfg.vocab_size,
        head_num: cfg.head_num,
        head_dim: cfg.head_dim,
        kv_head_num: cfg.kv_head_num,
        num_layers: cfg.layer_num,
        num_experts: cfg.num_experts,
        experts_per_tok: cfg.experts_per_tok,
        moe_intermediate_size: cfg.moe_intermediate_size,
        num_shared_experts: 0,
    };
    dims.validate()?;
    Ok(dims)
}

fn validate_global_geometry<T, D>(
    global: &Qwen3MoeGlobalWeights<T, D>,
    cfg: &LoadConfig,
) -> OpResult<()>
where
    T: Dtype,
    D: LlmBackend,
{
    if global.embed.table.shape().as_slice() != [cfg.vocab_size, cfg.dim] {
        return Err(OpError::Shape(format!(
            "qwen3_moe embedding must be [{},{}], got {:?}",
            cfg.vocab_size,
            cfg.dim,
            global.embed.table.shape().as_slice()
        )));
    }
    if global.embed.parallelism().tp().size != 1 {
        return Err(OpError::Kernel(
            "qwen3_moe decoder shell currently requires TP1 embedding".into(),
        ));
    }
    if global.norm.weight.shape().as_slice() != [cfg.dim] {
        return Err(OpError::Shape(format!(
            "qwen3_moe final norm must be [{}], got {:?}",
            cfg.dim,
            global.norm.weight.shape().as_slice()
        )));
    }
    validate_configured_f32("final norm epsilon", global.norm.eps, cfg.rms_norm_eps)?;
    let lm_head_weight = global.lm_head.proj.weight.as_dense().ok_or_else(|| {
        OpError::Shape("qwen3_moe LM head must be dense in the local path".into())
    })?;
    if lm_head_weight.shape().as_slice() != [cfg.vocab_size, cfg.dim] {
        return Err(OpError::Shape(format!(
            "qwen3_moe LM head must be [{},{}], got {:?}",
            cfg.vocab_size,
            cfg.dim,
            lm_head_weight.shape().as_slice()
        )));
    }
    if global.lm_head.proj.parallelism().tp().size != 1 {
        return Err(OpError::Kernel(
            "qwen3_moe decoder shell currently requires TP1 LM head".into(),
        ));
    }
    if let Some(bias) = &global.lm_head.proj.bias
        && bias.shape().as_slice() != [cfg.vocab_size]
    {
        return Err(OpError::Shape(format!(
            "qwen3_moe LM-head bias must be [{}], got {:?}",
            cfg.vocab_size,
            bias.shape().as_slice()
        )));
    }

    let device_id = <D as ExecDevice>::device_id(global.embed.table.device());
    validate_tensor_device(&global.norm.weight, "final norm", device_id)?;
    validate_tensor_device(lm_head_weight, "LM-head weight", device_id)?;
    if let Some(bias) = &global.lm_head.proj.bias {
        validate_tensor_device(bias, "LM-head bias", device_id)?;
    }
    Ok(())
}

fn validate_block_config<T, D>(
    block: &Qwen3MoeLayer<T, D>,
    cfg: &LoadConfig,
    layer_index: usize,
    expected_device: DeviceId,
) -> OpResult<()>
where
    T: Dtype,
    D: LlmBackend,
{
    let attention = &block.attention;
    if attention.head_num != cfg.head_num
        || attention.kv_head_num != cfg.kv_head_num
        || attention.head_dim != cfg.head_dim
    {
        return Err(OpError::Shape(format!(
            "qwen3_moe layer {} attention geometry [{},{},{}] != config [{},{},{}]",
            layer_index,
            attention.head_num,
            attention.kv_head_num,
            attention.head_dim,
            cfg.head_num,
            cfg.kv_head_num,
            cfg.head_dim
        )));
    }
    if attention.sin.shape().as_slice()[0] != cfg.seq_len {
        return Err(OpError::Shape(format!(
            "qwen3_moe layer {} RoPE cache length {} != config {}",
            layer_index,
            attention.sin.shape().as_slice()[0],
            cfg.seq_len
        )));
    }
    validate_configured_f32(
        &format!("layer {layer_index} input norm epsilon"),
        attention.input_layernorm.eps,
        cfg.rms_norm_eps,
    )?;
    validate_configured_f32(
        &format!("layer {layer_index} q_norm epsilon"),
        attention.q_norm.as_ref().unwrap().eps,
        cfg.rms_norm_eps,
    )?;
    validate_configured_f32(
        &format!("layer {layer_index} k_norm epsilon"),
        attention.k_norm.as_ref().unwrap().eps,
        cfg.rms_norm_eps,
    )?;
    validate_configured_f32(
        &format!("layer {layer_index} attention scale"),
        attention.scale,
        1.0 / (cfg.head_dim as f32).sqrt(),
    )?;

    let routed = &block.ffn.routed;
    if routed.hidden_features() != cfg.dim
        || routed.num_experts() != cfg.num_experts
        || routed.top_k() != cfg.experts_per_tok
        || routed.experts().intermediate_features() != cfg.moe_intermediate_size
    {
        return Err(OpError::Shape(format!(
            "qwen3_moe layer {} routed geometry [hidden={}, experts={}, top_k={}, intermediate={}] != config [{},{},{},{}]",
            layer_index,
            routed.hidden_features(),
            routed.num_experts(),
            routed.top_k(),
            routed.experts().intermediate_features(),
            cfg.dim,
            cfg.num_experts,
            cfg.experts_per_tok,
            cfg.moe_intermediate_size
        )));
    }
    if routed.router().renormalize() != cfg.norm_topk_prob {
        return Err(OpError::Shape(format!(
            "qwen3_moe layer {} router renormalize={} != config {}",
            layer_index,
            routed.router().renormalize(),
            cfg.norm_topk_prob
        )));
    }
    if block.ffn.post_attention_layernorm.weight.shape().as_slice() != [cfg.dim] {
        return Err(OpError::Shape(format!(
            "qwen3_moe layer {} post-attention norm must be [{}], got {:?}",
            layer_index,
            cfg.dim,
            block.ffn.post_attention_layernorm.weight.shape().as_slice()
        )));
    }
    validate_configured_f32(
        &format!("layer {layer_index} post-attention norm epsilon"),
        block.ffn.post_attention_layernorm.eps,
        cfg.rms_norm_eps,
    )?;

    let block_device =
        <D as ExecDevice>::device_id(block.attention.input_layernorm.weight.device());
    if block_device != expected_device {
        return Err(OpError::Shape(format!(
            "qwen3_moe layer {} belongs to device {}, expected global device {}",
            layer_index, block_device.0, expected_device.0
        )));
    }
    Ok(())
}

fn validate_configured_f32(name: &str, actual: f32, expected: f32) -> OpResult<()> {
    if actual.to_bits() != expected.to_bits() {
        return Err(OpError::Shape(format!(
            "qwen3_moe {name} {actual} != config {expected}"
        )));
    }
    Ok(())
}

fn component_rmsnorm<T, D>(norm: LayerRmsNorm<T, D>) -> RmsNorm<T, D>
where
    T: Dtype,
    D: OpBackend + LlmBackend,
{
    RmsNorm {
        weight: norm.weight,
        eps: norm.eps,
    }
}

fn checked_attention_width(label: &str, heads: usize, head_dim: usize) -> OpResult<usize> {
    heads.checked_mul(head_dim).ok_or_else(|| {
        OpError::Shape(format!(
            "qwen3_moe: {label} width overflows ({heads} heads * {head_dim})"
        ))
    })
}

fn validate_tensor_shape(
    loader: &WeightLoader<'_>,
    name: &str,
    expected: &[usize],
) -> OpResult<()> {
    let view = loader
        .read_view(name)
        .map_err(|error| OpError::Kernel(format!("tensor '{}': {}", name, error)))?;
    if view.shape() != expected {
        return Err(OpError::Shape(format!(
            "tensor '{}': expected {:?}, got {:?}",
            name,
            expected,
            view.shape()
        )));
    }
    Ok(())
}

fn validate_attention_config(cfg: &LoadConfig) -> OpResult<()> {
    if cfg.dim == 0 || cfg.head_num == 0 || cfg.kv_head_num == 0 || cfg.head_dim == 0 {
        return Err(OpError::Shape(format!(
            "qwen3_moe: attention dimensions must be nonzero, got hidden={}, heads={}, kv_heads={}, head_dim={}",
            cfg.dim, cfg.head_num, cfg.kv_head_num, cfg.head_dim
        )));
    }
    if !cfg.head_num.is_multiple_of(cfg.kv_head_num) {
        return Err(OpError::Shape(format!(
            "qwen3_moe: query head count {} must be divisible by KV head count {}",
            cfg.head_num, cfg.kv_head_num
        )));
    }
    if !cfg.head_dim.is_multiple_of(2) {
        return Err(OpError::Shape(format!(
            "qwen3_moe: head_dim must be even for RoPE, got {}",
            cfg.head_dim
        )));
    }
    if cfg.rotary_dim != cfg.head_dim {
        return Err(OpError::Kernel(format!(
            "qwen3_moe: partial rotary attention is not implemented (rotary_dim={}, head_dim={})",
            cfg.rotary_dim, cfg.head_dim
        )));
    }
    if cfg.seq_len == 0 {
        return Err(OpError::Shape(
            "qwen3_moe: sequence length must be nonzero".into(),
        ));
    }
    if !cfg.rms_norm_eps.is_finite() || cfg.rms_norm_eps <= 0.0 {
        return Err(OpError::Shape(format!(
            "qwen3_moe: rms_norm_eps must be finite and positive, got {}",
            cfg.rms_norm_eps
        )));
    }
    if !cfg.rope_theta.is_finite() || cfg.rope_theta <= 0.0 {
        return Err(OpError::Shape(format!(
            "qwen3_moe: rope_theta must be finite and positive, got {}",
            cfg.rope_theta
        )));
    }
    if cfg.attn_output_gate {
        return Err(OpError::Kernel(
            "qwen3_moe: attention output gating is not implemented".into(),
        ));
    }
    if cfg.linear_attn.is_some() {
        return Err(OpError::Kernel(
            "qwen3_moe: hybrid linear attention is not implemented".into(),
        ));
    }
    Ok(())
}

fn validate_attention_geometry<T, D>(
    attention: &Attention<T, D>,
    hidden_features: usize,
    qkv_features: usize,
) -> OpResult<()>
where
    T: Dtype,
    D: LlmBackend,
{
    if attention.head_num == 0
        || attention.kv_head_num == 0
        || attention.head_dim == 0
        || !attention.head_num.is_multiple_of(attention.kv_head_num)
        || !attention.head_dim.is_multiple_of(2)
    {
        return Err(OpError::Shape(format!(
            "qwen3_moe attention has invalid head geometry heads={}, kv_heads={}, head_dim={}",
            attention.head_num, attention.kv_head_num, attention.head_dim
        )));
    }
    if attention.input_layernorm.weight.shape().as_slice() != [hidden_features] {
        return Err(OpError::Shape(format!(
            "qwen3_moe attention input norm must be [{}], got {:?}",
            hidden_features,
            attention.input_layernorm.weight.shape().as_slice()
        )));
    }

    let qkv_weight = attention.qkv_proj.weight.as_dense().ok_or_else(|| {
        OpError::Shape("qwen3_moe attention QKV weight must be dense in the local path".into())
    })?;
    if qkv_weight.shape().as_slice() != [qkv_features, hidden_features] {
        return Err(OpError::Shape(format!(
            "qwen3_moe attention QKV weight must be [{},{}], got {:?}",
            qkv_features,
            hidden_features,
            qkv_weight.shape().as_slice()
        )));
    }
    let q_features = checked_attention_width("query", attention.head_num, attention.head_dim)?;
    let o_weight = attention.o_proj.weight.as_dense().ok_or_else(|| {
        OpError::Shape("qwen3_moe attention output weight must be dense in the local path".into())
    })?;
    if o_weight.shape().as_slice() != [hidden_features, q_features] {
        return Err(OpError::Shape(format!(
            "qwen3_moe attention output weight must be [{},{}], got {:?}",
            hidden_features,
            q_features,
            o_weight.shape().as_slice()
        )));
    }
    if attention.qkv_proj.bias.is_some() || attention.o_proj.bias.is_some() {
        return Err(OpError::Shape(
            "qwen3_moe attention projections must not have bias".into(),
        ));
    }
    if attention.qkv_proj.parallelism().tp().size != 1
        || attention.o_proj.parallelism().tp().size != 1
    {
        return Err(OpError::Kernel(
            "qwen3_moe attention assembly currently requires TP1".into(),
        ));
    }

    let (q_norm, k_norm) = match (&attention.q_norm, &attention.k_norm) {
        (Some(q_norm), Some(k_norm)) => (q_norm, k_norm),
        _ => {
            return Err(OpError::Shape(
                "qwen3_moe attention requires both q_norm and k_norm".into(),
            ));
        }
    };
    for (name, norm) in [("q_norm", q_norm), ("k_norm", k_norm)] {
        if norm.weight.shape().as_slice() != [attention.head_dim] {
            return Err(OpError::Shape(format!(
                "qwen3_moe attention {name} must be [{}], got {:?}",
                attention.head_dim,
                norm.weight.shape().as_slice()
            )));
        }
    }

    let sin_shape = attention.sin.shape().as_slice();
    let cos_shape = attention.cos.shape().as_slice();
    if sin_shape.len() != 2
        || sin_shape[0] == 0
        || sin_shape[1] != attention.head_dim / 2
        || cos_shape != sin_shape
    {
        return Err(OpError::Shape(format!(
            "qwen3_moe attention RoPE caches must share [seq,{}], got sin={:?}, cos={:?}",
            attention.head_dim / 2,
            sin_shape,
            cos_shape
        )));
    }
    if !attention.scale.is_finite() || attention.scale <= 0.0 {
        return Err(OpError::Shape(format!(
            "qwen3_moe attention scale must be finite and positive, got {}",
            attention.scale
        )));
    }

    let device_id = <D as ExecDevice>::device_id(attention.input_layernorm.weight.device());
    validate_tensor_device(qkv_weight, "QKV weight", device_id)?;
    validate_tensor_device(o_weight, "output weight", device_id)?;
    validate_tensor_device(&q_norm.weight, "q_norm", device_id)?;
    validate_tensor_device(&k_norm.weight, "k_norm", device_id)?;
    validate_tensor_device(&attention.sin, "sin cache", device_id)?;
    validate_tensor_device(&attention.cos, "cos cache", device_id)?;
    Ok(())
}

fn validate_tensor_device<T, D>(
    tensor: &crate::domain::tensor::Tensor<T, D>,
    name: &str,
    expected: DeviceId,
) -> OpResult<()>
where
    T: Dtype,
    D: LlmBackend,
{
    let actual = <D as ExecDevice>::device_id(tensor.device());
    if actual != expected {
        return Err(OpError::Shape(format!(
            "qwen3_moe {name} belongs to device {}, expected {}",
            actual.0, expected.0
        )));
    }
    Ok(())
}

fn layer_prefix(layer_index: usize) -> String {
    format!("model.layers.{layer_index}")
}

fn post_attention_norm_name(layer_index: usize) -> String {
    format!(
        "{}.post_attention_layernorm.weight",
        layer_prefix(layer_index)
    )
}

fn validate_layer_index(cfg: &LoadConfig, layer_index: usize) -> OpResult<()> {
    if layer_index >= cfg.layer_num {
        return Err(OpError::Shape(format!(
            "qwen3_moe: layer index {} is outside 0..{}",
            layer_index, cfg.layer_num
        )));
    }
    Ok(())
}

fn validate_layer_loading(
    loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    layer_index: usize,
) -> OpResult<()> {
    validate_model_loading(loader, cfg)?;
    validate_layer_index(cfg, layer_index)?;
    Ok(())
}

fn validate_model_loading(loader: &WeightLoader<'_>, cfg: &LoadConfig) -> OpResult<()> {
    validate_config(cfg)?;
    if loader.tensor_parallel().size != 1 {
        return Err(OpError::Kernel(format!(
            "qwen3_moe: local model loading requires TP1, got TP{}",
            loader.tensor_parallel().size
        )));
    }
    if cfg.mlp_quant.is_some() || cfg.fp8_block.is_some() {
        return Err(OpError::Kernel(
            "qwen3_moe: quantized local model loading is not implemented".into(),
        ));
    }
    Ok(())
}

fn expert_linear_spec(
    mlp_prefix: &str,
    num_experts: usize,
    projections: &[(&str, usize)],
    input_features: usize,
) -> OpResult<ExpertLinearLoadSpec> {
    let names = (0..num_experts)
        .map(|expert| {
            projections
                .iter()
                .map(|(suffix, _)| format!("{}.experts.{}.{}", mlp_prefix, expert, suffix))
                .collect()
        })
        .collect();
    let widths = projections.iter().map(|(_, rows)| *rows).collect();
    ExpertLinearLoadSpec::new(names, widths, input_features)
}

pub fn build<T, D>(
    loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    device: &D,
) -> OpResult<Qwen3MoeModel<T, D>>
where
    T: Dtype,
    D: OpBackend + LlmBackend,
{
    validate_model_loading(loader, cfg)?;
    validate_attention_config(cfg)?;
    let global = load_global_weights::<T, D>(loader, cfg, device)?;
    let mut blocks = Vec::with_capacity(cfg.layer_num);
    for layer_index in 0..cfg.layer_num {
        blocks.push(load_layer_block::<T, D>(loader, cfg, layer_index, device)?);
    }
    assemble_decoder_shell(global, blocks, cfg)
}

fn validate_config(cfg: &LoadConfig) -> OpResult<()> {
    if cfg.layer_num == 0 {
        return Err(OpError::Shape(
            "qwen3_moe: config layer_num must be > 0".into(),
        ));
    }
    if cfg.vocab_size == 0 {
        return Err(OpError::Shape(
            "qwen3_moe: config vocab_size must be > 0".into(),
        ));
    }
    if cfg.num_experts == 0 {
        return Err(OpError::Shape(
            "qwen3_moe: config num_experts must be > 0".into(),
        ));
    }
    if cfg.experts_per_tok == 0 || cfg.experts_per_tok > cfg.num_experts {
        return Err(OpError::Shape(format!(
            "qwen3_moe: invalid experts_per_tok={} for num_experts={}",
            cfg.experts_per_tok, cfg.num_experts
        )));
    }
    if cfg.moe_intermediate_size == 0 {
        return Err(OpError::Shape(
            "qwen3_moe: moe_intermediate_size must be > 0".into(),
        ));
    }
    if cfg.decoder_sparse_step != 1 {
        return Err(OpError::Kernel(format!(
            "qwen3_moe: only decoder_sparse_step=1 is scaffolded, got {}",
            cfg.decoder_sparse_step
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Qwen3MoeLayerLinears, assemble_decoder_shell, assemble_layer_block, assemble_layer_ffn,
        build, expert_linear_spec, load_global_weights, load_layer_attention, load_layer_block,
        load_layer_ffn, post_attention_norm_name,
    };
    use crate::components::{ExpertLinear, Linear, MoeRouter, RmsNorm};
    use crate::domain::tensor::Tensor;
    use crate::infrastructure::cpu::Cpu;
    use crate::infrastructure::io::SafetensorsReader;
    use crate::models::loader::{LoadConfig, WeightLoader};
    use half::bf16;
    use safetensors::{
        Dtype as SafeDtype,
        tensor::{TensorView, serialize},
    };
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    const EXPERTS: usize = 3;
    const HIDDEN: usize = 4;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 2;
    const Q_DIM: usize = HEADS * HEAD_DIM;
    const KV_DIM: usize = KV_HEADS * HEAD_DIM;
    const INTERMEDIATE: usize = 2;

    struct TempSafetensors(PathBuf);

    impl TempSafetensors {
        fn write(bytes: &[u8]) -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "rustinfer-qwen3-moe-{}-{}.safetensors",
                std::process::id(),
                NEXT_ID.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&path, bytes).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempSafetensors {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_le_bytes())
            .collect()
    }

    fn test_config() -> LoadConfig {
        LoadConfig {
            dim: HIDDEN,
            intermediate_size: 0,
            layer_num: 1,
            head_num: HEADS,
            kv_head_num: KV_HEADS,
            head_dim: HEAD_DIM,
            vocab_size: 8,
            seq_len: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            mlp_quant: None,
            fp8_block: None,
            rotary_dim: HEAD_DIM,
            attn_output_gate: false,
            linear_attn: None,
            num_experts: 2,
            experts_per_tok: 1,
            moe_intermediate_size: INTERMEDIATE,
            norm_topk_prob: true,
            decoder_sparse_step: 1,
        }
    }

    fn tiny_layer_file() -> TempSafetensors {
        let mut tensors = vec![
            (
                "model.embed_tokens.weight".to_string(),
                vec![8, HIDDEN],
                bf16_bytes(&[50.0; 8 * HIDDEN]),
            ),
            (
                "model.norm.weight".to_string(),
                vec![HIDDEN],
                bf16_bytes(&[60.0, 61.0, 62.0, 63.0]),
            ),
            (
                "lm_head.weight".to_string(),
                vec![8, HIDDEN],
                bf16_bytes(&[70.0; 8 * HIDDEN]),
            ),
            (
                "model.layers.0.input_layernorm.weight".to_string(),
                vec![HIDDEN],
                bf16_bytes(&[5.0, 6.0, 7.0, 8.0]),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                vec![Q_DIM, HIDDEN],
                bf16_bytes(&[10.0; Q_DIM * HIDDEN]),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".to_string(),
                vec![KV_DIM, HIDDEN],
                bf16_bytes(&[20.0; KV_DIM * HIDDEN]),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".to_string(),
                vec![KV_DIM, HIDDEN],
                bf16_bytes(&[30.0; KV_DIM * HIDDEN]),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".to_string(),
                vec![HIDDEN, Q_DIM],
                bf16_bytes(&[40.0; HIDDEN * Q_DIM]),
            ),
            (
                "model.layers.0.self_attn.q_norm.weight".to_string(),
                vec![HEAD_DIM],
                bf16_bytes(&[9.0, 10.0]),
            ),
            (
                "model.layers.0.self_attn.k_norm.weight".to_string(),
                vec![HEAD_DIM],
                bf16_bytes(&[13.0, 14.0]),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                vec![HIDDEN],
                bf16_bytes(&[1.0, 2.0, 3.0, 4.0]),
            ),
            (
                "model.layers.0.mlp.gate.weight".to_string(),
                vec![2, HIDDEN],
                bf16_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            ),
        ];
        for (expert, gate, up, down) in [(0, 1.0, 2.0, 3.0), (1, 4.0, 5.0, 6.0)] {
            tensors.push((
                format!("model.layers.0.mlp.experts.{expert}.gate_proj.weight"),
                vec![INTERMEDIATE, HIDDEN],
                bf16_bytes(&[gate; INTERMEDIATE * HIDDEN]),
            ));
            tensors.push((
                format!("model.layers.0.mlp.experts.{expert}.up_proj.weight"),
                vec![INTERMEDIATE, HIDDEN],
                bf16_bytes(&[up; INTERMEDIATE * HIDDEN]),
            ));
            tensors.push((
                format!("model.layers.0.mlp.experts.{expert}.down_proj.weight"),
                vec![HIDDEN, INTERMEDIATE],
                bf16_bytes(&[down; HIDDEN * INTERMEDIATE]),
            ));
        }
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(SafeDtype::BF16, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        TempSafetensors::write(&serialize(views, None).unwrap())
    }

    fn cpu_linears() -> Qwen3MoeLayerLinears<f32, Cpu> {
        let router = MoeRouter::new(
            Linear::new(
                Tensor::from_host_slice(&[0.0f32; EXPERTS * HIDDEN], [EXPERTS, HIDDEN], &Cpu)
                    .unwrap(),
                None,
            ),
            2,
            true,
        )
        .unwrap();
        let gate_up = ExpertLinear::new_dense(
            Tensor::from_host_slice(
                &[0.0f32; EXPERTS * 2 * INTERMEDIATE * HIDDEN],
                [EXPERTS, 2 * INTERMEDIATE, HIDDEN],
                &Cpu,
            )
            .unwrap(),
        )
        .unwrap();
        let down = ExpertLinear::new_dense(
            Tensor::from_host_slice(
                &[0.0f32; EXPERTS * HIDDEN * INTERMEDIATE],
                [EXPERTS, HIDDEN, INTERMEDIATE],
                &Cpu,
            )
            .unwrap(),
        )
        .unwrap();
        Qwen3MoeLayerLinears {
            router,
            gate_up,
            down,
        }
    }

    #[test]
    fn expert_manifest_uses_numeric_expert_order() {
        let spec = expert_linear_spec(
            "model.layers.7.mlp",
            12,
            &[("gate_proj.weight", 3), ("up_proj.weight", 3)],
            5,
        )
        .unwrap();

        assert_eq!(spec.num_experts(), 12);
        assert_eq!(spec.input_features(), 5);
        assert_eq!(spec.output_features(), 6);
        assert_eq!(
            spec.expert_projection_names()[2],
            [
                "model.layers.7.mlp.experts.2.gate_proj.weight",
                "model.layers.7.mlp.experts.2.up_proj.weight"
            ]
        );
        assert_eq!(
            spec.expert_projection_names()[10],
            [
                "model.layers.7.mlp.experts.10.gate_proj.weight",
                "model.layers.7.mlp.experts.10.up_proj.weight"
            ]
        );
        assert_eq!(
            post_attention_norm_name(7),
            "model.layers.7.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn assembles_loaded_linears_into_one_local_moe_ffn() {
        let ffn = assemble_layer_ffn(
            RmsNorm {
                weight: Tensor::from_host_slice(&[1.0f32; HIDDEN], [HIDDEN], &Cpu).unwrap(),
                eps: 1e-6,
            },
            cpu_linears(),
        )
        .unwrap();

        assert_eq!(
            ffn.post_attention_layernorm.weight.shape().as_slice(),
            [HIDDEN]
        );
        assert_eq!(ffn.post_attention_layernorm.eps, 1e-6);
        assert_eq!(ffn.routed.num_experts(), EXPERTS);
        assert_eq!(ffn.routed.top_k(), 2);
        assert_eq!(ffn.routed.hidden_features(), HIDDEN);
        assert_eq!(ffn.routed.experts().intermediate_features(), INTERMEDIATE);
        assert!(ffn.shared.is_none());
        assert!(ffn.scratch.is_none());
    }

    #[test]
    fn layer_assembly_rejects_wrong_norm_width() {
        let err = assemble_layer_ffn(
            RmsNorm {
                weight: Tensor::from_host_slice(&[1.0f32; HIDDEN - 1], [HIDDEN - 1], &Cpu).unwrap(),
                eps: 1e-6,
            },
            cpu_linears(),
        )
        .err()
        .unwrap();

        assert!(err.to_string().contains("norm weight must be [4]"));
    }

    #[test]
    fn loads_and_assembles_one_synthetic_layer() {
        let file = tiny_layer_file();
        let reader = SafetensorsReader::open(file.path()).unwrap();
        let loader = WeightLoader::new(&reader);
        let ffn = load_layer_ffn::<bf16, Cpu>(&loader, &test_config(), 0, &Cpu).unwrap();

        let norm = ffn
            .post_attention_layernorm
            .weight
            .to_host_vec()
            .unwrap()
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        assert_eq!(norm, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ffn.post_attention_layernorm.eps, 1e-6);
        assert_eq!(ffn.routed.num_experts(), 2);
        assert_eq!(ffn.routed.top_k(), 1);

        let gate_up = ffn
            .routed
            .experts()
            .gate_up()
            .weight()
            .to_host_vec()
            .unwrap()
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let mut expected_gate_up = Vec::new();
        for value in [1.0, 2.0, 4.0, 5.0] {
            expected_gate_up.extend(vec![value; INTERMEDIATE * HIDDEN]);
        }
        assert_eq!(gate_up, expected_gate_up);

        let down = ffn
            .routed
            .experts()
            .down()
            .weight()
            .to_host_vec()
            .unwrap()
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        assert_eq!(
            down,
            [
                vec![3.0; HIDDEN * INTERMEDIATE],
                vec![6.0; HIDDEN * INTERMEDIATE],
            ]
            .concat()
        );
        assert!(ffn.shared.is_none());
        assert!(ffn.scratch.is_none());
    }

    #[test]
    fn loads_and_assembles_one_synthetic_decoder_block() {
        let file = tiny_layer_file();
        let reader = SafetensorsReader::open(file.path()).unwrap();
        let loader = WeightLoader::new(&reader);
        let block = load_layer_block::<bf16, Cpu>(&loader, &test_config(), 0, &Cpu).unwrap();

        let input_norm = block
            .attention
            .input_layernorm
            .weight
            .to_host_vec()
            .unwrap()
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        assert_eq!(input_norm, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(block.attention.input_layernorm.eps, 1e-6);
        assert_eq!(block.attention.head_num, HEADS);
        assert_eq!(block.attention.kv_head_num, KV_HEADS);
        assert_eq!(block.attention.head_dim, HEAD_DIM);

        let qkv = block
            .attention
            .qkv_proj
            .weight
            .as_dense()
            .unwrap()
            .to_host_vec()
            .unwrap()
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        assert_eq!(
            qkv,
            [
                vec![10.0; Q_DIM * HIDDEN],
                vec![20.0; KV_DIM * HIDDEN],
                vec![30.0; KV_DIM * HIDDEN],
            ]
            .concat()
        );
        assert_eq!(
            block
                .attention
                .o_proj
                .weight
                .as_dense()
                .unwrap()
                .shape()
                .as_slice(),
            [HIDDEN, Q_DIM]
        );
        assert_eq!(
            block
                .attention
                .q_norm
                .as_ref()
                .unwrap()
                .weight
                .shape()
                .as_slice(),
            [HEAD_DIM]
        );
        assert_eq!(
            block
                .attention
                .k_norm
                .as_ref()
                .unwrap()
                .weight
                .shape()
                .as_slice(),
            [HEAD_DIM]
        );
        assert_eq!(block.attention.sin.shape().as_slice(), [8, HEAD_DIM / 2]);
        assert_eq!(block.attention.cos.shape().as_slice(), [8, HEAD_DIM / 2]);
        assert_eq!(block.ffn.routed.num_experts(), 2);
        assert_eq!(block.ffn.routed.top_k(), 1);
        assert!(block.attention.scratch.is_none());
        assert!(block.ffn.scratch.is_none());
    }

    #[test]
    fn block_assembly_requires_both_qk_norms() {
        let file = tiny_layer_file();
        let reader = SafetensorsReader::open(file.path()).unwrap();
        let loader = WeightLoader::new(&reader);
        let cfg = test_config();
        let mut attention = load_layer_attention::<bf16, Cpu>(&loader, &cfg, 0, &Cpu).unwrap();
        let ffn = load_layer_ffn::<bf16, Cpu>(&loader, &cfg, 0, &Cpu).unwrap();
        attention.k_norm = None;

        let err = assemble_layer_block(attention, ffn).err().unwrap();
        assert!(err.to_string().contains("requires both q_norm and k_norm"));
    }

    #[test]
    fn partial_rotary_attention_remains_disabled() {
        let file = tiny_layer_file();
        let reader = SafetensorsReader::open(file.path()).unwrap();
        let loader = WeightLoader::new(&reader);
        let mut cfg = test_config();
        cfg.rotary_dim = HEAD_DIM / 2;

        let err = load_layer_attention::<bf16, Cpu>(&loader, &cfg, 0, &Cpu)
            .err()
            .unwrap();
        assert!(err.to_string().contains("partial rotary attention"));
    }

    #[test]
    fn full_decoder_build_loads_synthetic_checkpoint() {
        let file = tiny_layer_file();
        let reader = SafetensorsReader::open(file.path()).unwrap();
        let loader = WeightLoader::new(&reader);

        let model = build::<bf16, Cpu>(&loader, &test_config(), &Cpu).unwrap();
        assert_eq!(model.blocks.len(), 1);
        assert_eq!(model.dims.num_layers, 1);
        assert_eq!(model.dims.num_experts, 2);
        assert_eq!(model.dims.experts_per_tok, 1);
        assert!(model.dims.is_moe());
    }

    #[test]
    fn loads_globals_and_assembles_one_layer_decoder_shell() {
        let file = tiny_layer_file();
        let reader = SafetensorsReader::open(file.path()).unwrap();
        let loader = WeightLoader::new(&reader);
        let cfg = test_config();
        let global = load_global_weights::<bf16, Cpu>(&loader, &cfg, &Cpu).unwrap();

        assert_eq!(global.embed.table.shape().as_slice(), [8, HIDDEN]);
        assert_eq!(
            global
                .norm
                .weight
                .to_host_vec()
                .unwrap()
                .into_iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>(),
            vec![60.0, 61.0, 62.0, 63.0]
        );
        let lm_head_weight = global.lm_head.proj.weight.as_dense().unwrap();
        assert_eq!(lm_head_weight.shape().as_slice(), [8, HIDDEN]);
        assert_eq!(lm_head_weight.to_host_vec().unwrap()[0].to_f32(), 70.0);

        let block = load_layer_block::<bf16, Cpu>(&loader, &cfg, 0, &Cpu).unwrap();
        let model = assemble_decoder_shell(global, vec![block], &cfg).unwrap();

        assert_eq!(model.blocks.len(), 1);
        assert_eq!(model.dims.dim, HIDDEN);
        assert_eq!(model.dims.q_dim, Q_DIM);
        assert_eq!(model.dims.kv_dim, KV_DIM);
        assert_eq!(model.dims.qkv_dim, Q_DIM + 2 * KV_DIM);
        assert_eq!(model.dims.intermediate_size, 0);
        assert_eq!(model.dims.vocab_size, 8);
        assert_eq!(model.dims.num_layers, 1);
        assert_eq!(model.dims.num_experts, 2);
        assert_eq!(model.dims.experts_per_tok, 1);
        assert_eq!(model.dims.moe_intermediate_size, INTERMEDIATE);
        assert_eq!(model.dims.num_shared_experts, 0);
        assert!(model.dims.is_moe());
        assert!(model.scratch.is_none());
        assert!(model.blocks[0].attention.scratch.is_none());
        assert!(model.blocks[0].ffn.scratch.is_none());
    }

    #[test]
    fn decoder_shell_requires_every_configured_block() {
        let file = tiny_layer_file();
        let reader = SafetensorsReader::open(file.path()).unwrap();
        let loader = WeightLoader::new(&reader);
        let cfg = test_config();
        let global = load_global_weights::<bf16, Cpu>(&loader, &cfg, &Cpu).unwrap();

        let err = assemble_decoder_shell(global, Vec::new(), &cfg)
            .err()
            .unwrap();
        assert!(err.to_string().contains("requires 1 blocks, got 0"));
    }
}
