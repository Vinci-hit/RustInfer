// src/model/layers/weight_mapping.rs
//
// Weight mapping for different model architectures
// Maps generic layer names to model-specific safetensor keys

/// Maps generic layer names to model-specific weight names in safetensors
///
/// Different model families use different naming conventions for their weights.
/// For example:
/// - Llama: "model.layers.{layer}.self_attn.q_proj.weight"
/// - Qwen2: "model.layers.{layer}.self_attn.q_proj.weight" (same as Llama)
///
/// This struct provides a centralized mapping to handle these variations.
#[derive(Debug, Clone)]
pub struct WeightMapping {
    /// Token embedding layer weight name
    /// e.g., "model.embed_tokens.weight"
    pub embedding: &'static str,

    /// Final RMSNorm layer weight name
    /// e.g., "model.norm.weight"
    pub rmsnorm_final: &'static str,

    /// Classification/language model head weight name
    /// e.g., "lm_head.weight"
    pub cls: &'static str,

    /// Prefix for transformer layers
    /// e.g., "model.layers" (will be formatted as "model.layers.{layer}")
    pub layer_prefix: &'static str,

    // Per-layer weight names (use {layer} placeholder for layer index)

    /// Query projection weight name
    /// e.g., "self_attn.q_proj.weight"
    pub attn_q: &'static str,

    /// Key projection weight name
    /// e.g., "self_attn.k_proj.weight"
    pub attn_k: &'static str,

    /// Value projection weight name
    /// e.g., "self_attn.v_proj.weight"
    pub attn_v: &'static str,

    /// Output projection weight name
    /// e.g., "self_attn.o_proj.weight"
    pub attn_o: &'static str,

    /// FFN gate projection weight name (w1)
    /// e.g., "mlp.gate_proj.weight"
    pub ffn_gate: &'static str,

    /// FFN up projection weight name (w3)
    /// e.g., "mlp.up_proj.weight"
    pub ffn_up: &'static str,

    /// FFN down projection weight name (w2)
    /// e.g., "mlp.down_proj.weight"
    pub ffn_down: &'static str,

    /// Attention RMSNorm weight name
    /// e.g., "input_layernorm.weight"
    pub rmsnorm_attn: &'static str,

    /// FFN RMSNorm weight name
    /// e.g., "post_attention_layernorm.weight"
    pub rmsnorm_ffn: &'static str,
}

impl WeightMapping {
    /// Llama3 weight mapping (standard Llama family)
    pub const LLAMA: WeightMapping = WeightMapping {
        embedding: "model.embed_tokens.weight",
        rmsnorm_final: "model.norm.weight",
        cls: "lm_head.weight",
        layer_prefix: "model.layers",
        attn_q: "self_attn.q_proj.weight",
        attn_k: "self_attn.k_proj.weight",
        attn_v: "self_attn.v_proj.weight",
        attn_o: "self_attn.o_proj.weight",
        ffn_gate: "mlp.gate_proj.weight",
        ffn_up: "mlp.up_proj.weight",
        ffn_down: "mlp.down_proj.weight",
        rmsnorm_attn: "input_layernorm.weight",
        rmsnorm_ffn: "post_attention_layernorm.weight",
    };

    /// Qwen2 weight mapping (uses same naming as Llama)
    pub const QWEN2: WeightMapping = WeightMapping {
        embedding: "model.embed_tokens.weight",
        rmsnorm_final: "model.norm.weight",
        cls: "lm_head.weight",
        layer_prefix: "model.layers",
        attn_q: "self_attn.q_proj.weight",
        attn_k: "self_attn.k_proj.weight",
        attn_v: "self_attn.v_proj.weight",
        attn_o: "self_attn.o_proj.weight",
        ffn_gate: "mlp.gate_proj.weight",
        ffn_up: "mlp.up_proj.weight",
        ffn_down: "mlp.down_proj.weight",
        rmsnorm_attn: "input_layernorm.weight",
        rmsnorm_ffn: "post_attention_layernorm.weight",
    };

    /// Mistral weight mapping (same as Llama)
    pub const MISTRAL: WeightMapping = Self::LLAMA;

    /// Format a layer-specific weight name
    ///
    /// # Arguments
    /// * `layer_idx` - The layer index
    /// * `weight_name` - The weight name pattern (e.g., "self_attn.q_proj.weight")
    ///
    /// # Returns
    /// Formatted weight name (e.g., "model.layers.0.self_attn.q_proj.weight")
    pub fn format_layer_weight(&self, layer_idx: usize, weight_name: &str) -> String {
        format!("{}.{}.{}", self.layer_prefix, layer_idx, weight_name)
    }
}
