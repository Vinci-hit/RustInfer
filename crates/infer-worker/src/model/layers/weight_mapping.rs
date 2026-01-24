// Weight mapping abstraction for different model architectures
//
// Each model architecture is responsible for implementing its own weight mapping
// strategy to handle differences in naming conventions across model families.

use crate::model::ModelLoader;
use crate::base::error::Result;

/// Abstract weight mapping interface for model architectures
///
/// Different model families use different naming conventions for their weights.
/// For example:
/// - Llama/Mistral: "model.layers.{layer}.self_attn.q_proj.weight"
/// - Qwen: Similar to Llama
/// - Phi: Different naming conventions
///
/// Each model architecture should implement this trait to define how to map
/// generic layer names to model-specific safetensor keys.
pub trait WeightMappingAdapter: Send + Sync {
    /// Get the embedding layer weight name
    /// e.g., "model.embed_tokens.weight"
    fn embedding(&self) -> &'static str;

    /// Get the final RMSNorm layer weight name
    /// e.g., "model.norm.weight"
    fn rmsnorm_final(&self) -> &'static str;

    /// Get the classification/language model head weight name
    /// e.g., "lm_head.weight"
    fn cls(&self) -> &'static str;

    /// Format a layer-specific weight name
    ///
    /// # Arguments
    /// * `layer_idx` - The layer index
    /// * `weight_name` - The weight name pattern (e.g., "self_attn.q_proj.weight")
    ///
    /// # Returns
    /// Formatted weight name (e.g., "model.layers.0.self_attn.q_proj.weight")
    fn format_layer_weight(&self, layer_idx: usize, weight_name: &str) -> String;

    /// Query projection weight name
    /// e.g., "self_attn.q_proj.weight"
    fn attn_q(&self) -> &'static str;

    /// Key projection weight name
    /// e.g., "self_attn.k_proj.weight"
    fn attn_k(&self) -> &'static str;

    /// Value projection weight name
    /// e.g., "self_attn.v_proj.weight"
    fn attn_v(&self) -> &'static str;

    /// Output projection weight name
    /// e.g., "self_attn.o_proj.weight"
    fn attn_o(&self) -> &'static str;

    /// FFN gate projection weight name (w1)
    /// e.g., "mlp.gate_proj.weight"
    fn ffn_gate(&self) -> &'static str;

    /// FFN up projection weight name (w3)
    /// e.g., "mlp.up_proj.weight"
    fn ffn_up(&self) -> &'static str;

    /// FFN down projection weight name (w2)
    /// e.g., "mlp.down_proj.weight"
    fn ffn_down(&self) -> &'static str;

    /// Attention RMSNorm weight name
    /// e.g., "input_layernorm.weight"
    fn rmsnorm_attn(&self) -> &'static str;

    /// FFN RMSNorm weight name
    /// e.g., "post_attention_layernorm.weight"
    fn rmsnorm_ffn(&self) -> &'static str;

    /// Verify that all required weights are present in the loader
    ///
    /// Optional method for stricter validation.
    /// Implement if the model needs to validate weight presence.
    fn verify_weights(&self, _loader: &ModelLoader) -> Result<()> {
        Ok(())
    }
}
