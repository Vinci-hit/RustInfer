//! Qwen3 MoE model.
//!
//! This file is the ownership boundary for Qwen3 MoE differences: HF identity,
//! MoE config validation, router/expert tensor naming, and eventually the MoE
//! FFN assembly. The shared `WeightLoader` stays model-agnostic.

use crate::components::MoeFfn;
use crate::domain::dtype::Dtype;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpBackend, OpError, OpResult};
use crate::models::decoder::Decoder;
use crate::models::loader::{LoadConfig, WeightLoader};

pub const MODEL_TYPE: &str = "qwen3_moe";
pub const HF_MODEL_TYPES: &[&str] = &["qwen3_moe"];
pub const HF_ARCHITECTURES: &[&str] = &["Qwen3MoeForCausalLM"];

pub type Qwen3MoeModel<T, D> = Decoder<T, D, MoeFfn<T, D>>;

pub fn build<T, D>(
    _loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    _device: &D,
) -> OpResult<Qwen3MoeModel<T, D>>
where
    T: Dtype,
    D: OpBackend + LlmBackend,
{
    validate_config(cfg)?;
    Err(OpError::Kernel(
        "qwen3_moe model scaffold is wired, but MoE weight assembly/forward is not implemented yet"
            .into(),
    ))
}

fn validate_config(cfg: &LoadConfig) -> OpResult<()> {
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
