//! Qwen3 dense model. Model-specific behavior belongs in this file; the shared
//! loader remains name-driven and model-agnostic.

use crate::models::decoder::{Decoder, build_dense_decoder};
use crate::models::loader::{LoadConfig, WeightLoader};
use infer_core::dtype::Dtype;
use infer_core::ports::backend::LlmBackend;
use infer_core::ports::{OpBackend, OpResult};

pub type Qwen3Model<T, D> = Decoder<T, D>;

pub fn build<T, D>(
    loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    device: &D,
) -> OpResult<Qwen3Model<T, D>>
where
    T: Dtype,
    D: OpBackend + LlmBackend,
{
    build_dense_decoder(loader, cfg, device)
}

pub mod eagle3;

pub mod dflash;
