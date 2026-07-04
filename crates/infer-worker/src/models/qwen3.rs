//! Qwen3 dense model. Model-specific behavior belongs in this file; the shared
//! loader remains name-driven and model-agnostic.

use crate::domain::dtype::Dtype;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpBackend, OpResult};
use crate::models::decoder::{build_dense_decoder, Decoder};
use crate::models::loader::{LoadConfig, WeightLoader};

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
