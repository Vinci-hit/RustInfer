//! Llama3 model — a dense decoder assembled from reusable components.
//!
//! Model-specific behavior belongs in this file. The shared loader remains
//! name-driven and model-agnostic.

use crate::domain::dtype::Dtype;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpBackend, OpResult};
use crate::models::decoder::{Decoder, build_dense_decoder};
use crate::models::loader::{LoadConfig, WeightLoader};

pub type Llama3Model<T, D> = Decoder<T, D>;

pub fn build<T, D>(
    loader: &WeightLoader<'_>,
    cfg: &LoadConfig,
    device: &D,
) -> OpResult<Llama3Model<T, D>>
where
    T: Dtype,
    D: OpBackend + LlmBackend,
{
    build_dense_decoder(loader, cfg, device)
}
