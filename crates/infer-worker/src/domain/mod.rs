//! Worker-specific model contracts, execution requests, and resource rules.
//!
//! Shared tensors, storage, execution types, and backend ports live in
//! `infer_core` and are imported directly from that crate. This module owns
//! the worker's model/cache contracts, KV allocation, and step request state.

pub mod cache;
pub mod draft;
pub mod features;
pub mod forward_scratch;
pub mod gdn_scratch;
pub mod global_kv_alloc;
#[cfg(test)]
mod kv_tests;
pub mod model;
pub mod plan;
pub mod speculative;
pub mod tensor_parallel;
#[cfg(test)]
mod tensor_tests;

// Worker-owned domain types.
pub use forward_scratch::ForwardScratch;
pub use global_kv_alloc::{AllocFull, GlobalKvAllocator};
pub use model::{DecoderModel, Logits, ModelDims, SampleRows};
pub use tensor_parallel::TensorParallelPlacement;

pub(crate) mod draft_scratch;
