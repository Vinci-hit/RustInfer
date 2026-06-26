//! Domain layer — pure business logic, zero FFI, zero I/O.
//!
//! This module defines WHAT the system IS, not HOW it does things.
//! All trait definitions (ports) live here. Infrastructure implements them.

pub use infer_core::component;
pub use infer_core::dtype;
pub use infer_core::exec;
pub mod forward_scratch;
pub mod global_kv_alloc;
pub use infer_core::kv;
#[cfg(test)]
mod kv_tests;
pub mod model;
pub mod plan;
pub use infer_core::ports;
pub use infer_core::storage;
pub use infer_core::tensor;
#[cfg(test)]
mod tensor_tests;
pub use infer_core::types;

// ─── Re-exports ──────────────────────────────────────────────────────────────
pub use component::{Component, Hidden, LayerRange, StageKind};
pub use dtype::{DTypeId, DTypeSpec, Fp8E4m3, Fp8E5m2};
pub use exec::{DeviceId, ExecScope, MaskHandle, QuantTier, Rank, TopologyShape};
pub use forward_scratch::ForwardScratch;
pub use global_kv_alloc::{AllocFull, GlobalKvAllocator};
pub use model::{DecoderModel, Logits, ModelDims, SampleRows};
pub use ports::{
    Allocator, CoreOps, Device, DiffusionOps, HostDevice, MemoryPort, OpBackend, OpError, OpResult,
};
pub use storage::Storage;
pub use tensor::Tensor;
pub use types::{DataType, Dims, Dtype, MAX_RANK, Shape, Strides};
