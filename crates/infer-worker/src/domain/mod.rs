//! Domain layer — pure business logic, zero FFI, zero I/O.
//!
//! This module defines WHAT the system IS, not HOW it does things.
//! All trait definitions (ports) live here. Infrastructure implements them.

pub mod component;
pub mod dtype;
pub mod exec;
pub mod global_kv_alloc;
pub mod kv;
pub mod model;
pub mod plan;
pub mod ports;
pub mod storage;
pub mod tensor;
pub mod types;

// ─── Re-exports ──────────────────────────────────────────────────────────────
pub use component::{Component, Hidden, LayerRange, StageKind};
pub use dtype::{DTypeId, DTypeSpec, Fp8E4m3, Fp8E5m2};
pub use exec::{DeviceId, ExecScope, MaskHandle, QuantTier, Rank, TopologyShape};
pub use global_kv_alloc::{AllocFull, GlobalKvAllocator};
pub use model::{DecoderModel, Logits, ModelDims, SampleRows};
pub use ports::{
    Allocator, CoreOps, Device, DiffusionOps, HostDevice, MemoryPort, OpBackend, OpError, OpResult,
    V2Backend, V2DiffusionOps,
};
pub use storage::Storage;
pub use tensor::Tensor;
pub use types::{DataType, Dims, Dtype, MAX_RANK, Shape, Strides};
