//! Domain layer — pure business logic, zero FFI, zero I/O.
//!
//! This module defines WHAT the system IS, not HOW it does things.
//! All trait definitions (ports) live here. Infrastructure implements them.

pub mod batch;
pub mod global_kv_alloc;
pub mod model;
pub mod ports;
pub mod storage;
pub mod tensor;
pub mod types;

// ─── Re-exports ──────────────────────────────────────────────────────────────
pub use batch::{BatchKind, BatchPlan, PagedKvLayer, PagedKvPool, RAGGED_Q_TILE};
pub use global_kv_alloc::{AllocFull, GlobalKvAllocator};
pub use model::{ForwardContext, LlmForwardWorkspace, LlmModel};
pub use ports::{
    Allocator, CoreOps, Device, DiffusionOps, HostDevice, LlmOps, MemoryPort, OpBackend, OpError,
    OpResult,
};
pub use storage::Storage;
pub use tensor::Tensor;
pub use types::{DataType, Dims, Dtype, MAX_RANK, Shape, Strides};
