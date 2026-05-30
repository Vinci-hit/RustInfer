//! Domain layer — pure business logic, zero FFI, zero I/O.
//!
//! This module defines WHAT the system IS, not HOW it does things.
//! All trait definitions (ports) live here. Infrastructure implements them.

pub mod types;
pub mod storage;
pub mod tensor;
pub mod ports;
pub mod ops;
pub mod batch;
pub mod batch_workspace;
pub mod global_kv_alloc;
pub mod forward_workspace;
pub mod model;
pub mod runtime;

// ─── Re-exports ──────────────────────────────────────────────────────────────
pub use types::{Shape, Strides, Dims, Dtype, DataType, MAX_RANK};
pub use storage::Storage;
pub use tensor::Tensor;
pub use ports::{Device, Allocator, HostDevice, MemoryPort, OpBackend, OpError, OpResult};
pub use batch::{BatchKind, BatchPlan, PagedKvLayer, PagedKvPool, RAGGED_Q_TILE};
pub use global_kv_alloc::{AllocFull, GlobalKvAllocator};
pub use batch_workspace::{BatchWorkspace, WsSeqStep};
pub use forward_workspace::{ForwardWorkspace, ModelDims};
pub use model::{LlmModel, ForwardContext};
pub use runtime::KvCache;
