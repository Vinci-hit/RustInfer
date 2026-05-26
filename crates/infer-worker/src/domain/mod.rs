//! Domain layer — pure business logic, zero FFI, zero I/O.
//!
//! This module defines WHAT the system IS, not HOW it does things.
//! All trait definitions (ports) live here. Infrastructure implements them.

pub mod types;
pub mod tensor;
pub mod ports;
pub mod ops;
pub mod model;
pub mod runtime;

// ─── Re-exports ──────────────────────────────────────────────────────────────
pub use types::{Shape, Strides, Dims, Dtype, DataType, MAX_RANK};
pub use tensor::Tensor;
pub use ports::{Device, Allocator, HostDevice, OpBackend, OpError, OpResult};
pub use model::{LlmModel, ForwardContext};
pub use runtime::KvCache;
