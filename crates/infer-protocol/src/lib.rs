//! RustInfer wire protocol — split by direction × plane.
//!
//! Each module name is the contract:
//!   * `_data` modules: high-volume payloads (batch in / tokens out).
//!   * `_control` modules: lifecycle, KV release, liveness, RPC.
//!
//! Every byte that crosses the control plane is wrapped in
//! [`ControlEnvelope`] with a [`RequestId`] for RPC correlation.

pub mod common;
pub mod config;
pub mod control_envelope;
pub mod scheduler_to_server;
pub mod scheduler_to_worker_control;
pub mod scheduler_to_worker_data;
pub mod server_to_scheduler;
pub mod worker_to_scheduler_control;
pub mod worker_to_scheduler_data;

#[cfg(test)]
mod syntax_test;

// Crate-root re-exports limited to cross-cutting types. Direction-specific
// types must be imported through their module to keep the plane visible at
// every use site.
pub use common::{ProtocolError, ProtocolResult};
pub use config::{
    CudaMemoryConfig, RustInferConfig, resolve_model_type, supported_model_types,
    supported_model_types_csv,
};
pub use control_envelope::{ControlEnvelope, RequestId};
pub use scheduler_to_server::{
    ChunkType, ImageOutput, InferenceMetrics, InferenceResponse, ResponseStatus, StreamChunk,
};
pub use server_to_scheduler::{
    CancelReason, CancelRequest as ServerCancelRequest, DiffusionRequest, InferenceModality,
    InferenceRequest, ServerCommand,
};
