//! RustInfer Scheduler — production-grade continuous batching scheduler.
//!
//! ## Architecture
//!
//! The scheduler is the central orchestrator for LLM inference:
//! - Receives tokenized requests from the HTTP server (via ZMQ)
//! - Manages KV cache allocation (Slot or Paged mode)
//! - Applies pluggable scheduling policies (continuous batching, priority, etc.)
//! - Sends batch commands to the GPU worker
//! - Processes step outputs and dispatches responses
//!
//! ## Key Design Principles
//!
//! - **Type-state lifecycle**: requests transition through compile-time verified states
//! - **Async event-driven**: tokio select! loop with zero-lock main task
//! - **Pluggable policies**: scheduling, caching, and transport are all trait-based
//! - **Dual KV mode**: Slot (non-paged) for current compat, Paged for future PagedAttention

pub mod config;
pub mod error;
pub mod core;
pub mod request;
pub mod policy;
pub mod cache;
pub mod transport;
pub mod metrics;
pub mod utils;
pub mod worker_group;

// Re-export key public types.
pub use config::{SchedulerConfig, KvCacheMode, SchedulerMode};
pub use core::SchedulerEngine;
pub use error::{SchedulerError, Result};
pub use worker_group::{WorkerGroup, WorkerGroupState};
