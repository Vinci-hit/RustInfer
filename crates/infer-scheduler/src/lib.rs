//! RustInfer Scheduler — production-grade continuous batching scheduler.
//!
//! ## Hexagonal architecture (DDD three-layer)
//!
//! ```text
//! src/
//! ├── domain/         pure business types — aggregate roots, value
//! │                   objects, NewType identifiers, policy traits.
//! │                   No IO, no async runtime.
//! ├── application/    orchestrators — SchedulerEngine + 5 Systems
//! │                   (Ingestion / Planning / Dispatch /
//! │                   OutputProcessing / ControlEvent) + the
//! │                   tokio event loop that drives them.
//! ├── infrastructure/ IO and runtime — ZMQ transports, Prometheus
//! │                   metrics, paged-KV physical block algorithms,
//! │                   control-plane router thread.
//! ├── config.rs       SchedulerConfig + KvCacheMode (paged-only)
//! ├── error.rs        SchedulerError + Result alias
//! ├── lib.rs          public surface
//! └── main.rs         binary bootstrap
//! ```
//!
//! ## Engine entrypoint
//!
//! [`SchedulerEngine`] is the only top-level type required to spin
//! up the scheduler binary; it is constructed from boxed
//! transport / KV / policy implementations and run via
//! [`SchedulerEngine::run`](application::SchedulerEngine::run).
//!
//! ## Key design principles
//!
//! - **Paged-only KV**: contiguous slot mode has been removed; all KV
//!   resource management goes through the paged
//!   [`KvCachePool`](domain::KvCachePool) trait + [`KvLease`](domain::KvLease)
//!   RAII guard.
//! - **Typestate lifecycle**: requests transition through compile-time
//!   verified states ([`InferenceSession<S>`](domain::InferenceSession)).
//! - **Async event-driven**: a single tokio `select!` loop with no
//!   shared mutable state on the hot path.
//! - **Pluggable policies**: [`SchedulingPolicy`](domain::policy::SchedulingPolicy),
//!   [`FrontendTransport`](infrastructure::transport::FrontendTransport),
//!   [`WorkerTransport`](infrastructure::transport::WorkerTransport) are
//!   all `Box<dyn>`.

// ─── Architectural rings ─────────────────────────────────────────────
pub mod application;
pub mod domain;
pub mod infrastructure;

// ─── Foundation ──────────────────────────────────────────────────────
pub mod config;
pub mod error;

// ─── Stable public API ───────────────────────────────────────────────
pub use application::SchedulerEngine;
pub use config::{KvCacheMode, SchedulerConfig, SchedulerMode};
pub use error::{Result, SchedulerError};
