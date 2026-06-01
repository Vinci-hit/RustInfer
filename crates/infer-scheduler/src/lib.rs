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
//! │                   metrics, scheduler-side prefix-cache index,
//! │                   control-plane router thread.
//! ├── config.rs       SchedulerConfig (paged_block_size, limits)
//! ├── error.rs        SchedulerError + Result alias
//! ├── lib.rs          public surface
//! └── main.rs         binary bootstrap
//! ```
//!
//! ## Engine entrypoint
//!
//! [`SchedulerEngine`] is the only top-level type required to spin
//! up the scheduler binary; it is constructed from boxed transport
//! and policy implementations and run via
//! [`SchedulerEngine::run`](application::SchedulerEngine::run).
//!
//! ## Key design principles
//!
//! - **Worker owns physical KV**. The worker's `GlobalKvAllocator`
//!   is the sole authority on per-token slot allocation; the
//!   scheduler never hands out block ids. `StepOutput.assigned_indices`
//!   is the worker's after-the-fact report of which global indices it
//!   consumed for each sequence this step.
//! - **`RadixTree` is the only prefix-reuse structure**. Nodes carry
//!   `owners: HashSet<SeqId>` as a reference count over live
//!   sequences walking the chain; a node is admitted to the LRU iff
//!   `owners.is_empty()` *and* it is a leaf. Eviction yields global
//!   KV indices that the scheduler returns to the worker via
//!   `FreeKvIndices`.
//! - **`KvBudget` is the single capacity gate**.
//!   [`KvBudget`](domain::KvBudget) tracks `outstanding ≤ capacity` over
//!   worker-reported `assigned_indices`. The scheduler does not run a
//!   pre-batch admission cascade — instead, the worker reports KV pool
//!   occupancy in every Heartbeat.
//! - **KV pressure is worker-driven, not scheduler-driven**. When
//!   `kv_free_slots / kv_total_slots` crosses a low-water threshold
//!   the scheduler evicts owners-empty leaves from `RadixTree` and
//!   replies with `FreeKvIndices`. This keeps the scheduler reactive
//!   instead of speculatively guessing pressure.
//! - **Preemption (under KV starvation) is handled inside the worker**;
//!   the scheduler is notified out-of-band (future work) and adjusts
//!   session state. The scheduler does not preempt live decoders.
//! - **Decoding is the worker's domain**. The scheduler tracks
//!   `Decoding` sessions only to route output tokens back to clients
//!   and respond to cancel — it does not pick which decodes run next.
//! - **Typestate lifecycle**. Requests transition through compile-time
//!   verified states ([`InferenceSession<S>`](domain::InferenceSession)
//!   = `Queued` / `Prefilling` / `Decoding` / `Finished` / `Failed`).
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
pub use config::{SchedulerConfig, SchedulerMode};
pub use error::{Result, SchedulerError};
