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
//! - **KV pressure is event-driven, not heartbeat-driven**. When the
//!   worker's `GlobalKvAllocator::alloc_indices` actually fails it
//!   emits an `AllocFailed{round}` control message. The scheduler
//!   responds at round 0 with RadixTree LRU eviction (≤5% of total
//!   capacity) sent back as `FreeKvIndices`, and at round 1 with
//!   victim preemption — picking decoding / chunked-prefilling
//!   sessions sorted by `(output_len desc, input_len asc)`, marking
//!   their chains finished, flipping them back to `Queued`, and
//!   replying with `Preempt(sequence_ids)`. Heartbeats are
//!   liveness-only and carry no KV occupancy info.
//! - **Preemption decisions live in the scheduler**, not the worker.
//!   The worker is purely passive — it asks for relief on alloc-fail
//!   and free()s the slots that come back.
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
