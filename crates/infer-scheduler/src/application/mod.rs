//! Application layer — engine orchestrator + 5 Systems.
//!
//! Middle ring of the hexagonal architecture. Consumes the
//! `domain::*` aggregate roots (Session / KvCachePool / WorkerNode /
//! Policy) and the `infrastructure::*` IO adapters, and exposes the
//! user-facing flows: ingest a request, schedule one iteration,
//! drain outputs, drive control events.
//!
//! ## Engine
//!
//! - [`SchedulerEngine`] — top-level orchestrator. Owns the 5
//!   Systems below as fields and drives them from a tokio
//!   event loop in [`event_loop`].
//!
//! ## 5 Systems
//!
//! - [`IngestionSystem`]  — accepts new `FrontendEvent::Infer`
//! - [`PlanningSystem`]   — owns `SchedulingPolicy` + `BatchBuilder`
//! - [`DispatchSystem`]   — owns the worker + frontend transports
//! - [`OutputProcessingSystem`] — terminal-state owner, single
//!   point of KV release
//! - [`ControlEventSystem`] — translates worker control events
//!   into `ControlOutcome` (P1-B)
//!
//! Plus [`cancel`] free helpers and the [`batch_builder`] wire
//! serializer used by `PlanningSystem`.

mod batch_builder;

pub mod admission;
pub mod cancel;
pub mod control_event;
pub mod dispatch;
pub mod engine;
pub mod event_loop;
pub mod ingestion;
pub mod outcomes;
pub mod output;
pub mod planning;

pub use control_event::ControlEventSystem;
pub use dispatch::DispatchSystem;
pub use engine::SchedulerEngine;
pub use ingestion::IngestionSystem;
pub use outcomes::{ControlFlow, ControlOutcome};
pub use output::OutputProcessingSystem;
pub use planning::PlanningSystem;
