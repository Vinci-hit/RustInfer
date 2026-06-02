//! Application layer — engine orchestrator + 5 Systems.
//!
//! Middle ring of the hexagonal architecture. Consumes the
//! `domain::*` aggregate roots (Session / KvBudget / RadixTree /
//! WorkerNode / Policy) and the `infrastructure::*` IO adapters, and
//! exposes the user-facing flows: ingest a request, schedule one
//! iteration, drain outputs, drive control events.
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
//! - [`OutputProcessingSystem`] — terminal-state owner; drives
//!   responses, error fan-out, and `RadixTree` extension from
//!   `StepOutput.assigned_indices`.
//! - [`ControlEventSystem`] — translates worker control events into
//!   a `ControlOutcome` the engine then dispatches; never touches
//!   `OutputProcessingSystem` directly so borrows stay disjoint.
//!   This is also where KV-pressure heartbeats from the worker
//!   trigger RadixTree LRU eviction + `FreeKvIndices` replies.
//!
//! Plus the [`cancel`] free helpers and an internal `batch_builder`
//! wire serializer used by `PlanningSystem`.

mod batch_builder;

pub mod cancel;
pub mod capacity;
pub mod control_event;
pub mod control_fns;
pub mod dispatch;
pub mod engine;
pub mod event_loop;
pub mod ingestion;
pub mod outcomes;
pub mod output;
pub mod output_fns;
pub mod planning;
pub mod scheduler_event;
pub mod workflow;

pub use capacity::{CapacityGate, DiffusionCapacityGate, LlmCapacityGate};
pub use control_event::ControlEventSystem;
pub use dispatch::DispatchSystem;
pub use engine::SchedulerEngine;
pub use ingestion::IngestionSystem;
pub use outcomes::{ControlFlow, ControlOutcome};
pub use output::OutputProcessingSystem;
pub use planning::PlanningSystem;
pub use workflow::{DiffusionWorkflow, EngineWorkflow, LlmWorkflow, ResourceContext};
