//! Outcomes returned by application-layer Systems.
//!
//! Systems can't directly mutate every other System (P1-B in the
//! refactor plan: `ControlEventSystem` cannot hold `&mut sessions`
//! and `&mut output_system` at the same time without aliasing
//! through `SchedulerEngine`). They return enriched outcome enums
//! that the engine orchestrator interprets and dispatches.
//!
//! ## `ControlOutcome` (P1-B)
//!
//! Result of `ControlEventSystem::handle()`. The engine inspects
//! the variant and either calls `OutputProcessingSystem::fail_sessions`
//! (with a fresh `&mut` borrow) for `Continue`, or unwinds the
//! event loop with `WorkerError` for `Terminate`.

use crate::domain::worker_node::{Lost, WorkerNode};
use crate::error::SchedulerError;
use crate::domain::inference_session::lifecycle::RequestId;

/// Result of processing a control-plane event.
#[derive(Debug)]
pub enum ControlOutcome {
    /// Engine continues. Optional list of sessions whose fail path
    /// the orchestrator must drive (calling `OutputProcessingSystem::
    /// fail_sessions` with the supplied message).
    ///
    /// `failed_request_ids` carries internal `RequestId` (uuid-backed,
    /// Step 8). Callers resolving these into the session repository
    /// hold the only authoritative mapping.
    Continue {
        failed_request_ids: Vec<RequestId>,
        fail_message: Option<String>,
    },
    /// Engine must terminate this iteration.
    ///
    /// Carries a `SchedulerError` that the orchestrator surfaces and
    /// then bubbles out of the event loop. The `lost` snapshot is
    /// optional today: until the engine swaps `WorkerGroup` for
    /// `WorkerNode<Ready>` (Step 18), the System cannot construct
    /// the snapshot itself; once that lands every `Terminate` will
    /// carry `Some(lost)` and downstream consumers can rely on it.
    Terminate {
        lost: Option<WorkerNode<Lost>>,
        error: SchedulerError,
    },
}

impl ControlOutcome {
    /// Convenience: a "no failures, no termination" outcome.
    pub fn noop() -> Self {
        Self::Continue {
            failed_request_ids: Vec::new(),
            fail_message: None,
        }
    }
}

/// Whether the engine event loop should keep iterating.
#[derive(Debug)]
pub enum ControlFlow {
    Continue,
    Terminate(SchedulerError),
}
