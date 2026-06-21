//! Outcomes returned by application-layer Systems.
//!
//! Systems can't directly mutate every other System without aliasing
//! through `SchedulerEngine` —
//! `control_fns::handle_control_event` cannot hold `&mut sessions` and
//! `&mut output_system` at the same time, for instance. Instead each
//! System returns an enriched outcome enum that the engine
//! orchestrator interprets and dispatches.
//!
//! ## `ControlOutcome`
//!
//! Result of `control_fns::handle_control_event()`. The engine inspects
//! the variant and either calls `output_fns::fail_sessions`
//! (with a fresh `&mut` borrow) for `Continue`, or unwinds the
//! event loop with `WorkerError` for `Terminate`.

use crate::domain::inference_session::lifecycle::RequestId;
use crate::error::SchedulerError;

/// Result of processing a control-plane event.
#[derive(Debug)]
pub enum ControlOutcome {
    /// Engine continues. Optional list of sessions whose fail path
    /// the orchestrator must drive (calling `output_fns::fail_sessions`
    /// with the supplied message).
    ///
    /// `failed_request_ids` carries internal `RequestId` (uuid-backed).
    /// Callers resolving these into the session repository hold the
    /// only authoritative mapping.
    Continue {
        failed_request_ids: Vec<RequestId>,
        fail_message: Option<String>,
    },
    /// Engine must terminate this iteration.
    ///
    /// Carries a `SchedulerError` that the orchestrator surfaces and
    /// then bubbles out of the event loop.
    Terminate { error: SchedulerError },
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
