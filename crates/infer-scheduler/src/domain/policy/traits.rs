//! Core scheduling policy trait and output types.

use std::ops::Range;

use crate::domain::inference_session::lifecycle::RequestId;
use crate::domain::inference_session::queue::WaitingQueue;
use crate::domain::policy::token_budget::TokenBudget;

/// A snapshot of currently-running sessions handed to the scheduling
/// policy.
///
/// The scheduler **does not schedule decodes** — the worker's
/// sub-scheduler decides which decoding sequences run next. We still
/// track `num_prefilling` so the policy knows how many sequence slots
/// are already burned, and `prefilling_continuations` so chunked
/// prefill can produce a continuation segment. Decoding sequences are
/// "owned" by the worker; the scheduler only routes their tokens back
/// to clients and answers cancel/heartbeat events for them.
pub struct RunningSet {
    /// Number of sequences currently in prefill phase.
    pub num_prefilling: usize,
    /// Prefilling sequences that need continuation chunks this iteration.
    /// Each entry is (request_id, remaining_tokens_in_prompt).
    pub prefilling_continuations: Vec<(RequestId, usize)>,
}

impl RunningSet {
    /// Total number of sessions the policy must reason about. Decoding
    /// is not part of this count — the worker schedules those itself.
    pub fn total(&self) -> usize {
        self.num_prefilling
    }
}

/// Output of a scheduling decision for one iteration.
#[derive(Debug)]
pub struct BatchPlan {
    /// Requests selected for prefill this iteration.
    pub prefill_batch: Vec<PrefillEntry>,
    /// Sequences to preempt. Currently unused on the worker-driven KV
    /// path (worker handles preemption internally); retained for
    /// backwards compatibility in test fixtures.
    pub preemptions: Vec<PreemptionAction>,
    /// Total tokens in this iteration (prefill only — the scheduler
    /// does not touch decode tokens).
    pub total_tokens: usize,
}

impl BatchPlan {
    /// Empty plan (nothing to do).
    pub fn empty() -> Self {
        Self {
            prefill_batch: vec![],
            preemptions: vec![],
            total_tokens: 0,
        }
    }

    /// Whether this plan has any work.
    pub fn has_work(&self) -> bool {
        !self.prefill_batch.is_empty()
    }
}

/// A request selected for prefill in this iteration.
#[derive(Debug)]
pub struct PrefillEntry {
    pub request_id: RequestId,
    /// Which token range of the prompt to prefill (for chunked prefill).
    pub token_range: Range<usize>,
    /// Whether this is a partial chunk (more chunks coming).
    pub is_partial: bool,
}

/// Preemption action to execute.
#[derive(Debug)]
pub enum PreemptionAction {
    /// Recompute: free all KV, move back to waiting queue.
    Recompute { request_id: RequestId },
    /// Swap to CPU (stub).
    Swap { request_id: RequestId },
}

/// The core trait for pluggable scheduling strategies.
///
/// Called once per iteration. The implementation inspects the current state
/// and returns a `BatchPlan` describing what to do.
pub trait SchedulingPolicy: Send + Sync {
    /// Decide what to include in the next batch.
    fn schedule(
        &self,
        waiting: &WaitingQueue,
        running: &RunningSet,
        budget: &TokenBudget,
    ) -> BatchPlan;

    /// Policy name for logging/metrics.
    fn name(&self) -> &'static str;
}

/// Blanket impl: `Box<dyn SchedulingPolicy>` is itself a SchedulingPolicy.
impl SchedulingPolicy for Box<dyn SchedulingPolicy> {
    fn schedule(
        &self,
        waiting: &WaitingQueue,
        running: &RunningSet,
        budget: &TokenBudget,
    ) -> BatchPlan {
        (**self).schedule(waiting, running, budget)
    }

    fn name(&self) -> &'static str {
        (**self).name()
    }
}
