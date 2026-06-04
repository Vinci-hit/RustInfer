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
/// track `num_prefilling` and `num_decoding` so the policy knows how
/// many sequence slots are already burned, and `prefilling_continuations`
/// so chunked prefill can produce a continuation segment. Decoding
/// sequences are "owned" by the worker for stepping, but they still
/// occupy a batch slot, so admission must count them.
pub struct RunningSet {
    /// Number of sequences currently in prefill phase.
    pub num_prefilling: usize,
    /// Number of sequences currently decoding on the worker. These still
    /// occupy a sequence slot in the worker's batch even though the worker
    /// drives their decode steps, so admission must count them.
    pub num_decoding: usize,
    /// Prefilling sequences that need continuation chunks this iteration.
    /// Each entry is (request_id, remaining_tokens_in_prompt).
    pub prefilling_continuations: Vec<(RequestId, usize)>,
}

impl RunningSet {
    /// Total number of sequence slots already occupied in the worker's
    /// batch. Both prefilling and decoding sequences hold a slot, so the
    /// seq-admission budget (`max_seqs - total()`) must include both —
    /// otherwise the worker's batch can overshoot `max_num_seqs` and the
    /// decode `build_plan` rejects the oversized batch, failing requests.
    pub fn total(&self) -> usize {
        self.num_prefilling + self.num_decoding
    }
}

/// Output of a scheduling decision for one iteration.
#[derive(Debug)]
pub struct BatchPlan {
    /// Requests selected for prefill this iteration.
    pub prefill_batch: Vec<PrefillEntry>,
    /// Total tokens in this iteration (prefill only — the scheduler
    /// does not touch decode tokens).
    pub total_tokens: usize,
}

impl BatchPlan {
    /// Empty plan (nothing to do).
    pub fn empty() -> Self {
        Self {
            prefill_batch: vec![],
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
