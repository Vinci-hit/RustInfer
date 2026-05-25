//! Core scheduling policy trait and output types.

use std::ops::Range;

use crate::infrastructure::kv_cache::traits::CacheState;
use crate::domain::inference_session::lifecycle::RequestId;
use crate::domain::inference_session::queue::WaitingQueue;
use crate::domain::policy::token_budget::TokenBudget;

/// A set of currently running (prefilling + decoding) sequences.
///
/// Provides read-only access to the scheduler's running state for policy decisions.
pub struct RunningSet {
    /// Number of sequences currently in prefill phase.
    pub num_prefilling: usize,
    /// Number of sequences currently in decode phase.
    pub num_decoding: usize,
    /// Total decode tokens this iteration (one per decoding sequence).
    pub decode_tokens: usize,
    /// Request IDs of running decode sequences.
    pub running_ids: Vec<RequestId>,
    /// Prefilling sequences that need continuation chunks this iteration.
    /// Each entry is (request_id, remaining_tokens_in_prompt).
    pub prefilling_continuations: Vec<(RequestId, usize)>,
}

impl RunningSet {
    /// Total number of running sequences.
    pub fn total(&self) -> usize {
        self.num_prefilling + self.num_decoding
    }
}

/// Output of a scheduling decision for one iteration.
#[derive(Debug)]
pub struct BatchPlan {
    /// Requests selected for prefill this iteration.
    pub prefill_batch: Vec<PrefillEntry>,
    /// Sequences continuing decode.
    pub decode_batch: Vec<DecodeEntry>,
    /// Sequences to preempt.
    pub preemptions: Vec<PreemptionAction>,
    /// Total tokens in this iteration (prefill + decode).
    pub total_tokens: usize,
}

impl BatchPlan {
    /// Empty plan (nothing to do).
    pub fn empty() -> Self {
        Self {
            prefill_batch: vec![],
            decode_batch: vec![],
            preemptions: vec![],
            total_tokens: 0,
        }
    }

    /// Whether this plan has any work.
    pub fn has_work(&self) -> bool {
        !self.prefill_batch.is_empty() || !self.decode_batch.is_empty()
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

/// A sequence continuing decode in this iteration.
#[derive(Debug)]
pub struct DecodeEntry {
    pub request_id: RequestId,
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
        cache_state: &CacheState,
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
        cache_state: &CacheState,
    ) -> BatchPlan {
        (**self).schedule(waiting, running, budget, cache_state)
    }

    fn name(&self) -> &'static str {
        (**self).name()
    }
}
