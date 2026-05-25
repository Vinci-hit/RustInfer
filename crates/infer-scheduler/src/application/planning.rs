//! `PlanningSystem` — turn `SessionRepository` state into a serialized
//! `BatchCommand` ready to ship to the worker.
//!
//! Owns:
//! - `policy: Box<dyn SchedulingPolicy>` — pluggable scheduling
//!   policy. Boxed (P2-D in the refactor plan) rather than generic
//!   so the engine isn't generic over a scheduling type parameter.
//! - `builder: BatchBuilder` — stateful command serializer; reuses
//!   internal staging buffers across iterations to avoid alloc churn.
//! - `current_chunk_sizes` — the per-iteration scratch list mapping
//!   `RequestId → tokens scheduled this iteration`. Used by both
//!   `execute_plan` (to populate) and `build_*_batch` (to consume).
//!
//! ## Surface
//!
//! - [`PlanningSystem::schedule`] — pure scheduling decision.
//! - [`PlanningSystem::execute_plan`] — drive KV allocation +
//!   `RequestTable` transitions for the prefill batch entries.
//! - [`PlanningSystem::build_llm_batch`] / `build_diffusion_batch`
//!   — wire-format serialization, with reusable buffers.
//! - [`PlanningSystem::scheduled_segments`] — read-only access to
//!   the current chunk-size list.

use crate::domain::kv_cache_pool::KvCachePool;
use crate::infrastructure::kv_cache::traits::CacheState;
use crate::config::SchedulerConfig;
use crate::domain::inference_session::lifecycle::{
    Decoding, InferenceSession, Prefilling, RequestId,
};
use crate::domain::inference_session::queue::WaitingQueue;
use crate::domain::inference_session::table::{PrefillStartOutcome, RequestLocation, RequestTable};
use crate::domain::policy::traits::{BatchPlan, RunningSet, SchedulingPolicy};
use crate::error::Result;
use crate::infrastructure::transport::codec::MsgPackCodec;
use crate::domain::policy::token_budget::TokenBudget;

use super::batch_builder::BatchBuilder;

/// Scheduling stage with internal builder buffers.
pub struct PlanningSystem {
    policy: Box<dyn SchedulingPolicy>,
    builder: BatchBuilder,
    /// Prefill segments scheduled in the current iteration:
    /// `(request_id, tokens_in_segment)`. Cleared at the head of
    /// every `execute_plan` call.
    current_chunk_sizes: Vec<(RequestId, usize)>,
}

impl PlanningSystem {
    pub fn new(policy: Box<dyn SchedulingPolicy>) -> Self {
        Self {
            policy,
            builder: BatchBuilder::new(),
            current_chunk_sizes: Vec::new(),
        }
    }

    /// Borrow the policy (for diagnostic / metrics paths that need
    /// `policy.name()` etc.). Not for hot-path use.
    pub fn policy(&self) -> &dyn SchedulingPolicy {
        &*self.policy
    }

    /// Run the policy. Pure: produces a [`BatchPlan`] without
    /// touching any state.
    pub fn schedule(
        &self,
        waiting: &WaitingQueue,
        running_set: &RunningSet,
        budget: &TokenBudget,
        cache_state: &CacheState,
    ) -> BatchPlan {
        self.policy.schedule(waiting, running_set, budget, cache_state)
    }

    /// Materialize the policy's `BatchPlan` into KV allocations and
    /// session transitions in the repository.
    ///
    /// For each prefill entry:
    /// - **Continuation**: ask the table to set the next chunk's
    ///   `inflight` window.
    /// - **Fresh prompt**: take the session out of the waiting
    ///   queue, ask the KV manager for blocks (with prefix reuse),
    ///   then commit the transition into `Prefilling`.
    ///
    /// Failures degrade gracefully: KV exhaustion restores the
    /// session to the front of the waiting queue and breaks out
    /// (we'll try again next iteration).
    pub fn execute_plan(
        &mut self,
        plan: &BatchPlan,
        sessions: &mut RequestTable,
        kv: &mut dyn KvCachePool,
    ) -> Result<()> {
        self.current_chunk_sizes.clear();

        for entry in &plan.prefill_batch {
            let scheduled_len = entry.token_range.len();
            if scheduled_len == 0 {
                tracing::warn!("Plan produced zero-length prefill for {}", entry.request_id);
                continue;
            }

            let is_continuation = sessions.location_for_request(&entry.request_id)
                == Some(RequestLocation::Prefilling);

            if is_continuation {
                match sessions.set_prefill_inflight(&entry.request_id, scheduled_len) {
                    Ok(segment) => self.current_chunk_sizes.push((
                        entry.request_id.clone(),
                        segment.segment_end - segment.segment_start,
                    )),
                    Err(e) => tracing::warn!(
                        "Failed to set prefill continuation for {}: {}",
                        entry.request_id,
                        e
                    ),
                }
            } else {
                let seq = match sessions.take_waiting(&entry.request_id) {
                    Ok(seq) => seq,
                    Err(e) => {
                        tracing::warn!(
                            "Plan references non-waiting request {}: {}",
                            entry.request_id,
                            e
                        );
                        continue;
                    }
                };

                let (kv_lease, prefix_match) =
                    match kv.allocate_with_prefix(&seq.meta.input_ids) {
                        Ok(result) => result,
                        Err(e) => {
                            tracing::warn!("KV allocation failed for {}: {}", entry.request_id, e);
                            sessions.restore_waiting_front(seq)?;
                            break;
                        }
                    };

                match sessions.commit_prefill_start(seq, kv_lease, prefix_match, scheduled_len)? {
                    PrefillStartOutcome::Scheduled {
                        request_id, segment, ..
                    } => {
                        self.current_chunk_sizes
                            .push((request_id, segment.segment_end - segment.segment_start));
                    }
                    PrefillStartOutcome::DecodeReady { request_id, .. } => {
                        tracing::debug!(
                            "Request {} moved directly to decoding from prefix cache",
                            request_id
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Read-only access to the iteration's scheduled chunk sizes.
    pub fn scheduled_segments(&self) -> &[(RequestId, usize)] {
        &self.current_chunk_sizes
    }

    /// Build the LLM-mode `Prefill` batch, reusing internal buffers.
    pub fn build_llm_batch(
        &mut self,
        prefilling: &[&InferenceSession<Prefilling>],
        decoding: &[&InferenceSession<Decoding>],
        config: &SchedulerConfig,
        codec: &MsgPackCodec,
    ) -> Result<Vec<u8>> {
        self.builder
            .build_llm_batch(prefilling, decoding, config, codec, &self.current_chunk_sizes)
    }

    /// Build the Diffusion-mode batch.
    pub fn build_diffusion_batch(
        &mut self,
        prefilling: &[&InferenceSession<Prefilling>],
        codec: &MsgPackCodec,
    ) -> Result<Vec<u8>> {
        self.builder
            .build_diffusion_batch(prefilling, codec, &self.current_chunk_sizes)
    }
}
