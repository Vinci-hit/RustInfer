//! `PlanningSystem` — turn `RequestTable` state into a serialized
//! `BatchCommand` ready to ship to the worker.
//!
//! Owns:
//! - `policy: Box<dyn SchedulingPolicy>` — pluggable scheduling
//!   policy. Boxed so the engine isn't generic over a scheduling
//!   type parameter.
//! - `builder: BatchBuilder` — stateful command serializer; reuses
//!   internal staging buffers across iterations to avoid alloc churn.
//! - `current_chunk_sizes` — the per-iteration scratch list mapping
//!   `RequestId → tokens scheduled this iteration`. Used by both
//!   `execute_plan` (to populate) and `build_*_batch` (to consume).
//!
//! ## Surface
//!
//! - [`PlanningSystem::schedule`] — pure scheduling decision.
//! - [`PlanningSystem::execute_plan`] — drive RadixTree prefix
//!   pinning + `RequestTable` transitions for the prefill batch
//!   entries.
//! - [`PlanningSystem::build_llm_batch`] / `build_diffusion_batch`
//!   — wire-format serialization, with reusable buffers.
//! - [`PlanningSystem::scheduled_segments`] — read-only access to
//!   the current chunk-size list.

use std::collections::HashMap;

use crate::config::{SchedulerConfig, SchedulerMode};
use crate::domain::inference_session::lifecycle::{InferenceSession, Prefilling, RequestId};
use crate::domain::inference_session::queue::WaitingQueue;
use crate::domain::inference_session::table::{Bucket, PrefillStartOutcome, RequestTable};
use crate::domain::policy::token_budget::TokenBudget;
use crate::domain::policy::traits::{BatchPlan, RunningSet, SchedulingPolicy};
use crate::error::Result;
use crate::infrastructure::kv_cache::radix_tree::{GlobalIndex, RadixTree};
use crate::infrastructure::kv_cache::traits::PrefixMatch;
use crate::infrastructure::transport::codec::MsgPackCodec;

use super::batch_builder::BatchBuilder;

/// Scheduling stage with internal builder buffers.
pub struct PlanningSystem {
    policy: Box<dyn SchedulingPolicy>,
    builder: BatchBuilder,
    /// Prefill segments scheduled in the current iteration:
    /// `(request_id, tokens_in_segment)`. Cleared at the head of
    /// every `execute_plan` call.
    current_chunk_sizes: Vec<(RequestId, usize)>,
    /// Per-iteration prefix-cache hits keyed by request id. Populated
    /// by `execute_plan` (fresh-prompt branch) from
    /// `RadixTree::lookup_prefix`; consumed by `build_llm_batch` to
    /// fill `PrefillSegmentMeta.prefix_hint`. Cleared alongside
    /// `current_chunk_sizes`.
    current_prefix_hints: Vec<(RequestId, Vec<GlobalIndex>)>,
}

impl PlanningSystem {
    pub fn new(policy: Box<dyn SchedulingPolicy>) -> Self {
        Self {
            policy,
            builder: BatchBuilder::new(),
            current_chunk_sizes: Vec::new(),
            current_prefix_hints: Vec::new(),
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
    ) -> BatchPlan {
        self.policy.schedule(waiting, running_set, budget)
    }

    /// Materialize the policy's `BatchPlan` into KV allocations and
    /// session transitions in the repository.
    ///
    /// For each prefill entry:
    /// - **Continuation**: ask the table to set the next chunk's
    ///   `inflight` window.
    /// - **Fresh prompt**: take the session out of the waiting
    ///   queue, ask the `RadixTree` for the longest prefix hit
    ///   (which also pins the matched chain by registering the new
    ///   `SeqId` as an owner), then commit the transition into
    ///   `Prefilling`. The matched indices are stashed for the
    ///   batch builder so they ride out as `prefix_hint` on the
    ///   wire — the worker uses them to skip recomputation.
    ///
    /// Failures degrade gracefully: a missing waiting entry is
    /// logged and skipped (we can't drop blocks on the floor any
    /// more — RadixTree pinning is idempotent on retry).
    pub fn execute_plan(
        &mut self,
        plan: &BatchPlan,
        sessions: &mut RequestTable,
        radix: &mut RadixTree,
    ) -> Result<()> {
        self.current_chunk_sizes.clear();
        self.current_prefix_hints.clear();

        for entry in &plan.prefill_batch {
            let scheduled_len = entry.token_range.len();
            if scheduled_len == 0 {
                tracing::warn!("Plan produced zero-length prefill for {}", entry.request_id);
                continue;
            }

            let is_continuation =
                sessions.location_for_request(&entry.request_id) == Some(Bucket::Prefilling);

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

                // RadixTree-driven prefix reuse. `lookup_prefix` both
                // returns the matched global indices *and* attaches
                // the new sequence as an owner on every node along
                // the matched chain — the chain is pinned for the
                // lifetime of this seq.
                let hit = radix.lookup_prefix(&seq.meta.input_ids, seq.meta.sequence_id.0);
                let mut matched_indices = hit.matched_indices;
                if matched_indices.len() >= seq.meta.input_ids.len() {
                    // A full prompt KV hit is not enough to start decoding:
                    // the worker would have no active sequence and no logits
                    // for the first generated token. Until cached logits or a
                    // no-write prefill path exists, recompute the prompt.
                    radix.mark_finished_chain(seq.meta.sequence_id.0);
                    matched_indices.clear();
                }
                let prefix_match = PrefixMatch {
                    num_cached_tokens: matched_indices.len(),
                };

                match sessions.commit_prefill_start(seq, prefix_match, scheduled_len)? {
                    PrefillStartOutcome::Scheduled {
                        request_id,
                        segment,
                        ..
                    } => {
                        self.current_chunk_sizes.push((
                            request_id.clone(),
                            segment.segment_end - segment.segment_start,
                        ));
                        if !matched_indices.is_empty() {
                            self.current_prefix_hints
                                .push((request_id, matched_indices));
                        }
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

    /// Number of new worker KV slots the current LLM plan needs.
    ///
    /// Prefix hits do not allocate new slots on the worker, so they are
    /// subtracted from the scheduled segment length. Continuation chunks
    /// never carry prefix hints and count in full.
    pub fn scheduled_new_kv_tokens(&self) -> usize {
        let prefix_lengths: HashMap<RequestId, usize> = self
            .current_prefix_hints
            .iter()
            .map(|(id, indices)| (*id, indices.len()))
            .collect();
        self.current_chunk_sizes
            .iter()
            .map(|(request_id, scheduled_len)| {
                let prefix_hit = prefix_lengths.get(request_id).copied().unwrap_or(0);
                scheduled_len.saturating_sub(prefix_hit)
            })
            .sum()
    }

    /// Build the LLM-mode `Prefill` batch, reusing internal buffers.
    /// Decoding is the worker's responsibility; the scheduler only
    /// emits prefill commands.
    pub fn build_llm_batch(
        &mut self,
        prefilling: &[&InferenceSession<Prefilling>],
        config: &SchedulerConfig,
        codec: &MsgPackCodec,
    ) -> Result<Vec<u8>> {
        self.builder.build_llm_batch(
            prefilling,
            config,
            codec,
            &self.current_chunk_sizes,
            &self.current_prefix_hints,
        )
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

    /// Unified batch build — dispatches to LLM or Diffusion internally.
    /// This is the preferred entry point; the mode-specific methods are
    /// retained for backward compat.
    pub fn build_batch(
        &mut self,
        prefilling: &[&InferenceSession<Prefilling>],
        config: &SchedulerConfig,
        codec: &MsgPackCodec,
    ) -> Result<Vec<u8>> {
        match config.mode {
            SchedulerMode::Llm => self.build_llm_batch(prefilling, config, codec),
            SchedulerMode::Diffusion => self.build_diffusion_batch(prefilling, codec),
        }
    }
}
