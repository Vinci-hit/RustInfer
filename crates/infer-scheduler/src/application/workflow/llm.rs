//! LLM workflow — continuous batching with KV cache management.

use async_trait::async_trait;
use infer_protocol::worker_to_scheduler_data::{AssignedIndices, StepOutput};
use std::borrow::Cow;
use std::collections::HashMap;

use crate::application::dispatch::DispatchSystem;
use crate::application::kv_reclaim::{KvReclaimer, SeqKv};
use crate::application::output_fns;
use crate::application::planning::PlanningSystem;
use crate::application::scheduler_event::SchedulerEvent;
use crate::application::workflow::{EngineWorkflow, ResourceContext};
use crate::domain::inference_session::lifecycle::SequenceId;
use crate::domain::inference_session::table::{Bucket, RequestTable, accounting};
use crate::domain::policy::token_budget::TokenBudget;
use crate::domain::policy::traits::{RunningSet, SchedulingPolicy};
use crate::error::Result;

/// Borrow-view for KV release (see `application::kv_reclaim`) — every KV
/// release in this workflow goes through it.
fn reclaimer<'x>(ctx: &'x mut ResourceContext<'_>) -> KvReclaimer<'x> {
    KvReclaimer {
        radix: &mut *ctx.radix,
        kv_budget: &mut *ctx.kv_budget,
        control_cmd: ctx.control_cmd,
        model_instance_id: &ctx.worker_group.model_instance_id,
        enable_prefix_caching: ctx.config.enable_prefix_caching,
    }
}

/// LLM workflow: continuous batching, KV cache management, chunked prefill.
///
/// Always schedulable — the engine's `run_iteration` guards on
/// `has_pending_work()` separately. The policy decides what fits.
pub struct LlmWorkflow {
    planning: PlanningSystem,
}

impl LlmWorkflow {
    pub fn new(policy: Box<dyn SchedulingPolicy>) -> Self {
        Self {
            planning: PlanningSystem::new(policy),
        }
    }

    pub fn planning(&self) -> &PlanningSystem {
        &self.planning
    }

    pub fn planning_mut(&mut self) -> &mut PlanningSystem {
        &mut self.planning
    }
}

#[async_trait]
impl EngineWorkflow for LlmWorkflow {
    fn can_schedule(&self, requests: &RequestTable) -> bool {
        // Continuous batching: schedulable whenever there is pending work.
        //
        // Concurrent prefills no longer need to be serialized: admission
        // reserves projected KV slots via `KvBudget::reserve_pending` when a
        // batch is dispatched, so back-to-back prefill scheduling sees a
        // pressure-accurate `headroom()` and cannot over-commit the worker
        // pool. (The old `!has_inflight_prefill()` gate serialized prefills
        // to dodge the pre-report over-commit window, at the cost of TTFT.)
        requests.has_pending_work()
    }

    fn has_in_flight_batch(&self) -> bool {
        false // LLM does not use in-flight gating
    }

    async fn try_schedule(
        &mut self,
        ctx: &mut ResourceContext<'_>,
        dispatch: &mut DispatchSystem,
    ) -> Result<()> {
        if !ctx.requests.has_pending_work() {
            return Ok(());
        }

        // Recompute the in-flight prefill reservation from live session
        // state before reading headroom. This is what lets us admit new
        // prefills while earlier ones are still unacked without
        // over-committing: the pending footprint of already-dispatched
        // prefills is subtracted from the headroom this decision sees.
        // Recomputing (rather than incrementing/decrementing a counter)
        // means cancelled / preempted / failed sequences self-heal out of
        // the reservation with no leak.
        ctx.kv_budget
            .set_pending_prefill(accounting::inflight_prefill_tokens(ctx.requests));

        let running = running_set(ctx.requests);
        let kv_limited_tokens = prefill_kv_budget(ctx);
        if kv_limited_tokens == 0
            && (!ctx.requests.waiting().is_empty() || !running.prefilling_continuations.is_empty())
        {
            tracing::debug!(
                headroom = ctx.kv_budget.headroom(),
                decoding = running.num_decoding,
                "KV headroom exhausted; deferring prefill scheduling"
            );
            return Ok(());
        }
        let token_budget = TokenBudget {
            max_tokens: ctx.config.max_batch_tokens.min(kv_limited_tokens),
            max_seqs: ctx.config.max_num_seqs,
        };

        let plan = self
            .planning
            .schedule(ctx.requests.waiting(), &running, &token_budget);
        if !plan.has_work()
            && ctx.requests.decoding_len() == 0
            && ctx.requests.prefilling_len() == 0
        {
            return Ok(());
        }
        self.planning.execute_plan(&plan, ctx.requests, ctx.radix)?;

        if ctx.config.enable_prefix_caching {
            let new_kv_tokens = self.planning.scheduled_new_kv_tokens() as u32;
            ensure_worker_kv_headroom(ctx, new_kv_tokens, "pre_dispatch_prefix_evict");
        }

        let prefilling_view = ctx.requests.prefilling();
        let batch_data = self
            .planning
            .build_llm_batch(&prefilling_view, ctx.config, ctx.codec)?;

        if !batch_data.is_empty() {
            if let Some(first) = prefilling_view.first() {
                tracing::debug!(
                    request_id = %first.meta.id,
                    sched_latency_ms = first.meta.arrival_time.elapsed().as_secs_f64() * 1000.0,
                    "TTFT_TRACE: batch sent to worker"
                );
            }
            dispatch.send_batch(batch_data).await?;
            // The pending KV reservation for this batch is not bumped here:
            // the segments are now recorded as `inflight` on their sessions,
            // so the next `try_schedule` recomputes `pending_prefill` from
            // `inflight_prefill_tokens()` and accounts for them automatically.
        }
        Ok(())
    }

    async fn handle_step_output(
        &mut self,
        ctx: &mut ResourceContext<'_>,
        dispatch: &mut DispatchSystem,
        event: SchedulerEvent,
    ) -> Result<()> {
        let SchedulerEvent::WorkerLlmStep(step) = event else {
            tracing::warn!("LlmWorkflow received non-LLM step event: {:?}", event);
            return Ok(());
        };
        handle_llm_step(ctx, dispatch, &step).await
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

fn running_set(requests: &RequestTable) -> RunningSet {
    RunningSet {
        num_prefilling: requests.prefilling_len(),
        num_decoding: requests.decoding_len(),
        prefilling_continuations: requests.prefilling_continuations(),
    }
}

fn prefill_kv_budget(ctx: &ResourceContext<'_>) -> usize {
    let future_decode_reserve = accounting::future_decode_reserve_tokens(ctx.requests);
    if ctx.config.enable_prefix_caching {
        let immediately_free = ctx.kv_budget.headroom() as usize;
        let reclaimable_cache = ctx.radix.lru_total_indices();
        immediately_free
            .saturating_add(reclaimable_cache)
            .saturating_sub(future_decode_reserve)
    } else {
        (ctx.kv_budget.headroom() as usize).saturating_sub(future_decode_reserve)
    }
}

fn ensure_worker_kv_headroom(
    ctx: &mut ResourceContext<'_>,
    new_kv_tokens: u32,
    reason: &'static str,
) {
    if new_kv_tokens == 0 {
        return;
    }
    let future_decode_reserve =
        u32::try_from(accounting::future_decode_reserve_tokens(ctx.requests)).unwrap_or(u32::MAX);
    let required = new_kv_tokens.saturating_add(future_decode_reserve);
    let headroom = ctx.kv_budget.headroom();
    if headroom >= required {
        return;
    }
    let missing = required - headroom;
    let default_worker = ctx.default_worker;
    let freed = reclaimer(ctx).evict_and_free(missing, default_worker, reason);
    if freed == 0 {
        tracing::debug!(
            required,
            headroom,
            missing,
            "prefix cache proactive eviction found no LRU entries"
        );
    }
}

/// Core LLM step-output processing. Separated from the trait impl
/// so it can be called from both the workflow and legacy code paths.
///
/// When `enable_prefix_caching` is true, assigned indices are fed into the
/// RadixTree for prefix-match reuse. When false, the scheduler skips the
/// RadixTree entirely and only tracks the KV budget — the worker owns KV
/// recycling via its `GlobalKvAllocator::release()` / `recycle()` path.
async fn handle_llm_step(
    ctx: &mut ResourceContext<'_>,
    dispatch: &mut DispatchSystem,
    step: &StepOutput,
) -> Result<()> {
    let enable_prefix = ctx.config.enable_prefix_caching;
    let sanitized_step = sanitize_step_output(ctx, step, enable_prefix);
    let step = &*sanitized_step;

    // KV budget tracking — always needed for admission control.
    if !step.assigned_indices.is_empty() {
        let total: u32 = step.assigned_indices.iter().map(|a| a.len as u32).sum();
        reserve_reported_kv(ctx.kv_budget, total);

        if enable_prefix {
            output_fns::feed_radix_assigned_indices(ctx.radix, step);
        }
    }

    // Snapshot `(sequence_id, kv slots)` for sequences finishing this step
    // **before** `process_llm_step_decoded` removes them from the table. Slot
    // counts include the slots assigned in this very step (reserved above but
    // not yet visible on the session).
    let assigned_slots = assigned_slots_by_sequence(step);
    let kv_by_sequence: HashMap<u64, SeqKv> = step
        .tokens
        .iter()
        .map(|tk| {
            let before_step = ctx
                .requests
                .kv_slots_for_sequence(SequenceId(tk.sequence_id))
                .unwrap_or(0);
            SeqKv {
                sequence_id: tk.sequence_id,
                kv_slots: before_step.saturating_add(
                    assigned_slots
                        .get(&tk.sequence_id)
                        .copied()
                        .unwrap_or_default(),
                ),
            }
        })
        .map(|seq| (seq.sequence_id, seq))
        .collect();

    let completed = output_fns::process_llm_step_decoded(
        ctx.requests,
        dispatch.frontend_mut(),
        ctx.metrics,
        step,
    )
    .await?;

    // A scheduler-matched stop sequence finishes before the worker marks the
    // row done. Explicitly cancel that row so worker-owned active/KV state is
    // released and late output is not produced indefinitely.
    for sequence in &completed {
        if sequence.stop_sequence_finished && !sequence.worker_finished {
            crate::application::cancel::send_cancel_to_worker(
                ctx.control_cmd,
                ctx.default_worker,
                sequence.sequence_id,
            )?;
        }
    }
    let finished: Vec<SeqKv> = completed
        .iter()
        .filter_map(|sequence| kv_by_sequence.get(&sequence.sequence_id.0).copied())
        .collect();

    // Terminated-KV reclamation, one entry point for both modes: with prefix
    // caching the chains are marked finished (KV stays cached, budget follows
    // LRU eviction); without it the budget is released now — the worker has
    // already recycled the physical slots on its side.
    if !finished.is_empty() {
        reclaimer(ctx).reclaim_terminated_collect(&finished, 0, "finished");
    }

    Ok(())
}

fn sanitize_step_output<'s>(
    ctx: &mut ResourceContext<'_>,
    step: &'s StepOutput,
    release_stale_indices: bool,
) -> Cow<'s, StepOutput> {
    // Fast path: when every reported sequence is still running there is
    // nothing to drop, so borrow the original instead of cloning the whole
    // StepOutput (assigned_indices carry `token_ids: Vec<i32>` — the bulk of
    // the per-step allocation). Most steps have no stale rows.
    let all_running = step
        .assigned_indices
        .iter()
        .all(|a| is_sequence_running(ctx.requests, a.sequence_id))
        && step
            .prefill_done
            .iter()
            .all(|sid| is_sequence_running(ctx.requests, *sid))
        && step
            .tokens
            .iter()
            .all(|tk| is_sequence_running(ctx.requests, tk.sequence_id));
    if all_running {
        return Cow::Borrowed(step);
    }

    // Slow path: at least one sequence went stale (late output for a
    // cancelled/finished request). Rebuild with the stale rows dropped and
    // optionally hand their KV indices back to the worker.
    let mut stale_indices = Vec::new();
    let assigned_indices: Vec<AssignedIndices> = step
        .assigned_indices
        .iter()
        .filter_map(|assigned| {
            if is_sequence_running(ctx.requests, assigned.sequence_id) {
                Some(assigned.clone())
            } else {
                stale_indices.extend(assigned_indices(assigned));
                None
            }
        })
        .collect();

    if !stale_indices.is_empty() {
        tracing::debug!(
            count = stale_indices.len(),
            release_stale_indices,
            "dropping stale KV indices from late StepOutput"
        );
        if release_stale_indices {
            let default_worker = ctx.default_worker;
            reclaimer(ctx).free_indices_to_worker(
                stale_indices,
                default_worker,
                "stale_step_output",
            );
        }
    }

    Cow::Owned(StepOutput {
        prefill_done: step
            .prefill_done
            .iter()
            .copied()
            .filter(|sid| is_sequence_running(ctx.requests, *sid))
            .collect(),
        tokens: step
            .tokens
            .iter()
            .filter(|tk| is_sequence_running(ctx.requests, tk.sequence_id))
            .cloned()
            .collect(),
        assigned_indices,
    })
}

fn is_sequence_running(requests: &RequestTable, sequence_id: u64) -> bool {
    matches!(
        requests.location_for_sequence(SequenceId(sequence_id)),
        Some(Bucket::Prefilling | Bucket::Decoding)
    )
}

fn assigned_slots_by_sequence(step: &StepOutput) -> HashMap<u64, u32> {
    let mut out = HashMap::new();
    for assigned in &step.assigned_indices {
        let slots = out.entry(assigned.sequence_id).or_insert(0u32);
        *slots = slots.saturating_add(assigned.len as u32);
    }
    out
}

fn assigned_indices(assigned: &AssignedIndices) -> impl Iterator<Item = u32> + '_ {
    assigned.base..assigned.end()
}

fn reserve_reported_kv(kv_budget: &mut crate::domain::kv_budget::KvBudget, requested: u32) {
    if requested == 0 {
        return;
    }
    // Book the worker-reported slots as confirmed outstanding. The matching
    // prefill segment is acked in the same step output, so the next
    // `try_schedule` recomputes `pending_prefill` without this segment —
    // headroom stays consistent (pending drops, outstanding rises).
    if let Err(err) = kv_budget.try_reserve(requested) {
        let headroom = kv_budget.headroom();
        tracing::warn!(
            requested,
            outstanding = err.outstanding,
            capacity = err.capacity,
            headroom,
            "worker reported KV slots beyond scheduler budget; clamping outstanding to capacity"
        );
        if headroom > 0 {
            let _ = kv_budget.try_reserve(headroom);
        }
    }
}
