//! LLM workflow — continuous batching with KV cache management.

use async_trait::async_trait;
use infer_protocol::scheduler_to_worker_control::{FreeKvIndices, SchedulerControlMessage};
use infer_protocol::worker_to_scheduler_data::{AssignedIndices, StepOutput};
use std::collections::HashMap;

use crate::application::dispatch::DispatchSystem;
use crate::application::outcomes::ControlOutcome;
use crate::application::output_fns;
use crate::application::planning::PlanningSystem;
use crate::application::scheduler_event::SchedulerEvent;
use crate::application::workflow::{EngineWorkflow, ResourceContext};
use crate::domain::inference_session::lifecycle::SequenceId;
use crate::domain::inference_session::table::{Bucket, RequestTable};
use crate::domain::policy::token_budget::TokenBudget;
use crate::domain::policy::traits::{RunningSet, SchedulingPolicy};
use crate::error::Result;
use crate::infrastructure::transport::control_plane::{
    ControlEvent, ControlPlaneCmdTx, WorkerGroup, WorkerId,
};

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
            .set_pending_prefill(ctx.requests.inflight_prefill_tokens());

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
            .build_batch(&prefilling_view, ctx.config, ctx.codec)?;

        if !batch_data.is_empty() {
            if let Some(first) = prefilling_view.first() {
                tracing::info!(
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

    fn handle_control_event(
        &self,
        event: ControlEvent,
        ctx: &mut ResourceContext<'_>,
        control_cmd: &ControlPlaneCmdTx,
        worker_group: &WorkerGroup,
        default_worker: &WorkerId,
    ) -> ControlOutcome {
        crate::application::control_fns::handle_control_event(
            event,
            ctx.requests,
            ctx.radix,
            ctx.kv_budget,
            control_cmd,
            worker_group,
            default_worker,
            ctx.config.enable_prefix_caching,
        )
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
    let future_decode_reserve = ctx.requests.future_decode_reserve_tokens();
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
        u32::try_from(ctx.requests.future_decode_reserve_tokens()).unwrap_or(u32::MAX);
    let required = new_kv_tokens.saturating_add(future_decode_reserve);
    let headroom = ctx.kv_budget.headroom();
    if headroom >= required {
        return;
    }
    let missing = required - headroom;
    let indices = ctx.radix.evict_collect_at_least(missing as usize);
    if indices.is_empty() {
        tracing::debug!(
            required,
            headroom,
            missing,
            "prefix cache proactive eviction found no LRU entries"
        );
        return;
    }
    release_budget_up_to(ctx.kv_budget, indices.len() as u32, reason);
    send_free_indices(ctx, indices, reason);
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
    let step = &sanitized_step;

    // KV budget tracking — always needed for admission control.
    if !step.assigned_indices.is_empty() {
        let total: u32 = step.assigned_indices.iter().map(|a| a.len as u32).sum();
        reserve_reported_kv(ctx.kv_budget, total);

        if enable_prefix {
            output_fns::feed_radix_assigned_indices(ctx.radix, ctx.kv_budget, step);
            for tk in &step.tokens {
                if tk.finished {
                    output_fns::radix_mark_finished(ctx.radix, tk.sequence_id);
                }
            }
        }
    }

    // Collect KV slot counts for finished sequences before they are
    // removed by `process_llm_step_decoded`. Only needed for the
    // real-time recycling path (prefix caching disabled), where the
    // scheduler must release the budget itself instead of waiting for
    // RadixTree LRU eviction.
    let finished_kv_slots: u32 = if !enable_prefix {
        let assigned_slots = assigned_slots_by_sequence(step);
        step.tokens
            .iter()
            .filter(|tk| tk.finished)
            .map(|tk| {
                let before_step = ctx
                    .requests
                    .kv_slots_for_sequence(SequenceId(tk.sequence_id))
                    .unwrap_or(0);
                before_step.saturating_add(
                    assigned_slots
                        .get(&tk.sequence_id)
                        .copied()
                        .unwrap_or_default(),
                )
            })
            .sum()
    } else {
        0
    };

    output_fns::process_llm_step_decoded(ctx.requests, dispatch.frontend_mut(), ctx.metrics, step)
        .await?;

    // Real-time recycling: release KV budget for finished sequences.
    // The worker has already moved these blocks to its released list or
    // recycled them into the free pool — the scheduler just aligns its
    // budget count.
    if finished_kv_slots > 0 {
        release_budget_up_to(ctx.kv_budget, finished_kv_slots, "finished_non_prefix");
    }

    Ok(())
}

fn sanitize_step_output(
    ctx: &mut ResourceContext<'_>,
    step: &StepOutput,
    release_stale_indices: bool,
) -> StepOutput {
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
            send_free_indices(ctx, stale_indices, "stale_step_output");
        }
    }

    StepOutput {
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
    }
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

fn send_free_indices(ctx: &ResourceContext<'_>, indices: Vec<u32>, reason: &'static str) {
    if indices.is_empty() {
        return;
    }
    let len = indices.len();
    let msg = SchedulerControlMessage::FreeKvIndices(FreeKvIndices {
        model_instance_id: ctx.worker_group.model_instance_id.clone(),
        indices,
    });
    if let Err(err) = ctx.control_cmd.send_to(ctx.default_worker, msg) {
        tracing::error!(
            count = len,
            reason,
            "failed to send FreeKvIndices to worker: {}",
            err
        );
    }
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

fn release_budget_up_to(
    kv_budget: &mut crate::domain::kv_budget::KvBudget,
    requested: u32,
    reason: &'static str,
) -> u32 {
    let releasable = requested.min(kv_budget.outstanding());
    if releasable < requested {
        tracing::warn!(
            requested,
            outstanding = kv_budget.outstanding(),
            released = releasable,
            reason,
            "KV budget release exceeds outstanding; clamping"
        );
    }
    if releasable > 0 {
        kv_budget.release(releasable);
    }
    releasable
}
