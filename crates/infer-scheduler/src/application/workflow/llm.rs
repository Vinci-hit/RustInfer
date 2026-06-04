//! LLM workflow — continuous batching with KV cache management.

use async_trait::async_trait;
use infer_protocol::worker_to_scheduler_data::StepOutput;

use crate::application::dispatch::DispatchSystem;
use crate::application::outcomes::ControlOutcome;
use crate::application::output_fns;
use crate::application::planning::PlanningSystem;
use crate::application::scheduler_event::SchedulerEvent;
use crate::application::workflow::{EngineWorkflow, ResourceContext};
use crate::domain::inference_session::lifecycle::SequenceId;
use crate::domain::inference_session::table::RequestTable;
use crate::domain::policy::traits::{RunningSet, SchedulingPolicy};
use crate::domain::policy::token_budget::TokenBudget;
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
    fn can_schedule(&self, _requests: &RequestTable) -> bool {
        true // LLM: continuous batching — always schedulable
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

        let plan = self.planning.schedule(
            ctx.requests.waiting(),
            &running_set(ctx.requests),
            &token_budget(ctx.config),
        );
        if !plan.has_work()
            && ctx.requests.decoding_len() == 0
            && ctx.requests.prefilling_len() == 0
        {
            return Ok(());
        }
        self.planning
            .execute_plan(&plan, ctx.requests, ctx.radix)?;

        let prefilling_view = ctx.requests.prefilling();
        let batch_data = self.planning.build_batch(
            &prefilling_view,
            ctx.config,
            ctx.codec,
        )?;

        if !batch_data.is_empty() {
            if let Some(first) = prefilling_view.first() {
                tracing::info!(
                    request_id = %first.meta.id,
                    sched_latency_ms = first.meta.arrival_time.elapsed().as_secs_f64() * 1000.0,
                    "TTFT_TRACE: batch sent to worker"
                );
            }
            dispatch.send_batch(batch_data).await?;
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

fn token_budget(config: &crate::config::SchedulerConfig) -> TokenBudget {
    TokenBudget {
        max_tokens: config.max_batch_tokens,
        max_seqs: config.max_num_seqs,
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

    // KV budget tracking — always needed for admission control.
    if !step.assigned_indices.is_empty() {
        let total: u32 = step.assigned_indices.iter().map(|a| a.len as u32).sum();
        let _ = ctx.kv_budget.try_reserve(total);

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
        step.tokens
            .iter()
            .filter(|tk| tk.finished)
            .filter_map(|tk| ctx.requests.kv_slots_for_sequence(SequenceId(tk.sequence_id)))
            .sum()
    } else {
        0
    };

    output_fns::process_llm_step_decoded(
        ctx.requests,
        dispatch.frontend_mut(),
        ctx.metrics,
        step,
    )
    .await?;

    // Real-time recycling: release KV budget for finished sequences.
    // The worker has already moved these blocks to its released list or
    // recycled them into the free pool — the scheduler just aligns its
    // budget count.
    if finished_kv_slots > 0 {
        ctx.kv_budget.release(finished_kv_slots);
    }

    Ok(())
}
