//! Diffusion workflow — batch-in/batch-out with in-flight gating.

use async_trait::async_trait;
use infer_protocol::worker_to_scheduler_data::DiffusionBatchOutput;

use crate::application::dispatch::DispatchSystem;
use crate::application::outcomes::ControlOutcome;
use crate::application::output_fns;
use crate::application::planning::PlanningSystem;
use crate::application::scheduler_event::SchedulerEvent;
use crate::application::workflow::{EngineWorkflow, ResourceContext};
use crate::domain::inference_session::table::RequestTable;
use crate::domain::policy::token_budget::TokenBudget;
use crate::domain::policy::traits::{RunningSet, SchedulingPolicy};
use crate::error::Result;
use crate::infrastructure::transport::control_plane::{
    ControlEvent, ControlPlaneCmdTx, WorkerGroup, WorkerId,
};

/// Diffusion workflow: at most one batch in-flight at a time.
///
/// After a batch is sent, `in_flight` is set and `can_schedule`
/// returns false. When the worker returns results, `in_flight`
/// is cleared and scheduling resumes.
pub struct DiffusionWorkflow {
    planning: PlanningSystem,
    in_flight: bool,
}

impl DiffusionWorkflow {
    pub fn new(policy: Box<dyn SchedulingPolicy>) -> Self {
        Self {
            planning: PlanningSystem::new(policy),
            in_flight: false,
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
impl EngineWorkflow for DiffusionWorkflow {
    fn can_schedule(&self, _requests: &RequestTable) -> bool {
        !self.in_flight
    }

    fn has_in_flight_batch(&self) -> bool {
        self.in_flight
    }

    async fn try_schedule(
        &mut self,
        ctx: &mut ResourceContext<'_>,
        dispatch: &mut DispatchSystem,
    ) -> Result<()> {
        if self.in_flight {
            return Ok(());
        }
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
        self.planning.execute_plan(&plan, ctx.requests, ctx.radix)?;

        let prefilling_view = ctx.requests.prefilling();
        let batch_data = self
            .planning
            .build_batch(&prefilling_view, ctx.config, ctx.codec)?;

        if !batch_data.is_empty() {
            if let Some(first) = prefilling_view.first() {
                tracing::debug!(
                    request_id = %first.meta.id,
                    sched_latency_ms = first.meta.arrival_time.elapsed().as_secs_f64() * 1000.0,
                    "TTFT_TRACE: batch sent to worker"
                );
            }
            dispatch.send_batch(batch_data).await?;
            self.in_flight = true;
        }
        Ok(())
    }

    async fn handle_step_output(
        &mut self,
        ctx: &mut ResourceContext<'_>,
        dispatch: &mut DispatchSystem,
        event: SchedulerEvent,
    ) -> Result<()> {
        let SchedulerEvent::WorkerDiffusionStep(output) = event else {
            tracing::warn!(
                "DiffusionWorkflow received non-Diffusion step event: {:?}",
                event
            );
            return Ok(());
        };
        self.in_flight = false;
        handle_diffusion_step(ctx, dispatch, &output).await
    }

    fn handle_control_event(
        &self,
        event: ControlEvent,
        ctx: &mut ResourceContext<'_>,
        control_cmd: &ControlPlaneCmdTx,
        worker_group: &WorkerGroup,
        default_worker: &WorkerId,
    ) -> ControlOutcome {
        // Diffusion mode doesn't use KV pressure relief (capacity=0),
        // but we still delegate to the same handler for StepError,
        // WorkerLost, etc.
        crate::application::control_fns::handle_control_event(
            event,
            ctx.requests,
            ctx.radix,
            ctx.kv_budget,
            control_cmd,
            worker_group,
            default_worker,
            false,
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

async fn handle_diffusion_step(
    ctx: &mut ResourceContext<'_>,
    dispatch: &mut DispatchSystem,
    output: &DiffusionBatchOutput,
) -> Result<()> {
    output_fns::process_diffusion_step_decoded(
        ctx.requests,
        dispatch.frontend_mut(),
        ctx.metrics,
        output,
    )
    .await
}
