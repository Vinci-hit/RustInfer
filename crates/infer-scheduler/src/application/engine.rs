//! SchedulerEngine — the top-level async orchestrator.
//!
//! Owns all mutable state and drives the event loop. Mode-specific
//! logic lives behind the `EngineWorkflow` trait (`LlmWorkflow` /
//! `DiffusionWorkflow`), so the engine itself is mode-agnostic.

use infer_protocol::server_to_scheduler::InferenceRequest;

use crate::domain::kv_budget::KvBudget;
use crate::infrastructure::kv_cache::radix_tree::RadixTree;
use crate::config::SchedulerConfig;
use crate::error::Result;
use crate::infrastructure::metrics::MetricsRecorder;
use crate::domain::policy::traits::SchedulingPolicy;
use crate::domain::inference_session::handle::ClientId;
use crate::domain::inference_session::lifecycle::RequestId;
use crate::domain::inference_session::table::RequestTable;
use crate::infrastructure::transport::codec::MsgPackCodec;
use crate::infrastructure::transport::control_plane::{
    ControlEvent, ControlPlaneCmdTx, ControlPlaneEventRx, WorkerId,
};
use crate::infrastructure::transport::traits::{FrontendTransport, WorkerTransport};
use crate::infrastructure::transport::control_plane::WorkerGroup;
use crate::application::workflow::EngineWorkflow;

/// The main scheduler engine.
pub struct SchedulerEngine {
    // ─── Workflow ───
    /// Mode-specific scheduling and output processing. Owns the
    /// `PlanningSystem` and any mode-specific state (e.g. Diffusion's
    /// in-flight flag).
    workflow: Box<dyn EngineWorkflow>,

    // ─── IO ───
    /// Outbound IO. Holds both transports as boxed trait objects so
    /// the engine can drop its `<F, W>` generics.
    dispatch: crate::application::DispatchSystem,

    // ─── Shared resources ───
    /// Scheduler-side prefix-cache index over worker-owned KV slots.
    radix: RadixTree,
    /// Single capacity gate over the worker's KV pool.
    kv_budget: KvBudget,
    worker_group: WorkerGroup,
    /// Request state.
    requests: RequestTable,

    // ─── Transport ───
    /// Control plane: cancel, drain, FreeKvIndices, Heartbeat, …
    control_cmd: ControlPlaneCmdTx,
    control_events: ControlPlaneEventRx,
    /// Worker identity used for runtime control unicast (single-rank
    /// deployment; populated at construction from the bootstrap
    /// registry).
    default_worker: WorkerId,
    codec: MsgPackCodec,

    // ─── Metrics ───
    metrics: MetricsRecorder,

    // ─── Config ───
    config: SchedulerConfig,

    // ─── State ───
    iteration_id: u64,
    /// New-request ingestion stage. Owns the monotonic SequenceId counter.
    ingestion: crate::application::IngestionSystem,
}

impl SchedulerEngine {
    /// Create a new scheduler engine.
    pub fn new<F, W>(
        config: SchedulerConfig,
        policy: Box<dyn SchedulingPolicy>,
        worker_group: WorkerGroup,
        frontend: F,
        worker: W,
        control_cmd: ControlPlaneCmdTx,
        control_events: ControlPlaneEventRx,
        default_worker: WorkerId,
    ) -> Self
    where
        F: FrontendTransport,
        W: WorkerTransport,
    {
        use crate::config::SchedulerMode;

        tracing::info!(
            "SchedulerEngine created: policy={}, worker_group={}, ranks={}, max_seqs={}, max_tokens={}",
            policy.name(),
            worker_group.group_id,
            worker_group.rank_count(),
            config.max_num_seqs,
            config.max_batch_tokens,
        );

        let workflow: Box<dyn EngineWorkflow> = match config.mode {
            SchedulerMode::Llm => Box::new(
                crate::application::workflow::LlmWorkflow::new(policy),
            ),
            SchedulerMode::Diffusion => Box::new(
                crate::application::workflow::DiffusionWorkflow::new(policy),
            ),
        };

        Self {
            workflow,
            dispatch: crate::application::DispatchSystem::new(
                Box::new(frontend),
                Box::new(worker),
            ),
            radix: RadixTree::new(),
            kv_budget: KvBudget::new(
                u32::try_from(
                    worker_group
                        .effective_capacity
                        .max_total_kv_tokens
                        .unwrap_or(0),
                )
                .unwrap_or(u32::MAX),
            ),
            worker_group,
            requests: RequestTable::new(),
            control_cmd,
            control_events,
            default_worker,
            codec: MsgPackCodec,
            metrics: MetricsRecorder::new(config.metrics_enabled),
            config,
            iteration_id: 0,
            ingestion: crate::application::IngestionSystem::new(),
        }
    }

    /// Run the scheduler event loop.
    pub async fn run(mut self) -> Result<()> {
        tracing::info!("SchedulerEngine starting event loop...");
        crate::application::event_loop::run_event_loop(&mut self).await
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Public methods used by event_loop
    // ═══════════════════════════════════════════════════════════════════════════

    /// Handle an incoming request from the frontend.
    pub(crate) fn handle_new_request(&mut self, client_id: ClientId, request: InferenceRequest) {
        use crate::application::ingestion::{IngestOutcome, RejectReason};

        let external_id = request.request_id.clone();
        let outcome = self.ingestion.ingest(client_id, request, &self.config, &mut self.requests);
        match outcome {
            IngestOutcome::Admitted { request_id, .. } => {
                tracing::info!(
                    %request_id,
                    %external_id,
                    "TTFT_TRACE: scheduler received request"
                );
                self.metrics.record_enqueue();
            }
            IngestOutcome::Rejected { reason, .. } => {
                let msg = reason.as_message();
                match reason {
                    RejectReason::Repository(_) => {
                        tracing::error!(%external_id, "ingestion repository rejection: {}", msg);
                    }
                    _ => {
                        tracing::warn!(%external_id, "rejecting request: {}", msg);
                    }
                }
            }
        }
    }

    /// Run one scheduling iteration — delegates to the workflow.
    pub(crate) async fn run_iteration(&mut self) -> Result<()> {
        if !self.workflow.can_schedule(&self.requests) {
            return Ok(());
        }
        self.iteration_id += 1;

        let SchedulerEngine {
            workflow,
            dispatch,
            requests,
            radix,
            kv_budget,
            metrics,
            codec,
            config,
            ..
        } = self;

        let mut ctx = crate::application::workflow::ResourceContext {
            requests,
            radix,
            kv_budget,
            metrics,
            codec,
            config,
        };
        workflow.try_schedule(&mut ctx, dispatch).await
    }

    /// Handle step output from the worker — delegates to the workflow.
    pub(crate) async fn handle_step_output(&mut self, data: Vec<u8>) -> Result<()> {
        use crate::config::SchedulerMode;
        use crate::infrastructure::transport::codec::Codec;
        use crate::application::scheduler_event::SchedulerEvent;

        let event = match self.config.mode {
            SchedulerMode::Llm => {
                let output: infer_protocol::worker_to_scheduler_data::StepOutput =
                    self.codec.decode(&data)?;
                SchedulerEvent::WorkerLlmStep(output)
            }
            SchedulerMode::Diffusion => {
                let output: infer_protocol::worker_to_scheduler_data::DiffusionBatchOutput =
                    self.codec.decode(&data)?;
                SchedulerEvent::WorkerDiffusionStep(output)
            }
        };

        let SchedulerEngine {
            workflow,
            dispatch,
            requests,
            radix,
            kv_budget,
            metrics,
            codec,
            config,
            ..
        } = self;

        let mut ctx = crate::application::workflow::ResourceContext {
            requests,
            radix,
            kv_budget,
            metrics,
            codec,
            config,
        };
        workflow
            .handle_step_output(&mut ctx, dispatch, event)
            .await
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Control-plane event dispatch
    // ═══════════════════════════════════════════════════════════════════════════

    /// Dispatch a single control event into the workflow.
    pub(crate) async fn on_control_event(&mut self, event: ControlEvent) -> Result<()> {
        use crate::application::ControlOutcome;

        let default_worker = self.default_worker.clone();
        let control_cmd = self.control_cmd.clone();

        let outcome = {
            let SchedulerEngine {
                workflow,
                requests,
                radix,
                kv_budget,
                metrics,
                codec,
                config,
                worker_group,
                ..
            } = self;

            let mut ctx = crate::application::workflow::ResourceContext {
                requests,
                radix,
                kv_budget,
                metrics,
                codec,
                config,
            };
            workflow.handle_control_event(
                event,
                &mut ctx,
                &control_cmd,
                worker_group,
                &default_worker,
            )
        };
        match outcome {
            ControlOutcome::Continue {
                failed_request_ids,
                fail_message,
            } => {
                if !failed_request_ids.is_empty() {
                    let msg = fail_message.unwrap_or_else(|| "worker step error".to_string());
                    self.fail_request_ids(&failed_request_ids, &msg).await?;
                }
                Ok(())
            }
            ControlOutcome::Terminate { lost: _, error } => {
                let running: Vec<RequestId> = self
                    .requests
                    .running_sequence_ids()
                    .into_iter()
                    .filter_map(|sid| self.requests.request_id_for_sequence(sid))
                    .collect();
                if !running.is_empty() {
                    let msg = error.to_string();
                    let _ = self.fail_request_ids(&running, &msg).await;
                }
                Err(error)
            }
        }
    }

    /// Drive the failure path for a list of internal `RequestId`s.
    async fn fail_request_ids(
        &mut self,
        failed_request_ids: &[RequestId],
        message: &str,
    ) -> Result<()> {
        crate::application::output_fns::fail_sessions(
            &mut self.requests,
            self.dispatch.frontend_mut(),
            failed_request_ids,
            message,
        )
        .await
    }

    /// LLM mode: process prefill segment ACKs and generated tokens from Worker.
    ///
    /// Retained for test compatibility. Production code uses
    /// [`Self::handle_step_output`] which dispatches via the workflow.
    #[cfg(test)]
    pub(crate) async fn handle_step_output_llm(&mut self, data: Vec<u8>) -> Result<()> {
        use crate::infrastructure::transport::codec::Codec;
        use crate::application::scheduler_event::SchedulerEvent;

        let output: infer_protocol::worker_to_scheduler_data::StepOutput =
            self.codec.decode(&data)?;
        let event = SchedulerEvent::WorkerLlmStep(output);

        let SchedulerEngine {
            workflow,
            dispatch,
            requests,
            radix,
            kv_budget,
            metrics,
            codec,
            config,
            ..
        } = self;

        let mut ctx = crate::application::workflow::ResourceContext {
            requests,
            radix,
            kv_budget,
            metrics,
            codec,
            config,
        };
        workflow
            .handle_step_output(&mut ctx, dispatch, event)
            .await
    }

    // ─── Accessors for event_loop ───

    pub(crate) fn has_pending_work(&self) -> bool {
        self.requests.has_pending_work()
    }

    pub(crate) fn has_in_flight_batch(&self) -> bool {
        self.workflow.has_in_flight_batch()
    }

    pub(crate) fn can_schedule(&self) -> bool {
        self.workflow.can_schedule(&self.requests)
    }

    /// Cancel by client-supplied external id.
    pub(crate) async fn cancel_request_by_external_id(&mut self, external_id: &str) -> Result<()> {
        crate::application::cancel::cancel_request_by_external_id(
            &mut self.requests,
            &self.control_cmd,
            &self.default_worker,
            external_id,
        )
        .await
    }

    /// Poll for the next event from frontend, worker, or the control plane.
    pub(crate) async fn poll_next_event(&mut self) -> crate::application::event_loop::EngineEvent {
        use crate::application::event_loop::EngineEvent;

        let has_work = self.has_pending_work() || self.has_in_flight_batch();

        if has_work {
            let (frontend, worker) = self.dispatch.borrow_both_mut();
            let control_events = &mut self.control_events;

            tokio::select! {
                biased;
                Some(ev) = control_events.recv() => EngineEvent::Control(ev),
                result = worker.recv_step_output() => EngineEvent::WorkerOutput(result),
                result = frontend.recv_event() => EngineEvent::Frontend(Box::new(result)),
            }
        } else {
            let frontend = self.dispatch.frontend_mut();
            let control_events = &mut self.control_events;

            tokio::select! {
                biased;
                Some(ev) = control_events.recv() => EngineEvent::Control(ev),
                result = frontend.recv_event() => EngineEvent::Frontend(Box::new(result)),
            }
        }
    }
}


#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
