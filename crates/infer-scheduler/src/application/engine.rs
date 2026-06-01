//! SchedulerEngine — the top-level async orchestrator.
//!
//! Owns all mutable state and drives the event loop.

use infer_protocol::server_to_scheduler::InferenceRequest;

use crate::domain::kv_budget::KvBudget;
use crate::infrastructure::kv_cache::radix_tree_v2::RadixTree;
use crate::config::{SchedulerConfig, SchedulerMode};
use crate::error::Result;
use crate::infrastructure::metrics::MetricsRecorder;
use crate::domain::policy::traits::{RunningSet, SchedulingPolicy};
use crate::domain::inference_session::handle::ClientId;
use crate::domain::inference_session::lifecycle::RequestId;
use crate::domain::inference_session::table::RequestTable;
use crate::infrastructure::transport::codec::MsgPackCodec;
use crate::infrastructure::transport::control_plane::{
    ControlEvent, ControlPlaneCmdTx, ControlPlaneEventRx, WorkerId,
};
use crate::infrastructure::transport::traits::{FrontendEvent, FrontendTransport, WorkerTransport};
use crate::domain::policy::token_budget::TokenBudget;
use crate::infrastructure::transport::control_plane::WorkerGroup;

/// The main scheduler engine.
pub struct SchedulerEngine {
    // ─── Subsystems ───
    /// Scheduling policy + batch builder. Owns the `Box<dyn
    /// SchedulingPolicy>` so the engine itself stays free of a
    /// scheduling type parameter.
    planning: crate::application::PlanningSystem,
    /// Outbound IO. Holds both transports as boxed trait objects so
    /// the engine can drop its `<F, W>` generics.
    dispatch: crate::application::DispatchSystem,
    /// Scheduler-side prefix-cache index over worker-owned KV slots.
    /// Populated from `StepOutput.assigned_indices`; consulted by
    /// `lookup_prefix` during planning and drained by the admission
    /// cascade's LRU eviction path.
    radix: RadixTree,
    /// Single capacity gate over the worker's KV pool. `try_reserve` /
    /// `release` / `headroom` are the admission cascade's only
    /// counters.
    kv_budget: KvBudget,
    worker_group: WorkerGroup,

    // ─── Request state ───
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
    /// Diffusion batch-in batch-out gate. LLM mode does not use this as backpressure.
    worker_busy: bool,
    /// New-request ingestion stage. Owns the monotonic SequenceId counter.
    ingestion: crate::application::IngestionSystem,
    /// Terminal-state output owner (success / failure / cleanup).
    /// Stateless today; engine still owns the borrowed resources.
    output: crate::application::OutputProcessingSystem,
    /// Control-plane event handler. Returns a `ControlOutcome` that
    /// the engine then dispatches; this split keeps `ControlEventSystem`
    /// from needing a simultaneous `&mut` to `OutputProcessingSystem`.
    control: crate::application::ControlEventSystem,
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
        tracing::info!(
            "SchedulerEngine created: policy={}, worker_group={}, ranks={}, max_seqs={}, max_tokens={}",
            policy.name(),
            worker_group.group_id,
            worker_group.rank_count(),
            config.max_num_seqs,
            config.max_batch_tokens,
        );

        Self {
            planning: crate::application::PlanningSystem::new(policy),
            dispatch: crate::application::DispatchSystem::new(
                Box::new(frontend),
                Box::new(worker),
            ),
            // RadixTree starts empty; admission populates it from
            // `StepOutput.assigned_indices`. KvBudget capacity is taken
            // from the worker's reported `max_total_kv_tokens` if known,
            // else 0 (a 0-capacity budget skips the admission cascade,
            // useful for diffusion mode and tests).
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
            worker_busy: false,
            ingestion: crate::application::IngestionSystem::new(),
            output: crate::application::OutputProcessingSystem::new(),
            control: crate::application::ControlEventSystem::new(),
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
    ///
    /// Delegates validation + admission to
    /// [`crate::application::IngestionSystem`]. The engine itself only
    /// stays responsible for metrics + tracing, which it drives off
    /// the returned [`crate::application::IngestOutcome`].
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

    /// Run one scheduling iteration.
    ///
    /// 1. Skip if diffusion-busy or no pending work.
    /// 2. Ask `PlanningSystem` for a [`BatchPlan`] (prefill only —
    ///    decoding is the worker's domain).
    /// 3. Execute the plan (RadixTree prefix-pinning + session
    ///    transitions).
    /// 4. Serialize batch and ship via `DispatchSystem`.
    ///
    /// KV pressure handling no longer happens here: the worker reports
    /// pool occupancy in every Heartbeat and the scheduler reacts to
    /// low-water signals out-of-band in
    /// [`Self::on_control_event`].
    pub(crate) async fn run_iteration(&mut self) -> Result<()> {
        if self.config.mode == SchedulerMode::Diffusion && self.worker_busy {
            return Ok(());
        }
        if !self.requests.has_pending_work() {
            return Ok(());
        }
        self.iteration_id += 1;

        let plan = self.planning.schedule(
            self.requests.waiting(),
            &self.running_set(),
            &self.token_budget(),
        );
        if !plan.has_work()
            && self.requests.decoding_len() == 0
            && self.requests.prefilling_len() == 0
        {
            return Ok(());
        }
        self.planning
            .execute_plan(&plan, &mut self.requests, &mut self.radix)?;

        let prefilling_view = self.requests.prefilling();
        let batch_data = match self.config.mode {
            SchedulerMode::Llm => self.planning.build_llm_batch(
                &prefilling_view,
                &self.config,
                &self.codec,
            )?,
            SchedulerMode::Diffusion => self
                .planning
                .build_diffusion_batch(&prefilling_view, &self.codec)?,
        };

        if !batch_data.is_empty() {
            if let Some(first) = prefilling_view.first() {
                tracing::info!(
                    request_id = %first.meta.id,
                    sched_latency_ms = first.meta.arrival_time.elapsed().as_secs_f64() * 1000.0,
                    "TTFT_TRACE: batch sent to worker"
                );
            }
            self.dispatch.send_batch(batch_data).await?;
            if self.config.mode == SchedulerMode::Diffusion {
                self.worker_busy = true;
            }
        }
        Ok(())
    }

    /// Snapshot of currently-prefilling sessions for the policy. The
    /// scheduler does not schedule decoding — that lives inside the
    /// worker — so the snapshot only carries prefill-relevant counters.
    fn running_set(&self) -> RunningSet {
        RunningSet {
            num_prefilling: self.requests.prefilling_len(),
            prefilling_continuations: self.requests.prefilling_continuations(),
        }
    }

    fn token_budget(&self) -> TokenBudget {
        TokenBudget {
            max_tokens: self.config.max_batch_tokens,
            max_seqs: self.config.max_num_seqs,
        }
    }

    /// Handle step output from the worker.
    pub(crate) async fn handle_step_output(&mut self, data: Vec<u8>) -> Result<()> {
        self.worker_busy = false;

        match self.config.mode {
            SchedulerMode::Llm => self.handle_step_output_llm(data).await,
            SchedulerMode::Diffusion => self.handle_step_output_diffusion(data).await,
        }
    }

    /// LLM mode: process prefill segment ACKs and generated tokens from Worker.
    ///
    /// Thin orchestrator: every state transition + IO happens inside
    /// [`crate::application::OutputProcessingSystem::process_llm_step`].
    async fn handle_step_output_llm(&mut self, data: Vec<u8>) -> Result<()> {
        // Peel `assigned_indices` off the StepOutput before
        // `process_llm_step` consumes the bytes. Each entry's slots feed
        // into the RadixTree (so prefix lookup can reuse them) and into
        // KvBudget (admission's accounting source of truth).
        if let Ok(parsed) = rmp_serde::from_slice::<
            infer_protocol::worker_to_scheduler_data::StepOutput,
        >(&data)
        {
            if !parsed.assigned_indices.is_empty() {
                // Account the slots that the worker just consumed.
                // Admission ran ahead of the batch send and ensured
                // headroom for at least `projected`; that headroom is
                // consumed *here* when the StepOutput proves the worker
                // actually wrote those slots. Admission itself never
                // calls `try_reserve` — that would double-count the same
                // slots once they show up here.
                let total: u32 = parsed
                    .assigned_indices
                    .iter()
                    .map(|a| a.len as u32)
                    .sum();
                let _ = self.kv_budget.try_reserve(total);
                self.output.feed_radix_assigned_indices(
                    &mut self.radix,
                    &mut self.kv_budget,
                    &parsed,
                );
                // Mark finished sequences' chains in the radix tree so
                // their slots become eligible for LRU eviction.
                for tk in &parsed.tokens {
                    if tk.finished {
                        self.output
                            .radix_mark_finished(&mut self.radix, tk.sequence_id);
                    }
                }
            }
        }
        self.output
            .process_llm_step(
                &mut self.requests,
                self.dispatch.frontend_mut(),
                &self.metrics,
                &self.codec,
                data,
            )
            .await
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Control-plane event dispatch
    // ═══════════════════════════════════════════════════════════════════════════

    /// Dispatch a single control event into the appropriate handler. Called by
    /// the event loop when `control_events.recv()` produces a message.
    ///
    /// Delegates the **decision** to
    /// [`crate::application::ControlEventSystem::handle`], then carries
    /// out the resulting [`ControlOutcome`] — driving
    /// `OutputProcessingSystem` for the failed-session list and
    /// unwinding on `Terminate`. The split keeps `ControlEventSystem`
    /// from needing simultaneous mutable access to both `requests`
    /// and `output`.
    pub(crate) async fn on_control_event(&mut self, event: ControlEvent) -> Result<()> {
        use crate::application::ControlOutcome;

        let outcome = self.control.handle(
            event,
            &mut self.requests,
            &mut self.radix,
            &mut self.kv_budget,
            &self.control_cmd,
            &self.worker_group,
            &self.default_worker,
        );
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
                // Drain every running session as failed before bubbling
                // the fatal error.
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

    /// Drive the OutputProcessingSystem for a list of internal
    /// request ids that the ControlEventSystem flagged as failed.
    /// Used by [`Self::on_control_event`].
    async fn fail_request_ids(
        &mut self,
        failed_request_ids: &[RequestId],
        message: &str,
    ) -> Result<()> {
        self.output
            .fail_sessions(
                &mut self.requests,
                self.dispatch.frontend_mut(),
                failed_request_ids,
                message,
            )
            .await
    }

    /// Diffusion mode: entire batch completes at once and returns image results.
    /// Diffusion mode: entire batch completes at once and returns image results.
    ///
    /// Thin orchestrator: full implementation lives in
    /// [`crate::application::OutputProcessingSystem::process_diffusion_step`].
    async fn handle_step_output_diffusion(&mut self, data: Vec<u8>) -> Result<()> {
        self.output
            .process_diffusion_step(
                &mut self.requests,
                self.dispatch.frontend_mut(),
                &self.metrics,
                &self.codec,
                data,
            )
            .await
    }

    // ─── Accessors for event_loop ───

    pub(crate) fn has_pending_work(&self) -> bool {
        self.requests.has_pending_work()
    }

    #[allow(dead_code)]
    pub(crate) fn is_idle(&self) -> bool {
        !self.has_pending_work() && !self.worker_busy()
    }

    pub(crate) fn worker_busy(&self) -> bool {
        self.config.mode == SchedulerMode::Diffusion && self.worker_busy
    }

    #[allow(dead_code)]
    pub(crate) fn worker_group(&self) -> &WorkerGroup {
        &self.worker_group
    }

    #[allow(dead_code)]
    pub(crate) fn active_request_count(&self) -> usize {
        self.requests.active_count()
    }

    #[allow(dead_code)]
    pub(crate) async fn cancel_request(&mut self, request_id: RequestId) -> Result<()> {
        crate::application::cancel::cancel_request(
            &mut self.requests,
            &self.output,
            &self.control_cmd,
            &self.default_worker,
            request_id,
        )
        .await
    }

    /// Cancel by client-supplied external id (delegates to
    /// [`crate::application::cancel`]).
    pub(crate) async fn cancel_request_by_external_id(&mut self, external_id: &str) -> Result<()> {
        crate::application::cancel::cancel_request_by_external_id(
            &mut self.requests,
            &self.output,
            &self.control_cmd,
            &self.default_worker,
            external_id,
        )
        .await
    }

    /// Pick the worker that should receive control traffic for an unspecified
    /// sequence. Single-rank deployment today; a future TP/PP variant will
    /// thread per-sequence affinity through here.
    #[allow(dead_code)]
    pub(crate) fn worker_id_for_default(&self) -> &WorkerId {
        &self.default_worker
    }

    #[allow(dead_code)]
    /// Receive an event from the frontend transport.
    pub(crate) async fn recv_frontend_event(&mut self) -> Result<FrontendEvent> {
        self.dispatch.recv_frontend().await
    }

    #[allow(dead_code)]
    /// Receive step output from the worker transport.
    pub(crate) async fn recv_worker_output(&mut self) -> Result<Vec<u8>> {
        self.dispatch.recv_worker_output().await
    }

    /// Poll for the next event from frontend, worker, or the control plane.
    ///
    /// `tokio::select!` is `biased` so the control plane wins ties — block
    /// grants and worker-lost notifications take priority over draining the
    /// next StepOutput from a possibly-wedged worker.
    pub(crate) async fn poll_next_event(&mut self) -> crate::application::event_loop::EngineEvent {
        use crate::application::event_loop::EngineEvent;

        let has_work = self.has_pending_work() || self.worker_busy();

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
