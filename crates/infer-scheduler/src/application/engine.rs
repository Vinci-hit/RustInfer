//! SchedulerEngine — the top-level async orchestrator.
//!
//! Owns all mutable state and drives the event loop. Mode-specific
//! logic lives behind the `EngineWorkflow` trait (`LlmWorkflow` /
//! `DiffusionWorkflow`), so the engine itself is mode-agnostic.
//!
//! ## Background decode
//!
//! MsgPack deserialization of worker output runs in a dedicated tokio
//! task, *not* on the main event loop. The ZMQ worker thread sends
//! raw `Vec<u8>` through an mpsc channel; the background decode task
//! receives those bytes, decodes them into typed `SchedulerEvent`
//! variants, and forwards them through a second mpsc channel. The
//! main event loop only ever processes fully-decoded events.

use infer_protocol::server_to_scheduler::InferenceRequest;
use tokio::sync::mpsc;

use crate::application::scheduler_event::SchedulerEvent;
use crate::application::workflow::EngineWorkflow;
use crate::config::{SchedulerConfig, SchedulerMode};
use crate::domain::inference_session::handle::ClientId;
use crate::domain::inference_session::lifecycle::RequestId;
use crate::domain::inference_session::table::RequestTable;
use crate::domain::kv_budget::KvBudget;
use crate::domain::policy::traits::SchedulingPolicy;
use crate::error::Result;
use crate::infrastructure::kv_cache::radix_tree::RadixTree;
use crate::infrastructure::metrics::{MetricsHandle, MetricsRecorder};
use crate::infrastructure::transport::codec::{Codec, MsgPackCodec};
use crate::infrastructure::transport::control_plane::WorkerGroup;
use crate::infrastructure::transport::control_plane::{
    ControlEvent, ControlPlaneCmdTx, ControlPlaneEventRx, WorkerId,
};
use crate::infrastructure::transport::traits::{FrontendEvent, FrontendTransport, WorkerTransport};

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
    metrics: MetricsHandle,

    // ─── Config ───
    config: SchedulerConfig,

    // ─── State ───
    iteration_id: u64,
    /// New-request ingestion stage. Owns the monotonic SequenceId counter.
    ingestion: crate::application::IngestionSystem,

    // ─── Batch-accumulation (throughput knob) ───
    /// Max time to hold freshly arrived requests while accumulating a larger
    /// prefill batch. `None` => low-latency mode (dispatch immediately).
    batch_wait: Option<std::time::Duration>,
    /// Absolute instant at which the currently-open accumulation window must
    /// flush. `Some` only while requests are being deliberately held.
    schedule_deadline: Option<tokio::time::Instant>,

    // ─── Background decode ───
    /// Raw worker output receiver — extracted from the `WorkerTransport`
    /// at construction time and consumed by `run()` to spawn the
    /// background decode task.
    worker_output_rx: Option<mpsc::UnboundedReceiver<Vec<u8>>>,
}

impl SchedulerEngine {
    /// Create a new scheduler engine.
    pub fn new<F, W>(
        config: SchedulerConfig,
        policy: Box<dyn SchedulingPolicy>,
        worker_group: WorkerGroup,
        frontend: F,
        mut worker: W,
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

        // Extract the raw byte receiver from the worker transport
        // before boxing it. The background decode task will consume
        // raw bytes from this channel, decode them off the main loop,
        // and forward typed SchedulerEvents.
        let worker_output_rx = worker.take_output_rx();

        let workflow: Box<dyn EngineWorkflow> = match config.mode {
            SchedulerMode::Llm => Box::new(crate::application::workflow::LlmWorkflow::new(policy)),
            SchedulerMode::Diffusion => {
                Box::new(crate::application::workflow::DiffusionWorkflow::new(policy))
            }
        };
        let max_total_kv_tokens = match worker_group.effective_capacity.max_total_kv_tokens {
            Some(tokens) => tokens,
            None => {
                tracing::warn!(
                    "worker capacity missing max_total_kv_tokens; initializing KV budget to 0"
                );
                0
            }
        };
        let kv_capacity = u32::try_from(max_total_kv_tokens).unwrap_or_else(|_| {
            tracing::warn!(
                "worker max_total_kv_tokens={} exceeds u32::MAX; clamping KV budget",
                max_total_kv_tokens
            );
            u32::MAX
        });

        Self {
            workflow,
            dispatch: crate::application::DispatchSystem::new(Box::new(frontend), Box::new(worker)),
            radix: RadixTree::new(),
            kv_budget: KvBudget::new(kv_capacity),
            worker_group,
            requests: RequestTable::new(),
            control_cmd,
            control_events,
            default_worker,
            codec: MsgPackCodec,
            metrics: std::sync::Arc::new(MetricsRecorder::new(config.metrics_enabled)),
            batch_wait: config.batch_wait,
            config,
            iteration_id: 0,
            ingestion: crate::application::IngestionSystem::new(),
            schedule_deadline: None,
            worker_output_rx,
        }
    }

    /// Run the scheduler event loop.
    ///
    /// Spawns a background decode task for worker output and then
    /// enters the main event loop. All MsgPack deserialization
    /// happens off the main async task.
    pub async fn run(mut self) -> Result<()> {
        tracing::info!("SchedulerEngine starting event loop...");

        // Extract the raw worker output receiver — present only for
        // real ZMQ transports. Mock/test transports return None.
        let raw_worker_rx = self.worker_output_rx.take().expect(
            "worker_output_rx already consumed or transport does not support background decode",
        );

        // Create the decoded event channel.
        let (decoded_tx, decoded_rx) = mpsc::unbounded_channel::<SchedulerEvent>();

        // Spawn the background decode task: recv raw bytes → decode
        // MsgPack → forward typed SchedulerEvent.
        let mode = self.config.mode;
        tokio::spawn(decode_worker_output(raw_worker_rx, decoded_tx, mode));

        crate::application::event_loop::run_event_loop(&mut self, decoded_rx).await
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Public methods used by event_loop
    // ═══════════════════════════════════════════════════════════════════════════

    /// Handle an incoming request from the frontend.
    pub(crate) fn handle_new_request(&mut self, client_id: ClientId, request: InferenceRequest) {
        use crate::application::ingestion::{IngestOutcome, RejectReason};

        let external_id = request.request_id.clone();
        let outcome = self
            .ingestion
            .ingest(client_id, request, &self.config, &mut self.requests);
        match outcome {
            IngestOutcome::Admitted { request_id, .. } => {
                tracing::debug!(
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

    /// Split disjoint borrows of `self` into the workflow handle, the dispatch
    /// system, and a freshly built [`ResourceContext`]. Collapses the 11-field
    /// destructure + 9-field context build that `run_iteration`,
    /// `handle_step_output`, and `on_control_event` would otherwise each repeat
    /// verbatim — the boilerplate the borrow checker forces out of methods that
    /// hand most of `self` to a `&mut`-borrowing callee.
    fn split_for_workflow(
        &mut self,
    ) -> (
        &mut Box<dyn EngineWorkflow>,
        &mut crate::application::DispatchSystem,
        crate::application::workflow::ResourceContext<'_>,
    ) {
        let SchedulerEngine {
            workflow,
            dispatch,
            requests,
            radix,
            kv_budget,
            metrics,
            codec,
            config,
            control_cmd,
            worker_group,
            default_worker,
            ..
        } = self;
        let ctx = crate::application::workflow::ResourceContext {
            requests,
            radix,
            kv_budget,
            metrics: &**metrics,
            codec,
            config,
            control_cmd,
            worker_group,
            default_worker,
        };
        (workflow, dispatch, ctx)
    }

    /// Run one scheduling iteration — delegates to the workflow.
    pub(crate) async fn run_iteration(&mut self) -> Result<()> {
        if !self.workflow.can_schedule(&self.requests) {
            return Ok(());
        }
        self.iteration_id += 1;

        let (workflow, dispatch, mut ctx) = self.split_for_workflow();
        workflow.try_schedule(&mut ctx, dispatch).await
    }

    /// Schedule prefills, honoring the batch-accumulation window.
    ///
    /// Low-latency mode (`batch_wait == None`) dispatches immediately, exactly
    /// as the eager path always did. Throughput mode holds freshly arrived
    /// requests until the accumulation window closes (deadline elapsed, batch
    /// full, or continuation work present), arming a timer so the held batch
    /// flushes even if no further events arrive.
    pub(crate) async fn maybe_schedule(&mut self) -> Result<()> {
        if !self.can_schedule() {
            return Ok(());
        }
        if self.should_defer_new_prefills() {
            // Open the accumulation window on the first held request; keep the
            // existing deadline otherwise so the window is measured from the
            // oldest waiter (bounded TTFT), not reset by later arrivals.
            if self.schedule_deadline.is_none() {
                if let Some(wait) = self.batch_wait {
                    self.schedule_deadline = Some(tokio::time::Instant::now() + wait);
                }
            }
            return Ok(());
        }
        self.schedule_deadline = None;
        self.run_iteration().await
    }

    /// Whether to hold freshly arrived prefills to accumulate a larger batch.
    ///
    /// Returns false (schedule now) in low-latency mode, when there is nothing
    /// fresh to batch, when the batch is already seq-saturated, when there is
    /// continuation work that must not be stalled, or once the accumulation
    /// window has elapsed.
    fn should_defer_new_prefills(&self) -> bool {
        if self.batch_wait.is_none() {
            return false;
        }
        let waiting = self.requests.waiting();
        if waiting.is_empty() || waiting.len() >= self.config.max_num_seqs {
            return false;
        }
        if self.requests.has_prefilling_continuations() {
            return false;
        }
        match self.schedule_deadline {
            // Window still open: keep holding.
            Some(deadline) => tokio::time::Instant::now() < deadline,
            // No window yet: open one and hold.
            None => true,
        }
    }

    /// Flush the accumulated prefill batch when the wait window elapses.
    pub(crate) async fn on_batch_timer(&mut self) -> Result<()> {
        self.schedule_deadline = None;
        if self.can_schedule() {
            self.run_iteration().await?;
        }
        Ok(())
    }

    pub(crate) fn metrics_handle(&self) -> MetricsHandle {
        self.metrics.clone()
    }

    /// Handle a decoded worker step output — delegates to the workflow.
    ///
    /// The event is already decoded by the background decode task;
    /// no MsgPack deserialization happens here.
    pub(crate) async fn handle_step_output(&mut self, event: SchedulerEvent) -> Result<()> {
        let (workflow, dispatch, mut ctx) = self.split_for_workflow();
        workflow.handle_step_output(&mut ctx, dispatch, event).await
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Control-plane event dispatch
    // ═══════════════════════════════════════════════════════════════════════════

    /// Dispatch a single control event.
    ///
    /// Control-plane handling is mode-independent, so the engine calls
    /// `control_fns::handle_control_event` directly with disjoint field borrows
    /// — no workflow indirection and no manual re-borrow dance. Prefix-cache
    /// relief only applies in LLM mode with caching enabled.
    pub(crate) async fn on_control_event(&mut self, event: ControlEvent) -> Result<()> {
        use crate::application::ControlOutcome;

        let enable_prefix_caching =
            matches!(self.config.mode, SchedulerMode::Llm) && self.config.enable_prefix_caching;
        let outcome = crate::application::control_fns::handle_control_event(
            event,
            &mut self.requests,
            &mut self.radix,
            &mut self.kv_budget,
            &self.control_cmd,
            &self.worker_group,
            &self.default_worker,
            enable_prefix_caching,
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
            ControlOutcome::Terminate { error } => {
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
    /// [`Self::handle_step_output`] which receives already-decoded
    /// `SchedulerEvent` from the background decode task.
    #[cfg(test)]
    pub(crate) async fn handle_step_output_llm(&mut self, data: Vec<u8>) -> Result<()> {
        let output: infer_protocol::worker_to_scheduler_data::StepOutput =
            self.codec.decode(&data)?;
        let event = SchedulerEvent::WorkerLlmStep(output);
        self.handle_step_output(event).await
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

    pub(crate) fn shutdown_worker_best_effort(&self) {
        let _ = self.control_cmd.send_to(
            &self.default_worker,
            infer_protocol::scheduler_to_worker_control::SchedulerControlMessage::Shutdown,
        );
    }

    /// Cancel by client-supplied external id.
    pub(crate) async fn cancel_request_by_external_id(&mut self, external_id: &str) -> Result<()> {
        crate::application::cancel::cancel_request_by_external_id_with_kv(
            &mut self.requests,
            &mut self.radix,
            &mut self.kv_budget,
            &self.control_cmd,
            &self.default_worker,
            external_id,
            self.config.enable_prefix_caching,
        )
        .await
    }

    /// Poll for the next typed `SchedulerEvent`.
    ///
    /// Translates raw channel events (frontend `Result<FrontendEvent>`,
    /// control `ControlEvent`, decoded worker `SchedulerEvent`) into
    /// the unified `SchedulerEvent` enum. No MsgPack deserialization
    /// happens here — worker output arrives already decoded via the
    /// background decode task.
    pub(crate) async fn poll_next_event(
        &mut self,
        decoded_rx: &mut mpsc::UnboundedReceiver<SchedulerEvent>,
    ) -> SchedulerEvent {
        let has_work = self.has_pending_work() || self.has_in_flight_batch();
        let deadline = self.schedule_deadline;
        let frontend = self.dispatch.frontend_mut();
        let control_events = &mut self.control_events;

        tokio::select! {
            biased;
            Some(ev) = control_events.recv() => SchedulerEvent::ControlSignal(ev),
            result = frontend.recv_event() => frontend_result_to_event(result),
            // Batch-accumulation deadline (throughput mode only). The deadline
            // is absolute, so re-arming this branch each poll fires at the same
            // wall-clock instant regardless of intervening events.
            _ = sleep_until_opt(deadline), if deadline.is_some() => SchedulerEvent::BatchTimer,
            Some(event) = decoded_rx.recv(), if has_work => event,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Background decode task
// ═══════════════════════════════════════════════════════════════════════════════

/// Background task: recv raw `Vec<u8>` from the ZMQ worker thread,
/// decode MsgPack into a typed `SchedulerEvent`, and forward it
/// through the decoded-event channel.
///
/// This is the key offload: MsgPack deserialization (which can be
/// expensive for large step outputs) never blocks the main event
/// loop.
async fn decode_worker_output(
    mut raw_rx: mpsc::UnboundedReceiver<Vec<u8>>,
    decoded_tx: mpsc::UnboundedSender<SchedulerEvent>,
    mode: SchedulerMode,
) {
    let codec = MsgPackCodec;

    while let Some(data) = raw_rx.recv().await {
        let event = match mode {
            SchedulerMode::Llm => {
                match codec.decode::<infer_protocol::worker_to_scheduler_data::StepOutput>(&data) {
                    Ok(output) => SchedulerEvent::WorkerLlmStep(output),
                    Err(e) => SchedulerEvent::WorkerDecodeError(e.to_string()),
                }
            }
            SchedulerMode::Diffusion => {
                match codec
                    .decode::<infer_protocol::worker_to_scheduler_data::DiffusionBatchOutput>(&data)
                {
                    Ok(output) => SchedulerEvent::WorkerDiffusionStep(output),
                    Err(e) => SchedulerEvent::WorkerDecodeError(e.to_string()),
                }
            }
        };
        if decoded_tx.send(event).is_err() {
            // Engine dropped the receiver — shut down.
            break;
        }
    }
    // Raw channel closed — worker transport shut down.
    let _ = decoded_tx.send(SchedulerEvent::WorkerShutdown);
}

/// Sleep until an optional absolute deadline.
///
/// `None` yields a future that never completes — safe because the
/// `select!` branch using it is disabled (`if deadline.is_some()`) in that
/// case, so it is constructed but never polled.
async fn sleep_until_opt(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(dl) => tokio::time::sleep_until(dl).await,
        None => std::future::pending::<()>().await,
    }
}

/// Translate a `Result<FrontendEvent>` from the frontend transport
/// into a `SchedulerEvent`.
fn frontend_result_to_event(result: crate::error::Result<FrontendEvent>) -> SchedulerEvent {
    match result {
        Ok(FrontendEvent::Infer { client_id, request }) => {
            SchedulerEvent::NewRequest { client_id, request }
        }
        Ok(FrontendEvent::Cancel {
            external_id,
            reason,
        }) => SchedulerEvent::Cancel {
            external_id,
            reason,
        },
        Err(crate::error::SchedulerError::Shutdown) => SchedulerEvent::FrontendShutdown,
        Err(e) => SchedulerEvent::FrontendError(e.to_string()),
    }
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
