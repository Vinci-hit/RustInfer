//! SchedulerEngine — the top-level async orchestrator.
//!
//! Owns all mutable state and drives the event loop.

use std::sync::Arc;
use std::time::Instant;

use infer_protocol::scheduler_to_server::{
    ChunkType, ImageOutput, InferenceMetrics, InferenceResponse, ResponseStatus, StreamChunk,
};
use infer_protocol::server_to_scheduler::{InferenceModality, InferenceRequest};
use infer_protocol::worker_to_scheduler::{
    DiffusionBatchOutput, DiffusionOutputStatus, StepOutput, WorkerStepError,
};

use crate::cache::kv_manager::KvManager;
use crate::cache::traits::CacheState;
use crate::config::{SchedulerConfig, SchedulerMode};
use crate::error::{Result, SchedulerError};
use crate::metrics::MetricsRecorder;
use crate::policy::traits::{BatchPlan, RunningSet, SchedulingPolicy};
use crate::request::handle::{ClientId, RequestHandle};
use crate::request::lifecycle::*;
use crate::request::{CancelOutcome, FailedOutcome, PrefillAckOutcome, PrefillStartOutcome, RequestLocation, RequestTable, TerminalReason};
use crate::transport::codec::MsgPackCodec;
use crate::transport::traits::{FrontendEvent, FrontendTransport, WorkerTransport};
use crate::utils::token_budget::TokenBudget;
use crate::worker_group::WorkerGroup;

/// The main scheduler engine.
pub struct SchedulerEngine<P, F, W>
where
    P: SchedulingPolicy,
    F: FrontendTransport,
    W: WorkerTransport,
{
    // ─── Subsystems ───
    policy: P,
    kv_manager: Box<dyn KvManager>,
    worker_group: WorkerGroup,

    // ─── Request state ───
    requests: RequestTable,

    // ─── Transport ───
    frontend: F,
    worker: W,
    codec: MsgPackCodec,

    // ─── Metrics ───
    metrics: MetricsRecorder,

    // ─── Config ───
    config: SchedulerConfig,

    // ─── State ───
    iteration_id: u64,
    /// Diffusion batch-in batch-out gate. LLM mode does not use this as backpressure.
    worker_busy: bool,
    next_sequence_id: u64,
    /// Prefill segments scheduled in the current Scheduler iteration.
    /// Key: request_id, Value: tokens sent in this segment.
    current_chunk_sizes: Vec<(RequestId, usize)>,
}

impl<P, F, W> SchedulerEngine<P, F, W>
where
    P: SchedulingPolicy,
    F: FrontendTransport,
    W: WorkerTransport,
{
    /// Create a new scheduler engine.
    pub fn new(
        config: SchedulerConfig,
        policy: P,
        kv_manager: Box<dyn KvManager>,
        worker_group: WorkerGroup,
        frontend: F,
        worker: W,
    ) -> Self {
        tracing::info!(
            "SchedulerEngine created: policy={}, kv_mode={}, worker_group={}, ranks={}, max_seqs={}, max_tokens={}",
            policy.name(),
            kv_manager.mode_name(),
            worker_group.group_id,
            worker_group.rank_count(),
            config.max_num_seqs,
            config.max_batch_tokens,
        );

        Self {
            policy,
            kv_manager,
            worker_group,
            requests: RequestTable::new(),
            frontend,
            worker,
            codec: MsgPackCodec,
            metrics: MetricsRecorder::new(config.metrics_enabled),
            config,
            iteration_id: 0,
            worker_busy: false,
            next_sequence_id: 1,
            current_chunk_sizes: Vec::new(),
        }
    }

    /// Run the scheduler event loop.
    pub async fn run(mut self) -> Result<()> {
        tracing::info!("SchedulerEngine starting event loop...");
        crate::core::event_loop::run_event_loop(&mut self).await
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Public methods used by event_loop
    // ═══════════════════════════════════════════════════════════════════════════

    /// Handle an incoming request from the frontend.
    pub(crate) fn handle_new_request(&mut self, client_id: ClientId, request: InferenceRequest) {
        let is_diffusion = request.modality == InferenceModality::Diffusion
            || matches!(self.config.mode, SchedulerMode::Diffusion);

        if is_diffusion {
            let Some(diffusion) = request.diffusion.as_ref() else {
                tracing::warn!(
                    "Rejecting diffusion request {}: missing diffusion payload",
                    request.request_id
                );
                return;
            };
            if diffusion.prompt.is_empty() {
                tracing::warn!(
                    "Rejecting diffusion request {}: empty prompt",
                    request.request_id
                );
                return;
            }
            if diffusion.prompt_input_ids.is_empty() {
                tracing::warn!(
                    "Rejecting diffusion request {}: empty server-tokenized prompt_input_ids",
                    request.request_id
                );
                return;
            }
        } else {
            if request.input_ids.is_empty() {
                tracing::warn!("Rejecting request {}: empty input_ids", request.request_id);
                return;
            }
            if request.input_ids.len() > self.config.max_model_len {
                tracing::warn!(
                    "Rejecting request {}: prompt length {} exceeds max_model_len {}",
                    request.request_id,
                    request.input_ids.len(),
                    self.config.max_model_len,
                );
                return;
            }
        }

        let sequence_id = SequenceId(self.next_sequence_id);
        self.next_sequence_id += 1;

        let input_ids = if is_diffusion && request.input_ids.is_empty() {
            vec![0]
        } else {
            request.input_ids
        };
        let max_tokens = if is_diffusion { 1 } else { request.max_tokens };

        let meta = Arc::new(RequestMeta {
            id: RequestId(request.request_id.clone()),
            sequence_id,
            input_ids,
            max_tokens,
            sampling: SamplingParams {
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: request.top_k,
            },
            priority: Priority(request.priority),
            stream: request.stream,
            stop_sequences: vec![],
            diffusion: request.diffusion,
            arrival_time: Instant::now(),
        });
        tracing::info!(
            request_id = %meta.id,
            "TTFT_TRACE: scheduler received request"
        );

        let handle = RequestHandle::new(client_id, request.stream);
        match self.requests.insert_new(meta.clone(), handle) {
            Ok(()) => self.metrics.record_enqueue(),
            Err(e) => tracing::error!("failed to insert request {}: {}", meta.id, e),
        }
    }

    /// Run one scheduling iteration.
    pub(crate) async fn run_iteration(&mut self) -> Result<()> {
        if self.config.mode == SchedulerMode::Diffusion && self.worker_busy {
            return Ok(());
        }
        if !self.requests.has_pending_work() {
            return Ok(());
        }

        self.iteration_id += 1;

        let running_set = RunningSet {
            num_prefilling: self.requests.prefilling().len(),
            num_decoding: self.requests.decoding().len(),
            decode_tokens: self.requests.decoding().len(),
            running_ids: self
                .requests
                .decoding()
                .iter()
                .map(|s| s.meta.id.clone())
                .collect(),
            prefilling_continuations: self.requests.prefilling_continuations(),
        };

        let budget = TokenBudget {
            max_tokens: self.config.max_batch_tokens,
            max_seqs: self.config.max_num_seqs,
        };

        let cache_state = CacheState {
            free_blocks: 0,
            total_blocks: 0,
            utilization: 0.0,
            evictable_blocks: 0,
        };

        let plan = self
            .policy
            .schedule(self.requests.waiting(), &running_set, &budget, &cache_state);

        if !plan.has_work()
            && self.requests.decoding().is_empty()
            && self.requests.prefilling().is_empty()
        {
            return Ok(());
        }

        // Execute plan: allocate KV for new prefills.
        self.execute_plan(&plan)?;

        // Build and send batch command.
        let batch_data = match self.config.mode {
            SchedulerMode::Llm => crate::core::batch_builder::build_batch(
                self.requests.prefilling(),
                self.requests.decoding(),
                &self.config,
                &self.codec,
                &self.current_chunk_sizes,
            )?,
            SchedulerMode::Diffusion => crate::core::batch_builder::build_diffusion_batch(
                self.requests.prefilling(),
                &self.codec,
                &self.current_chunk_sizes,
            )?,
        };

        if !batch_data.is_empty() {
            if let Some(first_prefill) = self.requests.prefilling().first() {
                let sched_latency = first_prefill.meta.arrival_time.elapsed();
                tracing::info!(
                    request_id = %first_prefill.meta.id,
                    sched_latency_ms = sched_latency.as_secs_f64() * 1000.0,
                    "TTFT_TRACE: batch sent to worker"
                );
            }
            self.worker.send_batch(batch_data).await?;
            if self.config.mode == SchedulerMode::Diffusion {
                self.worker_busy = true;
            }
        }

        Ok(())
    }

    /// Execute the scheduling plan: handle new prefills and continuation chunks.
    fn execute_plan(&mut self, plan: &BatchPlan) -> Result<()> {
        self.current_chunk_sizes.clear();

        for entry in &plan.prefill_batch {
            let scheduled_len = entry.token_range.len();
            if scheduled_len == 0 {
                tracing::warn!("Plan produced zero-length prefill for {}", entry.request_id);
                continue;
            }

            let is_continuation = self.requests.location_for_request(&entry.request_id)
                == Some(RequestLocation::Prefilling);

            if is_continuation {
                match self.requests.set_prefill_inflight(&entry.request_id, scheduled_len) {
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
                let seq = match self.requests.take_waiting(&entry.request_id) {
                    Ok(seq) => seq,
                    Err(e) => {
                        tracing::warn!("Plan references non-waiting request {}: {}", entry.request_id, e);
                        continue;
                    }
                };

                let prompt_len = seq.meta.input_ids.len();
                let (kv_alloc, prefix_match) = match self
                    .kv_manager
                    .allocate_with_prefix(prompt_len, &seq.meta.input_ids)
                {
                    Ok(result) => result,
                    Err(e) => {
                        tracing::warn!("KV allocation failed for {}: {}", entry.request_id, e);
                        self.requests.restore_waiting_front(seq)?;
                        break;
                    }
                };

                match self.requests.commit_prefill_start(seq, kv_alloc, prefix_match, scheduled_len)? {
                    PrefillStartOutcome::Scheduled { request_id, segment, .. } => {
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

    /// Handle step output from the worker.
    pub(crate) async fn handle_step_output(&mut self, data: Vec<u8>) -> Result<()> {
        self.worker_busy = false;

        match self.config.mode {
            SchedulerMode::Llm => self.handle_step_output_llm(data).await,
            SchedulerMode::Diffusion => self.handle_step_output_diffusion(data).await,
        }
    }

    /// LLM mode: process prefill segment ACKs and generated tokens from Worker.
    async fn handle_step_output_llm(&mut self, data: Vec<u8>) -> Result<()> {
        use crate::transport::codec::Codec;
        use infer_protocol::scheduler_to_worker::{BlockGrantCmd, WorkerCommand};
        let output: StepOutput = self.codec.decode(&data)?;

        if let Some(err) = output.error.clone() {
            self.handle_worker_step_error(err.clone()).await?;
            if err.fatal {
                return Err(SchedulerError::WorkerError(err.message));
            }
        }

        // 1. ACK prefill segments. Final segment moves the sequence to decoding before
        // processing the token generated by that final prefill.
        for sequence_id in output.prefill_done {
            match self.requests.ack_prefill(SequenceId(sequence_id)) {
                Ok(PrefillAckOutcome::MovedToDecoding { .. })
                | Ok(PrefillAckOutcome::Continue { .. }) => {}
                Err(e) => tracing::warn!("PrefillDone for sequence_id={} failed: {}", sequence_id, e),
            }
        }

        for need in &output.need_blocks {
            match self
                .kv_manager
                .allocate_decode_blocks(need.request_blocks as usize)
            {
                Ok(blocks) => {
                    if let Err(e) = self.requests.extend_decode_kv(SequenceId(need.sequence_id), blocks.clone()) {
                        tracing::debug!(
                            "NeedBlocks for non-decoding sequence_id={} ignored: {}",
                            need.sequence_id,
                            e
                        );
                        continue;
                    }
                    let cmd = WorkerCommand::GrantBlocks(BlockGrantCmd {
                        sequence_id: need.sequence_id,
                        block_ids: blocks.iter().map(|b| b.0).collect(),
                    });
                    let bytes = self.codec.encode(&cmd)?;
                    self.worker.send_batch(bytes).await?;
                }
                Err(e) => {
                    tracing::warn!(
                        "NeedBlocks denied: sequence_id={} request_blocks={} error={}",
                        need.sequence_id,
                        need.request_blocks,
                        e,
                    );
                }
            }
        }

        // 2. Process generated tokens. Worker owns EOS/max_tokens decision and
        // is the only component allowed to end decode. Scheduler mirrors the
        // worker-reported lifecycle instead of racing it with a second max-token
        // check.
        let mut finished_sequence_ids: Vec<SequenceId> = Vec::new();
        let mut token_chunks: Vec<(ClientId, StreamChunk)> = Vec::new();
        for token in &output.tokens {
            match self.requests.append_generated_token(
                SequenceId(token.sequence_id),
                token.token_id,
                token.finished,
            ) {
                Ok(outcome) => {
                    if outcome.stream {
                        token_chunks.push((
                            outcome.client_id,
                            StreamChunk {
                                request_id: outcome.request_id.0,
                                chunk_type: ChunkType::Token,
                                token_id: Some(outcome.token_id),
                                finish_reason: None,
                                metrics: None,
                            },
                        ));
                    }
                    if outcome.worker_finished {
                        finished_sequence_ids.push(outcome.sequence_id);
                    }
                }
                Err(e) => tracing::warn!(
                    "Generated token for sequence_id={} failed: {}",
                    token.sequence_id,
                    e
                ),
            }
        }

        // Send token chunks before done chunks for sequences that finish in this same step.
        for (client_id, chunk) in token_chunks {
            self.frontend.send_stream_chunk(&client_id, chunk).await?;
        }

        finished_sequence_ids.sort_unstable_by_key(|id| id.0);
        finished_sequence_ids.dedup();
        for sequence_id in finished_sequence_ids {
            let seq = self.requests.finish_decoding(sequence_id)?;
            self.complete_sequence(seq).await?;
        }

        Ok(())
    }

    async fn handle_worker_step_error(&mut self, err: WorkerStepError) -> Result<()> {
        let mut failed_ids = err.sequence_ids.clone();
        if err.fatal || failed_ids.is_empty() {
            failed_ids.extend(self.requests.running_sequence_ids().into_iter().map(|id| id.0));
        }
        failed_ids.sort_unstable();
        failed_ids.dedup();

        for sequence_id in failed_ids {
            match self.requests.fail_sequence(SequenceId(sequence_id), &err.message)? {
                FailedOutcome::RemovedPrefilling { sequence, .. } => {
                    self.fail_prefilling_sequence(sequence, &err.message).await?;
                }
                FailedOutcome::RemovedDecoding { sequence, .. } => {
                    self.fail_decoding_sequence(sequence, &err.message).await?;
                }
                FailedOutcome::NotFound { .. } => {}
            }
        }
        Ok(())
    }

    async fn fail_prefilling_sequence(
        &mut self,
        seq: Sequence<Prefilling>,
        message: &str,
    ) -> Result<()> {
        let request_id = seq.meta.id.clone();
        let client_id = ClientId(seq.handle.client_id.0.clone());
        let stream = seq.meta.stream;
        self.kv_manager.free(seq.state.kv_alloc);
        self.send_request_error(client_id, request_id, stream, message.to_string(), 0)
            .await
    }

    async fn fail_decoding_sequence(
        &mut self,
        seq: Sequence<Decoding>,
        message: &str,
    ) -> Result<()> {
        let request_id = seq.meta.id.clone();
        let client_id = ClientId(seq.handle.client_id.0.clone());
        let stream = seq.meta.stream;
        let num_tokens = seq.state.output_tokens.len() as u32;
        self.kv_manager.free(seq.state.kv_alloc);
        self.send_request_error(
            client_id,
            request_id,
            stream,
            message.to_string(),
            num_tokens,
        )
        .await
    }

    async fn send_request_error(
        &mut self,
        client_id: ClientId,
        request_id: RequestId,
        stream: bool,
        message: String,
        num_tokens: u32,
    ) -> Result<()> {
        let metrics = InferenceMetrics {
            total_ms: 0,
            num_tokens,
            tokens_per_second: 0.0,
        };
        if stream {
            self.frontend
                .send_stream_chunk(
                    &client_id,
                    StreamChunk {
                        request_id: request_id.0.clone(),
                        chunk_type: ChunkType::Error,
                        token_id: None,
                        finish_reason: Some(message),
                        metrics: Some(metrics),
                    },
                )
                .await
        } else {
            self.frontend
                .send_response(
                    &client_id,
                    InferenceResponse {
                        request_id: request_id.0.clone(),
                        status: ResponseStatus::Error,
                        output_token_ids: vec![],
                        images: vec![],
                        finish_reason: Some("error".to_string()),
                        error: Some(message),
                        metrics,
                    },
                )
                .await
        }
    }

    /// Diffusion mode: entire batch completes at once and returns image results.
    async fn handle_step_output_diffusion(&mut self, data: Vec<u8>) -> Result<()> {
        use crate::transport::codec::Codec;
        let output: DiffusionBatchOutput = self.codec.decode(&data)?;

        for item in output.results {
            let item_request_id = RequestId(item.request_id.clone());
            let Some(seq) = self.requests.take_prefilling_by_request(
                &item_request_id,
                if matches!(item.status, DiffusionOutputStatus::Success) {
                    TerminalReason::Finished
                } else {
                    TerminalReason::Failed(item.error.clone().unwrap_or_else(|| "diffusion error".to_string()))
                },
            )? else {
                tracing::warn!(
                    "Diffusion output for unknown request_id={}",
                    item.request_id
                );
                continue;
            };
            let request_id = seq.meta.id.clone();
            let client_id = ClientId(seq.handle.client_id.0.clone());
            self.kv_manager.free(seq.state.kv_alloc);

            let elapsed_ms = seq.meta.arrival_time.elapsed().as_millis() as u64;
            let (status, images, error) = match item.status {
                DiffusionOutputStatus::Success => {
                    let images = item
                        .image
                        .into_iter()
                        .map(|image| ImageOutput {
                            width: image.width,
                            height: image.height,
                            channels: image.channels,
                            format: image.format,
                            data: image.data,
                        })
                        .collect();
                    (ResponseStatus::Success, images, None)
                }
                DiffusionOutputStatus::Error => (ResponseStatus::Error, vec![], item.error),
            };

            let response = InferenceResponse {
                request_id: request_id.0.clone(),
                status,
                output_token_ids: vec![],
                images,
                finish_reason: Some("stop".to_string()),
                error,
                metrics: InferenceMetrics {
                    total_ms: elapsed_ms.max(item.metrics.total_ms),
                    num_tokens: 0,
                    tokens_per_second: 0.0,
                },
            };

            self.frontend.send_response(&client_id, response).await?;
            self.metrics.record_completion(elapsed_ms, 0);
            tracing::info!(
                "Completed diffusion request {} in {}ms",
                request_id,
                elapsed_ms
            );
        }

        Ok(())
    }

    /// Complete a sequence: send response, free KV, record metrics.
    async fn complete_sequence(&mut self, seq: Sequence<Decoding>) -> Result<()> {
        let request_id = seq.meta.id.clone();
        let client_id = ClientId(seq.handle.client_id.0.clone());
        let stream = seq.meta.stream;

        let reason = if seq.reached_max_tokens() {
            FinishReason::MaxTokens
        } else {
            FinishReason::Eos
        };

        // Extract KV allocation and prompt tokens BEFORE consuming the sequence.
        let kv_alloc = seq.state.kv_alloc.clone();
        let prompt_tokens = seq.meta.input_ids.clone();

        // Transition: Decoding → Finished.
        let finished = seq.finish(reason);

        // Free KV resources, or keep full prompt blocks in prefix cache for paged mode.
        self.kv_manager.free_finished(&prompt_tokens, kv_alloc);

        // Build response.
        let elapsed_ms = finished.state.metrics.e2e_latency.as_millis() as u64;
        let num_tokens = finished.state.metrics.num_output_tokens;
        let tokens_per_second = if elapsed_ms > 0 {
            (num_tokens as f64 / elapsed_ms as f64) * 1000.0
        } else {
            0.0
        };

        let metrics = InferenceMetrics {
            total_ms: elapsed_ms,
            num_tokens,
            tokens_per_second,
        };

        if stream {
            self.frontend
                .send_stream_chunk(
                    &client_id,
                    StreamChunk {
                        request_id: request_id.0.clone(),
                        chunk_type: ChunkType::Done,
                        token_id: None,
                        finish_reason: Some("stop".to_string()),
                        metrics: Some(metrics.clone()),
                    },
                )
                .await?;
        } else {
            let response = InferenceResponse {
                request_id: request_id.0.clone(),
                status: ResponseStatus::Success,
                output_token_ids: finished.state.output_tokens,
                images: vec![],
                finish_reason: Some("stop".to_string()),
                error: None,
                metrics: metrics.clone(),
            };
            self.frontend.send_response(&client_id, response).await?;
        }

        self.metrics.record_completion(elapsed_ms, num_tokens);

        tracing::info!(
            "Completed {}: {} tokens in {}ms ({:.1} tok/s)",
            request_id,
            num_tokens,
            elapsed_ms,
            tokens_per_second,
        );

        Ok(())
    }

    // ─── Accessors for event_loop ───

    pub(crate) fn has_pending_work(&self) -> bool {
        self.requests.has_pending_work()
    }

    #[allow(dead_code)]
    pub(crate) fn is_idle(&self) -> bool {
        !self.has_pending_work() && !self.worker_busy()
    }

    #[allow(dead_code)]
    pub(crate) fn frontend_mut(&mut self) -> &mut F {
        &mut self.frontend
    }

    #[allow(dead_code)]
    pub(crate) fn worker_mut(&mut self) -> &mut W {
        &mut self.worker
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
        match self.requests.cancel_request(&request_id)? {
            CancelOutcome::RemovedWaiting { .. } | CancelOutcome::NotFound => Ok(()),
            CancelOutcome::RemovedPrefilling {
                sequence_id,
                kv_alloc,
                ..
            } => {
                self.kv_manager.free(kv_alloc);
                let data = crate::core::batch_builder::build_cancel_request(sequence_id.0, &self.codec)?;
                self.worker.send_batch(data).await
            }
            CancelOutcome::RemovedDecoding {
                sequence_id,
                prompt_tokens,
                kv_alloc,
                ..
            } => {
                self.kv_manager.free_finished(&prompt_tokens, kv_alloc);
                let data = crate::core::batch_builder::build_cancel_request(sequence_id.0, &self.codec)?;
                self.worker.send_batch(data).await
            }
        }
    }

    #[allow(dead_code)]
    /// Receive an event from the frontend transport.
    pub(crate) async fn recv_frontend_event(&mut self) -> Result<FrontendEvent> {
        self.frontend.recv_event().await
    }

    #[allow(dead_code)]
    /// Receive step output from the worker transport.
    pub(crate) async fn recv_worker_output(&mut self) -> Result<Vec<u8>> {
        self.worker.recv_step_output().await
    }

    /// Poll for the next event from either frontend or worker.
    ///
    /// This method carefully splits the borrow so tokio::select! can
    /// poll both transports simultaneously.
    pub(crate) async fn poll_next_event(&mut self) -> crate::core::event_loop::EngineEvent {
        use crate::core::event_loop::EngineEvent;

        let has_work = self.has_pending_work() || self.worker_busy();

        if has_work {
            let frontend = &mut self.frontend;
            let worker = &mut self.worker;

            tokio::select! {
                result = worker.recv_step_output() => EngineEvent::WorkerOutput(result),
                result = frontend.recv_event() => EngineEvent::Frontend(Box::new(result)),
            }
        } else {
            let result = self.frontend.recv_event().await;
            EngineEvent::Frontend(Box::new(result))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use infer_protocol::scheduler_to_worker::WorkerCommand;
    use infer_protocol::worker_to_scheduler::{GeneratedToken, NeedBlocksRequest, StepOutput};

    use crate::cache::kv_manager::KvAllocation;
    use crate::cache::paged_kv_manager::PagedKvManager;
    use crate::cache::traits::PhysicalBlockId;
    use crate::config::{KvCacheMode, SchedulerConfig};
    use crate::policy::ContinuousBatchingPolicy;
    use crate::request::handle::{ClientId, RequestHandle};
    use crate::request::lifecycle::{
        InFlightPrefillSegment, Prefilling, Priority, RequestId, RequestMeta, SamplingParams,
        Sequence, SequenceId,
    };
    use crate::transport::codec::{Codec, MsgPackCodec};
    use crate::transport::traits::{FrontendEvent, FrontendTransport, WorkerTransport};
    use crate::worker_group::WorkerGroup;
    use infer_protocol::scheduler_to_server::{InferenceResponse, StreamChunk};
    use infer_protocol::worker_to_scheduler_control::{WorkerCapacity, WorkerReady};

    #[derive(Default)]
    struct MockFrontend;

    #[async_trait]
    impl FrontendTransport for MockFrontend {
        async fn recv_event(&mut self) -> Result<FrontendEvent> {
            Err(crate::error::SchedulerError::Shutdown)
        }

        async fn send_response(
            &mut self,
            _client: &ClientId,
            _response: InferenceResponse,
        ) -> Result<()> {
            Ok(())
        }

        async fn send_stream_chunk(
            &mut self,
            _client: &ClientId,
            _chunk: StreamChunk,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct MockWorker {
        sent: Arc<Mutex<Vec<Vec<u8>>>>,
    }

    #[async_trait]
    impl WorkerTransport for MockWorker {
        async fn send_batch(&mut self, cmd: Vec<u8>) -> Result<()> {
            self.sent.lock().unwrap().push(cmd);
            Ok(())
        }

        async fn recv_step_output(&mut self) -> Result<Vec<u8>> {
            Err(crate::error::SchedulerError::Shutdown)
        }
    }

    fn worker_group() -> WorkerGroup {
        WorkerGroup::from_single_ready(WorkerReady {
            worker_id: "worker-test".to_string(),
            model_instance_id: "default".to_string(),
            model_path: "model".to_string(),
            model_type: "llama3".to_string(),
            device: "cuda:0".to_string(),
            capacity: WorkerCapacity {
                max_batch_tokens: 256,
                max_batch_seqs: 4,
                max_running_requests: 4,
                max_total_kv_tokens: Some(32),
                free_mem_before_load_gb: None,
                free_mem_after_load_gb: None,
                weight_mem_usage_gb: None,
                workspace_mem_usage_gb: None,
                graph_mem_usage_gb: None,
            },
        })
    }

    fn prefilling_sequence() -> Sequence<Prefilling> {
        let meta = Arc::new(RequestMeta {
            id: RequestId("req-need-blocks".to_string()),
            sequence_id: SequenceId(7),
            input_ids: vec![1, 2, 3, 4],
            max_tokens: 8,
            sampling: SamplingParams::default(),
            priority: Priority::default(),
            stream: false,
            stop_sequences: vec![],
            diffusion: None,
            arrival_time: Instant::now(),
        });
        Sequence {
            meta,
            handle: RequestHandle::noop(),
            state: Prefilling {
                kv_alloc: KvAllocation::Blocks(vec![PhysicalBlockId(0)]),
                num_computed_tokens: 0,
                inflight: Some(InFlightPrefillSegment {
                    segment_start: 0,
                    segment_end: 4,
                    is_final: true,
                }),
                prompt_len: 4,
                prefill_start: Instant::now(),
            },
        }
    }

    #[tokio::test]
    async fn step_output_final_prefill_need_blocks_grants_after_decode_transition() -> Result<()> {
        let config = SchedulerConfig {
            kv_cache_mode: KvCacheMode::Paged { block_size: 4 },
            num_gpu_blocks: 4,
            ..Default::default()
        };
        let worker = MockWorker::default();
        let sent = Arc::clone(&worker.sent);
        let mut engine = SchedulerEngine::new(
            config,
            ContinuousBatchingPolicy::new(None),
            Box::new(PagedKvManager::new(4, 4)),
            worker_group(),
            MockFrontend,
            worker,
        );
        let seq = prefilling_sequence();
        let request_id = seq.meta.id.clone();
        engine
            .requests
            .insert_new(Arc::clone(&seq.meta), RequestHandle::noop())?;
        let queued = engine.requests.take_waiting(&request_id)?;
        engine.requests.commit_prefill_start(
            queued,
            KvAllocation::Blocks(vec![PhysicalBlockId(0)]),
            crate::cache::traits::PrefixMatch::none(),
            4,
        )?;

        let codec = MsgPackCodec;
        let output = StepOutput {
            prefill_done: vec![7],
            tokens: vec![GeneratedToken {
                sequence_id: 7,
                token_id: 42,
                finished: false,
            }],
            need_blocks: vec![NeedBlocksRequest {
                sequence_id: 7,
                current_blocks: 1,
                required_blocks: 2,
                request_blocks: 1,
            }],
            error: None,
        };
        engine
            .handle_step_output_llm(codec.encode(&output)?)
            .await?;

        assert!(engine.requests.prefilling().is_empty());
        assert_eq!(engine.requests.decoding().len(), 1);
        assert_eq!(engine.requests.decoding()[0].state.output_tokens, vec![42]);
        let KvAllocation::Blocks(blocks) = &engine.requests.decoding()[0].state.kv_alloc else {
            panic!("expected paged blocks allocation");
        };
        assert_eq!(blocks.len(), 2);

        let sent = sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        let cmd: WorkerCommand = codec.decode(&sent[0])?;
        let WorkerCommand::GrantBlocks(grant) = cmd else {
            panic!("expected GrantBlocks command");
        };
        assert_eq!(grant.sequence_id, 7);
        assert_eq!(grant.block_ids.len(), 1);
        Ok(())
    }
}
