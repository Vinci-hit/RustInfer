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
use crate::request::active_table::ActiveRequestTable;
use crate::request::handle::{ClientId, RequestHandle};
use crate::request::lifecycle::*;
use crate::request::queue::WaitingQueue;
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
    waiting_queue: WaitingQueue,
    prefilling: Vec<Sequence<Prefilling>>,
    decoding: Vec<Sequence<Decoding>>,
    active_requests: ActiveRequestTable,

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
            waiting_queue: WaitingQueue::new(),
            prefilling: Vec::new(),
            decoding: Vec::new(),
            active_requests: ActiveRequestTable::new(),
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

        self.active_requests
            .insert_waiting(meta.id.clone(), meta.input_ids.len(), meta.max_tokens);

        let handle = RequestHandle::new(client_id, request.stream);
        let seq = Sequence::new(meta.clone(), handle);

        self.waiting_queue.push(seq);
        self.metrics.record_enqueue();
    }

    /// Run one scheduling iteration.
    pub(crate) async fn run_iteration(&mut self) -> Result<()> {
        if self.config.mode == SchedulerMode::Diffusion && self.worker_busy {
            return Ok(());
        }
        if self.waiting_queue.is_empty() && self.prefilling.is_empty() && self.decoding.is_empty() {
            return Ok(());
        }

        self.iteration_id += 1;

        // Build continuation info for prefilling sequences that need more chunks.
        let prefilling_continuations: Vec<(RequestId, usize)> = self
            .prefilling
            .iter()
            .filter(|seq| !seq.has_inflight())
            .map(|seq| (seq.meta.id.clone(), seq.remaining_tokens()))
            .collect();

        let running_set = RunningSet {
            num_prefilling: self.prefilling.len(),
            num_decoding: self.decoding.len(),
            decode_tokens: self.decoding.len(),
            running_ids: self.decoding.iter().map(|s| s.meta.id.clone()).collect(),
            prefilling_continuations,
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
            .schedule(&self.waiting_queue, &running_set, &budget, &cache_state);

        if !plan.has_work() && self.decoding.is_empty() && self.prefilling.is_empty() {
            return Ok(());
        }

        // Execute plan: allocate KV for new prefills.
        self.execute_plan(&plan)?;

        // Build and send batch command.
        let batch_data = match self.config.mode {
            SchedulerMode::Llm => crate::core::batch_builder::build_batch(
                &self.prefilling,
                &self.decoding,
                &self.config,
                &self.codec,
                &self.current_chunk_sizes,
            )?,
            SchedulerMode::Diffusion => crate::core::batch_builder::build_diffusion_batch(
                &self.prefilling,
                &self.codec,
                &self.current_chunk_sizes,
            )?,
        };

        if !batch_data.is_empty() {
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
            // Check if this is a continuation (already in prefilling).
            let is_continuation = self
                .prefilling
                .iter()
                .any(|s| s.meta.id == entry.request_id);

            if is_continuation {
                // Continuation chunk: mark exact KV/prompt segment in flight.
                if let Some(seq) = self
                    .prefilling
                    .iter_mut()
                    .find(|s| s.meta.id == entry.request_id)
                {
                    if seq.has_inflight() {
                        continue;
                    }
                    let start = seq.state.num_computed_tokens;
                    let end = (start + entry.token_range.len()).min(seq.state.prompt_len);
                    if start >= end {
                        continue;
                    }
                    seq.set_inflight(start, end);
                    self.current_chunk_sizes
                        .push((entry.request_id.clone(), end - start));
                }
            } else {
                // New request: pop from waiting, allocate KV, move to prefilling.
                let seq = match self.waiting_queue.remove(&entry.request_id) {
                    Some(s) => s,
                    None => {
                        tracing::warn!("Plan references unknown request: {}", entry.request_id);
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
                        self.waiting_queue.push_front(seq);
                        break;
                    }
                };

                self.active_requests
                    .mark_prefilling(&entry.request_id, kv_alloc.clone());
                let mut prefilling_seq = seq.start_prefill(kv_alloc);
                prefilling_seq.state.num_computed_tokens = prefix_match
                    .num_cached_tokens
                    .min(prefilling_seq.state.prompt_len);
                let start = prefilling_seq.state.num_computed_tokens;
                let scheduled_len = entry.token_range.len();
                let end = (start + scheduled_len).min(prefilling_seq.state.prompt_len);
                if end == 0 || start >= end {
                    self.prefilling.push(prefilling_seq);
                    continue;
                }
                prefilling_seq.set_inflight(start, end);
                self.prefilling.push(prefilling_seq);

                self.current_chunk_sizes
                    .push((entry.request_id.clone(), end - start));
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
            let Some(idx) = self
                .prefilling
                .iter()
                .position(|s| s.meta.sequence_id.0 == sequence_id)
            else {
                tracing::warn!("PrefillDone for unknown sequence_id={}", sequence_id);
                continue;
            };

            let mut seq = self.prefilling.remove(idx);
            let Some(inflight) = seq.ack_inflight() else {
                tracing::warn!(
                    "PrefillDone for sequence_id={} without inflight segment",
                    sequence_id
                );
                self.prefilling.push(seq);
                continue;
            };
            if inflight.is_final || seq.is_complete() {
                self.active_requests.mark_decoding(&seq.meta.id);
                self.decoding.push(seq.start_decode());
            } else {
                self.prefilling.push(seq);
            }
        }

        for need in &output.need_blocks {
            let Some(seq_idx) = self
                .decoding
                .iter()
                .position(|s| s.meta.sequence_id.0 == need.sequence_id)
            else {
                tracing::debug!(
                    "NeedBlocks for non-decoding sequence_id={}",
                    need.sequence_id
                );
                continue;
            };
            match self
                .kv_manager
                .allocate_decode_blocks(need.request_blocks as usize)
            {
                Ok(blocks) => {
                    if let crate::cache::kv_manager::KvAllocation::Blocks(existing) =
                        &mut self.decoding[seq_idx].state.kv_alloc
                    {
                        existing.extend(blocks.iter().copied());
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
        let mut finished_indices: Vec<usize> = Vec::new();
        let mut token_chunks: Vec<(ClientId, StreamChunk)> = Vec::new();
        for token in &output.tokens {
            if let Some(seq) = self
                .decoding
                .iter_mut()
                .find(|s| s.meta.sequence_id.0 == token.sequence_id)
            {
                seq.append_token(token.token_id);
                self.active_requests.record_generated_token(&seq.meta.id);

                if seq.meta.stream {
                    token_chunks.push((
                        ClientId(seq.handle.client_id.0.clone()),
                        StreamChunk {
                            request_id: seq.meta.id.0.clone(),
                            chunk_type: ChunkType::Token,
                            token_id: Some(token.token_id),
                            finish_reason: None,
                            metrics: None,
                        },
                    ));
                }

                if token.finished
                    && let Some(idx) = self
                        .decoding
                        .iter()
                        .position(|s| s.meta.sequence_id.0 == token.sequence_id)
                {
                    finished_indices.push(idx);
                }
            } else {
                tracing::warn!(
                    "Generated token for unknown sequence_id={}",
                    token.sequence_id
                );
            }
        }

        // Send token chunks before done chunks for sequences that finish in this same step.
        for (client_id, chunk) in token_chunks {
            self.frontend.send_stream_chunk(&client_id, chunk).await?;
        }

        finished_indices.sort_unstable();
        finished_indices.dedup();
        for &idx in finished_indices.iter().rev() {
            if idx < self.decoding.len() {
                let seq = self.decoding.remove(idx);
                self.complete_sequence(seq).await?;
            }
        }

        Ok(())
    }

    async fn handle_worker_step_error(&mut self, err: WorkerStepError) -> Result<()> {
        let mut failed_ids = err.sequence_ids.clone();
        if err.fatal || failed_ids.is_empty() {
            failed_ids.extend(self.prefilling.iter().map(|seq| seq.meta.sequence_id.0));
            failed_ids.extend(self.decoding.iter().map(|seq| seq.meta.sequence_id.0));
        }
        failed_ids.sort_unstable();
        failed_ids.dedup();

        for sequence_id in failed_ids {
            if let Some(idx) = self
                .prefilling
                .iter()
                .position(|seq| seq.meta.sequence_id.0 == sequence_id)
            {
                let seq = self.prefilling.remove(idx);
                self.fail_prefilling_sequence(seq, &err.message).await?;
                continue;
            }
            if let Some(idx) = self
                .decoding
                .iter()
                .position(|seq| seq.meta.sequence_id.0 == sequence_id)
            {
                let seq = self.decoding.remove(idx);
                self.fail_decoding_sequence(seq, &err.message).await?;
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
        let _ = self.active_requests.finish(&request_id);
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
        let _ = self.active_requests.finish(&request_id);
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
            let Some(idx) = self
                .prefilling
                .iter()
                .position(|s| s.meta.id.0 == item.request_id)
            else {
                tracing::warn!(
                    "Diffusion output for unknown request_id={}",
                    item.request_id
                );
                continue;
            };
            let seq = self.prefilling.remove(idx);
            let request_id = seq.meta.id.clone();
            let client_id = ClientId(seq.handle.client_id.0.clone());
            self.kv_manager.free(seq.state.kv_alloc);
            let _ = self.active_requests.finish(&request_id);

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
        let _ = self.active_requests.finish(&request_id);

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
        !self.waiting_queue.is_empty() || !self.prefilling.is_empty() || !self.decoding.is_empty()
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
        self.active_requests.len()
    }

    #[allow(dead_code)]
    pub(crate) async fn cancel_request(&mut self, request_id: RequestId) -> Result<()> {
        self.active_requests.mark_cancelling(&request_id);

        if self.waiting_queue.remove(&request_id).is_some() {
            self.active_requests.finish(&request_id);
            return Ok(());
        }

        if let Some(idx) = self.prefilling.iter().position(|s| s.meta.id == request_id) {
            let seq = self.prefilling.remove(idx);
            let sequence_id = seq.meta.sequence_id.0;
            self.kv_manager.free(seq.state.kv_alloc);
            self.active_requests.finish(&request_id);
            let data = crate::core::batch_builder::build_cancel_request(sequence_id, &self.codec)?;
            return self.worker.send_batch(data).await;
        }

        if let Some(idx) = self.decoding.iter().position(|s| s.meta.id == request_id) {
            let seq = self.decoding.remove(idx);
            let sequence_id = seq.meta.sequence_id.0;
            let prompt_tokens = seq.meta.input_ids.clone();
            self.kv_manager
                .free_finished(&prompt_tokens, seq.state.kv_alloc);
            self.active_requests.finish(&request_id);
            let data = crate::core::batch_builder::build_cancel_request(sequence_id, &self.codec)?;
            return self.worker.send_batch(data).await;
        }

        Ok(())
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
        engine.prefilling.push(prefilling_sequence());

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

        assert!(engine.prefilling.is_empty());
        assert_eq!(engine.decoding.len(), 1);
        assert_eq!(engine.decoding[0].state.output_tokens, vec![42]);
        let KvAllocation::Blocks(blocks) = &engine.decoding[0].state.kv_alloc else {
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
