//! SchedulerEngine — the top-level async orchestrator.
//!
//! Owns all mutable state and drives the event loop.

use std::sync::Arc;
use std::time::Instant;

use infer_protocol::scheduler_to_server::{InferenceMetrics, InferenceResponse, ResponseStatus};
use infer_protocol::server_to_scheduler::InferenceRequest;
use infer_protocol::worker_to_scheduler::StepOutput;

use crate::cache::kv_manager::KvManager;
use crate::cache::traits::CacheState;
use crate::config::{SchedulerConfig, SchedulerMode};
use crate::error::Result;
use crate::metrics::MetricsRecorder;
use crate::policy::traits::{BatchPlan, RunningSet, SchedulingPolicy};
use crate::request::handle::{ClientId, RequestHandle};
use crate::request::active_table::ActiveRequestTable;
use crate::request::lifecycle::*;
use crate::request::queue::WaitingQueue;
use crate::transport::codec::MsgPackCodec;
use crate::transport::traits::{FrontendTransport, WorkerTransport};
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
    worker_busy: bool,
    /// Tokens scheduled for each prefilling sequence in the current batch.
    /// Used by handle_step_output to know how much to advance each seq.
    /// Key: request_id, Value: tokens processed this iteration.
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

        let meta = Arc::new(RequestMeta {
            id: RequestId(request.request_id.clone()),
            input_ids: request.input_ids,
            max_tokens: request.max_tokens,
            sampling: SamplingParams {
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: request.top_k,
            },
            priority: Priority(request.priority),
            stream: request.stream,
            stop_sequences: vec![],
            arrival_time: Instant::now(),
        });

        self.active_requests.insert_waiting(
            meta.id.clone(),
            meta.input_ids.len(),
            meta.max_tokens,
        );

        let handle = RequestHandle::new(client_id, request.stream);
        let seq = Sequence::new(meta.clone(), handle);

        tracing::debug!(
            "Enqueued request {}: {} input tokens, max_tokens={}",
            meta.id,
            meta.input_ids.len(),
            meta.max_tokens,
        );

        self.waiting_queue.push(seq);
        self.metrics.record_enqueue();
    }

    /// Run one scheduling iteration.
    pub(crate) async fn run_iteration(&mut self) -> Result<()> {
        if self.worker_busy {
            return Ok(());
        }
        if self.waiting_queue.is_empty() && self.prefilling.is_empty() && self.decoding.is_empty() {
            return Ok(());
        }

        self.iteration_id += 1;

        // Build continuation info for prefilling sequences that need more chunks.
        let prefilling_continuations: Vec<(RequestId, usize)> = self.prefilling
            .iter()
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

        let plan = self.policy.schedule(
            &self.waiting_queue,
            &running_set,
            &budget,
            &cache_state,
        );

        if !plan.has_work() && self.decoding.is_empty() && self.prefilling.is_empty() {
            return Ok(());
        }

        // Execute plan: allocate KV for new prefills.
        self.execute_plan(&plan)?;

        // Build and send batch command.
        let batch_data = crate::core::batch_builder::build_batch(
            &self.prefilling,
            &self.decoding,
            &self.config,
            &self.codec,
            &self.current_chunk_sizes,
        )?;

        if !batch_data.is_empty() {
            self.worker.send_batch(batch_data).await?;
            self.worker_busy = true;
        }

        Ok(())
    }

    /// Execute the scheduling plan: handle new prefills and continuation chunks.
    fn execute_plan(&mut self, plan: &BatchPlan) -> Result<()> {
        self.current_chunk_sizes.clear();

        for entry in &plan.prefill_batch {
            // Check if this is a continuation (already in prefilling).
            let is_continuation = self.prefilling.iter().any(|s| s.meta.id == entry.request_id);

            if is_continuation {
                // Continuation chunk: record how many tokens to process this iteration.
                self.current_chunk_sizes.push((
                    entry.request_id.clone(),
                    entry.token_range.len(),
                ));
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
                let kv_alloc = match self.kv_manager.allocate(prompt_len) {
                    Ok(alloc) => alloc,
                    Err(e) => {
                        tracing::warn!("KV allocation failed for {}: {}", entry.request_id, e);
                        self.waiting_queue.push_front(seq);
                        break;
                    }
                };

                self.active_requests
                    .mark_prefilling(&entry.request_id, kv_alloc.clone());
                let prefilling_seq = seq.start_prefill(kv_alloc);
                self.prefilling.push(prefilling_seq);

                // Record chunk size for this new entry.
                self.current_chunk_sizes.push((
                    entry.request_id.clone(),
                    entry.token_range.len(),
                ));
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

    /// LLM mode: process per-token outputs, advance chunks, transition prefill→decode→finish.
    async fn handle_step_output_llm(&mut self, data: Vec<u8>) -> Result<()> {
        use crate::transport::codec::Codec;
        let output: StepOutput = self.codec.decode(&data)?;

        let mut finished_indices: Vec<usize> = Vec::new();

        // Process tokens for decoding sequences.
        for seq_token in &output.tokens {
            if let Some(seq) = self.decoding.iter_mut().find(|s| s.meta.id.0 == seq_token.request_id) {
                seq.append_token(seq_token.token_id);
                self.active_requests.record_generated_token(&seq.meta.id);
                if seq_token.finished || seq.reached_max_tokens() {
                    if let Some(idx) = self.decoding.iter().position(|s| s.meta.id.0 == seq_token.request_id) {
                        finished_indices.push(idx);
                    }
                }
            }
        }

        // Move completed prefilling → decoding using recorded chunk sizes.
        let chunk_sizes = std::mem::take(&mut self.current_chunk_sizes);
        let mut new_decoding = Vec::new();
        let mut remaining_prefilling = Vec::new();
        for mut seq in self.prefilling.drain(..) {
            let chunk = chunk_sizes
                .iter()
                .find(|(id, _)| *id == seq.meta.id)
                .map(|(_, size)| *size)
                .unwrap_or(seq.state.prompt_len);

            seq.advance_chunk(chunk);
            if seq.is_complete() {
                self.active_requests.mark_decoding(&seq.meta.id);
                new_decoding.push(seq.start_decode());
            } else {
                remaining_prefilling.push(seq);
            }
        }
        self.prefilling = remaining_prefilling;
        self.decoding.extend(new_decoding);

        // Handle finished sequences.
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

    /// Diffusion mode: entire batch completes at once. All prefilling → finished directly.
    async fn handle_step_output_diffusion(&mut self, _data: Vec<u8>) -> Result<()> {
        // In Diffusion mode, Worker returning means the entire batch is done.
        // All prefilling sequences go directly to Finished (skip Decoding).
        for seq in self.prefilling.drain(..) {
            let request_id = seq.meta.id.clone();
            let client_id = ClientId(seq.handle.client_id.0.clone());

            // Mark prefill complete so we can transition.
            let prompt_len = seq.state.prompt_len;
            let mut seq = seq;
            seq.advance_chunk(prompt_len);

            // Prefilling → Decoding → Finished (instant transition, no tokens generated).
            let decoding_seq = seq.start_decode();
            let kv_alloc = decoding_seq.state.kv_alloc.clone();
            let finished = decoding_seq.finish(FinishReason::Eos);

            self.kv_manager.free(kv_alloc);
            let _ = self.active_requests.finish(&request_id);

            let elapsed_ms = finished.state.metrics.e2e_latency.as_millis() as u64;
            let response = InferenceResponse {
                request_id: request_id.0.clone(),
                status: ResponseStatus::Success,
                output_token_ids: vec![], // Diffusion output is image, not tokens
                finish_reason: Some("stop".to_string()),
                error: None,
                metrics: InferenceMetrics {
                    total_ms: elapsed_ms,
                    num_tokens: 0,
                    tokens_per_second: 0.0,
                },
            };

            self.frontend.send_response(&client_id, response).await?;
            self.metrics.record_completion(elapsed_ms, 0);

            tracing::info!("Completed diffusion request {} in {}ms", request_id, elapsed_ms);
        }

        Ok(())
    }

    /// Complete a sequence: send response, free KV, record metrics.
    async fn complete_sequence(&mut self, seq: Sequence<Decoding>) -> Result<()> {
        let request_id = seq.meta.id.clone();
        let client_id = ClientId(seq.handle.client_id.0.clone());
        let _ = self.active_requests.finish(&request_id);

        let reason = if seq.reached_max_tokens() {
            FinishReason::MaxTokens
        } else {
            FinishReason::Eos
        };

        // Extract KV allocation to free BEFORE consuming the sequence.
        let kv_alloc = seq.state.kv_alloc.clone();

        // Transition: Decoding → Finished.
        let finished = seq.finish(reason);

        // Free KV resources.
        self.kv_manager.free(kv_alloc);

        // Build response.
        let elapsed_ms = finished.state.metrics.e2e_latency.as_millis() as u64;
        let num_tokens = finished.state.metrics.num_output_tokens;
        let tokens_per_second = if elapsed_ms > 0 {
            (num_tokens as f64 / elapsed_ms as f64) * 1000.0
        } else {
            0.0
        };

        let response = InferenceResponse {
            request_id: request_id.0.clone(),
            status: ResponseStatus::Success,
            output_token_ids: finished.state.output_tokens,
            finish_reason: Some("stop".to_string()),
            error: None,
            metrics: InferenceMetrics {
                total_ms: elapsed_ms,
                num_tokens,
                tokens_per_second,
            },
        };

        self.frontend.send_response(&client_id, response).await?;
        self.metrics.record_completion(elapsed_ms, num_tokens);

        tracing::info!(
            "Completed {}: {} tokens in {}ms ({:.1} tok/s)",
            request_id, num_tokens, elapsed_ms, tokens_per_second,
        );

        Ok(())
    }

    // ─── Accessors for event_loop ───

    pub(crate) fn has_pending_work(&self) -> bool {
        !self.waiting_queue.is_empty() || !self.prefilling.is_empty() || !self.decoding.is_empty()
    }

    #[allow(dead_code)]
    pub(crate) fn is_idle(&self) -> bool {
        !self.has_pending_work() && !self.worker_busy
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
        self.worker_busy
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
            let _ = self.active_requests.finish(&request_id);
            return Ok(());
        }

        let data = crate::core::batch_builder::build_cancel_request(&request_id, &self.codec)?;
        self.worker.send_batch(data).await
    }

    #[allow(dead_code)]
    /// Receive a request from the frontend transport.
    pub(crate) async fn recv_frontend_request(&mut self) -> Result<(ClientId, InferenceRequest)> {
        self.frontend.recv_request().await
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

        let has_work = self.has_pending_work() || self.worker_busy;

        if has_work {
            // Both branches active.
            let frontend = &mut self.frontend;
            let worker = &mut self.worker;

            tokio::select! {
                biased;
                result = worker.recv_step_output() => EngineEvent::WorkerOutput(result),
                result = frontend.recv_request() => EngineEvent::NewRequest(result),
            }
        } else {
            // Idle: only listen for new requests.
            let result = self.frontend.recv_request().await;
            EngineEvent::NewRequest(result)
        }
    }
}
