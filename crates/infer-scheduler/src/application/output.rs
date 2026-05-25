//! `OutputProcessingSystem` — terminal-state owner.
//!
//! Step 15 ports the engine's response-emission + KV-cleanup paths
//! into a single System. Per the refactor plan (P2-F), this System
//! is the **sole owner** of returning KV resources to the pool —
//! no other System or engine method may call `kv.free*` directly.
//!
//! ## Surface
//!
//! - [`OutputProcessingSystem::send_request_error`] — construct and
//!   emit an error response (stream chunk vs. unary), no state change.
//! - [`OutputProcessingSystem::fail_prefilling_session`] — KV release
//!   for a Prefilling session + send error.
//! - [`OutputProcessingSystem::fail_decoding_session`] — same for
//!   Decoding (carries partial token count for metrics).
//! - [`OutputProcessingSystem::complete_session`] — emit success
//!   response, route prompt tokens through prefix cache, release KV,
//!   record completion metrics.
//!
//! ## Borrow shape
//!
//! Each method takes the resources it needs as `&mut`/`&` arguments
//! rather than holding them in `self`. This lets the engine keep its
//! current ownership of `RequestTable` / `KvManager` / transports
//! without forcing field aliasing through the System. Step 18 will
//! tighten this when the engine slim-down lands.

use infer_protocol::scheduler_to_server::{
    ChunkType, InferenceMetrics, InferenceResponse, ResponseStatus, StreamChunk,
};

use crate::domain::kv_cache_pool::{KvCachePool, KvLease};
use crate::error::Result;
use crate::infrastructure::metrics::MetricsRecorder;
use crate::domain::inference_session::handle::ClientId;
use crate::domain::inference_session::lifecycle::{Decoding, FinishReason, InferenceSession, Prefilling};
use crate::infrastructure::transport::traits::FrontendTransport;

/// Output / cleanup stage. Stateless today (the `MetricsRecorder` is
/// passed in per-call rather than owned, since the engine still has
/// the only reference); a future Step 19 will inject a
/// `MetricsHandle: Arc<MetricsRecorder>` here.
#[derive(Debug, Default)]
pub struct OutputProcessingSystem;

impl OutputProcessingSystem {
    pub fn new() -> Self {
        Self
    }

    /// Emit an error to the client. Does not touch KV (callers do).
    pub async fn send_request_error(
        &self,
        frontend: &mut dyn FrontendTransport,
        client_id: ClientId,
        external_id: String,
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
            frontend
                .send_stream_chunk(
                    &client_id,
                    StreamChunk {
                        request_id: external_id,
                        chunk_type: ChunkType::Error,
                        token_id: None,
                        finish_reason: Some(message),
                        metrics: Some(metrics),
                    },
                )
                .await
        } else {
            frontend
                .send_response(
                    &client_id,
                    InferenceResponse {
                        request_id: external_id,
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

    /// Fail a session caught in `Prefilling`. Owner of KV release.
    pub async fn fail_prefilling_session(
        &self,
        frontend: &mut dyn FrontendTransport,
        kv: &mut dyn KvCachePool,
        seq: InferenceSession<Prefilling>,
        message: &str,
    ) -> Result<()> {
        let external_id = seq.meta.external_id.clone();
        let client_id = ClientId::new(seq.handle.client_id.as_bytes().to_vec());
        let stream = seq.meta.stream;
        // Drop the lease without prefix-cache routing — failed
        // prompts shouldn't pollute the cache. The KvLease Drop
        // guard parks blocks in the return sink for the next
        // `flush_pending_returns` to drain.
        let _ = (kv, seq.state.kv_lease);
        self.send_request_error(frontend, client_id, external_id, stream, message.to_string(), 0)
            .await
    }

    /// Fail a session caught in `Decoding`. Carries the partial
    /// token count so error metrics reflect work already done.
    pub async fn fail_decoding_session(
        &self,
        frontend: &mut dyn FrontendTransport,
        kv: &mut dyn KvCachePool,
        seq: InferenceSession<Decoding>,
        message: &str,
    ) -> Result<()> {
        let external_id = seq.meta.external_id.clone();
        let client_id = ClientId::new(seq.handle.client_id.as_bytes().to_vec());
        let stream = seq.meta.stream;
        let num_tokens = seq.state.output_tokens.len() as u32;
        let _ = (kv, seq.state.kv_lease);
        self.send_request_error(
            frontend,
            client_id,
            external_id,
            stream,
            message.to_string(),
            num_tokens,
        )
        .await
    }

    /// Successfully complete a Decoding session. Routes the full
    /// prompt through the prefix cache (so a future request with the
    /// same prompt can short-circuit), then frees the lease blocks.
    pub async fn complete_session(
        &self,
        frontend: &mut dyn FrontendTransport,
        kv: &mut dyn KvCachePool,
        metrics: &MetricsRecorder,
        seq: InferenceSession<Decoding>,
    ) -> Result<CompleteOutcome> {
        let request_id_display = seq.meta.id.to_string();
        let external_id = seq.meta.external_id.clone();
        let client_id = ClientId::new(seq.handle.client_id.as_bytes().to_vec());
        let stream = seq.meta.stream;

        let reason = if seq.reached_max_tokens() {
            FinishReason::MaxTokens
        } else {
            FinishReason::Eos
        };

        let prompt_tokens = seq.meta.input_ids.clone();
        // `finish` consumes the session and yields the lease + output
        // bookkeeping. We then route the lease through `free_finished`
        // (P2-F: single owner of KV release).
        let finished = seq.finish(reason);
        kv.free_finished(&prompt_tokens, finished.state.kv_lease);

        let elapsed_ms = finished.state.metrics.e2e_latency.as_millis() as u64;
        let num_tokens = finished.state.metrics.num_output_tokens;
        let tokens_per_second = if elapsed_ms > 0 {
            (num_tokens as f64 / elapsed_ms as f64) * 1000.0
        } else {
            0.0
        };
        let response_metrics = InferenceMetrics {
            total_ms: elapsed_ms,
            num_tokens,
            tokens_per_second,
        };

        if stream {
            frontend
                .send_stream_chunk(
                    &client_id,
                    StreamChunk {
                        request_id: external_id,
                        chunk_type: ChunkType::Done,
                        token_id: None,
                        finish_reason: Some("stop".to_string()),
                        metrics: Some(response_metrics.clone()),
                    },
                )
                .await?;
        } else {
            let response = InferenceResponse {
                request_id: external_id,
                status: ResponseStatus::Success,
                output_token_ids: finished.state.output_tokens,
                images: vec![],
                finish_reason: Some("stop".to_string()),
                error: None,
                metrics: response_metrics.clone(),
            };
            frontend.send_response(&client_id, response).await?;
        }

        metrics.record_completion(elapsed_ms, num_tokens);

        Ok(CompleteOutcome {
            request_id_display,
            num_tokens,
            elapsed_ms,
            tokens_per_second,
        })
    }
}

/// Diagnostic bundle returned by [`OutputProcessingSystem::complete_session`].
///
/// The engine's tracing line wants all four fields together; instead
/// of forcing a destructure we hand back a small typed record so the
/// log line is unambiguous and future fields (e.g. `prefill_latency`)
/// extend without breaking callers.
#[derive(Debug)]
pub struct CompleteOutcome {
    pub request_id_display: String,
    pub num_tokens: u32,
    pub elapsed_ms: u64,
    pub tokens_per_second: f64,
}

impl OutputProcessingSystem {
    /// Release KV for a session canceled while still in `Prefilling`.
    /// No client response (the client already initiated the cancel
    /// and isn't waiting for an ack from us). Lease drop returns
    /// the blocks to the sink for the next `flush_pending_returns`.
    pub fn release_canceled_prefill(&self, kv: &mut dyn KvCachePool, kv_lease: KvLease) {
        // Cancel doesn't go through prefix cache: a partial prompt is
        // not worth caching. Just drop the lease and flush.
        let _ = (kv, kv_lease);
    }

    /// Release KV for a session canceled in `Decoding`. Routes the
    /// prompt through the prefix cache (so the work isn't wasted —
    /// a future request with the same prefix can reuse the blocks).
    pub fn release_canceled_decode(
        &self,
        kv: &mut dyn KvCachePool,
        prompt_tokens: &[i32],
        kv_lease: KvLease,
    ) {
        kv.free_finished(prompt_tokens, kv_lease);
    }

    /// Process one batch of worker step output (LLM mode).
    ///
    /// Drives three transitions per call:
    /// 1. ACK each `prefill_done` sequence — the table moves it
    ///    from `Prefilling` to `Decoding` (or restarts the next
    ///    prefill chunk for chunked prefill).
    /// 2. Append every emitted token to its session and emit a
    ///    streaming chunk for streaming clients.
    /// 3. Drive the success-completion path for any session that
    ///    the worker flagged as `finished` in this step.
    ///
    /// Token chunks are sent **before** completion responses so
    /// streaming clients see the final token before the `Done`
    /// chunk.
    #[allow(clippy::too_many_arguments)]
    pub async fn process_llm_step(
        &self,
        sessions: &mut crate::domain::inference_session::table::RequestTable,
        kv: &mut dyn KvCachePool,
        frontend: &mut dyn FrontendTransport,
        metrics: &MetricsRecorder,
        codec: &crate::infrastructure::transport::codec::MsgPackCodec,
        data: Vec<u8>,
    ) -> Result<()> {
        use crate::domain::inference_session::lifecycle::SequenceId;
        use crate::domain::inference_session::table::PrefillAckOutcome;
        use crate::infrastructure::transport::codec::Codec;
        use infer_protocol::worker_to_scheduler_data::StepOutput;

        let output: StepOutput = codec.decode(&data)?;

        // 1. ACK prefill segments. Final segment moves the sequence to
        // decoding before processing the token generated by that final prefill.
        for sequence_id in output.prefill_done {
            match sessions.ack_prefill(SequenceId(sequence_id)) {
                Ok(PrefillAckOutcome::MovedToDecoding { .. })
                | Ok(PrefillAckOutcome::Continue { .. }) => {}
                Err(e) => tracing::warn!(
                    "PrefillDone for sequence_id={} failed: {}",
                    sequence_id,
                    e
                ),
            }
        }

        // 2. Process generated tokens. Worker owns EOS / max_tokens
        // and is the only component allowed to end decode.
        let mut finished_sequence_ids: Vec<SequenceId> = Vec::new();
        let mut token_chunks: Vec<(ClientId, StreamChunk)> = Vec::new();
        for token in &output.tokens {
            match sessions.append_generated_token(
                SequenceId(token.sequence_id),
                token.token_id,
                token.finished,
            ) {
                Ok(outcome) => {
                    if outcome.stream {
                        token_chunks.push((
                            outcome.client_id,
                            StreamChunk {
                                request_id: outcome.external_id,
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

        // Send token chunks before completion responses for sequences
        // that finish in this same step.
        for (client_id, chunk) in token_chunks {
            frontend.send_stream_chunk(&client_id, chunk).await?;
        }

        finished_sequence_ids.sort_unstable_by_key(|id| id.0);
        finished_sequence_ids.dedup();
        for sequence_id in finished_sequence_ids {
            let seq = sessions.finish_decoding(sequence_id)?;
            let outcome = self.complete_session(frontend, kv, metrics, seq).await?;
            tracing::info!(
                "Completed {}: {} tokens in {}ms ({:.1} tok/s)",
                outcome.request_id_display,
                outcome.num_tokens,
                outcome.elapsed_ms,
                outcome.tokens_per_second,
            );
        }

        Ok(())
    }

    /// Drive the failure path for a list of internal `RequestId`s.
    ///
    /// Used by [`crate::application::ControlEventSystem`] consumers
    /// (when `ControlOutcome::Continue` carries failed ids) and by
    /// the engine's `Terminate` path that drains every running
    /// session before unwinding.
    pub async fn fail_sessions(
        &self,
        sessions: &mut crate::domain::inference_session::table::RequestTable,
        kv: &mut dyn KvCachePool,
        frontend: &mut dyn FrontendTransport,
        failed_request_ids: &[crate::domain::inference_session::lifecycle::RequestId],
        message: &str,
    ) -> Result<()> {
        use crate::domain::inference_session::table::FailedOutcome;

        for rid in failed_request_ids {
            let Some(sid) = sessions.sequence_id_for(rid) else {
                continue;
            };
            match sessions.fail_sequence(sid, message)? {
                FailedOutcome::RemovedPrefilling { sequence, .. } => {
                    self.fail_prefilling_session(frontend, kv, sequence, message)
                        .await?;
                }
                FailedOutcome::RemovedDecoding { sequence, .. } => {
                    self.fail_decoding_session(frontend, kv, sequence, message)
                        .await?;
                }
                FailedOutcome::NotFound { .. } => {}
            }
        }
        Ok(())
    }

    /// Process one batch of worker step output (Diffusion mode).
    ///
    /// Diffusion is batch-in/batch-out: every request in the batch
    /// completes (or errors) at once. We walk the result vector,
    /// route each item through `take_prefilling_by_request` to drop
    /// the session out of the repository with the right terminal
    /// reason, release its KV (via `release_canceled_prefill` —
    /// diffusion uses no real KV), and emit the final
    /// `InferenceResponse` with image bytes.
    pub async fn process_diffusion_step(
        &self,
        sessions: &mut crate::domain::inference_session::table::RequestTable,
        kv: &mut dyn KvCachePool,
        frontend: &mut dyn FrontendTransport,
        metrics: &MetricsRecorder,
        codec: &crate::infrastructure::transport::codec::MsgPackCodec,
        data: Vec<u8>,
    ) -> Result<()> {
        use crate::domain::inference_session::table::TerminalReason;
        use crate::infrastructure::transport::codec::Codec;
        use infer_protocol::scheduler_to_server::ImageOutput;
        use infer_protocol::worker_to_scheduler_data::{DiffusionBatchOutput, DiffusionOutputStatus};

        let output: DiffusionBatchOutput = codec.decode(&data)?;

        for item in output.results {
            // Worker echoed the external_id; resolve back to internal id.
            let Some(seq_id) = sessions.sequence_id_for_external(&item.request_id) else {
                tracing::warn!(
                    "Diffusion output for unknown external_id={}",
                    item.request_id
                );
                continue;
            };
            let Some(item_request_id) = sessions.request_id_for_sequence(seq_id) else {
                tracing::warn!(
                    "Diffusion output: sequence_id={} no longer active",
                    seq_id
                );
                continue;
            };
            let reason = if matches!(item.status, DiffusionOutputStatus::Success) {
                TerminalReason::Finished
            } else {
                TerminalReason::Failed(
                    item.error.clone().unwrap_or_else(|| "diffusion error".to_string()),
                )
            };
            let Some(seq) = sessions.take_prefilling_by_request(&item_request_id, reason)? else {
                tracing::warn!(
                    "Diffusion output for unknown request_id={}",
                    item.request_id
                );
                continue;
            };
            let request_id_display = seq.meta.id.to_string();
            let external_id = seq.meta.external_id.clone();
            let client_id = ClientId::new(seq.handle.client_id.as_bytes().to_vec());
            // P2-F: KV release goes through OutputSystem; engine never
            // touches `kv.free*` directly.
            self.release_canceled_prefill(kv, seq.state.kv_lease);

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
                request_id: external_id,
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

            frontend.send_response(&client_id, response).await?;
            metrics.record_completion(elapsed_ms, 0);
            tracing::info!(
                "Completed diffusion request {} in {}ms",
                request_id_display,
                elapsed_ms
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::kv_cache_pool::KvLease;
    use crate::infrastructure::kv_cache::traits::PhysicalBlockId;
    use crate::domain::inference_session::handle::RequestHandle;
    use crate::domain::inference_session::lifecycle::{
        Decoding, InferenceSession, Prefilling, Priority, RequestId, RequestMeta,
        SamplingParams, SequenceId,
    };
    use async_trait::async_trait;
    use infer_protocol::scheduler_to_server::{InferenceResponse, StreamChunk};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    /// Capturing FrontendTransport: records every send_* call so tests
    /// can assert on emitted responses/chunks.
    #[derive(Default, Clone)]
    struct CapturingFrontend {
        responses: Arc<Mutex<Vec<InferenceResponse>>>,
        chunks: Arc<Mutex<Vec<StreamChunk>>>,
    }

    #[async_trait]
    impl crate::infrastructure::transport::traits::FrontendTransport for CapturingFrontend {
        async fn recv_event(&mut self) -> Result<crate::infrastructure::transport::traits::FrontendEvent> {
            Err(crate::error::SchedulerError::Shutdown)
        }
        async fn send_response(
            &mut self,
            _client: &ClientId,
            response: InferenceResponse,
        ) -> Result<()> {
            self.responses.lock().unwrap().push(response);
            Ok(())
        }
        async fn send_stream_chunk(
            &mut self,
            _client: &ClientId,
            chunk: StreamChunk,
        ) -> Result<()> {
            self.chunks.lock().unwrap().push(chunk);
            Ok(())
        }
    }

    /// Capturing KvCachePool: records `free_finished` calls so tests
    /// can verify the success path routes prompts through the
    /// prefix cache. `allocate` issues empty leases so tests don't
    /// need a real block allocator.
    #[derive(Default)]
    struct CapturingKv {
        free_finished_calls: Vec<Vec<i32>>,
    }

    impl crate::domain::kv_cache_pool::KvCachePool for CapturingKv {
        fn allocate(&mut self, _: crate::domain::ids::TokenCount) -> Result<KvLease> {
            Ok(KvLease::empty())
        }
        fn allocate_with_prefix(
            &mut self,
            _: &[i32],
        ) -> Result<(KvLease, crate::infrastructure::kv_cache::PrefixMatch)> {
            Ok((
                KvLease::empty(),
                crate::infrastructure::kv_cache::PrefixMatch::none(),
            ))
        }
        fn allocate_decode_blocks(
            &mut self,
            _: crate::domain::ids::BlockCount,
        ) -> Result<Vec<crate::infrastructure::kv_cache::PhysicalBlockId>> {
            Ok(Vec::new())
        }
        fn free_finished(&mut self, prompt_tokens: &[i32], _lease: KvLease) {
            self.free_finished_calls.push(prompt_tokens.to_vec());
        }
        fn match_prefix(
            &mut self,
            _: &[i32],
        ) -> crate::infrastructure::kv_cache::PrefixMatch {
            crate::infrastructure::kv_cache::PrefixMatch::none()
        }
        fn flush_pending_returns(&mut self) {}
        fn block_size(&self) -> crate::domain::ids::BlockSize {
            crate::domain::ids::BlockSize::new(1)
        }
        fn total_blocks(&self) -> crate::domain::ids::BlockCount {
            crate::domain::ids::BlockCount::new(0)
        }
        fn available_blocks(&self) -> crate::domain::ids::BlockCount {
            crate::domain::ids::BlockCount::new(0)
        }
        fn mode_name(&self) -> &'static str {
            "capturing"
        }
    }

    fn meta_for_test(stream: bool, prompt: Vec<i32>) -> Arc<RequestMeta> {
        Arc::new(RequestMeta {
            id: RequestId::new_v4(),
            external_id: "test-ext".to_string(),
            sequence_id: SequenceId(1),
            input_ids: prompt,
            max_tokens: 8,
            sampling: SamplingParams::default(),
            priority: Priority::default(),
            stream,
            stop_sequences: vec![],
            diffusion: None,
            arrival_time: Instant::now(),
        })
    }

    fn prefilling_session(stream: bool) -> InferenceSession<Prefilling> {
        InferenceSession {
            meta: meta_for_test(stream, vec![1, 2, 3, 4]),
            handle: RequestHandle::noop(),
            state: Prefilling {
                kv_lease: KvLease::test_with_blocks(vec![PhysicalBlockId(7)]),
                num_computed_tokens: 0,
                inflight: None,
                prompt_len: 4,
                prefill_start: Instant::now(),
            },
        }
    }

    fn decoding_session(stream: bool, output: Vec<i32>) -> InferenceSession<Decoding> {
        InferenceSession {
            meta: meta_for_test(stream, vec![1, 2, 3, 4]),
            handle: RequestHandle::noop(),
            state: Decoding {
                kv_lease: KvLease::test_with_blocks(vec![PhysicalBlockId(11)]),
                output_tokens: output,
                seq_position: 4,
                prompt_len: 4,
                first_token_time: Instant::now(),
                preemption_count: 0,
            },
        }
    }

    #[tokio::test]
    async fn fail_prefilling_releases_kv_and_emits_error() {
        let sys = OutputProcessingSystem::new();
        let mut frontend = CapturingFrontend::default();
        let mut kv = CapturingKv::default();
        sys.fail_prefilling_session(&mut frontend, &mut kv, prefilling_session(false), "bad")
            .await
            .unwrap();
        // Failure path drops the lease; prefix cache is NOT touched
        // (we don't want a partial / failed prompt polluting the cache).
        assert_eq!(kv.free_finished_calls.len(), 0, "no prefix-cache routing on failure");
        let resps = frontend.responses.lock().unwrap();
        assert_eq!(resps.len(), 1);
        assert!(matches!(resps[0].status, ResponseStatus::Error));
        assert_eq!(resps[0].metrics.num_tokens, 0);
    }

    #[tokio::test]
    async fn fail_decoding_carries_partial_token_count_to_metrics() {
        let sys = OutputProcessingSystem::new();
        let mut frontend = CapturingFrontend::default();
        let mut kv = CapturingKv::default();
        sys.fail_decoding_session(
            &mut frontend,
            &mut kv,
            decoding_session(false, vec![1, 2, 3]),
            "bang",
        )
        .await
        .unwrap();
        let resps = frontend.responses.lock().unwrap();
        assert_eq!(resps.len(), 1);
        assert_eq!(resps[0].metrics.num_tokens, 3, "3 partial output tokens");
    }

    #[tokio::test]
    async fn fail_decoding_streams_error_chunk_when_stream_true() {
        let sys = OutputProcessingSystem::new();
        let mut frontend = CapturingFrontend::default();
        let mut kv = CapturingKv::default();
        sys.fail_decoding_session(
            &mut frontend,
            &mut kv,
            decoding_session(true, vec![42, 43]),
            "boom",
        )
        .await
        .unwrap();
        // Streaming path emits a chunk, not a unary response.
        assert_eq!(frontend.responses.lock().unwrap().len(), 0);
        let chunks = frontend.chunks.lock().unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(matches!(chunks[0].chunk_type, ChunkType::Error));
        assert_eq!(chunks[0].finish_reason.as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn complete_session_routes_through_free_finished() {
        let sys = OutputProcessingSystem::new();
        let mut frontend = CapturingFrontend::default();
        let mut kv = CapturingKv::default();
        let metrics = MetricsRecorder::new(false);
        let outcome = sys
            .complete_session(
                &mut frontend,
                &mut kv,
                &metrics,
                decoding_session(false, vec![10, 11, 12]),
            )
            .await
            .unwrap();
        // Single owner of KV release: free_finished routes through
        // the prefix cache. Exactly one call recorded.
        assert_eq!(kv.free_finished_calls.len(), 1);
        // Prompt forwarded to prefix cache routing.
        assert_eq!(kv.free_finished_calls[0], vec![1, 2, 3, 4]);
        // Response is success with the output token list.
        let resps = frontend.responses.lock().unwrap();
        assert_eq!(resps.len(), 1);
        assert!(matches!(resps[0].status, ResponseStatus::Success));
        assert_eq!(resps[0].output_token_ids, vec![10, 11, 12]);
        assert!(outcome.num_tokens >= 1);
    }
}
