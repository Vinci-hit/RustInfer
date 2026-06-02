//! `OutputProcessingSystem` — terminal-state owner.
//!
//! Single home for response emission and the scheduler-side cleanup
//! that follows session termination: error fan-out, success
//! completion, and the `RadixTree` extension that turns each
//! `StepOutput.assigned_indices` into prefix-cache entries the
//! next iteration can reuse.
//!
//! ## Surface
//!
//! - [`OutputProcessingSystem::send_request_error`] — construct and
//!   emit an error response (stream chunk vs. unary), no state change.
//! - [`OutputProcessingSystem::fail_prefilling_session`] — emit error
//!   for a session caught in `Prefilling`. RadixTree release is
//!   driven separately by the engine via `radix_mark_finished`.
//! - [`OutputProcessingSystem::fail_decoding_session`] — same for
//!   `Decoding` (carries partial token count for metrics).
//! - [`OutputProcessingSystem::complete_session`] — emit success
//!   response and record completion metrics.
//! - [`OutputProcessingSystem::feed_radix_assigned_indices`] /
//!   [`OutputProcessingSystem::radix_mark_finished`] — RadixTree
//!   maintenance helpers.
//!
//! ## Borrow shape
//!
//! Each method takes the resources it needs as `&mut`/`&` arguments
//! rather than holding them in `self`. The engine keeps ownership of
//! `RequestTable` / transports / metrics and hands non-aliasing
//! borrows in per call.

use infer_protocol::scheduler_to_server::{
    ChunkType, InferenceMetrics, InferenceResponse, ResponseStatus, StreamChunk,
};

use crate::config::SchedulerMode;
use crate::domain::kv_budget::KvBudget;
use crate::error::Result;
use crate::infrastructure::kv_cache::radix_tree::RadixTree;
use crate::infrastructure::metrics::MetricsRecorder;
use crate::domain::inference_session::handle::ClientId;
use crate::domain::inference_session::lifecycle::{Decoding, FinishReason, InferenceSession, Prefilling};
use crate::infrastructure::transport::traits::FrontendTransport;

/// Output / cleanup stage. Stateless; the engine owns the
/// `MetricsRecorder` and passes it in per call.
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

    /// Fail a session caught in `Prefilling`. Emits the error
    /// response; RadixTree release for the partially-filled chain
    /// is driven separately by the engine through
    /// `radix_mark_finished`.
    pub async fn fail_prefilling_session(
        &self,
        frontend: &mut dyn FrontendTransport,
        seq: InferenceSession<Prefilling>,
        message: &str,
    ) -> Result<()> {
        let external_id = seq.meta.external_id.clone();
        let client_id = ClientId::new(seq.handle.client_id.as_bytes().to_vec());
        let stream = seq.meta.stream;
        self.send_request_error(frontend, client_id, external_id, stream, message.to_string(), 0)
            .await
    }

    /// Fail a session caught in `Decoding`. Carries the partial
    /// token count so error metrics reflect work already done.
    pub async fn fail_decoding_session(
        &self,
        frontend: &mut dyn FrontendTransport,
        seq: InferenceSession<Decoding>,
        message: &str,
    ) -> Result<()> {
        let external_id = seq.meta.external_id.clone();
        let client_id = ClientId::new(seq.handle.client_id.as_bytes().to_vec());
        let stream = seq.meta.stream;
        let num_tokens = seq.state.output_tokens.len() as u32;
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

    /// Successfully complete a Decoding session. The `RadixTree`
    /// already owns prefix indexing for this session's tokens; slot
    /// release is driven separately by the engine through
    /// `radix_mark_finished`.
    pub async fn complete_session(
        &self,
        frontend: &mut dyn FrontendTransport,
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

        // `finish` consumes the session and yields output bookkeeping.
        let finished = seq.finish(reason);

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
    /// Feed `StepOutput.assigned_indices` into the scheduler-side
    /// `RadixTree` + `KvBudget`. Idempotent on
    /// `assigned_indices.is_empty()`.
    ///
    /// **Per-seq multi-segment correctness**: when the worker's
    /// `GlobalKvAllocator` is fragmented it returns non-contiguous indices,
    /// which the worker emits as multiple `AssignedIndices` entries with the
    /// same `sequence_id`. We must feed every entry's slots into the radix
    /// tree, not just the last one — keying a HashMap by `sequence_id`
    /// would silently drop duplicates and starve subsequent `evict()` calls
    /// of slots they should have surfaced.
    pub fn feed_radix_assigned_indices(
        &self,
        radix: &mut RadixTree,
        budget: &mut KvBudget,
        output: &infer_protocol::worker_to_scheduler_data::StepOutput,
    ) {
        let _ = budget; // engine reserves; we just append
        if output.assigned_indices.is_empty() {
            return;
        }

        // Build per-seq concatenated slot list: every AssignedIndices entry
        // for a given seq contributes its `[base..base+len)` slots, in
        // protocol order. Worker contract: order matches the seq's own
        // chain extension order (i.e. the i-th slot in this list is the
        // KV position for the i-th new token of that seq this step).
        let mut by_seq: std::collections::HashMap<u64, Vec<u32>> =
            std::collections::HashMap::new();
        for a in &output.assigned_indices {
            let entry = by_seq.entry(a.sequence_id).or_default();
            for k in 0..a.len as u32 {
                entry.push(a.base + k);
            }
        }

        // Walk tokens in order and pair each with the next slot for its
        // seq. We use a per-seq cursor to advance through the concatenated
        // slot list as tokens come in.
        let mut seq_cursor: std::collections::HashMap<u64, usize> =
            std::collections::HashMap::with_capacity(by_seq.len());
        for tk in &output.tokens {
            let Some(slots) = by_seq.get(&tk.sequence_id) else {
                continue;
            };
            let cursor = seq_cursor.entry(tk.sequence_id).or_insert(0);
            if *cursor < slots.len() {
                radix.append_token(tk.sequence_id, tk.token_id, slots[*cursor]);
                *cursor += 1;
            }
        }
        // Any remaining slots (e.g. prefill that wrote 100 KV slots but only
        // produced one sample token) are appended with placeholder token id
        // 0 — they represent prompt KV positions whose token ids we don't
        // re-derive on the scheduler side. Prefix re-use will still work
        // for the actual prefill path because `lookup_prefix` is driven by
        // the new seq's prompt, not by the placeholder tokens.
        for (sid, slots) in &by_seq {
            let cursor = seq_cursor.entry(*sid).or_insert(0);
            while *cursor < slots.len() {
                radix.append_token(*sid, 0, slots[*cursor]);
                *cursor += 1;
            }
        }
    }

    /// When a session terminates (success / fail / cancel / preempt),
    /// tell the `RadixTree` so its chain transitions to unowned and
    /// the chain's tail nodes become eligible for LRU eviction.
    ///
    /// Idempotent on unknown seq ids.
    pub fn radix_mark_finished(&self, radix: &mut RadixTree, sequence_id: u64) {
        radix.mark_finished_chain(sequence_id);
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
            let outcome = self.complete_session(frontend, metrics, seq).await?;
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

    /// Process one batch of worker step output (LLM mode) with a
    /// pre-decoded `StepOutput`. This is the preferred entry point
    /// when deserialization has already happened (e.g. in a background
    /// decode task). The `process_llm_step` method is retained for
    /// backward compat where raw bytes are still in use.
    #[allow(clippy::too_many_arguments)]
    pub async fn process_llm_step_decoded(
        &self,
        sessions: &mut crate::domain::inference_session::table::RequestTable,
        frontend: &mut dyn FrontendTransport,
        metrics: &MetricsRecorder,
        output: &infer_protocol::worker_to_scheduler_data::StepOutput,
    ) -> Result<()> {
        use crate::domain::inference_session::lifecycle::SequenceId;
        use crate::domain::inference_session::table::PrefillAckOutcome;

        // 1. ACK prefill segments.
        for sequence_id in &output.prefill_done {
            match sessions.ack_prefill(SequenceId(*sequence_id)) {
                Ok(PrefillAckOutcome::MovedToDecoding { .. })
                | Ok(PrefillAckOutcome::Continue { .. }) => {}
                Err(e) => tracing::warn!(
                    "PrefillDone for sequence_id={} failed: {}",
                    sequence_id,
                    e
                ),
            }
        }

        // 2. Process generated tokens.
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

        for (client_id, chunk) in token_chunks {
            frontend.send_stream_chunk(&client_id, chunk).await?;
        }

        finished_sequence_ids.sort_unstable_by_key(|id| id.0);
        finished_sequence_ids.dedup();
        for sequence_id in finished_sequence_ids {
            let seq = sessions.finish_decoding(sequence_id)?;
            let outcome = self.complete_session(frontend, metrics, seq).await?;
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
                    self.fail_prefilling_session(frontend, sequence, message)
                        .await?;
                }
                FailedOutcome::RemovedDecoding { sequence, .. } => {
                    self.fail_decoding_session(frontend, sequence, message)
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
    /// reason, and emit the final `InferenceResponse` with image
    /// bytes. Diffusion uses no real KV, so there is no
    /// scheduler-side slot release to do here.
    pub async fn process_diffusion_step(
        &self,
        sessions: &mut crate::domain::inference_session::table::RequestTable,
        frontend: &mut dyn FrontendTransport,
        metrics: &MetricsRecorder,
        codec: &crate::infrastructure::transport::codec::MsgPackCodec,
        data: Vec<u8>,
    ) -> Result<()> {
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
            let Some(seq) = sessions.take_prefilling_by_request(&item_request_id)? else {
                tracing::warn!(
                    "Diffusion output for unknown request_id={}",
                    item.request_id
                );
                continue;
            };
            let request_id_display = seq.meta.id.to_string();
            let external_id = seq.meta.external_id.clone();
            let client_id = ClientId::new(seq.handle.client_id.as_bytes().to_vec());
            // No scheduler-side KV cleanup for diffusion; the worker
            // reclaims any model-internal resources itself.

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

    /// Process one batch of worker step output (Diffusion mode) with a
    /// pre-decoded `DiffusionBatchOutput`. Preferred entry point when
    /// deserialization has already happened.
    pub async fn process_diffusion_step_decoded(
        &self,
        sessions: &mut crate::domain::inference_session::table::RequestTable,
        frontend: &mut dyn FrontendTransport,
        metrics: &MetricsRecorder,
        output: &infer_protocol::worker_to_scheduler_data::DiffusionBatchOutput,
    ) -> Result<()> {
        use infer_protocol::scheduler_to_server::ImageOutput;
        use infer_protocol::worker_to_scheduler_data::DiffusionOutputStatus;

        for item in &output.results {
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
            let Some(seq) = sessions.take_prefilling_by_request(&item_request_id)? else {
                tracing::warn!(
                    "Diffusion output for unknown request_id={}",
                    item.request_id
                );
                continue;
            };
            let request_id_display = seq.meta.id.to_string();
            let external_id = seq.meta.external_id.clone();
            let client_id = ClientId::new(seq.handle.client_id.as_bytes().to_vec());

            let elapsed_ms = seq.meta.arrival_time.elapsed().as_millis() as u64;
            let (status, images, error) = match item.status {
                DiffusionOutputStatus::Success => {
                    let images = item
                        .image
                        .iter()
                        .map(|image| ImageOutput {
                            width: image.width,
                            height: image.height,
                            channels: image.channels,
                            format: image.format.clone(),
                            data: image.data.clone(),
                        })
                        .collect();
                    (ResponseStatus::Success, images, None)
                }
                DiffusionOutputStatus::Error => (ResponseStatus::Error, vec![], item.error.clone()),
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

// ─── OutputRouter ──────────────────────────────────────────────────────

/// Mode-agnostic step-output dispatcher. Absorbs the `match config.mode`
/// branch so the engine's `handle_step_output` stays clean.
pub struct OutputRouter;

impl OutputRouter {
    /// Process worker step output, dispatching to the LLM or Diffusion
    /// path based on `mode`.
    ///
    /// Uses `output_fns` free functions internally.
    #[allow(clippy::too_many_arguments)]
    pub async fn process_step(
        mode: SchedulerMode,
        _output_sys: &OutputProcessingSystem,
        sessions: &mut crate::domain::inference_session::table::RequestTable,
        frontend: &mut dyn FrontendTransport,
        metrics: &MetricsRecorder,
        codec: &crate::infrastructure::transport::codec::MsgPackCodec,
        radix: &mut RadixTree,
        kv_budget: &mut KvBudget,
        data: Vec<u8>,
    ) -> Result<()> {
        match mode {
            SchedulerMode::Llm => {
                use crate::infrastructure::transport::codec::Codec;
                use infer_protocol::worker_to_scheduler_data::StepOutput;

                let output: StepOutput = codec.decode(&data)?;
                if !output.assigned_indices.is_empty() {
                    let total: u32 = output.assigned_indices.iter().map(|a| a.len as u32).sum();
                    let _ = kv_budget.try_reserve(total);
                    super::output_fns::feed_radix_assigned_indices(radix, kv_budget, &output);
                    for tk in &output.tokens {
                        if tk.finished {
                            super::output_fns::radix_mark_finished(radix, tk.sequence_id);
                        }
                    }
                }
                super::output_fns::process_llm_step_decoded(sessions, frontend, metrics, &output)
                    .await
            }
            SchedulerMode::Diffusion => {
                use crate::infrastructure::transport::codec::Codec;
                use infer_protocol::worker_to_scheduler_data::DiffusionBatchOutput;

                let output: DiffusionBatchOutput = codec.decode(&data)?;
                super::output_fns::process_diffusion_step_decoded(
                    sessions, frontend, metrics, &output,
                )
                .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
                output_tokens: output,
                seq_position: 4,
                prompt_len: 4,
                first_token_time: Instant::now(),
                preemption_count: 0,
            },
        }
    }

    #[tokio::test]
    async fn fail_prefilling_emits_error_response() {
        let sys = OutputProcessingSystem::new();
        let mut frontend = CapturingFrontend::default();
        sys.fail_prefilling_session(&mut frontend, prefilling_session(false), "bad")
            .await
            .unwrap();
        let resps = frontend.responses.lock().unwrap();
        assert_eq!(resps.len(), 1);
        assert!(matches!(resps[0].status, ResponseStatus::Error));
        assert_eq!(resps[0].metrics.num_tokens, 0);
    }

    #[tokio::test]
    async fn fail_decoding_carries_partial_token_count_to_metrics() {
        let sys = OutputProcessingSystem::new();
        let mut frontend = CapturingFrontend::default();
        sys.fail_decoding_session(
            &mut frontend,
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
        sys.fail_decoding_session(
            &mut frontend,
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
    async fn complete_session_emits_success_response() {
        let sys = OutputProcessingSystem::new();
        let mut frontend = CapturingFrontend::default();
        let metrics = MetricsRecorder::new(false);
        let outcome = sys
            .complete_session(
                &mut frontend,
                &metrics,
                decoding_session(false, vec![10, 11, 12]),
            )
            .await
            .unwrap();
        // Response is success with the output token list.
        let resps = frontend.responses.lock().unwrap();
        assert_eq!(resps.len(), 1);
        assert!(matches!(resps[0].status, ResponseStatus::Success));
        assert_eq!(resps[0].output_token_ids, vec![10, 11, 12]);
        assert!(outcome.num_tokens >= 1);
    }
}
