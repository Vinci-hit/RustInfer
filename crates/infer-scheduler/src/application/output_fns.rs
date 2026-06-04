//! Output processing free functions.
//!
//! All output/termination logic lives here as free functions.
//! Workflow implementations (`LlmWorkflow`, `DiffusionWorkflow`)
//! call these directly; `engine.rs` also calls `fail_sessions` for
//! the terminate-all path.

use infer_protocol::scheduler_to_server::{
    ChunkType, ImageOutput, InferenceMetrics, InferenceResponse, ResponseStatus, StreamChunk,
};
use infer_protocol::worker_to_scheduler_data::{DiffusionBatchOutput, DiffusionOutputStatus, StepOutput};

use crate::domain::inference_session::handle::ClientId;
use crate::domain::inference_session::lifecycle::{Decoding, InferenceSession, Prefilling};
use crate::domain::kv_budget::KvBudget;
use crate::error::Result;
use crate::infrastructure::kv_cache::radix_tree::RadixTree;
use crate::infrastructure::metrics::MetricsRecorder;
use crate::infrastructure::transport::traits::FrontendTransport;

pub use super::output::CompleteOutcome;

/// Emit an error to the client. Does not touch KV (callers do).
pub async fn send_request_error(
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

/// Fail a session caught in `Prefilling`.
pub async fn fail_prefilling_session(
    frontend: &mut dyn FrontendTransport,
    seq: InferenceSession<Prefilling>,
    message: &str,
) -> Result<()> {
    let external_id = seq.meta.external_id.clone();
    let client_id = ClientId::new(seq.handle.client_id.as_bytes().to_vec());
    let stream = seq.meta.stream;
    send_request_error(frontend, client_id, external_id, stream, message.to_string(), 0).await
}

/// Fail a session caught in `Decoding`.
pub async fn fail_decoding_session(
    frontend: &mut dyn FrontendTransport,
    seq: InferenceSession<Decoding>,
    message: &str,
) -> Result<()> {
    let external_id = seq.meta.external_id.clone();
    let client_id = ClientId::new(seq.handle.client_id.as_bytes().to_vec());
    let stream = seq.meta.stream;
    let num_tokens = seq.state.output_tokens.len() as u32;
    send_request_error(
        frontend,
        client_id,
        external_id,
        stream,
        message.to_string(),
        num_tokens,
    )
    .await
}

/// Successfully complete a Decoding session.
pub async fn complete_session(
    frontend: &mut dyn FrontendTransport,
    metrics: &MetricsRecorder,
    seq: InferenceSession<Decoding>,
) -> Result<CompleteOutcome> {
    use crate::domain::inference_session::lifecycle::FinishReason;

    let request_id_display = seq.meta.id.to_string();
    let external_id = seq.meta.external_id.clone();
    let client_id = ClientId::new(seq.handle.client_id.as_bytes().to_vec());
    let stream = seq.meta.stream;

    let reason = if seq.reached_max_tokens() {
        FinishReason::MaxTokens
    } else {
        FinishReason::Eos
    };

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

/// Feed `StepOutput.assigned_indices` into the `RadixTree` + `KvBudget`.
pub fn feed_radix_assigned_indices(
    radix: &mut RadixTree,
    budget: &mut KvBudget,
    output: &StepOutput,
) {
    let _ = budget;
    if output.assigned_indices.is_empty() {
        return;
    }

    let mut by_seq: std::collections::HashMap<u64, Vec<u32>> =
        std::collections::HashMap::new();
    for a in &output.assigned_indices {
        let entry = by_seq.entry(a.sequence_id).or_default();
        for k in 0..a.len as u32 {
            entry.push(a.base + k);
        }
    }

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
    for (sid, slots) in &by_seq {
        let cursor = seq_cursor.entry(*sid).or_insert(0);
        while *cursor < slots.len() {
            radix.append_token(*sid, 0, slots[*cursor]);
            *cursor += 1;
        }
    }
}

/// Mark a sequence's chain as finished in the `RadixTree`.
pub fn radix_mark_finished(radix: &mut RadixTree, sequence_id: u64) {
    radix.mark_finished_chain(sequence_id);
}

/// Process one batch of worker step output (LLM mode) with a
/// pre-decoded `StepOutput`.
pub async fn process_llm_step_decoded(
    sessions: &mut crate::domain::inference_session::table::RequestTable,
    frontend: &mut dyn FrontendTransport,
    metrics: &MetricsRecorder,
    output: &StepOutput,
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
        let outcome = complete_session(frontend, metrics, seq).await?;
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

/// Process one batch of worker step output (Diffusion mode) with a
/// pre-decoded `DiffusionBatchOutput`.
pub async fn process_diffusion_step_decoded(
    sessions: &mut crate::domain::inference_session::table::RequestTable,
    frontend: &mut dyn FrontendTransport,
    metrics: &MetricsRecorder,
    output: &DiffusionBatchOutput,
) -> Result<()> {
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

/// Drive the failure path for a list of internal `RequestId`s.
pub async fn fail_sessions(
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
                fail_prefilling_session(frontend, sequence, message).await?;
            }
            FailedOutcome::RemovedDecoding { sequence, .. } => {
                fail_decoding_session(frontend, sequence, message).await?;
            }
            FailedOutcome::NotFound { .. } => {}
        }
    }
    Ok(())
}
