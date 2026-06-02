//! Wire-format batch serializer used by [`PlanningSystem`].
//!
//! `BatchBuilder` keeps its staging vectors as fields and `clear()`s
//! them between iterations, so steady-state batching does not churn
//! `input_ids` / `q_start_loc` / `segments` allocations.
//!
//! The builder is internal to the application layer (visible only
//! to `PlanningSystem`); engines never see it directly.

use std::collections::HashSet;

use crate::config::SchedulerConfig;
use crate::error::{Result, SchedulerError};
use crate::domain::inference_session::lifecycle::{InferenceSession, Prefilling, RequestId};
use crate::infrastructure::kv_cache::radix_tree::GlobalIndex;
use crate::infrastructure::transport::codec::{Codec, MsgPackCodec};

use infer_protocol::scheduler_to_worker_data::{
    BatchCommand, DiffusionBatchCmd, DiffusionBatchItem, PrefillBatchCmd,
    PrefillSegmentCompletion, PrefillSegmentMeta, SamplingParams as WorkerSamplingParams,
};

/// Serializes scheduler output into wire-format `BatchCommand`s,
/// reusing internal staging buffers across calls.
#[derive(Debug, Default)]
pub(crate) struct BatchBuilder {
    /// Concatenated prefill input ids across the whole batch.
    input_ids_all: Vec<i32>,
    /// Per-segment offset into `input_ids_all`.
    q_start_loc: Vec<u32>,
    /// Per-segment metadata.
    segments: Vec<PrefillSegmentMeta>,
    /// Diffusion staging buffer (reused).
    diffusion_requests: Vec<DiffusionBatchItem>,
}

impl BatchBuilder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Build the LLM-mode `Prefill` batch.
    ///
    /// Worker owns the decode self-loop, so decode sessions are not
    /// serialized here. `scheduled_segments` is the scheduler's pick
    /// of which prefilling sessions go in this iteration — exactly
    /// the ones whose `inflight` matches the planned chunk.
    ///
    /// `prefix_hints` carries the per-request global KV indices that
    /// hit the scheduler's `RadixTree` cache. The worker uses them
    /// to skip recomputing those leading prompt tokens.
    pub(crate) fn build_llm_batch(
        &mut self,
        prefilling: &[&InferenceSession<Prefilling>],
        config: &SchedulerConfig,
        codec: &MsgPackCodec,
        scheduled_segments: &[(RequestId, usize)],
        prefix_hints: &[(RequestId, Vec<GlobalIndex>)],
    ) -> Result<Vec<u8>> {
        if prefilling.is_empty() {
            return Ok(Vec::new());
        }
        if scheduled_segments.is_empty() {
            return Ok(Vec::new());
        }
        self.build_prefill_cmd(prefilling, config, codec, scheduled_segments, prefix_hints)
    }

    /// Build the Diffusion-mode batch.
    pub(crate) fn build_diffusion_batch(
        &mut self,
        prefilling: &[&InferenceSession<Prefilling>],
        codec: &MsgPackCodec,
        scheduled_requests: &[(RequestId, usize)],
    ) -> Result<Vec<u8>> {
        if scheduled_requests.is_empty() {
            return Ok(Vec::new());
        }
        let selected: HashSet<&RequestId> =
            scheduled_requests.iter().map(|(id, _)| id).collect();
        self.diffusion_requests.clear();

        for seq in prefilling {
            if !selected.contains(&seq.meta.id) {
                continue;
            }
            let Some(req) = &seq.meta.diffusion else {
                return Err(SchedulerError::Internal(format!(
                    "scheduled diffusion request {} has no diffusion payload",
                    seq.meta.id
                )));
            };
            self.diffusion_requests.push(DiffusionBatchItem {
                // Worker echoes this back in DiffusionBatchOutput.results[i].request_id.
                // Use external_id so the engine can match the response to the
                // by_external_id index without holding extra mapping.
                request_id: seq.meta.external_id.clone(),
                prompt: req.prompt.clone(),
                prompt_input_ids: req.prompt_input_ids.clone(),
                negative_prompt: req.negative_prompt.clone(),
                negative_prompt_input_ids: req.negative_prompt_input_ids.clone(),
                height: req.height,
                width: req.width,
                num_inference_steps: req.num_inference_steps,
                sigmas: req.sigmas.clone(),
                guidance_scale: req.guidance_scale,
                seed: req.seed,
                output_format: req.output_format.clone(),
            });
        }

        if self.diffusion_requests.is_empty() {
            return Ok(Vec::new());
        }
        // We hand ownership to the wire frame to avoid a Vec clone.
        let requests = std::mem::take(&mut self.diffusion_requests);
        codec.encode(&BatchCommand::DiffusionBatch(DiffusionBatchCmd { requests }))
    }

    fn build_prefill_cmd(
        &mut self,
        prefilling: &[&InferenceSession<Prefilling>],
        config: &SchedulerConfig,
        codec: &MsgPackCodec,
        scheduled_segments: &[(RequestId, usize)],
        prefix_hints: &[(RequestId, Vec<GlobalIndex>)],
    ) -> Result<Vec<u8>> {
        self.input_ids_all.clear();
        self.q_start_loc.clear();
        self.segments.clear();
        let selected: HashSet<&RequestId> =
            scheduled_segments.iter().map(|(id, _)| id).collect();

        for seq in prefilling {
            if !selected.contains(&seq.meta.id) {
                continue;
            }
            let Some(inflight) = seq.state.inflight else {
                return Err(SchedulerError::Internal(format!(
                    "scheduled prefill {} has no inflight segment",
                    seq.meta.id
                )));
            };

            let start = inflight.segment_start;
            let end = inflight.segment_end.min(seq.state.prompt_len);
            if start >= end || end > seq.meta.input_ids.len() {
                return Err(SchedulerError::Internal(format!(
                    "invalid prefill segment for {}: [{}..{}) prompt_len={}",
                    seq.meta.id,
                    start,
                    end,
                    seq.meta.input_ids.len()
                )));
            }

            self.q_start_loc.push(self.input_ids_all.len() as u32);
            self.input_ids_all
                .extend_from_slice(&seq.meta.input_ids[start..end]);

            let block_size = config.paged_block_size.raw();
            // The scheduler does not allocate physical KV blocks — the
            // worker owns the pool. `block_table` ships empty; the
            // worker provisions slots via its `GlobalKvAllocator` at
            // step time.
            let block_table: Vec<u32> = Vec::new();

            // Pull the prefix hint (if any) from the parallel table
            // populated by `PlanningSystem::execute_plan`. Linear
            // scan is fine: prefix_hints.len() ≤ scheduled_segments.len()
            // which is bounded by the per-iteration batch size.
            let prefix_hint = prefix_hints
                .iter()
                .find(|(id, _)| id == &seq.meta.id)
                .map(|(_, indices)| indices.clone());

            self.segments.push(PrefillSegmentMeta {
                sequence_id: seq.meta.sequence_id.0,
                block_table,
                block_size,
                prompt_len: seq.state.prompt_len as u32,
                segment_start: start as u32,
                segment_end: end as u32,
                max_tokens: seq.meta.max_tokens,
                sampling_params: WorkerSamplingParams {
                    temperature: seq.meta.sampling.temperature,
                    top_p: seq.meta.sampling.top_p,
                    top_k: seq.meta.sampling.top_k,
                },
                completion: if inflight.is_final {
                    PrefillSegmentCompletion::FinishPrefillAndStartDecode
                } else {
                    PrefillSegmentCompletion::ContinuePrefill
                },
                prefix_hint,
            });
        }

        if self.segments.is_empty() {
            return Ok(Vec::new());
        }

        // Take ownership of the buffers so the encoder doesn't need
        // to clone them. They're re-allocated next call via clear()
        // — `mem::take` on a `Vec` reuses the backing allocation
        // because `Default::default()` returns an empty Vec with
        // zero capacity, but `clear()` at the head of the next call
        // restores capacity from a fresh Vec; we trade a tiny bit
        // of capacity churn for not having to clone.
        let cmd = PrefillBatchCmd {
            input_ids: std::mem::take(&mut self.input_ids_all),
            q_start_loc: std::mem::take(&mut self.q_start_loc),
            segments: std::mem::take(&mut self.segments),
        };

        codec.encode(&BatchCommand::Prefill(cmd))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Instant;

    use crate::domain::inference_session::handle::RequestHandle;
    use crate::domain::inference_session::lifecycle::{
        InFlightPrefillSegment, Priority, RequestMeta, SamplingParams, SequenceId,
    };
    use infer_protocol::scheduler_to_worker_data::BatchCommand;

    fn make_prefilling_with_blocks() -> InferenceSession<Prefilling> {
        let meta = Arc::new(RequestMeta {
            id: RequestId::new_v4(),
            external_id: "req-paged".to_string(),
            sequence_id: SequenceId(42),
            input_ids: vec![11, 22, 33, 44],
            max_tokens: 128,
            sampling: SamplingParams::default(),
            priority: Priority::default(),
            stream: false,
            stop_sequences: vec![],
            diffusion: None,
            arrival_time: Instant::now(),
        });
        InferenceSession {
            meta,
            handle: RequestHandle::noop(),
            state: Prefilling {
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

    #[test]
    fn build_prefill_batch_with_paged_placement() {
        let config = SchedulerConfig {
            paged_block_size: crate::domain::ids::BlockSize::new(16),
            max_model_len: 4096,
            ..Default::default()
        };
        let codec = MsgPackCodec;
        let seq = make_prefilling_with_blocks();
        let request_id = seq.meta.id.clone();

        let mut builder = BatchBuilder::new();
        let bytes = builder
            .build_llm_batch(&[&seq], &config, &codec, &[(request_id, 4)], &[])
            .unwrap();
        let cmd: BatchCommand = codec.decode(&bytes).unwrap();
        let BatchCommand::Prefill(prefill) = cmd else {
            panic!("expected prefill command");
        };
        assert_eq!(prefill.segments.len(), 1);
        let segment = &prefill.segments[0];
        assert_eq!(segment.block_size, 16);
        // Scheduler ships an empty block_table; the worker owns
        // physical block allocation.
        assert!(segment.block_table.is_empty());
        assert!(segment.prefix_hint.is_none());
    }

    #[test]
    fn build_prefill_batch_threads_prefix_hint_through() {
        let config = SchedulerConfig {
            paged_block_size: crate::domain::ids::BlockSize::new(16),
            max_model_len: 4096,
            ..Default::default()
        };
        let codec = MsgPackCodec;
        let seq = make_prefilling_with_blocks();
        let request_id = seq.meta.id.clone();

        let mut builder = BatchBuilder::new();
        let bytes = builder
            .build_llm_batch(
                &[&seq],
                &config,
                &codec,
                &[(request_id.clone(), 4)],
                &[(request_id, vec![100, 101, 102])],
            )
            .unwrap();
        let cmd: BatchCommand = codec.decode(&bytes).unwrap();
        let BatchCommand::Prefill(prefill) = cmd else {
            panic!("expected prefill command");
        };
        let segment = &prefill.segments[0];
        assert_eq!(segment.prefix_hint.as_deref(), Some(&[100u32, 101, 102][..]));
    }

    /// After a build, reusing the same builder for a second batch
    /// must yield correct output (buffers are properly cleared).
    #[test]
    fn builder_reuse_does_not_concatenate_old_state() {
        let config = SchedulerConfig {
            paged_block_size: crate::domain::ids::BlockSize::new(16),
            max_model_len: 4096,
            ..Default::default()
        };
        let codec = MsgPackCodec;
        let seq1 = make_prefilling_with_blocks();
        let id1 = seq1.meta.id.clone();
        let mut builder = BatchBuilder::new();
        let _bytes1 = builder
            .build_llm_batch(&[&seq1], &config, &codec, &[(id1, 4)], &[])
            .unwrap();

        // Second iteration: same builder, fresh session.
        let seq2 = make_prefilling_with_blocks();
        let id2 = seq2.meta.id.clone();
        let bytes2 = builder
            .build_llm_batch(&[&seq2], &config, &codec, &[(id2, 4)], &[])
            .unwrap();
        let cmd: BatchCommand = codec.decode(&bytes2).unwrap();
        let BatchCommand::Prefill(prefill) = cmd else {
            panic!("expected prefill");
        };
        // Exactly one segment, not two: state from iteration 1 was cleared.
        assert_eq!(prefill.segments.len(), 1);
        // input_ids carries only iter-2's prompt.
        assert_eq!(prefill.input_ids, vec![11, 22, 33, 44]);
    }
}
