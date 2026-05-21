//! Batch builder — assembles scheduling output into wire-format commands.

use std::collections::HashSet;

use crate::cache::kv_manager::KvAllocation;
use crate::config::{KvCacheMode, SchedulerConfig};
use crate::error::{Result, SchedulerError};
use crate::request::lifecycle::{Sequence, Prefilling, Decoding, RequestId};
use crate::transport::codec::{Codec, MsgPackCodec};

use infer_protocol::scheduler_to_worker::{
    CancelRequest, DiffusionBatchCmd, DiffusionBatchItem, KvPlacement, PrefillBatchCmd,
    PrefillSegmentCompletion, PrefillSegmentMeta, SamplingParams as WorkerSamplingParams,
    WorkerCommand,
};

/// Build a serialized cancel command.
pub fn build_cancel_request(sequence_id: u64, codec: &MsgPackCodec) -> Result<Vec<u8>> {
    codec.encode(&WorkerCommand::Cancel(CancelRequest { sequence_id }))
}

/// Build a serialized prefill segment batch from the currently scheduled prefilling sequences.
///
/// Decode sequences are intentionally not serialized: Worker owns the decode self-loop.
/// `scheduled_segments` maps request_id → token count selected for this Scheduler iteration;
/// only these sequences are included, preventing old in-flight chunks from being resent.
pub fn build_batch(
    prefilling: &[Sequence<Prefilling>],
    decoding: &[Sequence<Decoding>],
    config: &SchedulerConfig,
    codec: &MsgPackCodec,
    scheduled_segments: &[(RequestId, usize)],
) -> Result<Vec<u8>> {
    if prefilling.is_empty() && decoding.is_empty() {
        return Ok(Vec::new());
    }
    if scheduled_segments.is_empty() {
        return Ok(Vec::new());
    }

    build_prefill_batch_cmd(prefilling, config, codec, scheduled_segments)
}

pub fn build_diffusion_batch(
    prefilling: &[Sequence<Prefilling>],
    codec: &MsgPackCodec,
    scheduled_requests: &[(RequestId, usize)],
) -> Result<Vec<u8>> {
    if scheduled_requests.is_empty() {
        return Ok(Vec::new());
    }
    let selected: HashSet<&RequestId> = scheduled_requests.iter().map(|(id, _)| id).collect();
    let mut requests = Vec::new();

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
        requests.push(DiffusionBatchItem {
            request_id: seq.meta.id.0.clone(),
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

    if requests.is_empty() {
        return Ok(Vec::new());
    }

    codec.encode(&WorkerCommand::DiffusionBatch(DiffusionBatchCmd { requests }))
}

fn build_prefill_batch_cmd(
    prefilling: &[Sequence<Prefilling>],
    config: &SchedulerConfig,
    codec: &MsgPackCodec,
    scheduled_segments: &[(RequestId, usize)],
) -> Result<Vec<u8>> {
    let mut input_ids_all: Vec<i32> = Vec::new();
    let mut q_start_loc: Vec<u32> = Vec::new();
    let mut segments: Vec<PrefillSegmentMeta> = Vec::new();
    let selected: HashSet<&RequestId> = scheduled_segments.iter().map(|(id, _)| id).collect();

    for seq in prefilling {
        if !selected.contains(&seq.meta.id) {
            continue;
        }
        let inflight = match seq.state.inflight {
            Some(inflight) => inflight,
            None => {
                return Err(SchedulerError::Internal(format!(
                    "scheduled prefill {} has no inflight segment",
                    seq.meta.id
                )));
            }
        };

        let start = inflight.segment_start;
        let end = inflight.segment_end.min(seq.state.prompt_len);
        if start >= end || end > seq.meta.input_ids.len() {
            return Err(SchedulerError::Internal(format!(
                "invalid prefill segment for {}: [{}..{}) prompt_len={}",
                seq.meta.id, start, end, seq.meta.input_ids.len()
            )));
        }

        q_start_loc.push(input_ids_all.len() as u32);
        input_ids_all.extend_from_slice(&seq.meta.input_ids[start..end]);

        let (kv_slot, kv) = match &seq.state.kv_alloc {
            KvAllocation::Slot(id) => (*id, None),
            KvAllocation::Blocks(blocks) => {
                let block_size = match config.kv_cache_mode {
                    KvCacheMode::Paged { block_size } => block_size as u32,
                    KvCacheMode::Slot => {
                        return Err(SchedulerError::Internal(
                            "Blocks allocation used while config.kv_cache_mode is Slot".into(),
                        ));
                    }
                };
                (
                    0,
                    Some(KvPlacement::Paged {
                        block_table: blocks.iter().map(|b| b.0).collect(),
                        block_size,
                    }),
                )
            }
        };

        segments.push(PrefillSegmentMeta {
            sequence_id: seq.meta.sequence_id.0,
            kv_slot,
            kv,
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
        });
    }

    if segments.is_empty() {
        return Ok(Vec::new());
    }

    let cmd = PrefillBatchCmd {
        input_ids: input_ids_all,
        q_start_loc,
        segments,
    };

    codec.encode(&WorkerCommand::Prefill(cmd))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Instant;

    use crate::cache::traits::PhysicalBlockId;
    use crate::request::handle::RequestHandle;
    use crate::request::lifecycle::{InFlightPrefillSegment, Priority, RequestMeta, SamplingParams, SequenceId};
    use infer_protocol::scheduler_to_worker::WorkerCommand;

    fn make_prefilling_with_blocks() -> Sequence<Prefilling> {
        let meta = Arc::new(RequestMeta {
            id: RequestId("req-paged".to_string()),
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
        Sequence {
            meta,
            handle: RequestHandle::noop(),
            state: Prefilling {
                kv_alloc: KvAllocation::Blocks(vec![PhysicalBlockId(7), PhysicalBlockId(8)]),
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
            kv_cache_mode: KvCacheMode::Paged { block_size: 16 },
            max_model_len: 4096,
            ..Default::default()
        };
        let codec = MsgPackCodec;
        let seq = make_prefilling_with_blocks();
        let request_id = seq.meta.id.clone();

        let bytes = build_batch(&[seq], &[], &config, &codec, &[(request_id, 4)]).unwrap();
        let cmd: WorkerCommand = codec.decode(&bytes).unwrap();
        let WorkerCommand::Prefill(prefill) = cmd else {
            panic!("expected prefill command");
        };
        assert_eq!(prefill.segments.len(), 1);
        let segment = &prefill.segments[0];
        assert_eq!(segment.kv_slot, 0);
        match segment.kv_placement() {
            KvPlacement::Paged { block_table, block_size } => {
                assert_eq!(block_size, 16);
                assert_eq!(block_table, vec![7, 8]);
            }
            other => panic!("expected paged placement, got {other:?}"),
        }
    }
}
