//! Batch builder — assembles scheduling output into wire-format commands.

use std::collections::HashSet;

use crate::cache::kv_manager::KvAllocation;
use crate::config::SchedulerConfig;
use crate::error::{Result, SchedulerError};
use crate::request::lifecycle::{Sequence, Prefilling, Decoding, RequestId};
use crate::transport::codec::{Codec, MsgPackCodec};

use infer_protocol::scheduler_to_worker::{
    CancelRequest, DiffusionBatchCmd, DiffusionBatchItem, PrefillBatchCmd,
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
    _config: &SchedulerConfig,
    codec: &MsgPackCodec,
    scheduled_segments: &[(RequestId, usize)],
) -> Result<Vec<u8>> {
    if prefilling.is_empty() && decoding.is_empty() {
        return Ok(Vec::new());
    }
    if scheduled_segments.is_empty() {
        return Ok(Vec::new());
    }

    build_prefill_batch_cmd(prefilling, codec, scheduled_segments)
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

        let kv_slot = match &seq.state.kv_alloc {
            KvAllocation::Slot(id) => *id,
            KvAllocation::Blocks(_) => {
                return Err(SchedulerError::Internal(
                    "build_prefill_batch_cmd called with Blocks allocation".into(),
                ));
            }
        };

        segments.push(PrefillSegmentMeta {
            sequence_id: seq.meta.sequence_id.0,
            kv_slot,
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
