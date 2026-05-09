//! Batch builder — assembles scheduling output into wire-format commands.

use crate::cache::kv_manager::KvAllocation;
use crate::config::SchedulerConfig;
use crate::error::{Result, SchedulerError};
use crate::request::lifecycle::{Sequence, Prefilling, Decoding, RequestId};
use crate::transport::codec::{Codec, MsgPackCodec};

use infer_worker::worker::protocol::{PrefillBatchCmd, SamplingParams as WorkerSamplingParams, RequestMeta as WorkerRequestMeta};

/// Build a serialized batch command from the current prefilling + decoding sequences.
///
/// For chunked prefill: only sends the tokens for the CURRENT chunk
/// (input_ids[num_computed_tokens..num_computed_tokens+chunk_size]).
///
/// `chunk_sizes` maps request_id → tokens to process this iteration.
pub fn build_batch(
    prefilling: &[Sequence<Prefilling>],
    decoding: &[Sequence<Decoding>],
    _config: &SchedulerConfig,
    codec: &MsgPackCodec,
    chunk_sizes: &[(RequestId, usize)],
) -> Result<Vec<u8>> {
    if prefilling.is_empty() && decoding.is_empty() {
        return Ok(Vec::new());
    }

    // For now, we only send PrefillBatchCmd for new prefills / continuation chunks.
    // Decode sequences are handled by the worker's internal loop.
    if prefilling.is_empty() {
        return Ok(Vec::new());
    }

    build_prefill_batch_cmd(prefilling, codec, chunk_sizes)
}

/// Build the PrefillBatchCmd, sending only the current chunk's tokens.
fn build_prefill_batch_cmd(
    prefilling: &[Sequence<Prefilling>],
    codec: &MsgPackCodec,
    chunk_sizes: &[(RequestId, usize)],
) -> Result<Vec<u8>> {
    let mut input_ids_all: Vec<i32> = Vec::new();
    let mut q_start_loc: Vec<u32> = Vec::new();
    let mut num_computed_tokens: Vec<u32> = Vec::new();
    let mut kv_slots: Vec<u32> = Vec::new();
    let mut sampling_params: Vec<WorkerSamplingParams> = Vec::new();
    let mut request_metas: Vec<WorkerRequestMeta> = Vec::new();

    for seq in prefilling {
        // Determine this iteration's chunk size.
        let chunk_size = chunk_sizes
            .iter()
            .find(|(id, _)| *id == seq.meta.id)
            .map(|(_, size)| *size)
            .unwrap_or(seq.state.prompt_len - seq.state.num_computed_tokens);

        let start = seq.state.num_computed_tokens;
        let end = (start + chunk_size).min(seq.state.prompt_len);

        // Only send this chunk's tokens.
        q_start_loc.push(input_ids_all.len() as u32);
        input_ids_all.extend_from_slice(&seq.meta.input_ids[start..end]);
        num_computed_tokens.push(start as u32);

        // Extract slot id from KvAllocation.
        let slot_id = match &seq.state.kv_alloc {
            KvAllocation::Slot(id) => *id,
            KvAllocation::Blocks(_) => {
                return Err(SchedulerError::Internal(
                    "build_prefill_batch_cmd called with Blocks allocation".into(),
                ));
            }
        };
        kv_slots.push(slot_id);

        sampling_params.push(WorkerSamplingParams {
            temperature: seq.meta.sampling.temperature,
            top_p: seq.meta.sampling.top_p,
            top_k: seq.meta.sampling.top_k,
        });

        request_metas.push(WorkerRequestMeta {
            request_id: seq.meta.id.0.clone(),
            max_tokens: seq.meta.max_tokens,
        });
    }

    let cmd = PrefillBatchCmd {
        input_ids: input_ids_all,
        q_start_loc,
        num_computed_tokens,
        kv_slots,
        sampling_params,
        request_metas,
    };

    codec.encode(&cmd)
}
