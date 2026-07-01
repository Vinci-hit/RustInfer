//! KV-accounting projections and preemption scoring over [`RequestTable`].
//!
//! These are *scheduling-policy reads* (KV reservation math, victim scoring),
//! not storage mechanics, so they live here as free functions rather than on
//! the `RequestTable` aggregate. As a child module of `table` they may read the
//! table's private buckets directly, which keeps the storage type focused on
//! index lifecycle while the policy-flavored projections live on their own.

use super::RequestTable;
use crate::domain::inference_session::lifecycle::{Decoding, InferenceSession};

/// Minimal projection of an active session into the data the preemption policy
/// needs to score it.
///
/// `kv_used` is the source of truth for "how many KV slots will be freed if we
/// preempt this seq" — that's what the scheduler counts toward its 5%-of-total
/// target.
#[derive(Debug, Clone, Copy)]
pub struct PreemptCandidate {
    pub sequence_id: u64,
    /// `Decoding`: emitted-token count. `Prefilling`: 0.
    pub output_len: u32,
    /// Prompt length (input_ids.len).
    pub input_len: u32,
    /// `Decoding`: `seq_position`. `Prefilling`: `num_computed_tokens`.
    pub kv_used: u32,
}

/// KV slots a decoding session currently occupies.
///
/// Excludes the latest unwritten output token: prefill produced the first
/// output token without writing it into KV, so `output_len - 1` decode slots
/// are live on top of the prompt.
pub(crate) fn decoding_kv_slots(seq: &InferenceSession<Decoding>) -> usize {
    seq.state
        .prompt_len
        .saturating_add(seq.state.output_tokens.len().saturating_sub(1))
}

/// Total prompt tokens currently in flight across all prefilling sequences
/// whose segment has been dispatched but not yet acked.
///
/// This is the projected worker KV-slot footprint of unreported prefill work.
/// It is recomputed from live session state, so sequences that are cancelled /
/// preempted / failed before their ack simply stop contributing — there is no
/// separate counter to leak.
///
/// Slightly conservative: it counts the full segment width and ignores
/// prefix-cache hits (which consume no new worker slots), so it can only
/// *over*-estimate pending pressure, never under-estimate. Over-estimating is
/// the safe direction for over-commit protection.
pub(crate) fn inflight_prefill_tokens(table: &RequestTable) -> u32 {
    table
        .prefilling
        .values()
        .filter_map(|seq| seq.inflight_segment())
        .map(|seg| (seg.segment_end - seg.segment_start) as u32)
        .sum()
}

/// KV slots that must stay free for already-admitted requests to finish
/// decoding without relying on worker-side emergency relief.
///
/// Prefill produces the first output token without writing that generated token
/// into KV, so a request with `max_tokens = N` needs at most `N - 1` future
/// decode allocations after prefill. For sessions already decoding, subtract
/// the output tokens the scheduler has observed.
pub(crate) fn future_decode_reserve_tokens(table: &RequestTable) -> usize {
    let prefilling_reserve: usize = table
        .prefilling
        .values()
        .map(|seq| seq.meta.max_tokens.saturating_sub(1))
        .sum();
    let decoding_reserve: usize = table
        .decoding
        .values()
        .map(|seq| {
            seq.meta
                .max_tokens
                .saturating_sub(seq.state.output_tokens.len())
        })
        .sum();
    prefilling_reserve.saturating_add(decoding_reserve)
}

/// Collect all currently-active sequences (decoding + chunked prefilling) along
/// with the data the scheduler needs to score them as preemption victims.
///
/// `Prefilling` sequences whose `num_computed_tokens == 0` are excluded — they
/// have no KV state on the worker yet, so preempting them frees nothing. Any
/// `Prefilling` with `num_computed_tokens > 0` (chunked prefill in progress) IS
/// included: its KV slots are real and worth recovering.
pub(crate) fn preemption_candidates(table: &RequestTable) -> Vec<PreemptCandidate> {
    // Token counts are bounded by `max_model_len` in practice, well below
    // `u32::MAX`, but clamp explicitly so an unexpected value degrades the
    // preemption score gracefully instead of wrapping to a tiny count.
    let clamp = |n: usize| -> u32 { u32::try_from(n).unwrap_or(u32::MAX) };
    let mut out = Vec::with_capacity(table.decoding.len() + table.prefilling.len());
    for seq in table.decoding.values() {
        out.push(PreemptCandidate {
            sequence_id: seq.meta.sequence_id.0,
            output_len: clamp(seq.state.output_tokens.len()),
            input_len: clamp(seq.meta.input_ids.len()),
            kv_used: clamp(decoding_kv_slots(seq)),
        });
    }
    for seq in table.prefilling.values() {
        if seq.state.num_computed_tokens > 0 {
            out.push(PreemptCandidate {
                sequence_id: seq.meta.sequence_id.0,
                output_len: 0,
                input_len: clamp(seq.meta.input_ids.len()),
                kv_used: clamp(seq.state.num_computed_tokens),
            });
        }
    }
    out
}
