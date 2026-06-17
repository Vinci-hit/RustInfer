//! Default FCFS continuous batching policy.
//!
//! Selects waiting requests in FIFO order (respecting priority from WaitingQueue)
//! until the token/sequence budget is exhausted.
//! Supports chunked prefill: long prompts are split across iterations.
//!
//! ## Decoding is not the scheduler's job
//!
//! The worker owns the decode self-loop: which decoding sequences run
//! next is decided inside the worker's sub-scheduler, not here. This
//! policy only drives prefill scheduling — both fresh prompts and
//! continuation chunks for already-prefilling sequences.

use crate::domain::inference_session::queue::WaitingQueue;
use crate::domain::policy::token_budget::TokenBudget;
use crate::domain::policy::traits::{BatchPlan, PrefillEntry, RunningSet, SchedulingPolicy};

/// Must match infer-worker's ragged prefill Q tile size.
const PREFILL_Q_TILE: usize = 128;

/// FCFS continuous batching policy.
///
/// Scheduling order per iteration:
/// 1. Continuation chunks for already-prefilling sequences (priority: don't waste KV).
/// 2. New requests from the waiting queue (FCFS within priority tier).
pub struct ContinuousBatchingPolicy {
    /// Max tokens per prefill chunk (None = no chunking, entire prompt at once).
    pub chunked_prefill_size: Option<usize>,
    /// Max NEW prefill sequences admitted per iteration (B1). 0 = unlimited.
    pub max_new_prefills_per_iter: usize,
    /// Shortest-job-first ordering of new prefills within an iteration (B2).
    pub sjf: bool,
}

impl ContinuousBatchingPolicy {
    pub fn new(chunked_prefill_size: Option<usize>) -> Self {
        Self {
            chunked_prefill_size,
            max_new_prefills_per_iter: 0,
            sjf: false,
        }
    }

    /// Configure admission control (B1) and shortest-job-first ordering (B2).
    /// Both default off in [`Self::new`]; this opt-in keeps strict FCFS and
    /// unbounded admission as the default behaviour.
    pub fn with_admission(mut self, max_new_prefills_per_iter: usize, sjf: bool) -> Self {
        self.max_new_prefills_per_iter = max_new_prefills_per_iter;
        self.sjf = sjf;
        self
    }

    /// Compute the chunk size for a given remaining token count.
    fn chunk_tokens(&self, remaining: usize) -> usize {
        match self.chunked_prefill_size {
            Some(max_chunk) => remaining.min(max_chunk),
            None => remaining,
        }
    }
}

impl SchedulingPolicy for ContinuousBatchingPolicy {
    fn schedule(
        &self,
        waiting: &WaitingQueue,
        running: &RunningSet,
        budget: &TokenBudget,
    ) -> BatchPlan {
        // The scheduler does not schedule decoding — the worker's
        // sub-scheduler decides that. We only plan prefills here.
        let mut kv_budget_remaining = budget.max_tokens;
        let mut tile_budget_remaining = prefill_tile_budget(budget);
        // Seq budget: continuation chunks don't consume new seq slots (already counted).
        let seq_budget = budget.max_seqs.saturating_sub(running.total());

        let mut prefill_batch: Vec<PrefillEntry> = Vec::new();

        // 1. Schedule continuation chunks for already-prefilling sequences.
        // These MUST run — they already have KV allocated. Priority over new requests.
        for (req_id, remaining) in &running.prefilling_continuations {
            if kv_budget_remaining == 0 || tile_budget_remaining == 0 {
                break;
            }

            let chunk = fit_prefill_tokens(
                self.chunk_tokens(*remaining),
                kv_budget_remaining,
                tile_budget_remaining,
            );
            if chunk == 0 {
                break;
            }
            let start = 0; // engine will compute actual start from num_computed_tokens
            let is_partial = chunk < *remaining;

            prefill_batch.push(PrefillEntry {
                request_id: req_id.clone(),
                token_range: start..(start + chunk),
                is_partial,
            });

            kv_budget_remaining = kv_budget_remaining.saturating_sub(chunk);
            tile_budget_remaining =
                tile_budget_remaining.saturating_sub(prefill_tiles_for_tokens(chunk));
        }

        // 2. Select new requests from waiting queue.
        if kv_budget_remaining > 0
            && tile_budget_remaining > 0
            && seq_budget > 0
            && !waiting.is_empty()
        {
            // P1: Only collect into a Vec when SJF sorting is needed.
            // For the common FCFS path, iterate the queue directly.
            if self.sjf {
                let mut candidates: Vec<_> = waiting.iter().collect();
                candidates.sort_by_key(|seq| seq.meta.input_ids.len());

                let mut new_admitted = 0usize;
                for (seqs_used, seq) in candidates.into_iter().enumerate() {
                    if seqs_used >= seq_budget || kv_budget_remaining == 0 || tile_budget_remaining == 0
                    {
                        break;
                    }
                    if self.max_new_prefills_per_iter != 0
                        && new_admitted >= self.max_new_prefills_per_iter
                    {
                        break;
                    }

                    let prompt_len = seq.meta.input_ids.len();
                    let requested_tokens = match self.chunked_prefill_size {
                        Some(_) => self.chunk_tokens(prompt_len).min(kv_budget_remaining),
                        None => {
                            prompt_len.min(kv_budget_remaining)
                        }
                    };
                    let decode_reserve = decode_reserve_for_new(seq.meta.max_tokens);
                    let tokens_to_prefill = fit_new_prefill_tokens(
                        requested_tokens,
                        kv_budget_remaining,
                        tile_budget_remaining,
                        decode_reserve,
                    );
                    if tokens_to_prefill == 0 {
                        break;
                    }

                    let is_partial = tokens_to_prefill < prompt_len;
                    prefill_batch.push(PrefillEntry {
                        request_id: seq.meta.id.clone(),
                        token_range: 0..tokens_to_prefill,
                        is_partial,
                    });
                    new_admitted += 1;

                    kv_budget_remaining =
                        kv_budget_remaining.saturating_sub(tokens_to_prefill + decode_reserve);
                    tile_budget_remaining = tile_budget_remaining
                        .saturating_sub(prefill_tiles_for_tokens(tokens_to_prefill));
                }
            } else {
                // FCFS: iterate queue directly without collecting.
                let mut new_admitted = 0usize;
                let mut seqs_used = 0usize;
                for seq in waiting.iter() {
                    if seqs_used >= seq_budget || kv_budget_remaining == 0 || tile_budget_remaining == 0
                    {
                        break;
                    }
                    if self.max_new_prefills_per_iter != 0
                        && new_admitted >= self.max_new_prefills_per_iter
                    {
                        break;
                    }

                    let prompt_len = seq.meta.input_ids.len();
                    let requested_tokens = match self.chunked_prefill_size {
                        Some(_) => self.chunk_tokens(prompt_len).min(kv_budget_remaining),
                        None => {
                            prompt_len.min(kv_budget_remaining)
                        }
                    };
                    let decode_reserve = decode_reserve_for_new(seq.meta.max_tokens);
                    let tokens_to_prefill = fit_new_prefill_tokens(
                        requested_tokens,
                        kv_budget_remaining,
                        tile_budget_remaining,
                        decode_reserve,
                    );
                    if tokens_to_prefill == 0 {
                        break;
                    }

                    let is_partial = tokens_to_prefill < prompt_len;
                    prefill_batch.push(PrefillEntry {
                        request_id: seq.meta.id.clone(),
                        token_range: 0..tokens_to_prefill,
                        is_partial,
                    });
                    new_admitted += 1;
                    seqs_used += 1;

                    kv_budget_remaining =
                        kv_budget_remaining.saturating_sub(tokens_to_prefill + decode_reserve);
                    tile_budget_remaining = tile_budget_remaining
                        .saturating_sub(prefill_tiles_for_tokens(tokens_to_prefill));
                }
            }
        }

        let total_tokens: usize = prefill_batch.iter().map(|e| e.token_range.len()).sum();

        BatchPlan {
            prefill_batch,
            total_tokens,
        }
    }

    fn name(&self) -> &'static str {
        "continuous_batching"
    }
}

fn prefill_tile_budget(budget: &TokenBudget) -> usize {
    if budget.max_tokens == 0 || budget.max_seqs == 0 {
        return 0;
    }
    div_ceil(budget.max_tokens, PREFILL_Q_TILE)
        .max(budget.max_seqs)
        .max(1)
}

fn fit_prefill_tokens(
    requested: usize,
    kv_budget_remaining: usize,
    tile_budget_remaining: usize,
) -> usize {
    requested
        .min(kv_budget_remaining)
        .min(tile_budget_remaining.saturating_mul(PREFILL_Q_TILE))
}

fn fit_new_prefill_tokens(
    requested: usize,
    kv_budget_remaining: usize,
    tile_budget_remaining: usize,
    decode_reserve: usize,
) -> usize {
    let Some(prefill_budget) = kv_budget_remaining.checked_sub(decode_reserve) else {
        return 0;
    };
    fit_prefill_tokens(requested, prefill_budget, tile_budget_remaining)
}

fn decode_reserve_for_new(max_tokens: usize) -> usize {
    max_tokens.saturating_sub(1)
}

fn prefill_tiles_for_tokens(tokens: usize) -> usize {
    if tokens == 0 {
        0
    } else {
        div_ceil(tokens, PREFILL_Q_TILE)
    }
}

fn div_ceil(n: usize, d: usize) -> usize {
    n.saturating_add(d - 1) / d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::inference_session::handle::RequestHandle;
    use crate::domain::inference_session::lifecycle::*;
    use std::sync::Arc;
    use std::time::Instant;

    fn make_waiting(ids: &[(&str, usize)]) -> WaitingQueue {
        make_waiting_with_max_tokens(ids, 1)
    }

    fn make_waiting_with_max_tokens(ids: &[(&str, usize)], max_tokens: usize) -> WaitingQueue {
        let mut q = WaitingQueue::new();
        for (id, len) in ids {
            let meta = Arc::new(RequestMeta {
                id: RequestId::new_v4(),
                external_id: id.to_string(),
                sequence_id: SequenceId(1),
                input_ids: vec![1i32; *len],
                max_tokens,
                sampling: SamplingParams::default(),
                priority: Priority(0),
                stream: false,
                stop_sequences: vec![],
                ignore_eos: false,
                diffusion: None,
                arrival_time: Instant::now(),
            });
            q.push(InferenceSession::new(meta, RequestHandle::noop()));
        }
        q
    }

    fn empty_running() -> RunningSet {
        RunningSet {
            num_prefilling: 0,
            num_decoding: 0,
            prefilling_continuations: vec![],
        }
    }

    #[test]
    fn empty_waiting_returns_empty_plan() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = WaitingQueue::new();
        let running = empty_running();
        let budget = TokenBudget {
            max_tokens: 512,
            max_seqs: 4,
        };

        let plan = policy.schedule(&waiting, &running, &budget);
        assert!(!plan.has_work());
    }

    #[test]
    fn selects_requests_within_budget() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = make_waiting(&[("a", 10), ("b", 10), ("c", 10)]);
        let running = empty_running();
        let budget = TokenBudget {
            max_tokens: 25,
            max_seqs: 4,
        };

        let plan = policy.schedule(&waiting, &running, &budget);
        // Budget = 25 tokens, each request is 10. The third request is
        // admitted as a 5-token partial chunk instead of being starved.
        assert_eq!(plan.prefill_batch.len(), 3);
        assert_eq!(plan.total_tokens, 25);
        assert_eq!(plan.prefill_batch[2].token_range, 0..5);
        assert!(plan.prefill_batch[2].is_partial);
    }

    #[test]
    fn respects_seq_budget() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = make_waiting(&[("a", 5), ("b", 5), ("c", 5)]);
        let running = empty_running();
        let budget = TokenBudget {
            max_tokens: 512,
            max_seqs: 2,
        };

        let plan = policy.schedule(&waiting, &running, &budget);
        assert_eq!(plan.prefill_batch.len(), 2);
    }

    #[test]
    fn chunked_prefill_splits_long_prompt() {
        let policy = ContinuousBatchingPolicy::new(Some(10));
        let waiting = make_waiting(&[("long", 25)]);
        let running = empty_running();
        let budget = TokenBudget {
            max_tokens: 512,
            max_seqs: 4,
        };

        let plan = policy.schedule(&waiting, &running, &budget);
        // chunk_size=10, prompt=25 → first chunk is 10 tokens, is_partial=true
        assert_eq!(plan.prefill_batch.len(), 1);
        assert_eq!(plan.prefill_batch[0].token_range, 0..10);
        assert!(plan.prefill_batch[0].is_partial);
        assert_eq!(plan.total_tokens, 10);
    }

    #[test]
    fn chunked_prefill_continuation_has_priority() {
        let policy = ContinuousBatchingPolicy::new(Some(10));
        let waiting = make_waiting(&[("new", 5)]);
        let new_id = waiting.iter().next().unwrap().meta.id.clone();
        // Simulate a sequence already in prefilling with 15 tokens remaining.
        let cont_id = RequestId::new_v4();
        let running = RunningSet {
            num_prefilling: 1,
            num_decoding: 0,
            prefilling_continuations: vec![(cont_id, 15)],
        };
        let budget = TokenBudget {
            max_tokens: 12,
            max_seqs: 4,
        };

        let plan = policy.schedule(&waiting, &running, &budget);
        // Continuation chunk takes 10 tokens (chunk_size), leaving 2 for new.
        // In chunked mode the new request can consume the remaining 2-token budget.
        assert_eq!(plan.prefill_batch.len(), 2);
        assert_eq!(plan.prefill_batch[0].request_id, cont_id);
        assert_eq!(plan.prefill_batch[0].token_range.len(), 10);
        assert_eq!(plan.prefill_batch[1].request_id, new_id);
        assert_eq!(plan.prefill_batch[1].token_range.len(), 2);
        assert!(plan.prefill_batch[1].is_partial);
    }

    #[test]
    fn no_chunking_sends_full_prompt() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = make_waiting(&[("full", 100)]);
        let running = empty_running();
        let budget = TokenBudget {
            max_tokens: 512,
            max_seqs: 4,
        };

        let plan = policy.schedule(&waiting, &running, &budget);
        assert_eq!(plan.prefill_batch.len(), 1);
        assert_eq!(plan.prefill_batch[0].token_range, 0..100);
        assert!(!plan.prefill_batch[0].is_partial);
    }

    #[test]
    fn no_chunking_adapts_to_runtime_token_budget() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = make_waiting(&[("large", 100)]);
        let running = empty_running();
        let budget = TokenBudget {
            max_tokens: 32,
            max_seqs: 4,
        };

        let plan = policy.schedule(&waiting, &running, &budget);
        assert_eq!(plan.prefill_batch.len(), 1);
        assert_eq!(plan.prefill_batch[0].token_range, 0..32);
        assert!(plan.prefill_batch[0].is_partial);
        assert_eq!(plan.total_tokens, 32);
    }

    #[test]
    fn respects_ragged_prefill_tile_budget() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = make_waiting(&[("a", 2936), ("b", 2936), ("c", 2936)]);
        let running = empty_running();
        let budget = TokenBudget {
            max_tokens: 8192,
            max_seqs: 32,
        };

        let plan = policy.schedule(&waiting, &running, &budget);
        let total_tiles: usize = plan
            .prefill_batch
            .iter()
            .map(|entry| prefill_tiles_for_tokens(entry.token_range.len()))
            .sum();

        assert_eq!(plan.prefill_batch.len(), 3);
        assert_eq!(plan.prefill_batch[0].token_range, 0..2936);
        assert_eq!(plan.prefill_batch[1].token_range, 0..2936);
        assert_eq!(plan.prefill_batch[2].token_range, 0..2304);
        assert_eq!(plan.total_tokens, 8176);
        assert_eq!(total_tiles, 64);
        assert!(plan.prefill_batch[2].is_partial);
    }

    #[test]
    fn new_requests_reserve_future_decode_slots() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = make_waiting_with_max_tokens(&[("a", 100), ("b", 100), ("c", 100)], 33);
        let running = empty_running();
        let budget = TokenBudget {
            max_tokens: 264,
            max_seqs: 32,
        };

        let plan = policy.schedule(&waiting, &running, &budget);
        assert_eq!(plan.prefill_batch.len(), 2);
        assert_eq!(plan.total_tokens, 200);
    }
}
