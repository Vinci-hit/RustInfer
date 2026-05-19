//! Default FCFS continuous batching policy.
//!
//! Selects waiting requests in FIFO order (respecting priority from WaitingQueue)
//! until the token/sequence budget is exhausted.
//! Supports chunked prefill: long prompts are split across iterations.

use crate::cache::traits::CacheState;
use crate::policy::traits::{
    BatchPlan, DecodeEntry, PrefillEntry, RunningSet, SchedulingPolicy,
};
use crate::request::queue::WaitingQueue;
use crate::utils::token_budget::TokenBudget;

/// FCFS continuous batching policy.
///
/// Scheduling order per iteration:
/// 1. All decode sequences continue (1 token each).
/// 2. Continuation chunks for already-prefilling sequences (priority: don't waste KV).
/// 3. New requests from the waiting queue (FCFS within priority tier).
pub struct ContinuousBatchingPolicy {
    /// Max tokens per prefill chunk (None = no chunking, entire prompt at once).
    pub chunked_prefill_size: Option<usize>,
}

impl ContinuousBatchingPolicy {
    pub fn new(chunked_prefill_size: Option<usize>) -> Self {
        Self { chunked_prefill_size }
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
        _cache_state: &CacheState,
    ) -> BatchPlan {
        // Step 1: All running decode sequences continue (each uses 1 token).
        let decode_batch: Vec<DecodeEntry> = running
            .running_ids
            .iter()
            .take(running.num_decoding)
            .map(|id| DecodeEntry {
                request_id: id.clone(),
            })
            .collect();

        let decode_tokens = decode_batch.len();

        // Token budget remaining after decode.
        let mut token_budget_remaining = budget.max_tokens.saturating_sub(decode_tokens);
        // Seq budget: continuation chunks don't consume new seq slots (already counted).
        let seq_budget = budget.max_seqs.saturating_sub(running.total());

        let mut prefill_batch: Vec<PrefillEntry> = Vec::new();

        // Step 2: Schedule continuation chunks for already-prefilling sequences.
        // These MUST run — they already have KV allocated. Priority over new requests.
        for (req_id, remaining) in &running.prefilling_continuations {
            if token_budget_remaining == 0 {
                break;
            }

            let chunk = self.chunk_tokens(*remaining).min(token_budget_remaining);
            let start = 0; // engine will compute actual start from num_computed_tokens
            let is_partial = chunk < *remaining;

            prefill_batch.push(PrefillEntry {
                request_id: req_id.clone(),
                token_range: start..(start + chunk),
                is_partial,
            });

            token_budget_remaining = token_budget_remaining.saturating_sub(chunk);
        }

        // Step 3: Select new requests from waiting queue.
        if token_budget_remaining > 0 && seq_budget > 0 && !waiting.is_empty() {
            let mut seqs_used = 0usize;

            for seq in waiting.iter() {
                if seqs_used >= seq_budget || token_budget_remaining == 0 {
                    break;
                }

                let prompt_len = seq.meta.input_ids.len();
                let tokens_to_prefill = match self.chunked_prefill_size {
                    Some(_) => self.chunk_tokens(prompt_len).min(token_budget_remaining),
                    None => {
                        if prompt_len > token_budget_remaining {
                            break;
                        }
                        prompt_len
                    }
                };
                if tokens_to_prefill == 0 {
                    break;
                }

                let is_partial = tokens_to_prefill < prompt_len;
                prefill_batch.push(PrefillEntry {
                    request_id: seq.meta.id.clone(),
                    token_range: 0..tokens_to_prefill,
                    is_partial,
                });

                token_budget_remaining = token_budget_remaining.saturating_sub(tokens_to_prefill);
                seqs_used += 1;
            }
        }

        let prefill_tokens: usize = prefill_batch.iter().map(|e| e.token_range.len()).sum();
        let total_tokens = decode_tokens + prefill_tokens;

        BatchPlan {
            prefill_batch,
            decode_batch,
            preemptions: vec![],
            total_tokens,
        }
    }

    fn name(&self) -> &'static str {
        "continuous_batching"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::lifecycle::*;
    use crate::request::handle::RequestHandle;
    use std::sync::Arc;
    use std::time::Instant;

    fn make_waiting(ids: &[(&str, usize)]) -> WaitingQueue {
        let mut q = WaitingQueue::new();
        for (id, len) in ids {
            let meta = Arc::new(RequestMeta {
                id: RequestId(id.to_string()),
                sequence_id: SequenceId(1),
                input_ids: vec![1i32; *len],
                max_tokens: 100,
                sampling: SamplingParams::default(),
                priority: Priority(0),
                stream: false,
                stop_sequences: vec![],
                diffusion: None,
                arrival_time: Instant::now(),
            });
            q.push(Sequence::new(meta, RequestHandle::noop()));
        }
        q
    }

    fn empty_running() -> RunningSet {
        RunningSet {
            num_prefilling: 0,
            num_decoding: 0,
            decode_tokens: 0,
            running_ids: vec![],
            prefilling_continuations: vec![],
        }
    }

    fn cache_state() -> CacheState {
        CacheState {
            free_blocks: 100,
            total_blocks: 256,
            utilization: 0.0,
            evictable_blocks: 0,
        }
    }

    #[test]
    fn empty_waiting_returns_empty_plan() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = WaitingQueue::new();
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 512, max_seqs: 4 };

        let plan = policy.schedule(&waiting, &running, &budget, &cache_state());
        assert!(!plan.has_work());
    }

    #[test]
    fn selects_requests_within_budget() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = make_waiting(&[("a", 10), ("b", 10), ("c", 10)]);
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 25, max_seqs: 4 };

        let plan = policy.schedule(&waiting, &running, &budget, &cache_state());
        // Budget = 25 tokens, each request is 10 → can fit 2.
        assert_eq!(plan.prefill_batch.len(), 2);
        assert_eq!(plan.total_tokens, 20);
    }

    #[test]
    fn respects_seq_budget() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = make_waiting(&[("a", 5), ("b", 5), ("c", 5)]);
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 512, max_seqs: 2 };

        let plan = policy.schedule(&waiting, &running, &budget, &cache_state());
        assert_eq!(plan.prefill_batch.len(), 2);
    }

    #[test]
    fn chunked_prefill_splits_long_prompt() {
        let policy = ContinuousBatchingPolicy::new(Some(10));
        let waiting = make_waiting(&[("long", 25)]);
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 512, max_seqs: 4 };

        let plan = policy.schedule(&waiting, &running, &budget, &cache_state());
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
        // Simulate a sequence already in prefilling with 15 tokens remaining.
        let running = RunningSet {
            num_prefilling: 1,
            num_decoding: 0,
            decode_tokens: 0,
            running_ids: vec![],
            prefilling_continuations: vec![
                (RequestId("cont".to_string()), 15),
            ],
        };
        let budget = TokenBudget { max_tokens: 12, max_seqs: 4 };

        let plan = policy.schedule(&waiting, &running, &budget, &cache_state());
        // Continuation chunk takes 10 tokens (chunk_size), leaving 2 for new.
        // In chunked mode the new request can consume the remaining 2-token budget.
        assert_eq!(plan.prefill_batch.len(), 2);
        assert_eq!(plan.prefill_batch[0].request_id.0, "cont");
        assert_eq!(plan.prefill_batch[0].token_range.len(), 10);
        assert_eq!(plan.prefill_batch[1].request_id.0, "new");
        assert_eq!(plan.prefill_batch[1].token_range.len(), 2);
        assert!(plan.prefill_batch[1].is_partial);
    }

    #[test]
    fn chunked_prefill_with_decode_mixed() {
        let policy = ContinuousBatchingPolicy::new(Some(10));
        let waiting = make_waiting(&[("new", 8)]);
        let running = RunningSet {
            num_prefilling: 1,
            num_decoding: 2,
            decode_tokens: 2,
            running_ids: vec![
                RequestId("d1".to_string()),
                RequestId("d2".to_string()),
            ],
            prefilling_continuations: vec![
                (RequestId("cont".to_string()), 20),
            ],
        };
        // Budget = 20 tokens, 4 seqs.
        // decode takes 2 tokens → 18 remaining.
        // continuation chunk takes 10 → 8 remaining.
        // "new" takes 8 → fits!
        let budget = TokenBudget { max_tokens: 20, max_seqs: 4 };

        let plan = policy.schedule(&waiting, &running, &budget, &cache_state());
        assert_eq!(plan.decode_batch.len(), 2);
        // 2 prefill entries: continuation + new
        assert_eq!(plan.prefill_batch.len(), 2);
        assert_eq!(plan.prefill_batch[0].request_id.0, "cont");
        assert_eq!(plan.prefill_batch[1].request_id.0, "new");
        assert_eq!(plan.total_tokens, 2 + 10 + 8);
    }

    #[test]
    fn no_chunking_sends_full_prompt() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = make_waiting(&[("full", 100)]);
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 512, max_seqs: 4 };

        let plan = policy.schedule(&waiting, &running, &budget, &cache_state());
        assert_eq!(plan.prefill_batch.len(), 1);
        assert_eq!(plan.prefill_batch[0].token_range, 0..100);
        assert!(!plan.prefill_batch[0].is_partial);
    }
}
