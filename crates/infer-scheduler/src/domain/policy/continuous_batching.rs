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

use crate::domain::policy::traits::{
    BatchPlan, PrefillEntry, RunningSet, SchedulingPolicy,
};
use crate::domain::inference_session::queue::WaitingQueue;
use crate::domain::policy::token_budget::TokenBudget;

/// FCFS continuous batching policy.
///
/// Scheduling order per iteration:
/// 1. Continuation chunks for already-prefilling sequences (priority: don't waste KV).
/// 2. New requests from the waiting queue (FCFS within priority tier).
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
    ) -> BatchPlan {
        // The scheduler does not schedule decoding — the worker's
        // sub-scheduler decides that. We only plan prefills here.
        let mut token_budget_remaining = budget.max_tokens;
        // Seq budget: continuation chunks don't consume new seq slots (already counted).
        let seq_budget = budget.max_seqs.saturating_sub(running.total());

        let mut prefill_batch: Vec<PrefillEntry> = Vec::new();

        // 1. Schedule continuation chunks for already-prefilling sequences.
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

        // 2. Select new requests from waiting queue.
        if token_budget_remaining > 0 && seq_budget > 0 && !waiting.is_empty() {
            for (seqs_used, seq) in waiting.iter().enumerate() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::inference_session::lifecycle::*;
    use crate::domain::inference_session::handle::RequestHandle;
    use std::sync::Arc;
    use std::time::Instant;

    fn make_waiting(ids: &[(&str, usize)]) -> WaitingQueue {
        let mut q = WaitingQueue::new();
        for (id, len) in ids {
            let meta = Arc::new(RequestMeta {
                id: RequestId::new_v4(), external_id: id.to_string(),
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
            q.push(InferenceSession::new(meta, RequestHandle::noop()));
        }
        q
    }

    fn empty_running() -> RunningSet {
        RunningSet {
            num_prefilling: 0,
            prefilling_continuations: vec![],
        }
    }

    #[test]
    fn empty_waiting_returns_empty_plan() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = WaitingQueue::new();
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 512, max_seqs: 4 };

        let plan = policy.schedule(&waiting, &running, &budget);
        assert!(!plan.has_work());
    }

    #[test]
    fn selects_requests_within_budget() {
        let policy = ContinuousBatchingPolicy::new(None);
        let waiting = make_waiting(&[("a", 10), ("b", 10), ("c", 10)]);
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 25, max_seqs: 4 };

        let plan = policy.schedule(&waiting, &running, &budget);
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

        let plan = policy.schedule(&waiting, &running, &budget);
        assert_eq!(plan.prefill_batch.len(), 2);
    }

    #[test]
    fn chunked_prefill_splits_long_prompt() {
        let policy = ContinuousBatchingPolicy::new(Some(10));
        let waiting = make_waiting(&[("long", 25)]);
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 512, max_seqs: 4 };

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
            prefilling_continuations: vec![(cont_id, 15)],
        };
        let budget = TokenBudget { max_tokens: 12, max_seqs: 4 };

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
        let budget = TokenBudget { max_tokens: 512, max_seqs: 4 };

        let plan = policy.schedule(&waiting, &running, &budget);
        assert_eq!(plan.prefill_batch.len(), 1);
        assert_eq!(plan.prefill_batch[0].token_range, 0..100);
        assert!(!plan.prefill_batch[0].is_partial);
    }
}
