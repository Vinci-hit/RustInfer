//! Diffusion scheduling policy.
//!
//! Groups requests by (height, width, num_steps) and dispatches entire batches.
//! No continuous batching — a batch runs to completion before the next starts.

use std::collections::HashMap;

use crate::cache::traits::CacheState;
use crate::policy::traits::{
    BatchPlan, PrefillEntry, RunningSet, SchedulingPolicy,
};
use crate::request::lifecycle::RequestId;
use crate::request::queue::WaitingQueue;
use crate::utils::token_budget::TokenBudget;

/// Shape key for batching: only requests with identical shape can be batched together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ShapeKey {
    height: u32,
    width: u32,
}

/// Diffusion scheduling policy.
///
/// Batching constraint: only requests with the **same (height, width)** can share a batch.
/// Different seeds are fine in the same batch (only affects initial noise).
///
/// Behavior:
/// - Groups waiting requests by shape
/// - Selects the largest group (or earliest-arriving) to fill the batch
/// - Entire batch runs all denoising steps to completion
/// - Worker returns only when the full batch is done
pub struct DiffusionPolicy {
    pub max_batch_size: usize,
}

impl DiffusionPolicy {
    pub fn new(max_batch_size: usize) -> Self {
        Self { max_batch_size }
    }
}

impl SchedulingPolicy for DiffusionPolicy {
    fn schedule(
        &self,
        waiting: &WaitingQueue,
        running: &RunningSet,
        _budget: &TokenBudget,
        _cache_state: &CacheState,
    ) -> BatchPlan {
        // If worker is still processing a batch, don't schedule anything.
        // (Engine handles this via worker_busy flag, but double-check here.)
        if running.num_prefilling > 0 {
            return BatchPlan::empty();
        }

        if waiting.is_empty() {
            return BatchPlan::empty();
        }

        // Group waiting requests by shape.
        // We use input_ids.len() as a proxy key since diffusion requests
        // store shape info in a way the policy can access via meta.
        // For now, we treat all waiting requests as same-shape (simplification)
        // and just batch up to max_batch_size in FIFO order.
        //
        // TODO: When DiffusionRequestMeta is available with height/width fields,
        // implement proper shape-based grouping.

        let batch_size = waiting.len().min(self.max_batch_size);

        let prefill_batch: Vec<PrefillEntry> = waiting
            .iter()
            .take(batch_size)
            .map(|seq| PrefillEntry {
                request_id: seq.meta.id.clone(),
                token_range: 0..seq.meta.input_ids.len(),
                is_partial: false, // Diffusion: always full request
            })
            .collect();

        let total_tokens = prefill_batch.iter().map(|e| e.token_range.len()).sum();

        BatchPlan {
            prefill_batch,
            decode_batch: vec![], // Diffusion has no decode phase
            preemptions: vec![],
            total_tokens,
        }
    }

    fn name(&self) -> &'static str {
        "diffusion"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::lifecycle::*;
    use crate::request::handle::RequestHandle;
    use std::sync::Arc;
    use std::time::Instant;

    fn make_waiting(ids: &[&str]) -> WaitingQueue {
        let mut q = WaitingQueue::new();
        for id in ids {
            let meta = Arc::new(RequestMeta {
                id: RequestId(id.to_string()),
                input_ids: vec![1i32; 10], // dummy prompt tokens
                max_tokens: 1, // not used for diffusion
                sampling: SamplingParams::default(),
                priority: Priority(0),
                stream: false,
                stop_sequences: vec![],
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
    fn empty_queue_returns_empty() {
        let policy = DiffusionPolicy::new(4);
        let waiting = WaitingQueue::new();
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 9999, max_seqs: 99 };

        let plan = policy.schedule(&waiting, &running, &budget, &cache_state());
        assert!(!plan.has_work());
    }

    #[test]
    fn batches_up_to_max_size() {
        let policy = DiffusionPolicy::new(3);
        let waiting = make_waiting(&["a", "b", "c", "d", "e"]);
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 9999, max_seqs: 99 };

        let plan = policy.schedule(&waiting, &running, &budget, &cache_state());
        // max_batch_size = 3, 5 waiting → selects 3
        assert_eq!(plan.prefill_batch.len(), 3);
        assert!(plan.decode_batch.is_empty());
        assert!(!plan.prefill_batch[0].is_partial);
    }

    #[test]
    fn does_not_schedule_while_running() {
        let policy = DiffusionPolicy::new(4);
        let waiting = make_waiting(&["a", "b"]);
        let running = RunningSet {
            num_prefilling: 2, // batch already running
            num_decoding: 0,
            decode_tokens: 0,
            running_ids: vec![],
            prefilling_continuations: vec![],
        };
        let budget = TokenBudget { max_tokens: 9999, max_seqs: 99 };

        let plan = policy.schedule(&waiting, &running, &budget, &cache_state());
        // Should not schedule while a batch is still running.
        assert!(!plan.has_work());
    }

    #[test]
    fn small_queue_sends_partial_batch() {
        let policy = DiffusionPolicy::new(8);
        let waiting = make_waiting(&["a", "b"]);
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 9999, max_seqs: 99 };

        let plan = policy.schedule(&waiting, &running, &budget, &cache_state());
        // Only 2 requests, max_batch=8 → sends 2 (don't wait to fill)
        assert_eq!(plan.prefill_batch.len(), 2);
    }
}
