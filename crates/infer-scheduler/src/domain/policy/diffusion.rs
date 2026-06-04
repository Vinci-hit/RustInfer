//! Diffusion scheduling policy.
//!
//! Groups requests by compatible image generation shape and dispatches an entire
//! batch to the Worker. No continuous batching — a diffusion batch runs to
//! completion before the next starts.

use std::collections::HashMap;

use crate::domain::policy::traits::{BatchPlan, PrefillEntry, RunningSet, SchedulingPolicy};
use crate::domain::inference_session::queue::WaitingQueue;
use crate::domain::policy::token_budget::TokenBudget;

/// Shape/schedule key for batching: only requests with identical generation
/// geometry and denoise schedule can share one diffusion batch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DiffusionBatchKey {
    height: u32,
    width: u32,
    num_inference_steps: usize,
    sigmas_bits: Option<Vec<u32>>,
}

/// Diffusion scheduling policy.
///
/// Behavior:
/// - Groups waiting requests by `(height, width, num_inference_steps, sigmas-kind)`
/// - Selects the largest compatible group, tie-breaking by earliest queue order
/// - Dispatches up to `max_batch_size`
/// - Entire batch runs to completion before the next one starts
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
    ) -> BatchPlan {
        if running.num_prefilling > 0 {
            return BatchPlan::empty();
        }
        if waiting.is_empty() {
            return BatchPlan::empty();
        }

        let mut groups: HashMap<DiffusionBatchKey, Vec<_>> = HashMap::new();
        let mut key_order: Vec<DiffusionBatchKey> = Vec::new();

        for seq in waiting.iter() {
            let Some(req) = &seq.meta.diffusion else {
                continue;
            };
            let key = DiffusionBatchKey {
                height: req.height,
                width: req.width,
                num_inference_steps: req.num_inference_steps,
                sigmas_bits: req.sigmas.as_ref().map(|sigmas| {
                    sigmas.iter().map(|sigma| sigma.to_bits()).collect()
                }),
            };
            if !groups.contains_key(&key) {
                key_order.push(key.clone());
            }
            groups.entry(key).or_default().push(seq.meta.id.clone());
        }

        let Some(best_key) = key_order
            .iter()
            .max_by_key(|key| groups.get(*key).map(|v| v.len()).unwrap_or(0))
        else {
            return BatchPlan::empty();
        };

        let selected = groups
            .get(best_key)
            .map(|ids| ids.iter().take(self.max_batch_size).cloned().collect::<Vec<_>>())
            .unwrap_or_default();

        let prefill_batch: Vec<PrefillEntry> = selected
            .into_iter()
            .map(|request_id| PrefillEntry {
                request_id,
                token_range: 0..1, // Diffusion uses a separate WorkerCommand; token range is only a selection marker.
                is_partial: false,
            })
            .collect();

        let total_tokens = prefill_batch.len();
        BatchPlan {
            prefill_batch,
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
    use crate::domain::inference_session::handle::RequestHandle;
    use crate::domain::inference_session::lifecycle::*;
    use std::sync::Arc;
    use std::time::Instant;

    fn make_waiting(ids: &[&str]) -> WaitingQueue {
        let mut q = WaitingQueue::new();
        for id in ids {
            let meta = Arc::new(RequestMeta {
                id: RequestId::new_v4(), external_id: id.to_string(),
                sequence_id: SequenceId(1),
                input_ids: vec![0],
                max_tokens: 1,
                sampling: SamplingParams::default(),
                priority: Priority(0),
                stream: false,
                stop_sequences: vec![],
                ignore_eos: false,
                diffusion: Some(infer_protocol::server_to_scheduler::DiffusionRequest {
                    prompt: id.to_string(),
                    prompt_input_ids: vec![1, 2, 3],
                    height: 1024,
                    width: 1024,
                    num_inference_steps: 8,
                    ..Default::default()
                }),
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
    fn empty_queue_returns_empty() {
        let policy = DiffusionPolicy::new(4);
        let waiting = WaitingQueue::new();
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 9999, max_seqs: 99 };

        let plan = policy.schedule(&waiting, &running, &budget);
        assert!(!plan.has_work());
    }

    #[test]
    fn batches_up_to_max_size() {
        let policy = DiffusionPolicy::new(3);
        let waiting = make_waiting(&["a", "b", "c", "d", "e"]);
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 9999, max_seqs: 99 };

        let plan = policy.schedule(&waiting, &running, &budget);
        assert_eq!(plan.prefill_batch.len(), 3);
        assert!(!plan.prefill_batch[0].is_partial);
    }

    #[test]
    fn does_not_schedule_while_running() {
        let policy = DiffusionPolicy::new(4);
        let waiting = make_waiting(&["a", "b"]);
        let running = RunningSet {
            num_prefilling: 2,
            num_decoding: 0,
            prefilling_continuations: vec![],
        };
        let budget = TokenBudget { max_tokens: 9999, max_seqs: 99 };

        let plan = policy.schedule(&waiting, &running, &budget);
        assert!(!plan.has_work());
    }

    #[test]
    fn small_queue_sends_partial_batch() {
        let policy = DiffusionPolicy::new(8);
        let waiting = make_waiting(&["a", "b"]);
        let running = empty_running();
        let budget = TokenBudget { max_tokens: 9999, max_seqs: 99 };

        let plan = policy.schedule(&waiting, &running, &budget);
        assert_eq!(plan.prefill_batch.len(), 2);
    }
}
