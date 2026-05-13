//! Worker Group abstraction.
//!
//! A Worker Group is the scheduler-visible unit that serves one model instance.
//! Today it contains one rank on one GPU. The shape deliberately matches the
//! future TP/PP case where multiple ranks must become ready as a group.

use infer_protocol::worker_to_scheduler_control::{WorkerCapacity, WorkerReady};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerGroupState {
    Pending,
    Ready,
    Draining,
    Error,
}

#[derive(Debug, Clone)]
pub struct WorkerRank {
    pub worker_id: String,
    pub device: String,
    pub tp_rank: usize,
    pub tp_size: usize,
    pub pp_rank: usize,
    pub pp_size: usize,
    pub capacity: WorkerCapacity,
}

#[derive(Debug, Clone)]
pub struct EffectiveCapacity {
    pub max_batch_tokens: usize,
    pub max_batch_seqs: usize,
    pub max_running_requests: usize,
    pub max_total_kv_tokens: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct WorkerGroup {
    pub group_id: String,
    pub model_instance_id: String,
    pub model_type: String,
    pub state: WorkerGroupState,
    pub ranks: Vec<WorkerRank>,
    pub effective_capacity: EffectiveCapacity,
}

impl WorkerGroup {
    pub fn from_single_ready(ready: WorkerReady) -> Self {
        let rank = WorkerRank {
            worker_id: ready.worker_id,
            device: ready.device,
            tp_rank: 0,
            tp_size: 1,
            pp_rank: 0,
            pp_size: 1,
            capacity: ready.capacity,
        };
        Self::from_ready_ranks(
            format!("group-{}", ready.model_instance_id),
            ready.model_instance_id,
            ready.model_type,
            vec![rank],
        )
    }

    pub fn from_ready_ranks(
        group_id: String,
        model_instance_id: String,
        model_type: String,
        ranks: Vec<WorkerRank>,
    ) -> Self {
        let effective_capacity = EffectiveCapacity::from_ranks(&ranks);
        Self {
            group_id,
            model_instance_id,
            model_type,
            state: WorkerGroupState::Ready,
            ranks,
            effective_capacity,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.state == WorkerGroupState::Ready && !self.ranks.is_empty()
    }

    pub fn rank_count(&self) -> usize {
        self.ranks.len()
    }
}

impl EffectiveCapacity {
    pub fn from_ranks(ranks: &[WorkerRank]) -> Self {
        let max_batch_tokens = ranks
            .iter()
            .map(|rank| rank.capacity.max_batch_tokens)
            .min()
            .unwrap_or(0);
        let max_batch_seqs = ranks
            .iter()
            .map(|rank| rank.capacity.max_batch_seqs)
            .min()
            .unwrap_or(0);
        let max_running_requests = ranks
            .iter()
            .map(|rank| rank.capacity.max_running_requests)
            .min()
            .unwrap_or(0);
        let max_total_kv_tokens = ranks
            .iter()
            .map(|rank| rank.capacity.max_total_kv_tokens)
            .collect::<Option<Vec<_>>>()
            .and_then(|tokens| tokens.into_iter().min());

        Self {
            max_batch_tokens,
            max_batch_seqs,
            max_running_requests,
            max_total_kv_tokens,
        }
    }
}
