//! KV reclamation — the single owner of "a sequence is done, reclaim its KV".
//!
//! Termination reaches the scheduler through four paths (step completion,
//! cancel, worker step error, preemption relief), and each used to re-derive
//! the `enable_prefix_caching` fork inline — the exact shape from which
//! double-release/leak bugs grow. Every termination path now goes through
//! [`KvReclaimer::reclaim_terminated_collect`]; the proactive-eviction paths
//! share [`KvReclaimer::evict_and_free`].
//!
//! Semantics of the two modes, in one place:
//! - **Prefix caching on**: a finished sequence's KV stays in the RadixTree for
//!   reuse — the chain is only marked finished (unpinned). Budget is reclaimed
//!   later by LRU eviction (`evict_target > 0` evicts immediately and returns
//!   the freed indices for the worker).
//! - **Prefix caching off**: the worker recycles physical slots autonomously on
//!   finish/cancel/preempt; the scheduler only releases the sequences' slots
//!   from its budget (clamped to outstanding).
//!
//! This is a borrow-view over the engine's resources, constructed on the fly
//! at each call site (the engine keeps single ownership of the underlying
//! state; this struct just fixes the *procedure*).

use infer_protocol::scheduler_to_worker_control::{FreeKvIndices, SchedulerControlMessage};

use crate::domain::kv_budget::KvBudget;
use crate::infrastructure::kv_cache::radix_tree::{GlobalIndex, RadixTree};
use crate::infrastructure::transport::control_plane::{ControlPlaneCmdTx, WorkerId};

/// A terminated sequence and the KV slot count it held, captured **while the
/// session was still resolvable** (before table removal).
#[derive(Debug, Clone, Copy)]
pub struct SeqKv {
    pub sequence_id: u64,
    pub kv_slots: u32,
}

pub struct KvReclaimer<'a> {
    pub radix: &'a mut RadixTree,
    pub kv_budget: &'a mut KvBudget,
    pub control_cmd: &'a ControlPlaneCmdTx,
    pub model_instance_id: &'a str,
    pub enable_prefix_caching: bool,
}

impl KvReclaimer<'_> {
    /// The one termination entry point. Marks/releases the sequences' KV per
    /// the active mode; with prefix caching, additionally evicts at least
    /// `evict_target` slots from the LRU and returns the freed global indices
    /// (the caller decides the wire message: `FreeKvIndices` vs
    /// `Preempt.free_indices`). Without prefix caching the returned list is
    /// always empty — the worker frees its own slots.
    pub fn reclaim_terminated_collect(
        &mut self,
        seqs: &[SeqKv],
        evict_target: u32,
        reason: &'static str,
    ) -> Vec<GlobalIndex> {
        if self.enable_prefix_caching {
            for seq in seqs {
                self.radix.mark_finished_chain(seq.sequence_id);
            }
            let indices = self.radix.evict_collect_at_least(evict_target as usize);
            if !indices.is_empty() {
                self.release_slots(indices.len() as u32, reason);
            }
            indices
        } else {
            let total: u32 = seqs
                .iter()
                .fold(0u32, |acc, seq| acc.saturating_add(seq.kv_slots));
            if total > 0 {
                self.release_slots(total, reason);
            }
            Vec::new()
        }
    }

    /// Proactive relief: evict at least `min_slots` from the prefix-cache LRU,
    /// release the freed budget, and tell `worker` to free the physical slots.
    /// Returns the number of slots actually freed (0 when the LRU is empty).
    /// Deliberately not gated on `enable_prefix_caching`: with caching off the
    /// tree is never fed, so this is naturally a no-op — and if entries exist
    /// anyway (drift), freeing them under pressure is the right call.
    pub fn evict_and_free(
        &mut self,
        min_slots: u32,
        worker: &WorkerId,
        reason: &'static str,
    ) -> usize {
        if min_slots == 0 {
            return 0;
        }
        let indices = self.radix.evict_collect_at_least(min_slots as usize);
        if indices.is_empty() {
            return 0;
        }
        let freed = indices.len();
        self.release_slots(freed as u32, reason);
        self.free_indices_to_worker(indices, worker, reason);
        freed
    }

    /// Send a `FreeKvIndices` control message; failures are logged (the
    /// heartbeat drift recalibration is the designed safety net for a lost
    /// free).
    pub fn free_indices_to_worker(
        &self,
        indices: Vec<GlobalIndex>,
        worker: &WorkerId,
        reason: &'static str,
    ) {
        if indices.is_empty() {
            return;
        }
        let len = indices.len();
        let msg = SchedulerControlMessage::FreeKvIndices(FreeKvIndices {
            model_instance_id: self.model_instance_id.to_string(),
            indices,
        });
        if let Err(err) = self.control_cmd.send_to(worker, msg) {
            tracing::error!(
                count = len,
                reason,
                "failed to send FreeKvIndices to worker: {}",
                err
            );
        }
    }

    /// Release `requested` slots from the budget, clamped to outstanding (a
    /// clamp indicates double-release or drift — warn loudly). Returns the
    /// amount actually released.
    pub fn release_slots(&mut self, requested: u32, reason: &'static str) -> u32 {
        let releasable = requested.min(self.kv_budget.outstanding());
        if releasable < requested {
            tracing::warn!(
                requested,
                outstanding = self.kv_budget.outstanding(),
                released = releasable,
                reason,
                "KV budget release exceeds outstanding; clamping"
            );
        }
        if releasable > 0 {
            self.kv_budget.release(releasable);
        }
        releasable
    }
}
