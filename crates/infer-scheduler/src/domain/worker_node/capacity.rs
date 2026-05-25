//! `Capacity` value object for `WorkerNode`.
//!
//! Type-driven replacement for the legacy `EffectiveCapacity` struct in
//! `worker_group.rs`. Wraps the protocol's flat `usize` fields into the
//! NewTypes from `domain::ids` so arithmetic mistakes (mixing token
//! counts with sequence counts) become compile errors.
//!
//! ## Aggregation across ranks
//!
//! For TP/PP groups, the effective capacity is the *minimum* across
//! ranks: a single weak rank caps the whole group. `from_protocol_ranks`
//! folds an iterator of `WorkerCapacity` accordingly.
//!
//! ## Why no `mem_profile`
//!
//! The original `EffectiveCapacity` had no memory fields. The protocol's
//! `WorkerCapacity` does carry `*_mem_*_gb` fields, but those are
//! diagnostic — used for logging at handshake — not consumed by the
//! scheduler's planning. We keep them out of the domain `Capacity`
//! (P3-I in the refactor plan: "delete `mem_profile`").

use infer_protocol::worker_to_scheduler_control::WorkerCapacity;

use crate::domain::ids::{SeqCount, TokenCount};

/// Aggregate capacity across one or more worker ranks.
///
/// Construct via [`Capacity::from_protocol_single`] or
/// [`Capacity::from_protocol_ranks`]; the constructors enforce
/// "minimum across ranks" semantics.
#[derive(Debug, Clone)]
pub struct Capacity {
    pub max_batch_tokens: TokenCount,
    pub max_batch_seqs: SeqCount,
    pub max_running_requests: SeqCount,
    /// Total KV cache tokens, if the worker reports a finite limit.
    /// `None` means "unbounded" (worker doesn't enforce a token cap).
    pub max_total_kv_tokens: Option<TokenCount>,
}

impl Capacity {
    /// Capacity for a single-rank worker (current production shape).
    pub fn from_protocol_single(cap: &WorkerCapacity) -> Self {
        Self {
            max_batch_tokens: TokenCount::new(cap.max_batch_tokens),
            max_batch_seqs: SeqCount::new(cap.max_batch_seqs),
            max_running_requests: SeqCount::new(cap.max_running_requests),
            max_total_kv_tokens: cap.max_total_kv_tokens.map(TokenCount::new),
        }
    }

    /// Capacity for a multi-rank group (TP/PP). Folds via *minimum*:
    /// the slowest/smallest rank dictates group capacity.
    ///
    /// Returns `None` if `ranks` is empty.
    pub fn from_protocol_ranks<'a, I>(ranks: I) -> Option<Self>
    where
        I: IntoIterator<Item = &'a WorkerCapacity>,
    {
        let mut iter = ranks.into_iter();
        let first = iter.next()?;
        let mut acc = Self::from_protocol_single(first);
        for cap in iter {
            acc = acc.fold_min(&Self::from_protocol_single(cap));
        }
        Some(acc)
    }

    fn fold_min(self, other: &Self) -> Self {
        Self {
            max_batch_tokens: TokenCount::new(
                self.max_batch_tokens.raw().min(other.max_batch_tokens.raw()),
            ),
            max_batch_seqs: SeqCount::new(
                self.max_batch_seqs.raw().min(other.max_batch_seqs.raw()),
            ),
            max_running_requests: SeqCount::new(
                self.max_running_requests
                    .raw()
                    .min(other.max_running_requests.raw()),
            ),
            max_total_kv_tokens: match (self.max_total_kv_tokens, other.max_total_kv_tokens) {
                (Some(a), Some(b)) => Some(TokenCount::new(a.raw().min(b.raw()))),
                // If any rank is unbounded but another is bounded, the bounded
                // value must dominate: the group is no better than its tightest
                // bounded rank.
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(tokens: usize, seqs: usize, running: usize, kv: Option<usize>) -> WorkerCapacity {
        WorkerCapacity {
            max_batch_tokens: tokens,
            max_batch_seqs: seqs,
            max_running_requests: running,
            max_total_kv_tokens: kv,
            free_mem_before_load_gb: None,
            free_mem_after_load_gb: None,
            weight_mem_usage_gb: None,
            workspace_mem_usage_gb: None,
            graph_mem_usage_gb: None,
        }
    }

    #[test]
    fn single_rank_passes_through() {
        let proto = cap(1024, 32, 64, Some(8192));
        let c = Capacity::from_protocol_single(&proto);
        assert_eq!(c.max_batch_tokens.raw(), 1024);
        assert_eq!(c.max_batch_seqs.raw(), 32);
        assert_eq!(c.max_running_requests.raw(), 64);
        assert_eq!(c.max_total_kv_tokens.unwrap().raw(), 8192);
    }

    #[test]
    fn multi_rank_folds_to_minimum() {
        let a = cap(2048, 32, 64, Some(8192));
        let b = cap(1024, 16, 32, Some(4096));
        let c = Capacity::from_protocol_ranks([&a, &b]).unwrap();
        assert_eq!(c.max_batch_tokens.raw(), 1024);
        assert_eq!(c.max_batch_seqs.raw(), 16);
        assert_eq!(c.max_running_requests.raw(), 32);
        assert_eq!(c.max_total_kv_tokens.unwrap().raw(), 4096);
    }

    #[test]
    fn empty_ranks_returns_none() {
        let empty: [&WorkerCapacity; 0] = [];
        assert!(Capacity::from_protocol_ranks(empty).is_none());
    }

    #[test]
    fn unbounded_rank_does_not_swallow_bounded_neighbor() {
        // If any rank reports a finite KV limit, the group must respect it.
        let bounded = cap(1024, 32, 64, Some(4096));
        let unbounded = cap(1024, 32, 64, None);
        let folded = Capacity::from_protocol_ranks([&bounded, &unbounded]).unwrap();
        assert_eq!(folded.max_total_kv_tokens.unwrap().raw(), 4096);
    }
}
