//! Aggregate Root #3: `WorkerNode<S>`.
//!
//! Typestate model: a `WorkerNode<Ready>` is statically guaranteed to
//! be a healthy, dispatchable worker. A `WorkerNode<Lost>` is a
//! terminal snapshot that exists solely to carry diagnostic context
//! (reason string, last-seen timestamp) into error paths and the
//! eventual `WorkerError` propagation.
//!
//! ## `snapshot_as_lost` does **not** consume `self`
//!
//! A naive consuming-transition `mark_lost(self) -> WorkerNode<Lost>`
//! cannot live behind `&mut self` borrow on a struct field — the
//! engine needs to keep the `Ready` worker reachable while it builds
//! up the error path, drains pending sessions, and returns control to
//! the supervisor. `snapshot_as_lost(&self, reason)` produces a fresh
//! `WorkerNode<Lost>` without disturbing the live `Ready` instance;
//! the `Ready` value is then dropped naturally when the engine's
//! event loop exits with `Err`.

use std::marker::PhantomData;
use std::time::Instant;

use crate::domain::ids::{LastSeenAt, ModelInstanceId, WorkerNodeId};

use super::capacity::Capacity;
use super::states::{Lost, NodeState, Ready};

/// A worker rank (or rank group) in a known state.
#[derive(Debug)]
pub struct WorkerNode<S: NodeState> {
    id: WorkerNodeId,
    model_instance_id: ModelInstanceId,
    capacity: Capacity,
    last_seen: LastSeenAt,
    state: S,
    _marker: PhantomData<fn() -> S>,
}

impl WorkerNode<Ready> {
    /// Construct a fresh ready worker.
    ///
    /// Caller is responsible for verifying handshake completion before
    /// calling this — the typestate is a *promise*, not a check.
    pub fn new_ready(
        id: WorkerNodeId,
        model_instance_id: ModelInstanceId,
        capacity: Capacity,
    ) -> Self {
        Self {
            id,
            model_instance_id,
            capacity,
            last_seen: LastSeenAt::now(),
            state: Ready,
            _marker: PhantomData,
        }
    }

    /// Refresh liveness (call on heartbeat / batch-ack).
    pub fn touch(&mut self, now: Instant) {
        self.last_seen = LastSeenAt::from_instant(now);
    }

    /// Build a `Lost` snapshot for diagnostics **without consuming** the
    /// live `Ready` value.
    ///
    /// The snapshot inherits id / capacity / last_seen from `self` so
    /// downstream error paths have full context. The live worker
    /// continues to occupy its slot until the engine itself unwinds.
    pub fn snapshot_as_lost(&self, reason: impl Into<String>) -> WorkerNode<Lost> {
        WorkerNode {
            id: self.id.clone(),
            model_instance_id: self.model_instance_id.clone(),
            capacity: self.capacity.clone(),
            last_seen: self.last_seen,
            state: Lost {
                reason: reason.into(),
            },
            _marker: PhantomData,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Accessors common to both states.
// ─────────────────────────────────────────────────────────────────────────────

impl<S: NodeState> WorkerNode<S> {
    pub fn id(&self) -> &WorkerNodeId {
        &self.id
    }

    pub fn model_instance_id(&self) -> &ModelInstanceId {
        &self.model_instance_id
    }

    pub fn capacity(&self) -> &Capacity {
        &self.capacity
    }

    pub fn last_seen(&self) -> LastSeenAt {
        self.last_seen
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Lost-only accessor.
// ─────────────────────────────────────────────────────────────────────────────

impl WorkerNode<Lost> {
    pub fn reason(&self) -> &str {
        &self.state.reason
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ids::{SeqCount, TokenCount};

    fn dummy_capacity() -> Capacity {
        Capacity {
            max_batch_tokens: TokenCount::new(2048),
            max_batch_seqs: SeqCount::new(32),
            max_running_requests: SeqCount::new(64),
            max_total_kv_tokens: Some(TokenCount::new(8192)),
        }
    }

    fn dummy_worker_id() -> WorkerNodeId {
        // WorkerNodeId is a re-export of transport::control_plane::WorkerId.
        // Use the crate-private constructor (we are in the same crate).
        crate::infrastructure::transport::control_plane::WorkerId::from_identity(b"worker-test")
    }

    #[test]
    fn ready_snapshot_does_not_consume_self() {
        let mut ready = WorkerNode::new_ready(
            dummy_worker_id(),
            ModelInstanceId::new("inst-A"),
            dummy_capacity(),
        );
        // touching is allowed and does not invalidate the snapshot.
        ready.touch(Instant::now());
        let lost = ready.snapshot_as_lost("heartbeat timeout");
        assert_eq!(lost.reason(), "heartbeat timeout");
        // ready is STILL usable — that's the whole point of the
        // non-consuming snapshot.
        assert_eq!(ready.model_instance_id().as_str(), "inst-A");
        assert_eq!(ready.capacity().max_batch_tokens.raw(), 2048);
    }

    #[test]
    fn lost_carries_full_context() {
        let ready = WorkerNode::new_ready(
            dummy_worker_id(),
            ModelInstanceId::new("inst-B"),
            dummy_capacity(),
        );
        let lost = ready.snapshot_as_lost("transport closed");
        // Lost snapshot inherits id/capacity/model_instance_id from the
        // ready value: the engine's error path can build full diagnostics.
        assert_eq!(lost.model_instance_id().as_str(), "inst-B");
        assert_eq!(lost.capacity().max_batch_seqs.raw(), 32);
        assert_eq!(lost.reason(), "transport closed");
    }

    #[test]
    fn touch_advances_last_seen() {
        let mut ready = WorkerNode::new_ready(
            dummy_worker_id(),
            ModelInstanceId::new("inst-C"),
            dummy_capacity(),
        );
        let before = ready.last_seen().raw();
        std::thread::sleep(std::time::Duration::from_millis(2));
        ready.touch(Instant::now());
        let after = ready.last_seen().raw();
        assert!(after > before);
    }
}
