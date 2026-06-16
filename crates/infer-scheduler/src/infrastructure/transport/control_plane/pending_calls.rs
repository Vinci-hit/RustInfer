//! RPC correlation table.
//!
//! Maps a [`RequestId`] handed out by the scheduler to the awaiting `oneshot`
//! (single-worker call) or fan-in collector (broadcast call). The router
//! thread resolves entries on reply; a periodic deadline sweep moves expired
//! entries into `Timeout`.
//!
//! ## Threading model
//!
//! The router runs on a dedicated std thread; engine code runs on tokio tasks.
//! Both call into [`PendingCalls`] but **only one can be inside at a time** in
//! practice — engine registers and the router completes/sweeps. Critical
//! sections are O(1) HashMap operations + `oneshot::send`, all non-blocking.
//!
//! Therefore [`PendingCalls`] uses `std::sync::Mutex` (not `tokio::sync`):
//! - Lock holders never `.await` (only HashMap mutations + `oneshot::send`,
//!   which is sync and non-blocking).
//! - The router thread is a std thread that cannot use `.await` anyway.
//! - Avoids the `try_lock` + `yield_now` busy-loop dance the tokio mutex
//!   forced on us.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use infer_protocol::RequestId;
use infer_protocol::worker_to_scheduler_control::WorkerControlMessage;
use tokio::sync::oneshot;

use super::handle::{ControlError, ControlResult, WorkerId};

// ─────────────────────────────────────────────────────────────────────────────
//  Public surface (used by ControlPlaneCmdTx)
// ─────────────────────────────────────────────────────────────────────────────

/// Awaitable reply for [`super::handle::ControlPlaneCmdTx::call_one`].
pub(crate) type OneRx = oneshot::Receiver<ControlResult<WorkerControlMessage>>;

/// Awaitable reply set for [`super::handle::ControlPlaneCmdTx::call_all`].
pub(crate) type AllRx = oneshot::Receiver<Vec<(WorkerId, ControlResult<WorkerControlMessage>)>>;

pub(crate) struct PendingCalls {
    next_id: AtomicU64,
    inner: Mutex<HashMap<RequestId, PendingEntry>>,
}

enum PendingEntry {
    One {
        deadline: Instant,
        tx: oneshot::Sender<ControlResult<WorkerControlMessage>>,
    },
    All {
        deadline: Instant,
        expected: Vec<WorkerId>,
        collected: Vec<(WorkerId, ControlResult<WorkerControlMessage>)>,
        tx: oneshot::Sender<Vec<(WorkerId, ControlResult<WorkerControlMessage>)>>,
    },
}

/// Lock the inner HashMap. Panic on poisoning is acceptable because a poisoned
/// pending-calls table means the router or engine has already crashed and the
/// process must restart.
fn lock_inner<'a>(
    m: &'a Mutex<HashMap<RequestId, PendingEntry>>,
) -> std::sync::MutexGuard<'a, HashMap<RequestId, PendingEntry>> {
    m.lock().expect("pending calls mutex poisoned")
}

impl PendingCalls {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            // RequestId(0) is reserved for uncorrelated traffic, so allocators
            // start at 1.
            next_id: AtomicU64::new(1),
            inner: Mutex::new(HashMap::new()),
        })
    }

    /// Register a single-target call. Returns the freshly allocated id and a
    /// oneshot receiver the caller awaits.
    pub(crate) fn register_one(&self, deadline: Instant) -> (RequestId, OneRx) {
        let id = self.alloc_id();
        let (tx, rx) = oneshot::channel();
        let entry = PendingEntry::One { deadline, tx };
        lock_inner(&self.inner).insert(id, entry);
        (id, rx)
    }

    /// Register a fan-in call against a snapshot of expected workers.
    pub(crate) fn register_all(&self, deadline: Instant) -> (RequestId, AllRx) {
        let id = self.alloc_id();
        let (tx, rx) = oneshot::channel();
        // Workers vector is filled in by the router thread when it knows the
        // current registry; for now we leave it empty and the router thread
        // populates it via `set_expected`.
        let entry = PendingEntry::All {
            deadline,
            expected: Vec::new(),
            collected: Vec::new(),
            tx,
        };
        lock_inner(&self.inner).insert(id, entry);
        (id, rx)
    }

    /// Called by the router thread once it knows the registered worker set
    /// for an `All` RPC.
    pub(crate) fn set_expected(&self, id: RequestId, workers: Vec<WorkerId>) {
        let mut g = lock_inner(&self.inner);
        if let Some(PendingEntry::All { expected, .. }) = g.get_mut(&id) {
            *expected = workers;
        }
    }

    /// Called by the router thread when a reply arrives.
    pub(crate) fn complete(&self, id: RequestId, worker: WorkerId, reply: WorkerControlMessage) {
        let mut g = lock_inner(&self.inner);

        let finished = match g.get_mut(&id) {
            Some(PendingEntry::One { .. }) => {
                // One-shot: take the entry out and fire.
                let Some(PendingEntry::One { tx, .. }) = g.remove(&id) else {
                    return;
                };
                let _ = tx.send(Ok(reply));
                true
            }
            Some(PendingEntry::All {
                expected,
                collected,
                ..
            }) => {
                collected.push((worker, Ok(reply)));
                expected.len() == collected.len()
            }
            None => return,
        };

        if finished && let Some(PendingEntry::All { collected, tx, .. }) = g.remove(&id) {
            let _ = tx.send(collected);
        }
    }

    /// Resolve a single-target call immediately with an error, e.g. when the
    /// router fails to put the request on the wire (H2). Without this the
    /// caller would block until the deadline sweep — wasting a full RPC
    /// timeout on a failure the router already knows about. No-op for `All`
    /// entries or unknown ids.
    pub(crate) fn fail_one(&self, id: RequestId, error: ControlError) {
        let mut g = lock_inner(&self.inner);
        if matches!(g.get(&id), Some(PendingEntry::One { .. })) {
            if let Some(PendingEntry::One { tx, .. }) = g.remove(&id) {
                let _ = tx.send(Err(error));
            }
        }
    }

    /// Periodic sweep for deadlines. Returns the count of fired entries.
    pub(crate) fn sweep_expired(&self, now: Instant) -> usize {
        let mut g = lock_inner(&self.inner);
        let expired: Vec<RequestId> = g
            .iter()
            .filter_map(|(id, entry)| {
                let dl = match entry {
                    PendingEntry::One { deadline, .. } => *deadline,
                    PendingEntry::All { deadline, .. } => *deadline,
                };
                (now >= dl).then_some(*id)
            })
            .collect();
        let count = expired.len();
        for id in expired {
            match g.remove(&id) {
                Some(PendingEntry::One { tx, deadline, .. }) => {
                    let elapsed = now.saturating_duration_since(deadline);
                    let _ = tx.send(Err(ControlError::Timeout(elapsed)));
                }
                Some(PendingEntry::All { tx, collected, .. }) => {
                    // Send what we have; missing replies surface as missing entries.
                    let _ = tx.send(collected);
                }
                None => {}
            }
        }
        count
    }

    /// Drop every pending entry with [`ControlError::Shutdown`]. Called when
    /// the router thread exits.
    pub(crate) fn shutdown(&self) {
        let mut g = lock_inner(&self.inner);
        for (_, entry) in g.drain() {
            match entry {
                PendingEntry::One { tx, .. } => {
                    let _ = tx.send(Err(ControlError::Shutdown));
                }
                PendingEntry::All { tx, collected, .. } => {
                    let _ = tx.send(collected);
                }
            }
        }
    }

    fn alloc_id(&self) -> RequestId {
        let raw = self.next_id.fetch_add(1, Ordering::Relaxed);
        // Allocator starts at 1 and increments; overflow back to 0 is
        // astronomically unlikely (>500 yrs at 1M RPCs/s) but we still skip 0
        // to preserve the "0 == uncorrelated" invariant.
        let raw = if raw == 0 {
            self.next_id.fetch_add(1, Ordering::Relaxed)
        } else {
            raw
        };
        RequestId(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn register_one_resolves_on_complete() {
        let pc = PendingCalls::new();
        let dl = Instant::now() + Duration::from_secs(5);
        let (id, rx) = pc.register_one(dl);
        let worker = WorkerId::from_identity(b"w1");
        pc.complete(id, worker, WorkerControlMessage::Pong);
        let reply = rx.await.unwrap().unwrap();
        assert!(matches!(reply, WorkerControlMessage::Pong));
    }

    #[tokio::test]
    async fn fail_one_resolves_immediately_with_error() {
        // H2: router send failure resolves the pending call now, not at the
        // deadline. Deadline is far in the future so a pass proves immediacy.
        let pc = PendingCalls::new();
        let dl = Instant::now() + Duration::from_secs(3600);
        let (id, rx) = pc.register_one(dl);
        pc.fail_one(id, ControlError::Router("send failed".into()));
        let err = rx.await.unwrap().unwrap_err();
        assert!(matches!(err, ControlError::Router(_)));
        // Idempotent: failing an already-resolved id is a no-op.
        pc.fail_one(id, ControlError::Router("again".into()));
    }

    #[tokio::test]
    async fn register_one_times_out() {
        let pc = PendingCalls::new();
        let dl = Instant::now() - Duration::from_millis(1); // already expired
        let (_id, rx) = pc.register_one(dl);
        pc.sweep_expired(Instant::now());
        let err = rx.await.unwrap().unwrap_err();
        assert!(matches!(err, ControlError::Timeout(_)));
    }

    #[tokio::test]
    async fn ids_are_monotonic_and_skip_zero() {
        let pc = PendingCalls::new();
        let dl = Instant::now() + Duration::from_secs(5);
        let (id1, _) = pc.register_one(dl);
        let (id2, _) = pc.register_one(dl);
        assert!(id1.is_correlated());
        assert!(id2.is_correlated());
        assert!(id2.0 > id1.0);
    }

    #[tokio::test]
    async fn shutdown_drains_with_error() {
        let pc = PendingCalls::new();
        let dl = Instant::now() + Duration::from_secs(5);
        let (_id, rx) = pc.register_one(dl);
        pc.shutdown();
        let err = rx.await.unwrap().unwrap_err();
        assert!(matches!(err, ControlError::Shutdown));
    }
}
