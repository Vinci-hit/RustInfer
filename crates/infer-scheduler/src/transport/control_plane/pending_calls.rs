//! RPC correlation table.
//!
//! Maps a [`RequestId`] handed out by the scheduler to the awaiting `oneshot`
//! (single-worker call) or fan-in collector (broadcast call). The router
//! thread resolves entries on reply; a periodic deadline sweep moves expired
//! entries into `Timeout`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use infer_protocol::RequestId;
use infer_protocol::worker_to_scheduler_control::WorkerControlMessage;
use tokio::sync::{Mutex, oneshot};

use super::handle::{ControlError, ControlResult, WorkerId};

// ─────────────────────────────────────────────────────────────────────────────
//  Public surface (used by ControlPlaneCmdTx)
// ─────────────────────────────────────────────────────────────────────────────

/// Awaitable reply for [`super::handle::ControlPlaneCmdTx::call_one`].
pub(crate) type OneRx = oneshot::Receiver<ControlResult<WorkerControlMessage>>;

/// Awaitable reply set for [`super::handle::ControlPlaneCmdTx::call_all`].
pub(crate) type AllRx =
    oneshot::Receiver<Vec<(WorkerId, ControlResult<WorkerControlMessage>)>>;

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
        // Hold the lock briefly to insert; never await with it held.
        let entry = PendingEntry::One { deadline, tx };
        // Best-effort: use blocking_lock would be wrong from async; we use a
        // sync mutex pattern by tokio::sync::Mutex but only inside a non-async
        // path. Since this method is sync and called from sync contexts (the
        // engine event-loop path is async but does not hold this), we
        // tokio::task::block_in_place style: but we are already cheap, so
        // perform the insert via a try_lock-blocking_recv ladder isn't needed.
        // Use try_lock; if contended, fall back to a small busy loop. In
        // practice this map is contended only during peak control RPC bursts.
        loop {
            if let Ok(mut g) = self.inner.try_lock() {
                g.insert(id, entry);
                break;
            }
            std::thread::yield_now();
        }
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
        loop {
            if let Ok(mut g) = self.inner.try_lock() {
                g.insert(id, entry);
                break;
            }
            std::thread::yield_now();
        }
        (id, rx)
    }

    /// Called by the router thread once it knows the registered worker set
    /// for an `All` RPC.
    pub(crate) fn set_expected(&self, id: RequestId, workers: Vec<WorkerId>) {
        let mut g = match self.inner.try_lock() {
            Ok(g) => g,
            Err(_) => loop {
                if let Ok(g) = self.inner.try_lock() {
                    break g;
                }
                std::thread::yield_now();
            },
        };
        if let Some(PendingEntry::All { expected, .. }) = g.get_mut(&id) {
            *expected = workers;
        }
    }

    /// Called by the router thread when a reply arrives.
    pub(crate) fn complete(
        &self,
        id: RequestId,
        worker: WorkerId,
        reply: WorkerControlMessage,
    ) {
        let mut g = match self.inner.try_lock() {
            Ok(g) => g,
            Err(_) => loop {
                if let Ok(g) = self.inner.try_lock() {
                    break g;
                }
                std::thread::yield_now();
            },
        };

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

        if finished
            && let Some(PendingEntry::All { collected, tx, .. }) = g.remove(&id) {
                let _ = tx.send(collected);
            }
    }

    /// Periodic sweep for deadlines. Returns ids that fired.
    pub(crate) fn sweep_expired(&self, now: Instant) -> usize {
        let mut g = match self.inner.try_lock() {
            Ok(g) => g,
            Err(_) => return 0,
        };
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
        let mut g = match self.inner.try_lock() {
            Ok(g) => g,
            Err(_) => return,
        };
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
        let raw = if raw == 0 { self.next_id.fetch_add(1, Ordering::Relaxed) } else { raw };
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
