//! Worker registry.
//!
//! Two-direction lookup table maintained by the router thread. The liveness
//! watchdog reads a shared snapshot through `Arc<RwLock<RegistryView>>`; only
//! the router thread mutates it.
//!
//! The `*_id`/`*_identity` separation keeps the engine API free of raw ZMQ
//! identity bytes — engine code uses [`WorkerId`] only.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use infer_protocol::worker_to_scheduler_control::WorkerState;

use super::handle::WorkerId;

/// Authoritative entry for one worker.
#[derive(Debug, Clone)]
pub(crate) struct RegisteredWorker {
    /// Stable id. Currently only read in tests; consumed by the
    /// multi-worker selection path once TP/PP support lands.
    #[allow(dead_code)]
    pub(crate) worker_id: WorkerId,
    pub(crate) last_seen: Instant,
    pub(crate) state: WorkerState,
}

/// Shared snapshot of the registry. Mutated by the router thread, read by the
/// liveness watchdog.
#[derive(Debug, Default)]
pub(crate) struct RegistryView {
    /// `WorkerId → RegisteredWorker` for liveness sweeps and broadcast iteration.
    pub(crate) by_worker_id: HashMap<WorkerId, RegisteredWorker>,
}

/// Mutable side of the registry, owned exclusively by the router thread.
pub(crate) struct Registry {
    /// Reverse lookup: ZMQ identity bytes → `WorkerId`. Used when we receive
    /// a frame and need to surface it as a typed worker.
    by_identity: HashMap<Vec<u8>, WorkerId>,
    /// Shared view used by the liveness task.
    pub(crate) view: Arc<RwLock<RegistryView>>,
}

impl Registry {
    pub(crate) fn new() -> Self {
        Self {
            by_identity: HashMap::new(),
            view: Arc::new(RwLock::new(RegistryView::default())),
        }
    }

    /// Look up the `WorkerId` for an inbound ZMQ identity, registering it on
    /// first sight. Returns `Err(())` if the worker is already known but has
    /// been marked lost — silent re-registration is refused.
    pub(crate) fn intern(
        &mut self,
        identity: &[u8],
        now: Instant,
        state: WorkerState,
    ) -> Result<WorkerId, RegistrationRefused> {
        if let Some(wid) = self.by_identity.get(identity) {
            // Update last_seen + state in place.
            let mut view = self.view.write().expect("registry view poisoned");
            if let Some(entry) = view.by_worker_id.get_mut(wid) {
                entry.last_seen = now;
                entry.state = state;
                return Ok(wid.clone());
            }
            // Identity known but no entry → was removed by liveness watchdog.
            // Refuse silent re-registration.
            return Err(RegistrationRefused::Reconnect);
        }

        let wid = WorkerId::from_identity(identity);
        self.by_identity.insert(identity.to_vec(), wid.clone());
        let mut view = self.view.write().expect("registry view poisoned");
        view.by_worker_id.insert(
            wid.clone(),
            RegisteredWorker {
                worker_id: wid.clone(),
                last_seen: now,
                state,
            },
        );
        Ok(wid)
    }

    /// Snapshot the current set of registered workers. Held briefly under the
    /// read lock; allocates a `Vec` so the caller doesn't keep the lock.
    pub(crate) fn current_workers(&self) -> Vec<WorkerId> {
        self.view
            .read()
            .expect("registry view poisoned")
            .by_worker_id
            .keys()
            .cloned()
            .collect()
    }

    /// Forget a worker entirely. Called by the liveness watchdog when a
    /// worker times out, and by `Drop` paths.
    #[allow(dead_code)]
    pub(crate) fn evict(&mut self, worker: &WorkerId) {
        let mut view = self.view.write().expect("registry view poisoned");
        view.by_worker_id.remove(worker);
        // We deliberately DO NOT remove from `by_identity` so that a later
        // Hello from the same identity is recognized as a refused reconnect
        // (RegistrationRefused::Reconnect) instead of silently rebinding.
    }

    /// Resolve an inbound identity to its `WorkerId` without mutating state.
    /// Used for routing frames that don't update `last_seen` themselves.
    #[allow(dead_code)]
    pub(crate) fn resolve(&self, identity: &[u8]) -> Option<WorkerId> {
        self.by_identity.get(identity).cloned()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RegistrationRefused {
    /// Identity matches a worker that was previously evicted by the liveness
    /// watchdog. We refuse silent rebinding.
    Reconnect,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_registers_first_seen() {
        let mut reg = Registry::new();
        let now = Instant::now();
        let wid = reg.intern(b"w1", now, WorkerState::Ready).unwrap();
        assert_eq!(reg.current_workers(), vec![wid.clone()]);
        let view = reg.view.read().unwrap();
        let entry = view.by_worker_id.get(&wid).unwrap();
        assert_eq!(entry.last_seen, now);
    }

    #[test]
    fn intern_updates_last_seen() {
        let mut reg = Registry::new();
        let t0 = Instant::now();
        let wid = reg.intern(b"w1", t0, WorkerState::Ready).unwrap();
        let t1 = t0 + std::time::Duration::from_millis(50);
        let wid2 = reg.intern(b"w1", t1, WorkerState::Running).unwrap();
        assert_eq!(wid, wid2);
        let view = reg.view.read().unwrap();
        let entry = view.by_worker_id.get(&wid).unwrap();
        assert_eq!(entry.last_seen, t1);
        assert_eq!(entry.state, WorkerState::Running);
    }

    #[test]
    fn evict_then_reconnect_is_refused() {
        let mut reg = Registry::new();
        let now = Instant::now();
        let wid = reg.intern(b"w1", now, WorkerState::Ready).unwrap();
        reg.evict(&wid);
        let again = reg.intern(b"w1", now, WorkerState::Ready);
        assert!(matches!(again, Err(RegistrationRefused::Reconnect)));
    }
}
