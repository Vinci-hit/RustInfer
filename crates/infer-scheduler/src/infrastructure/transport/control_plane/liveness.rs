//! Heartbeat watchdog.
//!
//! Owns a tokio `interval` task spawned by `ControlPlane::bootstrap`. Reads
//! the shared `RegistryView`, removes any worker whose `last_seen` has aged
//! past the configured timeout, and emits [`ControlEvent::WorkerLost`].

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};

use super::handle::{ControlEvent, WorkerId};
use super::registry::RegistryView;

/// Spawn the watchdog task. The returned `cancel_tx` lets the owner stop the
/// watchdog cooperatively at shutdown.
pub(crate) fn spawn(
    interval: Duration,
    timeout: Duration,
    view: Arc<RwLock<RegistryView>>,
    event_tx: mpsc::UnboundedSender<ControlEvent>,
) -> oneshot::Sender<()> {
    let (cancel_tx, cancel_rx) = oneshot::channel();
    tokio::spawn(run(interval, timeout, view, event_tx, cancel_rx));
    cancel_tx
}

async fn run(
    interval: Duration,
    timeout: Duration,
    view: Arc<RwLock<RegistryView>>,
    event_tx: mpsc::UnboundedSender<ControlEvent>,
    mut cancel: oneshot::Receiver<()>,
) {
    let mut tick = tokio::time::interval(interval);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = &mut cancel => {
                tracing::debug!("liveness watchdog shutting down");
                return;
            }
            _ = tick.tick() => {
                let now = Instant::now();
                let lost = collect_lost(&view, now, timeout);
                for (worker, last_seen) in lost {
                    let last_seen_ms = now.saturating_duration_since(last_seen).as_millis() as u64;
                    tracing::error!(
                        "worker liveness timeout: worker={} last_seen_ms={}",
                        worker,
                        last_seen_ms
                    );
                    if event_tx.send(ControlEvent::WorkerLost { worker, last_seen_ms }).is_err() {
                        // Engine receiver dropped — nothing to do.
                        return;
                    }
                }
            }
        }
    }
}

/// Drain timed-out workers from the registry view, returning the evicted
/// (id, last_seen) pairs.
fn collect_lost(
    view: &Arc<RwLock<RegistryView>>,
    now: Instant,
    timeout: Duration,
) -> Vec<(WorkerId, Instant)> {
    let mut g = view.write().expect("registry view poisoned");
    let lost_ids: Vec<WorkerId> = g
        .by_worker_id
        .iter()
        .filter_map(|(id, w)| {
            (now.saturating_duration_since(w.last_seen) > timeout).then_some(id.clone())
        })
        .collect();
    let mut out = Vec::with_capacity(lost_ids.len());
    for id in lost_ids {
        if let Some(w) = g.by_worker_id.remove(&id) {
            out.push((id, w.last_seen));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::transport::control_plane::registry::RegisteredWorker;
    use infer_protocol::worker_to_scheduler_control::WorkerState;

    fn put(view: &Arc<RwLock<RegistryView>>, id: &[u8], last_seen: Instant) -> WorkerId {
        let wid = WorkerId::from_identity(id);
        view.write().unwrap().by_worker_id.insert(
            wid.clone(),
            RegisteredWorker {
                worker_id: wid.clone(),
                last_seen,
                state: WorkerState::Running,
            },
        );
        wid
    }

    #[test]
    fn collect_lost_evicts_only_overdue() {
        let view = Arc::new(RwLock::new(RegistryView::default()));
        let now = Instant::now();
        let _fresh = put(&view, b"fresh", now - Duration::from_millis(50));
        let stale = put(&view, b"stale", now - Duration::from_secs(10));

        let lost = collect_lost(&view, now, Duration::from_secs(1));
        assert_eq!(lost.len(), 1);
        assert_eq!(lost[0].0, stale);

        let view = view.read().unwrap();
        assert_eq!(view.by_worker_id.len(), 1);
    }
}
