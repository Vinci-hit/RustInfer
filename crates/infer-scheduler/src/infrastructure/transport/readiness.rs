//! Shared lifecycle snapshot for the frontend's readiness replies.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use infer_protocol::scheduler_to_server::{
    FRONTEND_PROTOCOL_VERSION, SchedulerMetricsSnapshot, SchedulerPong, SchedulerReadiness,
};

/// The idle engine refreshes once per second. A stalled engine must not be
/// kept ready by the independent frontend I/O thread's ability to send Pongs.
const ENGINE_STALE_AFTER: Duration = Duration::from_secs(10);

#[derive(Clone, Default)]
pub struct ReadinessHandle(Arc<Mutex<Snapshot>>);

#[derive(Default)]
struct Snapshot {
    state: SchedulerReadiness,
    last_engine_tick: Option<Instant>,
    metrics: Option<SchedulerMetricsSnapshot>,
}

impl ReadinessHandle {
    pub fn set(&self, state: SchedulerReadiness) {
        if let Ok(mut snapshot) = self.0.lock() {
            snapshot.state = state;
            snapshot.last_engine_tick = Some(Instant::now());
        }
    }

    pub fn tick(&self, metrics: SchedulerMetricsSnapshot) {
        if let Ok(mut snapshot) = self.0.lock() {
            snapshot.last_engine_tick = Some(Instant::now());
            snapshot.metrics = Some(metrics);
        }
    }

    pub fn state(&self) -> SchedulerReadiness {
        self.state_at(Instant::now())
    }

    fn state_at(&self, now: Instant) -> SchedulerReadiness {
        let Ok(snapshot) = self.0.lock() else {
            return SchedulerReadiness::Failed;
        };
        snapshot.state_at(now)
    }

    pub fn pong(&self) -> SchedulerPong {
        let (readiness, metrics) = self
            .0
            .lock()
            .map(|snapshot| {
                let state = snapshot.state_at(Instant::now());
                let metrics = (state == SchedulerReadiness::Ready)
                    .then(|| snapshot.metrics.clone())
                    .flatten();
                (state, metrics)
            })
            .unwrap_or((SchedulerReadiness::Failed, None));
        SchedulerPong {
            protocol_version: FRONTEND_PROTOCOL_VERSION,
            readiness,
            metrics,
        }
    }

    /// Invalidate readiness even if the engine future panics or is cancelled.
    pub fn guard(&self) -> ReadinessGuard {
        ReadinessGuard(self.clone())
    }
}

impl Snapshot {
    fn state_at(&self, now: Instant) -> SchedulerReadiness {
        if self.state == SchedulerReadiness::Ready
            && self
                .last_engine_tick
                .is_none_or(|tick| now.saturating_duration_since(tick) >= ENGINE_STALE_AFTER)
        {
            SchedulerReadiness::Failed
        } else {
            self.state
        }
    }
}

pub struct ReadinessGuard(ReadinessHandle);

impl Drop for ReadinessGuard {
    fn drop(&mut self) {
        if let Ok(mut snapshot) = self.0.0.lock()
            && snapshot.state != SchedulerReadiness::Draining
        {
            snapshot.state = SchedulerReadiness::Failed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_requires_engine_start_and_fresh_ticks() {
        let readiness = ReadinessHandle::default();
        readiness.tick(Default::default());
        assert_eq!(readiness.state(), SchedulerReadiness::Loading);
        readiness.set(SchedulerReadiness::Ready);
        assert_eq!(readiness.state(), SchedulerReadiness::Ready);
        assert_eq!(
            readiness.state_at(Instant::now() + ENGINE_STALE_AFTER),
            SchedulerReadiness::Failed
        );
        readiness.set(SchedulerReadiness::Draining);
        readiness.tick(Default::default());
        assert_eq!(readiness.state(), SchedulerReadiness::Draining);
    }

    #[test]
    fn engine_drop_fails_closed_but_preserves_explicit_drain() {
        let readiness = ReadinessHandle::default();
        readiness.set(SchedulerReadiness::Ready);
        drop(readiness.guard());
        assert_eq!(readiness.state(), SchedulerReadiness::Failed);
        readiness.set(SchedulerReadiness::Draining);
        drop(readiness.guard());
        assert_eq!(readiness.state(), SchedulerReadiness::Draining);
    }

    #[test]
    fn stale_engine_pong_never_reports_live_resource_metrics() {
        let readiness = ReadinessHandle::default();
        readiness.set(SchedulerReadiness::Ready);
        readiness.tick(SchedulerMetricsSnapshot {
            queued_requests: 4,
            ..Default::default()
        });
        assert_eq!(readiness.pong().metrics.unwrap().queued_requests, 4);
        readiness.0.lock().unwrap().last_engine_tick = Some(Instant::now() - ENGINE_STALE_AFTER);
        let pong = readiness.pong();
        assert_eq!(pong.readiness, SchedulerReadiness::Failed);
        assert!(pong.metrics.is_none());
    }
}
