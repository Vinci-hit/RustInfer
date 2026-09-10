//! Shared execution contracts and allocation-free phase accounting.
//!
//! Plans describe operations; BatchPlan still owns model-specific indexing.
//! Host spans include existing waits; optional GPU spans use nonblocking events.
//! No timer adds a device synchronization or changes buffer ownership.
use crate::domain::ports::{OpError, OpResult};
use infer_core::exec::{ExecScope, ScopeTimer};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Phase {
    Prefill,
    Decode,
    Mixed,
    Draft,
    Verify,
    Snapshot,
    Restore,
    Replay,
    CatchUp,
    Readout,
    Wait,
    Commit,
}
const PHASES: [Phase; 12] = [
    Phase::Prefill,
    Phase::Decode,
    Phase::Mixed,
    Phase::Draft,
    Phase::Verify,
    Phase::Snapshot,
    Phase::Restore,
    Phase::Replay,
    Phase::CatchUp,
    Phase::Readout,
    Phase::Wait,
    Phase::Commit,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    Eager,
    DecodeGraph { slot: usize },
    PrefillGraph { tokens: usize },
    MixedGraph { key: u64 },
}

/// References to existing allocations, never a request for new storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceUse {
    /// Runtime owns staging, hidden and sample output until this call returns.
    Runtime,
    /// A/B/C remain borrowed until the matching decode finalize completes.
    Abc,
    /// Proposer owns token tape and ping-pong hidden; consumed before next draft.
    Proposer,
    /// Runtime borrows the proposer's tape for verification/replay synchronously.
    BorrowedTape,
    /// Persistent recurrent snapshot; overwritten by the next transaction.
    Recurrent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub phase: Phase,
    pub mode: ExecutionMode,
    pub batch: usize,
    pub tokens: usize,
    pub workspace: WorkspaceUse,
}

impl ExecutionPlan {
    pub fn eager(phase: Phase, batch: usize, tokens: usize, workspace: WorkspaceUse) -> Self {
        Self {
            phase,
            mode: ExecutionMode::Eager,
            batch,
            tokens,
            workspace,
        }
    }

    pub fn validate(&self) -> OpResult<()> {
        let valid = match self.mode {
            ExecutionMode::Eager => true,
            ExecutionMode::DecodeGraph { slot } => {
                self.phase == Phase::Decode
                    && self.tokens == self.batch
                    && slot >= self.batch
                    && slot > 0
            }
            ExecutionMode::MixedGraph { .. } => self.phase == Phase::Mixed && self.tokens > 0,
            ExecutionMode::PrefillGraph { tokens } => {
                self.phase == Phase::Prefill && tokens == self.tokens && tokens > 0
            }
        };
        let workspace_valid = match self.workspace {
            WorkspaceUse::Runtime => true,
            WorkspaceUse::Abc => matches!(
                self.phase,
                Phase::Decode | Phase::Mixed | Phase::Wait | Phase::Commit
            ),
            WorkspaceUse::Proposer => {
                matches!(self.phase, Phase::Draft | Phase::CatchUp | Phase::Wait)
            }
            WorkspaceUse::BorrowedTape => {
                matches!(self.phase, Phase::Verify | Phase::Replay | Phase::Wait)
            }
            WorkspaceUse::Recurrent => matches!(self.phase, Phase::Snapshot | Phase::Restore),
        };
        if (self.batch == 0 && self.phase != Phase::Wait) || !valid || !workspace_valid {
            return Err(OpError::Shape("invalid execution phase/graph shape".into()));
        }
        Ok(())
    }

    /// One entry point for ordinary and speculative operations. The closure
    /// receives the validated plan so graph dispatch uses the recorded decision.
    pub fn execute<R>(
        self,
        metrics: &ExecutionMetrics,
        op: impl FnOnce(Self) -> OpResult<R>,
    ) -> OpResult<R> {
        self.validate()?;
        let started = metrics.0.as_ref().map(|_| Instant::now());
        let gpu_sample = metrics.begin_gpu(self.phase);
        let result = op(self);
        if gpu_sample {
            metrics.end_gpu(self.phase, result.is_ok());
        }
        if let Some(started) = started {
            metrics.record(self, started.elapsed(), result.is_ok());
        }
        result
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseStats {
    pub calls: u64,
    pub failures: u64,
    pub tokens: u64,
    pub graph_calls: u64,
    pub host_ns: u64,
    pub max_host_ns: u64,
    pub gpu_samples: u64,
    pub gpu_total_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TokenStats {
    pub rounds: u64,
    pub proposed: u64,
    pub accepted: u64,
    pub emitted: u64,
    pub materialized: u64,
}

struct TimingSlot {
    timer: Option<Box<dyn ScopeTimer>>,
    pending: bool,
    active: bool,
    success: bool,
    calls: u64,
}
struct GpuTimings {
    sample_every: u64,
    slots: [TimingSlot; PHASES.len()],
}
struct MetricsInner {
    owner: &'static str,
    report_every: u64,
    phases: Mutex<[PhaseStats; PHASES.len()]>,
    tokens: Mutex<TokenStats>,
    gpu: Mutex<Option<GpuTimings>>,
}

/// Disabled by default: no clock reads, locks or per-step allocations.
/// Enabled storage is fixed-size and allocated once with its owner at startup.
#[derive(Clone, Default)]
pub struct ExecutionMetrics(Option<Arc<MetricsInner>>);

impl ExecutionMetrics {
    pub fn from_env(owner: &'static str) -> Self {
        let every = std::env::var("RUSTINFER_EXECUTION_STATS_EVERY")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        Self::new(owner, every)
    }

    pub fn new(owner: &'static str, report_every: u64) -> Self {
        Self((report_every > 0).then(|| {
            Arc::new(MetricsInner {
                owner,
                report_every,
                phases: Mutex::new([PhaseStats::default(); PHASES.len()]),
                tokens: Mutex::new(TokenStats::default()),
                gpu: Mutex::new(None),
            })
        }))
    }

    /// Reserve event pairs before graph priming/serving. Host-only backends
    /// keep returning None. Allocation errors fail startup, not a live request.
    pub fn prepare_gpu<S: ExecScope>(&self, scope: &S) -> OpResult<()> {
        let Some(m) = &self.0 else {
            return Ok(());
        };
        let every = std::env::var("RUSTINFER_EXECUTION_GPU_SAMPLE_EVERY")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        if every == 0 {
            return Ok(());
        }
        let mut storage = m.gpu.lock().unwrap_or_else(|e| e.into_inner());
        if storage.is_some() {
            return Ok(());
        }
        let mut slots = std::array::from_fn(|_| TimingSlot {
            timer: None,
            pending: false,
            active: false,
            success: false,
            calls: 0,
        });
        for phase in PHASES {
            // Host waits and commits have no compute region of their own.
            if !matches!(phase, Phase::Wait | Phase::Commit) {
                slots[phase as usize].timer = scope.create_timer()?;
            }
        }
        *storage = Some(GpuTimings {
            sample_every: every,
            slots,
        });
        Ok(())
    }

    fn begin_gpu(&self, phase: Phase) -> bool {
        let Some(m) = &self.0 else {
            return false;
        };
        let mut storage = m.gpu.lock().unwrap_or_else(|e| e.into_inner());
        let Some(gpu) = storage.as_mut() else {
            return false;
        };
        let slot = &mut gpu.slots[phase as usize];
        if slot.active {
            return false;
        }
        let Some(timer) = slot.timer.as_mut() else {
            return false;
        };
        let result = (|| -> OpResult<bool> {
            if slot.pending {
                match timer.elapsed_ms()? {
                    None => return Ok(false), // Never overwrite an unfinished pair.
                    Some(ms) => {
                        slot.pending = false;
                        if slot.success {
                            let mut phases = m.phases.lock().unwrap_or_else(|e| e.into_inner());
                            let s = &mut phases[phase as usize];
                            s.gpu_samples = s.gpu_samples.saturating_add(1);
                            s.gpu_total_ms += ms as f64;
                        }
                    }
                }
            }
            slot.calls = slot.calls.wrapping_add(1);
            if (slot.calls - 1).is_multiple_of(gpu.sample_every) {
                timer.start()?;
                slot.active = true;
                return Ok(true);
            }
            Ok(false)
        })();
        match result {
            Ok(active) => active,
            Err(error) => {
                tracing::warn!(target: "execution_stats", ?phase, %error, "GPU timer disabled");
                slot.timer = None;
                false
            }
        }
    }

    fn end_gpu(&self, phase: Phase, success: bool) {
        let Some(m) = &self.0 else {
            return;
        };
        let mut storage = m.gpu.lock().unwrap_or_else(|e| e.into_inner());
        let Some(gpu) = storage.as_mut() else {
            return;
        };
        let slot = &mut gpu.slots[phase as usize];
        slot.active = false;
        if let Some(timer) = slot.timer.as_mut() {
            match timer.stop() {
                Ok(()) => {
                    slot.pending = true;
                    slot.success = success;
                }
                Err(error) => {
                    tracing::warn!(target: "execution_stats", ?phase, %error, "GPU timer disabled");
                    slot.timer = None;
                }
            }
        }
    }

    pub fn snapshot(&self, phase: Phase) -> PhaseStats {
        self.0.as_ref().map_or(PhaseStats::default(), |m| {
            m.phases.lock().unwrap_or_else(|e| e.into_inner())[phase as usize]
        })
    }

    pub fn token_snapshot(&self) -> TokenStats {
        self.0.as_ref().map_or(TokenStats::default(), |m| {
            *m.tokens.lock().unwrap_or_else(|e| e.into_inner())
        })
    }

    fn record(&self, plan: ExecutionPlan, elapsed: Duration, success: bool) {
        let Some(m) = &self.0 else {
            return;
        };
        let stats = {
            let mut phases = m.phases.lock().unwrap_or_else(|e| e.into_inner());
            let s = &mut phases[plan.phase as usize];
            let ns = elapsed.as_nanos().min(u64::MAX as u128) as u64;
            s.calls = s.calls.saturating_add(1);
            s.failures = s.failures.saturating_add(u64::from(!success));
            if success {
                s.tokens = s.tokens.saturating_add(plan.tokens as u64);
            }
            s.graph_calls = s
                .graph_calls
                .saturating_add(u64::from(plan.mode != ExecutionMode::Eager));
            s.host_ns = s.host_ns.saturating_add(ns);
            s.max_host_ns = s.max_host_ns.max(ns);
            *s
        };
        tracing::debug!(target: "execution_stats", owner = m.owner, phase = ?plan.phase,
            mode = ?plan.mode, workspace = ?plan.workspace, batch = plan.batch,
            tokens = plan.tokens, success, host_ms = elapsed.as_secs_f64() * 1e3,
            "execution phase");
        if stats.calls.is_multiple_of(m.report_every) {
            log_phase(m.owner, plan.phase, stats);
        }
    }

    /// Call only after a successful state/output commit. Accepted drafts and
    /// emitted tokens are separate: the bonus token is not an accepted draft.
    pub fn committed(&self, proposed: usize, accepted: usize, emitted: usize, materialized: usize) {
        let Some(m) = &self.0 else {
            return;
        };
        let s = {
            let mut s = m.tokens.lock().unwrap_or_else(|e| e.into_inner());
            s.rounds = s.rounds.saturating_add(1);
            s.proposed = s.proposed.saturating_add(proposed as u64);
            s.accepted = s.accepted.saturating_add(accepted as u64);
            s.emitted = s.emitted.saturating_add(emitted as u64);
            s.materialized = s.materialized.saturating_add(materialized as u64);
            *s
        };
        if s.rounds.is_multiple_of(m.report_every) {
            log_tokens(m.owner, s);
        }
    }
}

fn log_phase(owner: &str, phase: Phase, s: PhaseStats) {
    tracing::info!(target: "execution_stats", owner, phase = ?phase, calls = s.calls,
        failures = s.failures, tokens = s.tokens, graph_calls = s.graph_calls,
        host_total_ms = s.host_ns as f64 / 1e6,
        host_mean_ms = s.host_ns as f64 / s.calls.max(1) as f64 / 1e6,
        host_max_ms = s.max_host_ns as f64 / 1e6,
        gpu_samples = s.gpu_samples, gpu_total_ms = s.gpu_total_ms,
        gpu_mean_ms = (s.gpu_samples > 0).then(|| s.gpu_total_ms / s.gpu_samples as f64), "execution summary");
}
fn log_tokens(owner: &str, s: TokenStats) {
    tracing::info!(target: "execution_stats", owner, rounds = s.rounds,
        proposed = s.proposed, accepted = s.accepted, emitted = s.emitted,
        materialized = s.materialized,
        acceptance_rate = (s.proposed > 0).then(|| s.accepted as f64 / s.proposed as f64),
        "execution tokens");
}
impl Drop for MetricsInner {
    fn drop(&mut self) {
        let phases = self.phases.get_mut().unwrap_or_else(|e| e.into_inner());
        for (phase, stats) in PHASES.into_iter().zip(phases.iter()) {
            if stats.calls > 0 {
                log_phase(self.owner, phase, *stats);
            }
        }
        let tokens = *self.tokens.get_mut().unwrap_or_else(|e| e.into_inner());
        if tokens.rounds > 0 {
            log_tokens(self.owner, tokens);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unfinished_gpu_samples_are_not_overwritten() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Timer {
            starts: Arc<AtomicUsize>,
            polls: usize,
        }
        impl ScopeTimer for Timer {
            fn start(&mut self) -> OpResult<()> {
                self.starts.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            fn stop(&mut self) -> OpResult<()> {
                Ok(())
            }
            fn elapsed_ms(&mut self) -> OpResult<Option<f32>> {
                self.polls += 1;
                Ok((self.polls >= 3).then_some(2.5))
            }
        }
        let starts = Arc::new(AtomicUsize::new(0));
        let m = ExecutionMetrics::new("test", 100);
        let mut slots = std::array::from_fn(|_| TimingSlot {
            timer: None,
            pending: false,
            active: false,
            success: false,
            calls: 0,
        });
        slots[Phase::Decode as usize].timer = Some(Box::new(Timer {
            starts: starts.clone(),
            polls: 0,
        }));
        *m.0.as_ref().unwrap().gpu.lock().unwrap() = Some(GpuTimings {
            sample_every: 1,
            slots,
        });
        let p = ExecutionPlan::eager(Phase::Decode, 1, 1, WorkspaceUse::Abc);
        for _ in 0..4 {
            p.execute(&m, |_| Ok(())).unwrap();
        }
        assert_eq!(starts.load(Ordering::Relaxed), 2);
        let s = m.snapshot(Phase::Decode);
        assert_eq!(s.calls, 4);
        assert_eq!(s.gpu_samples, 1);
        assert_eq!(s.gpu_total_ms, 2.5);
    }

    #[test]
    fn graph_contract_rejects_verification_and_undersized_slots() {
        let p = ExecutionPlan {
            mode: ExecutionMode::DecodeGraph { slot: 2 },
            ..ExecutionPlan::eager(Phase::Verify, 1, 2, WorkspaceUse::BorrowedTape)
        };
        assert!(
            p.execute(&ExecutionMetrics::default(), |_| -> OpResult<()> {
                panic!("must not execute")
            })
            .is_err()
        );
        let p = ExecutionPlan {
            phase: Phase::Decode,
            workspace: WorkspaceUse::Abc,
            batch: 3,
            tokens: 3,
            ..p
        };
        assert!(p.validate().is_err());
        assert!(
            ExecutionPlan::eager(Phase::Decode, 1, 1, WorkspaceUse::BorrowedTape)
                .validate()
                .is_err()
        );
        assert!(
            ExecutionPlan {
                batch: 1,
                tokens: 1,
                ..p
            }
            .validate()
            .is_ok()
        );
    }
    #[test]
    fn metrics_count_failures_without_committing_tokens() {
        let m = ExecutionMetrics::new("test", 100);
        let p = ExecutionPlan::eager(Phase::Verify, 1, 4, WorkspaceUse::Runtime);
        p.execute(&m, |_| Ok(())).unwrap();
        assert!(
            p.execute(&m, |_| Err::<(), _>(OpError::Shape("failed".into())))
                .is_err()
        );
        let s = m.snapshot(Phase::Verify);
        assert_eq!((s.calls, s.failures, s.tokens), (2, 1, 4));
        m.committed(3, 2, 3, 3);
        m.committed(0, 0, 1, 1);
        let t = m.token_snapshot();
        assert_eq!((t.rounds, t.proposed, t.accepted, t.emitted), (2, 3, 2, 4));
    }
    #[test]
    fn disabled_stats_still_execute_and_propagate_errors() {
        let m = ExecutionMetrics::default();
        let p = ExecutionPlan::eager(Phase::Draft, 1, 0, WorkspaceUse::Proposer);
        assert_eq!(p.execute(&m, |_| Ok(42)).unwrap(), 42);
        assert_eq!(m.snapshot(Phase::Draft).calls, 0);
    }
}
