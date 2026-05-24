//! Worker-side control-plane pump.
//!
//! Owns the DEALER socket once bootstrap is done. Runs in a dedicated std
//! thread alongside [`super::sub_scheduler::SubScheduler`] / the diffusion
//! server, and bridges the control plane through two `std::sync::mpsc`
//! channels:
//!
//! - **down**: pump → sub-scheduler. Carries `SchedulerControlMessage`
//!   payloads (GrantBlocks / Cancel / Drain / UnloadModel).
//! - **up**: sub-scheduler → pump. Carries `WorkerControlMessage`
//!   payloads (NeedBlocks / StepError, plus the bootstrap-style messages
//!   if anyone needs them post-Ready).
//!
//! Heartbeats are emitted by the pump itself on a wall-clock tick, driven
//! by the poll timeout. This keeps the worker "alive" in the scheduler's
//! eyes even if the sub-scheduler is parked on a CUDA wait.
//!
//! `Ping` is answered directly inside the pump thread, never crossing the
//! mpsc boundary, so liveness stays decoupled from sub-scheduler progress.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::time::{Duration, Instant};

use anyhow::Result;
use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
use infer_protocol::worker_to_scheduler_control::{
    WorkerControlMessage, WorkerHeartbeat, WorkerState,
};
use infer_protocol::{ControlEnvelope, RequestId};

use super::control_client::WorkerControlClient;

// ─────────────────────────────────────────────────────────────────────────────
//  Liveness atomics
// ─────────────────────────────────────────────────────────────────────────────

/// Sub-scheduler-owned snapshot read by the pump on every heartbeat tick.
/// Keeping it lock-free means a long CUDA call never starves the heartbeat.
#[derive(Clone)]
pub struct WorkerLiveState {
    state: Arc<AtomicU8>,
    active_requests: Arc<AtomicUsize>,
}

impl Default for WorkerLiveState {
    fn default() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(WorkerStateAtomic::from(WorkerState::Running).0)),
            active_requests: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl WorkerLiveState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_state(&self, state: WorkerState) {
        self.state
            .store(WorkerStateAtomic::from(state).0, Ordering::Relaxed);
    }

    pub fn set_active_requests(&self, n: usize) {
        self.active_requests.store(n, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> (WorkerState, usize) {
        let raw = self.state.load(Ordering::Relaxed);
        let state = WorkerStateAtomic(raw).into();
        let active = self.active_requests.load(Ordering::Relaxed);
        (state, active)
    }
}

/// Compact `WorkerState ↔ u8` mapping for atomic storage.
#[derive(Copy, Clone)]
struct WorkerStateAtomic(u8);

impl From<WorkerState> for WorkerStateAtomic {
    fn from(s: WorkerState) -> Self {
        let v = match s {
            WorkerState::Spawned => 0,
            WorkerState::Connecting => 1,
            WorkerState::Registered => 2,
            WorkerState::LoadingModel => 3,
            WorkerState::ProfilingMemory => 4,
            WorkerState::AllocatingRuntime => 5,
            WorkerState::Warmup => 6,
            WorkerState::Ready => 7,
            WorkerState::Running => 8,
            WorkerState::Draining => 9,
            WorkerState::Error => 10,
            WorkerState::Stopped => 11,
        };
        Self(v)
    }
}

impl From<WorkerStateAtomic> for WorkerState {
    fn from(s: WorkerStateAtomic) -> WorkerState {
        match s.0 {
            0 => WorkerState::Spawned,
            1 => WorkerState::Connecting,
            2 => WorkerState::Registered,
            3 => WorkerState::LoadingModel,
            4 => WorkerState::ProfilingMemory,
            5 => WorkerState::AllocatingRuntime,
            6 => WorkerState::Warmup,
            7 => WorkerState::Ready,
            8 => WorkerState::Running,
            9 => WorkerState::Draining,
            10 => WorkerState::Error,
            _ => WorkerState::Stopped,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Channel handles
// ─────────────────────────────────────────────────────────────────────────────

/// Sub-scheduler-side handles. Held by `SubScheduler` (or `DiffusionServer`).
pub struct ControlPumpHandles {
    /// Receive scheduler-originated runtime messages (GrantBlocks, Cancel, …).
    pub down_rx: Receiver<SchedulerControlMessage>,
    /// Send worker-originated runtime events (NeedBlocks, StepError, …).
    pub up_tx: SyncSender<WorkerControlMessage>,
    /// Atomics the pump reads when emitting heartbeats. Sub-scheduler updates
    /// these at known sync points.
    pub live: WorkerLiveState,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pump
// ─────────────────────────────────────────────────────────────────────────────

const DOWN_CHANNEL_CAPACITY: usize = 1024;
const UP_CHANNEL_CAPACITY: usize = 1024;

/// Owns the DEALER socket. Runs in a dedicated std thread spawned by
/// [`ControlPump::spawn`].
pub struct ControlPump {
    socket: zmq::Socket,
    worker_id: String,
    heartbeat_interval: Duration,
    down_tx: SyncSender<SchedulerControlMessage>,
    up_rx: Receiver<WorkerControlMessage>,
    live: WorkerLiveState,
}

impl ControlPump {
    /// Construct a `ControlPump` from a bootstrapped client. Returns the pump
    /// and the handles the sub-scheduler / diffusion server will use to talk
    /// to it.
    pub fn from_bootstrapped_client(
        client: WorkerControlClient,
        heartbeat_interval: Duration,
    ) -> (Self, ControlPumpHandles) {
        let (_ctx, socket, worker_id) = client.into_parts();
        let (down_tx, down_rx) = std::sync::mpsc::sync_channel(DOWN_CHANNEL_CAPACITY);
        let (up_tx, up_rx) = std::sync::mpsc::sync_channel(UP_CHANNEL_CAPACITY);
        let live = WorkerLiveState::new();
        let pump = Self {
            socket,
            worker_id,
            heartbeat_interval,
            down_tx,
            up_rx,
            live: live.clone(),
        };
        let handles = ControlPumpHandles {
            down_rx,
            up_tx,
            live,
        };
        // `_ctx` is dropped here; the socket holds its own ref so the context
        // stays alive for the socket's lifetime via ZMQ's internal refcount.
        drop(_ctx);
        (pump, handles)
    }

    /// Spawn the pump thread. Returns its `JoinHandle` so callers can wait on
    /// shutdown if they wish.
    pub fn spawn(self) -> std::thread::JoinHandle<()> {
        std::thread::Builder::new()
            .name("worker-control-pump".into())
            .spawn(move || {
                if let Err(e) = self.run() {
                    tracing::error!("worker control pump exited: {:?}", e);
                }
            })
            .expect("spawn worker-control-pump")
    }

    fn run(mut self) -> Result<()> {
        let mut last_hb = Instant::now() - self.heartbeat_interval;
        loop {
            // Compute remaining time until the next heartbeat tick. This
            // doubles as the poll timeout so the pump wakes up promptly to
            // flush outbound messages too.
            let elapsed = last_hb.elapsed();
            let timeout = if elapsed >= self.heartbeat_interval {
                Duration::from_millis(0)
            } else {
                self.heartbeat_interval - elapsed
            };
            let timeout_ms: i64 = timeout.as_millis().min(i64::MAX as u128) as i64;
            let mut items = [self.socket.as_poll_item(zmq::POLLIN)];
            if let Err(e) = zmq::poll(&mut items, timeout_ms) {
                tracing::error!("control pump poll: {:?}", e);
                continue;
            }

            // 1. Drain inbound from scheduler.
            loop {
                match self.socket.recv_bytes(zmq::DONTWAIT) {
                    Ok(bytes) => {
                        match rmp_serde::from_slice::<ControlEnvelope<SchedulerControlMessage>>(&bytes) {
                            Ok(env) => self.dispatch_inbound(env),
                            Err(e) => tracing::error!("control envelope decode: {}", e),
                        }
                    }
                    Err(zmq::Error::EAGAIN) => break,
                    Err(e) => {
                        tracing::error!("control pump recv: {:?}", e);
                        break;
                    }
                }
            }

            // 2. Drain outbound from sub-scheduler.
            while let Ok(msg) = self.up_rx.try_recv() {
                if let Err(e) = self.send(RequestId::NONE, msg) {
                    tracing::error!("control pump up-flush: {:?}", e);
                }
            }

            // 3. Heartbeat tick.
            if last_hb.elapsed() >= self.heartbeat_interval {
                let (state, active) = self.live.snapshot();
                let hb = WorkerHeartbeat {
                    worker_id: self.worker_id.clone(),
                    state,
                    active_requests: active,
                };
                if let Err(e) = self.send(RequestId::NONE, WorkerControlMessage::Heartbeat(hb)) {
                    tracing::error!("control pump heartbeat: {:?}", e);
                }
                last_hb = Instant::now();
            }
        }
    }

    fn dispatch_inbound(&mut self, env: ControlEnvelope<SchedulerControlMessage>) {
        // Ping: answer directly without touching sub-scheduler.
        if matches!(env.payload, SchedulerControlMessage::Ping) {
            if let Err(e) = self.send(env.request_id, WorkerControlMessage::Pong) {
                tracing::error!("control pump pong: {:?}", e);
            }
            return;
        }
        // Shutdown: forward then exit (sub-scheduler observes via its own
        // shutdown path; here we just make sure the message reaches it).
        if matches!(env.payload, SchedulerControlMessage::Shutdown) {
            tracing::info!("control pump received Shutdown");
            let _ = self.down_tx.try_send(env.payload);
            return;
        }
        // Everything else gets handed to the sub-scheduler via the bounded
        // channel. If the channel is full we drop the payload and log; the
        // scheduler will time out and surface the failure on its side.
        match self.down_tx.try_send(env.payload) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                tracing::warn!("control pump down channel full; dropping message");
            }
            Err(TrySendError::Disconnected(_)) => {
                tracing::error!("control pump down channel disconnected; pump exiting");
            }
        }
    }

    fn send(&self, request_id: RequestId, payload: WorkerControlMessage) -> Result<()> {
        let env = ControlEnvelope { request_id, payload };
        let bytes = rmp_serde::to_vec(&env)?;
        self.socket.send(&bytes, 0)?;
        Ok(())
    }
}
