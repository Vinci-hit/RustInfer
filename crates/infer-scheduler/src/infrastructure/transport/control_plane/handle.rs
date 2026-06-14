//! Public-facing control-plane handle types.
//!
//! `ControlPlaneCmdTx` is the cheap-clone send side used by the scheduler
//! engine and helper tasks. `ControlPlaneEventRx` is the single-consumer
//! receive side, owned by the engine and never cloned.

use std::sync::Arc;
use std::time::{Duration, Instant};

use infer_protocol::worker_to_scheduler_control::WorkerControlMessage;
use infer_protocol::{
    ControlEnvelope,
    scheduler_to_worker_control::SchedulerControlMessage,
    worker_to_scheduler_control::{AllocFailed, WorkerHeartbeat, WorkerStepError},
};
use tokio::sync::{mpsc, oneshot};

use super::pending_calls::PendingCalls;

// ─────────────────────────────────────────────────────────────────────────────
//  WorkerId
// ─────────────────────────────────────────────────────────────────────────────

/// Stable scheduler-side identifier for a worker.
///
/// Wraps the ZMQ ROUTER identity bytes assigned at connect time. We never
/// expose the raw bytes through the engine API — engine code references
/// workers exclusively by `WorkerId`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct WorkerId(Arc<[u8]>);

impl WorkerId {
    /// Build a `WorkerId` from raw ZMQ identity bytes (router-internal use).
    pub(crate) fn from_identity(bytes: &[u8]) -> Self {
        Self(Arc::from(bytes))
    }

    /// Raw identity bytes for emitting through ZMQ.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl std::fmt::Debug for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Most ZMQ identities are short ASCII-ish; format as hex for clarity.
        write!(f, "WorkerId(")?;
        for b in self.0.iter() {
            write!(f, "{:02x}", b)?;
        }
        write!(f, ")")
    }
}

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in self.0.iter() {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  ControlEvent
// ─────────────────────────────────────────────────────────────────────────────

/// Events surfaced from the control plane to the scheduler engine.
///
/// Bootstrap-phase progress is consumed inside `ControlPlane::bootstrap`
/// and never crosses into this enum.
#[derive(Debug)]
pub enum ControlEvent {
    /// Worker heartbeat received. The router thread also updates the
    /// liveness clock; the engine only sees this for observability.
    Heartbeat {
        worker: WorkerId,
        hb: WorkerHeartbeat,
    },

    /// Per-step execution error reported by the worker.
    StepError {
        worker: WorkerId,
        err: WorkerStepError,
    },

    /// Liveness watchdog tripped. Worker is considered gone; the registry
    /// has already been updated to refuse subsequent unicasts.
    WorkerLost { worker: WorkerId, last_seen_ms: u64 },

    /// Out-of-band fatal error reported by the worker itself.
    WorkerError {
        worker: WorkerId,
        message: String,
        fatal: bool,
    },

    /// Worker-driven KV pressure-relief request. `req.round` selects
    /// the relief level (0 = LRU evict, 1 = victim preempt). Sent only
    /// when `kv_allocator.alloc_indices()` actually fails on the
    /// worker — never on a periodic schedule.
    AllocFailed { worker: WorkerId, req: AllocFailed },
}

// ─────────────────────────────────────────────────────────────────────────────
//  Errors
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ControlError {
    #[error("control plane shut down")]
    Shutdown,

    #[error("rpc timed out after {0:?}")]
    Timeout(Duration),

    #[error("unknown worker {0}")]
    UnknownWorker(WorkerId),

    #[error("encode: {0}")]
    Encode(String),

    #[error("decode: {0}")]
    Decode(String),

    #[error("router thread died: {0}")]
    Router(String),

    #[error("unexpected reply type: {0}")]
    UnexpectedReply(String),
}

pub type ControlResult<T> = std::result::Result<T, ControlError>;

// ─────────────────────────────────────────────────────────────────────────────
//  Internal command surface (router thread)
// ─────────────────────────────────────────────────────────────────────────────

/// Internal commands sent from `ControlPlaneCmdTx` to the router thread.
///
/// The router thread owns the ZMQ ROUTER socket and is the only place that
/// can call `socket.send_multipart`.
pub(crate) enum RouterCommand {
    /// Fire-and-forget unicast.
    SendTo {
        worker: WorkerId,
        env: ControlEnvelope<SchedulerControlMessage>,
    },

    /// Fire-and-forget broadcast to every registered worker.
    Broadcast {
        env: ControlEnvelope<SchedulerControlMessage>,
    },

    /// RPC: send envelope to one worker, reply hooked up via `PendingCalls`.
    CallOne {
        worker: WorkerId,
        env: ControlEnvelope<SchedulerControlMessage>,
        /// Absolute deadline. Currently unused: timeout enforcement
        /// happens centrally via [`PendingCalls::sweep_expired`]
        /// rather than per-command in the router. Kept on the
        /// command for future per-call deadline overrides.
        #[allow(dead_code)]
        deadline: Instant,
    },

    /// RPC: broadcast to every registered worker, fan-in collected via
    /// `PendingCalls`.
    CallAll {
        env: ControlEnvelope<SchedulerControlMessage>,
        /// See `CallOne::deadline` — same rationale.
        #[allow(dead_code)]
        deadline: Instant,
    },

    /// Cooperative shutdown — drains pending RPCs with `ControlError::Shutdown`
    /// and exits the router loop.
    Shutdown,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public send/receive handles
// ─────────────────────────────────────────────────────────────────────────────

/// Cheap-clone send side of the control plane.
///
/// Holds the channel to the router thread plus a shared handle to
/// `PendingCalls` so RPCs can register their oneshot before the message
/// hits the wire.
#[derive(Clone)]
pub struct ControlPlaneCmdTx {
    pub(crate) tx: mpsc::UnboundedSender<RouterCommand>,
    pub(crate) pending: Arc<PendingCalls>,
    pub(crate) default_rpc_deadline: Duration,
}

impl ControlPlaneCmdTx {
    /// Fire-and-forget message to one worker.
    pub fn send_to(&self, worker: &WorkerId, msg: SchedulerControlMessage) -> ControlResult<()> {
        let env = ControlEnvelope::oneway(msg);
        self.tx
            .send(RouterCommand::SendTo {
                worker: worker.clone(),
                env,
            })
            .map_err(|_| ControlError::Shutdown)
    }

    /// Fire-and-forget broadcast to every registered worker.
    pub fn broadcast(&self, msg: SchedulerControlMessage) -> ControlResult<()> {
        let env = ControlEnvelope::oneway(msg);
        self.tx
            .send(RouterCommand::Broadcast { env })
            .map_err(|_| ControlError::Shutdown)
    }

    /// RPC: unicast a message and await the typed reply.
    pub async fn call_one<R>(
        &self,
        worker: &WorkerId,
        msg: SchedulerControlMessage,
    ) -> ControlResult<R>
    where
        R: TryFromControlReply,
    {
        let deadline = Instant::now() + self.default_rpc_deadline;
        let (rid, rx) = self.pending.register_one(deadline);
        let env = ControlEnvelope::rpc(rid, msg);
        self.tx
            .send(RouterCommand::CallOne {
                worker: worker.clone(),
                env,
                deadline,
            })
            .map_err(|_| ControlError::Shutdown)?;
        let reply = rx.await.map_err(|_| ControlError::Shutdown)??;
        R::try_from_reply(reply)
    }

    /// RPC: broadcast and collect every reply (or per-worker timeout).
    pub async fn call_all<R>(
        &self,
        msg: SchedulerControlMessage,
    ) -> ControlResult<Vec<(WorkerId, ControlResult<R>)>>
    where
        R: TryFromControlReply,
    {
        let deadline = Instant::now() + self.default_rpc_deadline;
        let (rid, rx) = self.pending.register_all(deadline);
        let env = ControlEnvelope::rpc(rid, msg);
        self.tx
            .send(RouterCommand::CallAll { env, deadline })
            .map_err(|_| ControlError::Shutdown)?;
        let replies = rx.await.map_err(|_| ControlError::Shutdown)?;
        Ok(replies
            .into_iter()
            .map(|(w, r)| (w, r.and_then(R::try_from_reply)))
            .collect())
    }
}

/// Single-consumer receive side. Held by the scheduler engine.
pub struct ControlPlaneEventRx {
    pub(crate) rx: mpsc::UnboundedReceiver<ControlEvent>,
}

impl ControlPlaneEventRx {
    /// Await the next control event. Returns `None` when the plane shut down.
    pub async fn recv(&mut self) -> Option<ControlEvent> {
        self.rx.recv().await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  RPC reply projection
// ─────────────────────────────────────────────────────────────────────────────

/// Project a wire-level [`WorkerControlMessage`] into the typed reply expected
/// by a particular RPC. Implementations live next to the call sites that need
/// them (e.g. `Pong` for `Ping`, `DrainAck` for `Drain`).
pub trait TryFromControlReply: Sized {
    fn try_from_reply(msg: WorkerControlMessage) -> ControlResult<Self>;
}

// Built-in projections for the most common reply shapes.

impl TryFromControlReply for () {
    fn try_from_reply(msg: WorkerControlMessage) -> ControlResult<Self> {
        match msg {
            WorkerControlMessage::Pong => Ok(()),
            other => Err(ControlError::UnexpectedReply(format!("{:?}", other))),
        }
    }
}

impl TryFromControlReply for infer_protocol::worker_to_scheduler_control::DrainAck {
    fn try_from_reply(msg: WorkerControlMessage) -> ControlResult<Self> {
        match msg {
            WorkerControlMessage::DrainAck(a) => Ok(a),
            other => Err(ControlError::UnexpectedReply(format!("{:?}", other))),
        }
    }
}

impl TryFromControlReply for infer_protocol::worker_to_scheduler_control::CancelAck {
    fn try_from_reply(msg: WorkerControlMessage) -> ControlResult<Self> {
        match msg {
            WorkerControlMessage::CancelAck(a) => Ok(a),
            other => Err(ControlError::UnexpectedReply(format!("{:?}", other))),
        }
    }
}

impl TryFromControlReply for infer_protocol::worker_to_scheduler_control::UnloadAck {
    fn try_from_reply(msg: WorkerControlMessage) -> ControlResult<Self> {
        match msg {
            WorkerControlMessage::UnloadAck(a) => Ok(a),
            other => Err(ControlError::UnexpectedReply(format!("{:?}", other))),
        }
    }
}

// Suppress `unused_import` until call sites exist:
#[allow(dead_code)]
fn _hint(_: oneshot::Sender<()>) {}
