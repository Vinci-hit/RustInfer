//! Control-plane RPC envelope.
//!
//! Every byte that crosses the scheduler↔worker control plane is wrapped in
//! [`ControlEnvelope`]. The envelope carries a [`RequestId`] used to correlate
//! RPC requests with their replies. `RequestId(0)` (== [`RequestId::NONE`]) is
//! reserved for spontaneous, uncorrelated events such as `Heartbeat`,
//! `StepError`, and bootstrap progress messages.
//!
//! ## RequestId allocation
//!
//! Today only the scheduler initiates RPCs (e.g. `Ping`, `Drain`,
//! `UnloadModel`). The scheduler holds an `AtomicU64` starting at 1 and hands
//! out monotonic ids. The worker echoes the originating id back on its reply.
//! A future variant that lets workers initiate RPCs must partition the id
//! space so the two sides do not collide; for now the worker never allocates.

use serde::{Deserialize, Serialize};

/// Monotonically increasing per-control-plane RPC id.
///
/// The all-zero value [`RequestId::NONE`] marks an uncorrelated message and is
/// never produced by an allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub u64);

impl RequestId {
    /// Sentinel value meaning "not associated with any pending RPC".
    pub const NONE: RequestId = RequestId(0);

    /// `true` when the id corresponds to a pending RPC.
    #[inline]
    pub fn is_correlated(self) -> bool {
        self.0 != 0
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 0 {
            f.write_str("rpc:NONE")
        } else {
            write!(f, "rpc:{}", self.0)
        }
    }
}

/// Wire-level wrapper carried over the control plane in both directions.
///
/// `T` is one of [`SchedulerControlMessage`] or [`WorkerControlMessage`].
///
/// [`SchedulerControlMessage`]: crate::scheduler_to_worker_control::SchedulerControlMessage
/// [`WorkerControlMessage`]: crate::worker_to_scheduler_control::WorkerControlMessage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlEnvelope<T> {
    pub request_id: RequestId,
    pub payload: T,
}

impl<T> ControlEnvelope<T> {
    /// Build an envelope for a fire-and-forget message.
    #[inline]
    pub fn oneway(payload: T) -> Self {
        Self {
            request_id: RequestId::NONE,
            payload,
        }
    }

    /// Build an envelope for an RPC request or its matching reply.
    ///
    /// Panics in debug builds if `request_id == RequestId::NONE`.
    #[inline]
    pub fn rpc(request_id: RequestId, payload: T) -> Self {
        debug_assert!(
            request_id.is_correlated(),
            "ControlEnvelope::rpc requires a correlated RequestId"
        );
        Self {
            request_id,
            payload,
        }
    }
}
