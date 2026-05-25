//! Sealed state markers for `WorkerNode<S>`.
//!
//! Following the same pattern as `request::lifecycle::SessionState`, only
//! the types defined in this module may implement `NodeState`. This makes
//! the typestate machine total: every reachable variant is enumerated
//! at compile time.

mod sealed {
    pub trait Sealed {}
}

/// Marker trait for valid `WorkerNode` states.
///
/// Sealed: external crates and other modules cannot add new states.
/// `'static + Send + Sync` so a `WorkerNode<S>` is freely passable
/// between async tasks.
pub trait NodeState: sealed::Sealed + 'static + Send + Sync {}

/// Worker has completed handshake and is accepting batches.
#[derive(Debug, Clone, Copy)]
pub struct Ready;

/// Worker has been declared lost (heartbeat timeout, transport error,
/// explicit terminal control message).
///
/// Carries the human-readable reason so the engine can surface it in
/// `WorkerError` propagation and tracing.
#[derive(Debug, Clone)]
pub struct Lost {
    pub reason: String,
}

impl sealed::Sealed for Ready {}
impl sealed::Sealed for Lost {}
impl NodeState for Ready {}
impl NodeState for Lost {}
