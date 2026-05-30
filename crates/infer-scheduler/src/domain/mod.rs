//! Domain layer — pure business types with no IO dependencies.
//!
//! This is the inner ring of the hexagonal architecture: it owns the
//! invariants (typestate, value objects, NewTypes) and exposes them to
//! the application layer through traits.
//!
//! Sub-modules:
//! - [`ids`]                  — NewType dictionary
//! - [`inference_session`]    — typestate `InferenceSession<S>` + repository
//! - [`kv_cache_pool`]        — paged KV pool + `KvLease` RAII guard
//! - [`worker_node`]          — typestate `WorkerNode<S>`
//! - [`policy`]               — scheduling policy domain service

pub mod ids;
pub mod inference_session;
pub mod kv_budget;
pub mod kv_cache_pool;
pub mod policy;
pub mod preemption;
pub mod worker_node;

// Re-exports for the most-used types at the domain root, so callers
// don't need to dive into sub-modules for the canonical surface.
pub use ids::{
    BlockCount, BlockSize, InferenceRequestId, LastSeenAt, ModelInstanceId, SeqCount, SequenceId,
    TokenCount, WorkerNodeId,
};
pub use inference_session::lifecycle::InferenceSession;
pub use kv_budget::{KvBudget, KvBudgetFull};
pub use kv_cache_pool::{KvCachePool, KvLease, NoopKvPool, PagedKvPool};
pub use preemption::{PreemptionConfig, RunningSnap, select_preempt_ids};
pub use worker_node::{Capacity, Lost, NodeState, Ready, WorkerNode};
