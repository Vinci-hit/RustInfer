//! Domain layer — pure business types with no IO dependencies.
//!
//! This is the inner ring of the hexagonal architecture: it owns the
//! invariants (typestate, value objects, NewType identifiers) and exposes
//! them to the application layer through traits.
//!
//! Sub-modules:
//! - [`ids`]                  — NewType dictionary
//! - [`inference_session`]    — typestate `InferenceSession<S>` + repository
//! - [`kv_budget`]            — `KvBudget` capacity gate over
//!   worker-reported global KV slots
//! - [`policy`]               — scheduling policy domain service

pub mod ids;
pub mod inference_session;
pub mod kv_budget;
pub mod policy;
pub mod prefix;

// Re-exports for the most-used types at the domain root, so callers
// don't need to dive into sub-modules for the canonical surface.
pub use ids::{
    BlockCount, BlockSize, InferenceRequestId, LastSeenAt, ModelInstanceId, SeqCount, SequenceId,
    TokenCount, WorkerId,
};
pub use inference_session::lifecycle::InferenceSession;
pub use kv_budget::{KvBudget, KvBudgetFull};
pub use prefix::PrefixMatch;
