//! Scheduler-side KV bookkeeping.
//!
//! Physical block allocation lives entirely in the worker; the scheduler's
//! only KV-related state is a token-prefix index over the global KV slots
//! the worker has reported via `StepOutput.assigned_indices`. This module
//! holds that index and the trait helpers that planning consumes.
//!
//! - [`RadixTree`] — token-granularity prefix tree mapping prompt-token
//!   sequences to global KV indices. Reference-counted by live sequences
//!   (`Node.owners`); a node enters the LRU only when
//!   `owners.is_empty() && children.is_empty()`. Eviction yields the
//!   indices the scheduler returns to the worker via `FreeKvIndices`.
//! - [`traits`] — small helper types (e.g. [`PrefixMatch`]) used between
//!   planning and the radix tree.

pub mod radix_tree;
pub mod traits;

pub use radix_tree::{GlobalIndex as TokenSlotIndex, PrefixHit, RadixTree, SeqId as RadixSeqId};
pub use traits::PrefixMatch;
