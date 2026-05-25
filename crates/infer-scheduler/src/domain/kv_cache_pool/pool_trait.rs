//! `KvCachePool` trait — domain-layer paged KV cache pool.
//!
//! This is the Step 11 introduction of the AR #2 abstraction.
//! It does **not** yet replace `cache::KvManager`/`cache::PagedKvManager`
//! at the engine call sites — that swap happens during the
//! `application/` layer split (Step 13+). Establishing the trait
//! here lets later Steps wire `Session<S>` to `KvLease` without
//! touching the legacy cache module.
//!
//! ## Surface
//!
//! - `allocate(tokens)` / `allocate_with_prefix(tokens)` return
//!   `KvLease` (RAII), not raw `Vec<PhysicalBlockId>`.
//! - `allocate_decode_blocks(n)` returns raw blocks for the
//!   block-table-extension path: those are appended onto an
//!   existing lease via `KvLease::extend`.
//! - `free_finished` consumes a `KvLease` and routes its blocks
//!   through the prefix cache.
//! - `flush_pending_returns()` drains the RAII return sink — call
//!   once per scheduling iteration, before allocating new leases.
//! - `block_size()` / `total_blocks()` / `available_blocks()`
//!   expose pool capacity for `PlanningSystem`.
//!
//! ## Why `&mut self` everywhere
//!
//! Single-engine ownership: the scheduler runs one event loop with
//! exclusive mutable access to the pool. No `Mutex` is needed on
//! the *pool*; the only mutex in the system is the small
//! `ReturnSink` that lease drops park into, and it's only contended
//! by drop-vs-flush, not the inference hot path.

use crate::infrastructure::kv_cache::traits::{PhysicalBlockId, PrefixMatch};
use crate::domain::ids::{BlockCount, BlockSize, TokenCount};
use crate::error::Result;

use super::lease::KvLease;

/// Domain-layer paged KV cache pool.
///
/// `Send` (not `Sync`) — only the engine task touches the pool.
pub trait KvCachePool: Send {
    /// Allocate enough blocks to host `tokens` tokens of KV state.
    ///
    /// The returned lease holds the blocks; if it is dropped without
    /// going through `free_finished`, the blocks return to the pool
    /// on the next `flush_pending_returns()`.
    fn allocate(&mut self, tokens: TokenCount) -> Result<KvLease>;

    /// Allocate with prefix-cache reuse. The `PrefixMatch` describes
    /// the cached portion already covered by the returned lease's
    /// initial blocks; the engine uses that to skip recomputation.
    fn allocate_with_prefix(&mut self, tokens: &[i32]) -> Result<(KvLease, PrefixMatch)>;

    /// Allocate `n` raw blocks for an in-flight decode that needs
    /// to extend its block table. Caller appends via `KvLease::extend`.
    fn allocate_decode_blocks(&mut self, n: BlockCount) -> Result<Vec<PhysicalBlockId>>;

    /// Release a finished sequence. Routes `prompt_tokens` and the
    /// lease's blocks through the prefix cache (if enabled), so a
    /// future request with the same prefix can reuse them.
    ///
    /// Implementations MUST consume the lease's blocks via
    /// `into_blocks_no_free` so the RAII path doesn't double-handle.
    fn free_finished(&mut self, prompt_tokens: &[i32], lease: KvLease);

    /// Match the longest cached prefix of `tokens`.
    ///
    /// Takes `&mut self` because `RadixTreeCache` updates LRU clocks
    /// on every match.
    fn match_prefix(&mut self, tokens: &[i32]) -> PrefixMatch;

    /// Drain any blocks that were parked by `KvLease::Drop`. Should
    /// be called at the head of every scheduling iteration so the
    /// freed-from-drop blocks become re-allocatable.
    fn flush_pending_returns(&mut self);

    fn block_size(&self) -> BlockSize;
    fn total_blocks(&self) -> BlockCount;
    fn available_blocks(&self) -> BlockCount;

    /// Mode name for logging / metrics.
    fn mode_name(&self) -> &'static str;
}
