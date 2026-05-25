//! `KvLease` — RAII drop guard for paged KV blocks.
//!
//! ## Why
//!
//! Phase 2 of the refactor (cosmic-cascade-tesla v5.1, P.4 in the
//! "Four Guardians" appendix) treats KV physical block leakage as a
//! production-class hazard: any panic, task cancellation, or early
//! return between `allocate` and `free` would otherwise permanently
//! lose blocks. A single recurring panic could deadlock the engine
//! by exhausting the pool.
//!
//! `KvLease` solves this with classic RAII: the lease *owns* the
//! blocks and returns them via `Drop`. Even an unwinding panic walks
//! the drop chain, so blocks always reach the pool's return sink.
//!
//! ## Sink + Weak design
//!
//! The lease holds a `Weak<Mutex<Vec<PhysicalBlockId>>>` rather than
//! an `Arc`. Two reasons:
//!
//! 1. **Pool shutdown safety.** When the engine tears down, the
//!    `Arc<Mutex<...>>` inside `PagedKvPool` is dropped. Any lingering
//!    `KvLease` whose Drop fires later finds `upgrade()` returns
//!    `None` and silently discards the blocks instead of panicking.
//!    This is the correct behavior at process exit.
//!
//! 2. **No reference cycle.** The pool owns the strong `Arc` to the
//!    sink; leases hold `Weak`. Pool drop is deterministic.
//!
//! ## Explicit-bypass path: `into_blocks_no_free`
//!
//! Sometimes blocks must *not* be returned to the free list — the
//! prefix cache wants them registered as cached entries instead.
//! `into_blocks_no_free` extracts the inner `Vec` and `mem::forget`s
//! the lease, so Drop never fires.

use std::sync::{Arc, Mutex, Weak};

use crate::infrastructure::kv_cache::traits::PhysicalBlockId;
use crate::domain::ids::BlockCount;

/// Shared, mutex-guarded buffer where lease drops park returned blocks.
///
/// `PagedKvPool` owns the strong `Arc`; leases keep a `Weak`. The
/// inner mutex is only contended by lease drops (push) and the
/// pool's `flush_pending_returns` (drain) — both rare relative to
/// the inference hot path, so `std::sync::Mutex` is cheap enough
/// to avoid pulling in `parking_lot`.
pub(crate) type ReturnSink = Arc<Mutex<Vec<PhysicalBlockId>>>;

/// Weak handle held by every outstanding `KvLease`.
pub(crate) type ReturnSinkWeak = Weak<Mutex<Vec<PhysicalBlockId>>>;

/// RAII guard for a contiguous run of paged KV blocks.
///
/// Drop deposits the held blocks into the pool's return sink; the
/// pool drains the sink on its next `flush_pending_returns` call.
///
/// Panic-safe by construction: any unwind between allocate and the
/// natural Drop site still returns the blocks.
#[must_use = "dropping a KvLease without using it leaks an iteration of latency"]
pub struct KvLease {
    blocks: Vec<PhysicalBlockId>,
    sink: ReturnSinkWeak,
}

impl KvLease {
    /// Construct a new lease. Crate-private — leases come from
    /// `KvCachePool::allocate*` only.
    pub(crate) fn new(blocks: Vec<PhysicalBlockId>, sink: &ReturnSink) -> Self {
        Self {
            blocks,
            sink: Arc::downgrade(sink),
        }
    }

    /// Detached lease with no return path. Used only for `NoopKvPool`
    /// (diffusion mode) where there is no physical block pool to
    /// return to. Drop is a no-op because the inner Vec is empty.
    pub fn empty() -> Self {
        Self {
            blocks: Vec::new(),
            sink: Weak::new(),
        }
    }

    /// Build a detached lease holding pre-known blocks. **For tests only**:
    /// no return sink, so drop simply discards the blocks. Production code
    /// must obtain leases through [`KvCachePool::allocate`] / etc.
    #[cfg(test)]
    pub fn test_with_blocks(blocks: Vec<PhysicalBlockId>) -> Self {
        Self {
            blocks,
            sink: Weak::new(),
        }
    }

    pub fn blocks(&self) -> &[PhysicalBlockId] {
        &self.blocks
    }

    pub fn len(&self) -> BlockCount {
        BlockCount::new(self.blocks.len())
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Append more blocks (decode-time block-table extension).
    pub fn extend(&mut self, more: Vec<PhysicalBlockId>) {
        self.blocks.extend(more);
    }

    /// Take the inner blocks **without triggering Drop's return path**.
    ///
    /// Use this when blocks are being handed off to the prefix cache
    /// (which adopts them as cached entries) or to a different
    /// allocator path. The caller becomes responsible for ensuring
    /// the blocks are accounted for — the lease's safety net is
    /// disabled.
    pub fn into_blocks_no_free(mut self) -> Vec<PhysicalBlockId> {
        let blocks = std::mem::take(&mut self.blocks);
        // Skip Drop: blocks are now owned by the caller's pathway.
        std::mem::forget(self);
        blocks
    }
}

impl std::fmt::Debug for KvLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvLease")
            .field("len", &self.blocks.len())
            .field("sink_alive", &(self.sink.strong_count() > 0))
            .finish()
    }
}

impl Drop for KvLease {
    fn drop(&mut self) {
        if self.blocks.is_empty() {
            return;
        }
        // upgrade() == None means the pool has already been dropped
        // (process is tearing down). Discarding the blocks is the
        // only sane action.
        let Some(sink) = self.sink.upgrade() else {
            return;
        };
        // .lock() can fail only on poison — we still want to capture
        // the blocks rather than leak them, so we recover via
        // `into_inner` of the poison error.
        let mut guard = match sink.lock() {
            Ok(g) => g,
            Err(poison) => poison.into_inner(),
        };
        guard.extend(self.blocks.drain(..));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_sink() -> ReturnSink {
        Arc::new(Mutex::new(Vec::new()))
    }

    #[test]
    fn drop_returns_blocks_to_sink() {
        let sink = fresh_sink();
        {
            let _lease = KvLease::new(
                vec![PhysicalBlockId(1), PhysicalBlockId(2), PhysicalBlockId(3)],
                &sink,
            );
            // lease still alive: sink is empty.
            assert!(sink.lock().unwrap().is_empty());
        }
        let returned = sink.lock().unwrap();
        assert_eq!(
            returned.iter().map(|b| b.0).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn into_blocks_no_free_skips_drop() {
        let sink = fresh_sink();
        let lease = KvLease::new(vec![PhysicalBlockId(7)], &sink);
        let blocks = lease.into_blocks_no_free();
        assert_eq!(blocks, vec![PhysicalBlockId(7)]);
        // Sink must NOT have received the blocks: caller adopted them.
        assert!(sink.lock().unwrap().is_empty());
    }

    /// The whole point of P.4: blocks must come back even when the
    /// owning task panics partway through use.
    #[test]
    fn drop_returns_blocks_even_under_panic() {
        let sink = fresh_sink();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lease = KvLease::new(
                vec![PhysicalBlockId(11), PhysicalBlockId(12)],
                &sink,
            );
            panic!("simulated worker error");
        }));
        assert!(result.is_err());
        let returned = sink.lock().unwrap();
        assert_eq!(
            returned.iter().map(|b| b.0).collect::<Vec<_>>(),
            vec![11, 12]
        );
    }

    #[test]
    fn drop_after_pool_gone_does_not_panic() {
        // Simulate engine shutdown: drop the sink before the lease.
        let sink = fresh_sink();
        let lease = KvLease::new(vec![PhysicalBlockId(99)], &sink);
        drop(sink);
        // Dropping the lease now must NOT panic — Weak::upgrade returns None.
        drop(lease);
    }

    #[test]
    fn empty_lease_drop_is_noop() {
        let lease = KvLease::empty();
        assert!(lease.is_empty());
        drop(lease);
    }

    #[test]
    fn extend_grows_block_table() {
        let sink = fresh_sink();
        let mut lease = KvLease::new(vec![PhysicalBlockId(1)], &sink);
        lease.extend(vec![PhysicalBlockId(2), PhysicalBlockId(3)]);
        assert_eq!(lease.len().raw(), 3);
        assert_eq!(lease.blocks()[2].0, 3);
    }
}
