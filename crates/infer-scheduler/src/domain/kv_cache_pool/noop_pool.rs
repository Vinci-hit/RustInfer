//! `NoopKvPool` — diffusion-mode KV pool that allocates nothing.
//!
//! Diffusion models don't use KV cache on the scheduler side; the
//! Worker pipeline manages whatever caches the model itself needs.
//! `NoopKvPool` exists so the scheduler engine can be parameterized
//! over `Box<dyn KvCachePool>` regardless of mode without sprinkling
//! conditionals.
//!
//! Every allocation returns an [`KvLease::empty`] — zero blocks, no
//! return sink. Drop is a no-op. `match_prefix` always misses.

use crate::infrastructure::kv_cache::traits::{PhysicalBlockId, PrefixMatch};
use crate::domain::ids::{BlockCount, BlockSize, TokenCount};
use crate::error::Result;

use super::lease::KvLease;
use super::pool_trait::KvCachePool;

/// Diffusion-mode no-op KV pool.
#[derive(Debug, Default)]
pub struct NoopKvPool;

impl NoopKvPool {
    pub fn new() -> Self {
        Self
    }
}

impl KvCachePool for NoopKvPool {
    fn allocate(&mut self, _tokens: TokenCount) -> Result<KvLease> {
        Ok(KvLease::empty())
    }

    fn allocate_with_prefix(&mut self, _tokens: &[i32]) -> Result<(KvLease, PrefixMatch)> {
        Ok((KvLease::empty(), PrefixMatch::none()))
    }

    fn allocate_decode_blocks(&mut self, _n: BlockCount) -> Result<Vec<PhysicalBlockId>> {
        // Diffusion mode never extends KV — the planner shouldn't ask.
        // Return an empty vec rather than erroring so the trait stays
        // total; callers that actually depend on blocks should never
        // be wired up against NoopKvPool in the first place.
        Ok(Vec::new())
    }

    fn free_finished(&mut self, _prompt_tokens: &[i32], _lease: KvLease) {
        // empty lease, drop is no-op anyway
    }

    fn match_prefix(&mut self, _tokens: &[i32]) -> PrefixMatch {
        PrefixMatch::none()
    }

    fn flush_pending_returns(&mut self) {
        // No sink — nothing to flush.
    }

    fn block_size(&self) -> BlockSize {
        // Diffusion has no real "block"; return 1 to keep arithmetic
        // safe (callers compute `tokens / block_size`).
        BlockSize::new(1)
    }

    fn total_blocks(&self) -> BlockCount {
        BlockCount::new(0)
    }

    fn available_blocks(&self) -> BlockCount {
        BlockCount::new(0)
    }

    fn mode_name(&self) -> &'static str {
        "noop (diffusion)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_yields_empty_lease() {
        let mut pool = NoopKvPool::new();
        let lease = pool.allocate(TokenCount::new(100)).unwrap();
        assert!(lease.is_empty());
        // Drop must not panic / not touch any sink.
        drop(lease);
    }

    #[test]
    fn match_prefix_always_misses() {
        let mut pool = NoopKvPool::new();
        let pm = pool.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(pm.num_cached_tokens, 0);
        assert!(pm.cached_blocks.is_empty());
    }

    #[test]
    fn capacity_methods_return_zero_blocks() {
        let pool = NoopKvPool::new();
        assert_eq!(pool.total_blocks().raw(), 0);
        assert_eq!(pool.available_blocks().raw(), 0);
        assert_eq!(pool.block_size().raw(), 1);
    }
}
