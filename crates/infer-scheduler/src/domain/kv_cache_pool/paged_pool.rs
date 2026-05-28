//! `PagedKvPool` — production paged KV pool.
//!
//! Owns the `PagedBlockAllocator` (free / used / cached block sets)
//! and the `RadixTreeCache` (prefix→blocks map) directly. There is
//! no longer an intervening `KvManager` trait or `KvAllocation`
//! enum: the pool returns `KvLease` (RAII guard) and the rest of the
//! engine treats KV as opaque.

use std::sync::{Arc, Mutex};

use crate::infrastructure::kv_cache::block_allocator::PagedBlockAllocator;
use crate::infrastructure::kv_cache::block_table::BlockTable;
use crate::infrastructure::kv_cache::radix_tree::RadixTreeCache;
use crate::infrastructure::kv_cache::traits::{BlockAllocator, CacheStrategy, PhysicalBlockId, PrefixMatch};
use crate::domain::ids::{BlockCount, BlockSize, TokenCount};
use crate::error::{Result, SchedulerError};

use super::lease::{KvLease, ReturnSink};
use super::pool_trait::KvCachePool;

/// Production paged-KV pool.
pub struct PagedKvPool {
    allocator: PagedBlockAllocator,
    prefix_cache: RadixTreeCache,
    block_size: usize,
    enable_prefix_cache: bool,
    return_sink: ReturnSink,
}

impl PagedKvPool {
    /// Construct directly from the physical pool parameters.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self::with_prefix_cache(num_blocks, block_size, false)
    }

    /// Construct with prefix-cache toggle.
    pub fn with_prefix_cache(num_blocks: usize, block_size: usize, enable_prefix_cache: bool) -> Self {
        Self {
            allocator: PagedBlockAllocator::new(num_blocks, block_size),
            prefix_cache: RadixTreeCache::new(block_size),
            block_size,
            enable_prefix_cache,
            return_sink: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Translate token count → matching block count (always ≥ 1).
    fn blocks_for(&self, tokens: TokenCount) -> usize {
        tokens.raw().div_ceil(self.block_size).max(1)
    }

    /// Allocate an exact number of physical blocks; eviction-aware.
    fn raw_allocate_blocks(&mut self, num_blocks: usize) -> Result<Vec<PhysicalBlockId>> {
        if num_blocks == 0 {
            return Ok(Vec::new());
        }
        if self.allocator.num_free_blocks() < num_blocks && self.enable_prefix_cache {
            let need = num_blocks - self.allocator.num_free_blocks();
            let evicted = self.prefix_cache.evict_entries(need);
            self.allocator.free_cached_blocks(&evicted);
        }
        let available = self.allocator.num_free_blocks();
        self.allocator
            .allocate(num_blocks)
            .ok_or(SchedulerError::CacheExhausted {
                needed: num_blocks,
                available,
            })
    }

    fn raw_match_prefix(&mut self, tokens: &[i32]) -> PrefixMatch {
        if self.enable_prefix_cache {
            self.prefix_cache.match_prefix(tokens)
        } else {
            PrefixMatch::none()
        }
    }

    /// Insert a completed prompt's blocks into the radix-tree prefix
    /// cache so a future request with the same prefix can short-circuit.
    /// Partial tail blocks are released to the free list rather than cached.
    fn insert_completed_prefix(&mut self, tokens: &[i32], blocks: &[PhysicalBlockId]) {
        if !self.enable_prefix_cache || blocks.is_empty() {
            // Prefix cache disabled (or empty allocation): just free.
            self.allocator.free(blocks);
            return;
        }
        let full_blocks = (tokens.len() / self.block_size).min(blocks.len());
        if full_blocks == 0 {
            self.allocator.free(blocks);
            return;
        }
        let table = BlockTable::from_blocks(
            blocks[..full_blocks].to_vec(),
            self.block_size,
            self.block_size,
        );
        self.prefix_cache.insert(tokens, &table);
        self.allocator.release_to_cache(&blocks[..full_blocks]);
        if full_blocks < blocks.len() {
            self.allocator.free(&blocks[full_blocks..]);
        }
    }
}

impl KvCachePool for PagedKvPool {
    fn allocate(&mut self, tokens: TokenCount) -> Result<KvLease> {
        // Reclaim drop-returned blocks first so the new alloc can use them.
        self.flush_pending_returns();
        let blocks = self.raw_allocate_blocks(self.blocks_for(tokens))?;
        Ok(KvLease::new(blocks, &self.return_sink))
    }

    fn allocate_with_prefix(&mut self, tokens: &[i32]) -> Result<(KvLease, PrefixMatch)> {
        self.flush_pending_returns();
        let mut prefix = self.raw_match_prefix(tokens);
        // Allocate prompt blocks + 1 extra block for decode headroom.
        // This ensures the worker has at least 1 block of space (block_size tokens)
        // after prefill before it needs to send NeedBlocks, avoiding a race
        // condition where NeedBlocks arrives at scheduler before the prefill
        // StepOutput transitions the session to Decoding state.
        let total_blocks = (tokens.len().div_ceil(self.block_size) + 1).max(1);
        // Keep at least one prompt block writable for the incoming
        // request: avoids mutating the final cached block in-place
        // when the prompt is a full block-cache hit.
        let max_reusable_blocks = total_blocks.saturating_sub(1);
        if prefix.cached_blocks.len() > max_reusable_blocks {
            prefix.cached_blocks.truncate(max_reusable_blocks);
            prefix.num_cached_tokens = prefix.cached_blocks.len() * self.block_size;
            if prefix.cached_blocks.is_empty() {
                prefix.last_block_hash = None;
            }
        }
        if !prefix.cached_blocks.is_empty() {
            self.allocator.retain_blocks(&prefix.cached_blocks);
        }
        let missing_blocks = total_blocks.saturating_sub(prefix.cached_blocks.len());
        let mut blocks = prefix.cached_blocks.clone();
        blocks.extend(self.raw_allocate_blocks(missing_blocks)?);
        Ok((KvLease::new(blocks, &self.return_sink), prefix))
    }

    fn allocate_decode_blocks(&mut self, n: BlockCount) -> Result<Vec<PhysicalBlockId>> {
        self.flush_pending_returns();
        self.raw_allocate_blocks(n.raw())
    }

    fn free_finished(&mut self, prompt_tokens: &[i32], lease: KvLease) {
        // Take ownership of the blocks without re-entering the Drop sink.
        let blocks = lease.into_blocks_no_free();
        self.insert_completed_prefix(prompt_tokens, &blocks);
    }

    fn match_prefix(&mut self, tokens: &[i32]) -> PrefixMatch {
        self.raw_match_prefix(tokens)
    }

    fn flush_pending_returns(&mut self) {
        let mut guard = match self.return_sink.lock() {
            Ok(g) => g,
            Err(poison) => poison.into_inner(),
        };
        if guard.is_empty() {
            return;
        }
        let drained: Vec<PhysicalBlockId> = guard.drain(..).collect();
        drop(guard);
        self.allocator.free(&drained);
    }

    fn block_size(&self) -> BlockSize {
        BlockSize::new(self.block_size as u32)
    }

    fn total_blocks(&self) -> BlockCount {
        BlockCount::new(self.allocator.total_blocks())
    }

    fn available_blocks(&self) -> BlockCount {
        BlockCount::new(self.allocator.num_free_blocks())
    }

    fn mode_name(&self) -> &'static str {
        "paged"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool_with_blocks(n: usize, bs: u32) -> PagedKvPool {
        PagedKvPool::new(n, bs as usize)
    }

    #[test]
    fn allocate_returns_lease_holding_blocks() {
        let mut pool = pool_with_blocks(8, 4);
        let lease = pool.allocate(TokenCount::new(8)).unwrap();
        assert_eq!(lease.len().raw(), 2);
        assert_eq!(pool.available_blocks().raw(), 6);
    }

    #[test]
    fn lease_drop_returns_blocks_via_flush() {
        let mut pool = pool_with_blocks(4, 4);
        {
            let _lease = pool.allocate(TokenCount::new(8)).unwrap();
            assert_eq!(pool.available_blocks().raw(), 2);
        }
        assert_eq!(pool.available_blocks().raw(), 2);
        pool.flush_pending_returns();
        assert_eq!(pool.available_blocks().raw(), 4);
    }

    #[test]
    fn allocate_implicitly_flushes_pending_returns() {
        let mut pool = pool_with_blocks(2, 4);
        {
            let _lease = pool.allocate(TokenCount::new(8)).unwrap();
        }
        let lease2 = pool.allocate(TokenCount::new(4)).unwrap();
        assert_eq!(lease2.len().raw(), 1);
    }

    #[test]
    fn panic_during_lease_use_does_not_leak_blocks() {
        let mut pool = pool_with_blocks(4, 4);
        let pre = pool.available_blocks().raw();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _lease = pool.allocate(TokenCount::new(16)).unwrap();
            assert_eq!(pool.available_blocks().raw(), 0);
            panic!("simulated downstream failure");
        }));
        assert!(result.is_err());
        pool.flush_pending_returns();
        assert_eq!(pool.available_blocks().raw(), pre);
    }

    #[test]
    fn free_finished_consumes_lease_no_double_return() {
        let mut pool = pool_with_blocks(4, 4);
        let lease = pool.allocate(TokenCount::new(8)).unwrap();
        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8];
        pool.free_finished(&prompt, lease);
        let parked = pool.return_sink.lock().unwrap().len();
        assert_eq!(parked, 0);
        assert_eq!(pool.available_blocks().raw(), 4);
    }

    #[test]
    fn block_size_and_capacity_round_trip() {
        let pool = pool_with_blocks(16, 32);
        assert_eq!(pool.block_size().raw(), 32);
        assert_eq!(pool.total_blocks().raw(), 16);
        assert_eq!(pool.available_blocks().raw(), 16);
    }
}
