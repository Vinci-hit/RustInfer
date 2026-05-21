//! Paged KV manager.
//!
//! The scheduler owns logical block allocation. It never touches GPU memory;
//! it only hands physical block IDs to the worker. The worker has already
//! allocated the physical Paged KV pool during startup.

use crate::cache::block_allocator::PagedBlockAllocator;
use crate::cache::block_table::BlockTable;
use crate::cache::kv_manager::{KvAllocation, KvManager};
use crate::cache::radix_tree::RadixTreeCache;
use crate::cache::traits::{BlockAllocator, CacheStrategy, PhysicalBlockId, PrefixMatch};
use crate::error::{Result, SchedulerError};

/// Paged KV manager — manages fixed-size physical block IDs.
pub struct PagedKvManager {
    allocator: PagedBlockAllocator,
    prefix_cache: RadixTreeCache,
    block_size: usize,
    enable_prefix_cache: bool,
}

impl PagedKvManager {
    /// Create a new paged KV manager.
    ///
    /// - `num_blocks`: number of physical blocks already available in the worker's KV pool.
    /// - `block_size`: tokens per block.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self::new_with_prefix_cache(num_blocks, block_size, false)
    }

    pub fn new_with_prefix_cache(num_blocks: usize, block_size: usize, enable_prefix_cache: bool) -> Self {
        Self {
            allocator: PagedBlockAllocator::new(num_blocks, block_size),
            prefix_cache: RadixTreeCache::new(block_size),
            block_size,
            enable_prefix_cache,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn total_blocks(&self) -> usize {
        self.allocator.total_blocks()
    }

    pub fn free_blocks(&self) -> usize {
        self.allocator.num_free_blocks()
    }

    fn blocks_for_tokens(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_size).max(1)
    }

    /// Allocate an exact number of physical blocks. Used by decode-time grants
    /// (`NeedBlocks -> GrantBlocks`) where the worker asks for block-table extension.
    pub fn allocate_blocks(&mut self, num_blocks: usize) -> Result<Vec<PhysicalBlockId>> {
        if num_blocks == 0 {
            return Ok(Vec::new());
        }
        if self.allocator.num_free_blocks() < num_blocks && self.enable_prefix_cache {
            let need = num_blocks - self.allocator.num_free_blocks();
            let evicted = self.prefix_cache.evict_entries(need);
            self.allocator.free_cached_blocks(&evicted);
        }
        let available = self.allocator.num_free_blocks();
        self.allocator.allocate(num_blocks).ok_or(SchedulerError::CacheExhausted {
            needed: num_blocks,
            available,
        })
    }

    /// Free an exact block list back to the allocator.
    pub fn free_block_list(&mut self, blocks: &[PhysicalBlockId]) {
        self.allocator.free(blocks);
    }

    pub fn match_prefix(&self, tokens: &[i32]) -> PrefixMatch {
        if self.enable_prefix_cache {
            self.prefix_cache.match_prefix(tokens)
        } else {
            PrefixMatch::none()
        }
    }

    /// Allocate blocks for a prompt while reusing a cached prefix when available.
    /// Returns the full block table and the matched prefix metadata.
    pub fn allocate_for_tokens_with_prefix(&mut self, tokens: &[i32]) -> Result<(KvAllocation, PrefixMatch)> {
        let mut prefix = self.match_prefix(tokens);
        let total_blocks = self.blocks_for_tokens(tokens.len());
        // Keep at least one prompt block writable for the incoming request. This
        // avoids mutating the final cached block in-place when a prompt is fully
        // block-cache-hit; copy-on-write can relax this later.
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
        blocks.extend(self.allocate_blocks(missing_blocks)?);
        Ok((KvAllocation::Blocks(blocks), prefix))
    }

    /// Insert a completed prompt into the radix-tree prefix cache and release its
    /// full-block KV blocks into cache-owned state. Partial tail blocks are not cached.
    pub fn insert_completed_prefix(&mut self, tokens: &[i32], alloc: &KvAllocation) {
        if !self.enable_prefix_cache {
            return;
        }
        let KvAllocation::Blocks(blocks) = alloc else {
            return;
        };
        let full_blocks = (tokens.len() / self.block_size).min(blocks.len());
        if full_blocks == 0 {
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

impl KvManager for PagedKvManager {
    fn allocate(&mut self, num_tokens: usize) -> Result<KvAllocation> {
        let num_blocks = self.blocks_for_tokens(num_tokens);
        Ok(KvAllocation::Blocks(self.allocate_blocks(num_blocks)?))
    }

    fn allocate_with_prefix(&mut self, num_tokens: usize, tokens: &[i32]) -> Result<(KvAllocation, PrefixMatch)> {
        if self.enable_prefix_cache {
            self.allocate_for_tokens_with_prefix(tokens)
        } else {
            Ok((self.allocate(num_tokens)?, PrefixMatch::none()))
        }
    }

    fn allocate_decode_blocks(&mut self, num_blocks: usize) -> Result<Vec<PhysicalBlockId>> {
        self.allocate_blocks(num_blocks)
    }

    fn insert_prefix_cache(&mut self, tokens: &[i32], alloc: &KvAllocation) {
        self.insert_completed_prefix(tokens, alloc);
    }

    fn free_finished(&mut self, tokens: &[i32], alloc: KvAllocation) {
        if self.enable_prefix_cache {
            self.insert_completed_prefix(tokens, &alloc);
        } else {
            self.free(alloc);
        }
    }

    fn extend(&mut self, alloc: &mut KvAllocation, additional_tokens: usize) -> Result<()> {
        if additional_tokens == 0 {
            return Ok(());
        }
        let add_blocks = self.blocks_for_tokens(additional_tokens);
        let new_blocks = self.allocate_blocks(add_blocks)?;
        match alloc {
            KvAllocation::Blocks(blocks) => {
                blocks.extend(new_blocks);
                Ok(())
            }
            KvAllocation::Slot(_) => Err(SchedulerError::Internal(
                "PagedKvManager::extend called with Slot allocation".into(),
            )),
        }
    }

    fn free(&mut self, alloc: KvAllocation) {
        match alloc {
            KvAllocation::Blocks(blocks) => self.free_block_list(&blocks),
            KvAllocation::Slot(slot) => {
                tracing::warn!("PagedKvManager::free called with Slot allocation {}", slot);
            }
        }
    }

    fn available_tokens(&self) -> usize {
        self.allocator.num_free_blocks() * self.block_size
    }

    fn total_capacity_tokens(&self) -> usize {
        self.allocator.total_blocks() * self.block_size
    }

    fn mode_name(&self) -> &'static str {
        "paged"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_prompt_blocks_by_token_count() {
        let mut mgr = PagedKvManager::new(8, 16);
        let alloc = mgr.allocate(17).unwrap();
        let blocks = alloc.as_blocks();
        assert_eq!(blocks.len(), 2);
        assert_eq!(mgr.free_blocks(), 6);
        assert_eq!(mgr.available_tokens(), 96);
    }

    #[test]
    fn grant_decode_blocks_and_free() {
        let mut mgr = PagedKvManager::new(8, 16);
        let initial = mgr.allocate(16).unwrap();
        assert_eq!(initial.as_blocks().len(), 1);

        let granted = mgr.allocate_blocks(2).unwrap();
        assert_eq!(granted.len(), 2);
        assert_eq!(mgr.free_blocks(), 5);

        mgr.free(initial);
        mgr.free_block_list(&granted);
        assert_eq!(mgr.free_blocks(), 8);
    }

    #[test]
    fn extend_blocks_allocation() {
        let mut mgr = PagedKvManager::new(8, 16);
        let mut alloc = mgr.allocate(16).unwrap();
        mgr.extend(&mut alloc, 33).unwrap();
        assert_eq!(alloc.as_blocks().len(), 4);
        assert_eq!(mgr.free_blocks(), 4);
    }

    #[test]
    fn exhaustion_reports_available_blocks() {
        let mut mgr = PagedKvManager::new(2, 16);
        let _ = mgr.allocate(32).unwrap();
        let err = mgr.allocate_blocks(1).unwrap_err();
        match err {
            SchedulerError::CacheExhausted { needed, available } => {
                assert_eq!(needed, 1);
                assert_eq!(available, 0);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn reuses_prefix_blocks_from_radix_tree() {
        let mut mgr = PagedKvManager::new_with_prefix_cache(8, 4, true);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let alloc = mgr.allocate(tokens.len()).unwrap();
        assert_eq!(alloc.as_blocks().len(), 2);
        let cached_blocks = alloc.as_blocks().to_vec();
        mgr.insert_prefix_cache(&tokens, &alloc);

        let (alloc2, prefix) = mgr.allocate_for_tokens_with_prefix(&[1, 2, 3, 4, 9, 9, 9, 9]).unwrap();
        assert_eq!(prefix.num_cached_tokens, 4);
        assert_eq!(prefix.cached_blocks, vec![cached_blocks[0]]);
        assert_eq!(alloc2.as_blocks()[0], cached_blocks[0]);
        assert_eq!(alloc2.as_blocks().len(), 2);
    }

    #[test]
    fn lru_eviction_reclaims_cached_blocks_for_new_allocations() {
        let mut mgr = PagedKvManager::new_with_prefix_cache(2, 4, true);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let alloc = mgr.allocate(tokens.len()).unwrap();
        mgr.insert_prefix_cache(&tokens, &alloc);
        assert_eq!(mgr.free_blocks(), 0);

        let new_blocks = mgr.allocate_blocks(1).unwrap();
        assert_eq!(new_blocks.len(), 1);
        assert_eq!(mgr.free_blocks(), 0);
    }
}
