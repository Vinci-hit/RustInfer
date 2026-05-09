//! Paged block allocator.
//!
//! Manages physical KV cache blocks with O(1) allocation (free-list stack),
//! reference counting for prefix sharing, and LRU eviction.

use std::collections::{HashMap, VecDeque};

use crate::cache::traits::{BlockAllocator, BlockHash, PhysicalBlockId};

/// Paged block allocator — manages physical KV cache block metadata.
///
/// Does NOT touch GPU memory. Only manages IDs and metadata that tell
/// the worker which physical blocks to use.
pub struct PagedBlockAllocator {
    /// Total number of physical blocks.
    num_blocks: usize,
    /// Tokens per block.
    block_size: usize,
    /// Free list (stack — O(1) alloc/free).
    free_list: Vec<PhysicalBlockId>,
    /// Reference count per block (index = block id).
    ref_counts: Vec<u32>,
    /// Block hash → physical block id (for prefix cache lookups).
    hash_to_block: HashMap<BlockHash, PhysicalBlockId>,
    /// LRU eviction queue for cached (refcount=0) blocks.
    eviction_order: VecDeque<PhysicalBlockId>,
}

impl PagedBlockAllocator {
    /// Create a new allocator with `num_blocks` physical blocks.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        // Initialize free list with all block ids (reversed so pop gives lowest first).
        let free_list: Vec<PhysicalBlockId> = (0..num_blocks as u32)
            .rev()
            .map(PhysicalBlockId)
            .collect();

        Self {
            num_blocks,
            block_size,
            free_list,
            ref_counts: vec![0; num_blocks],
            hash_to_block: HashMap::new(),
            eviction_order: VecDeque::new(),
        }
    }

    /// Block size in tokens.
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

impl BlockAllocator for PagedBlockAllocator {
    fn allocate(&mut self, num_blocks: usize) -> Option<Vec<PhysicalBlockId>> {
        if num_blocks > self.free_list.len() {
            // Try eviction first.
            let evicted = self.evict(num_blocks - self.free_list.len());
            if self.free_list.len() + evicted < num_blocks {
                return None;
            }
        }

        let mut allocated = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let block_id = self.free_list.pop()?;
            self.ref_counts[block_id.0 as usize] = 1;
            allocated.push(block_id);
        }
        Some(allocated)
    }

    fn free(&mut self, blocks: &[PhysicalBlockId]) {
        for &block_id in blocks {
            let idx = block_id.0 as usize;
            if idx < self.num_blocks {
                let rc = &mut self.ref_counts[idx];
                if *rc > 0 {
                    *rc -= 1;
                }
                if *rc == 0 {
                    self.free_list.push(block_id);
                    // Remove from eviction order if present.
                    self.eviction_order.retain(|&b| b != block_id);
                }
            }
        }
    }

    fn num_free_blocks(&self) -> usize {
        self.free_list.len()
    }

    fn total_blocks(&self) -> usize {
        self.num_blocks
    }

    fn evict(&mut self, num_needed: usize) -> usize {
        let mut evicted = 0;
        while evicted < num_needed {
            match self.eviction_order.pop_front() {
                Some(block_id) => {
                    let idx = block_id.0 as usize;
                    if self.ref_counts[idx] == 0 {
                        // Remove from hash map.
                        self.hash_to_block.retain(|_, &mut v| v != block_id);
                        self.free_list.push(block_id);
                        evicted += 1;
                    }
                }
                None => break,
            }
        }
        evicted
    }

    fn lookup_cached(&self, hash: BlockHash) -> Option<PhysicalBlockId> {
        self.hash_to_block.get(&hash).copied()
    }

    fn register_hash(&mut self, block_id: PhysicalBlockId, hash: BlockHash) {
        self.hash_to_block.insert(hash, block_id);
        // Add to eviction order (LRU: most recently accessed at the back).
        self.eviction_order.retain(|&b| b != block_id);
        self.eviction_order.push_back(block_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_alloc_free() {
        let mut alloc = PagedBlockAllocator::new(8, 16);
        assert_eq!(alloc.num_free_blocks(), 8);

        let blocks = alloc.allocate(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(alloc.num_free_blocks(), 5);

        alloc.free(&blocks);
        assert_eq!(alloc.num_free_blocks(), 8);
    }

    #[test]
    fn exhaustion() {
        let mut alloc = PagedBlockAllocator::new(4, 16);
        let _b = alloc.allocate(4).unwrap();
        assert_eq!(alloc.num_free_blocks(), 0);
        assert!(alloc.allocate(1).is_none());
    }

    #[test]
    fn hash_lookup() {
        let mut alloc = PagedBlockAllocator::new(8, 16);
        let blocks = alloc.allocate(1).unwrap();
        let block_id = blocks[0];
        let hash = BlockHash(0xDEADBEEF);

        alloc.register_hash(block_id, hash);
        assert_eq!(alloc.lookup_cached(hash), Some(block_id));
        assert_eq!(alloc.lookup_cached(BlockHash(0x12345)), None);
    }
}
