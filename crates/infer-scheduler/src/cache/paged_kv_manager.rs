//! Paged KV manager (stub).
//!
//! When implemented, this will manage KV cache via fixed-size physical blocks
//! with PagedAttention semantics. Currently returns NotImplemented.

use crate::cache::block_allocator::PagedBlockAllocator;
use crate::cache::kv_manager::{KvAllocation, KvManager};
use crate::cache::traits::BlockAllocator;
use crate::error::{Result, SchedulerError};

/// Paged KV manager — manages cache as fixed-size blocks.
///
/// **Current status: STUB.** All allocation methods return `NotImplemented`.
pub struct PagedKvManager {
    #[allow(dead_code)]
    allocator: PagedBlockAllocator,
    #[allow(dead_code)]
    block_size: usize,
}

impl PagedKvManager {
    /// Create a new paged KV manager.
    ///
    /// - `num_blocks`: total number of physical blocks available on the GPU.
    /// - `block_size`: tokens per block.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            allocator: PagedBlockAllocator::new(num_blocks, block_size),
            block_size,
        }
    }
}

impl KvManager for PagedKvManager {
    fn allocate(&mut self, _num_tokens: usize) -> Result<KvAllocation> {
        Err(SchedulerError::NotImplemented(
            "Paged KV cache mode not yet implemented. Use --kv-cache-mode slot".into(),
        ))
    }

    fn extend(&mut self, _alloc: &mut KvAllocation, _additional_tokens: usize) -> Result<()> {
        Err(SchedulerError::NotImplemented(
            "Paged KV extend not yet implemented".into(),
        ))
    }

    fn free(&mut self, _alloc: KvAllocation) {
        // Stub: nothing to free since we never successfully allocate.
        tracing::warn!("PagedKvManager::free called but paged mode is not implemented");
    }

    fn available_tokens(&self) -> usize {
        0
    }

    fn total_capacity_tokens(&self) -> usize {
        self.allocator.total_blocks() * self.block_size
    }

    fn mode_name(&self) -> &'static str {
        "paged (stub)"
    }
}
