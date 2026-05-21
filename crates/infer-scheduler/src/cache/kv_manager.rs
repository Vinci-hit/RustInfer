//! Unified KV resource management trait.
//!
//! Abstracts over Slot (non-paged) and Paged modes so the scheduler engine
//! can work with either without knowing the concrete type at compile time.

use crate::cache::traits::{PhysicalBlockId, PrefixMatch};
use crate::error::Result;

/// KV allocation result — tells the batch builder what to send to the worker.
#[derive(Debug, Clone)]
pub enum KvAllocation {
    /// Non-Paged: a single slot id (compatible with existing PrefillBatchCmd.kv_slots).
    Slot(u32),
    /// Paged: physical block id list (for BatchCommand.block_tables).
    Blocks(Vec<PhysicalBlockId>),
}

impl KvAllocation {
    /// Get the slot id (panics if this is a Blocks allocation).
    pub fn as_slot(&self) -> u32 {
        match self {
            Self::Slot(id) => *id,
            Self::Blocks(_) => panic!("KvAllocation::as_slot called on Blocks variant"),
        }
    }

    /// Get the block ids (panics if this is a Slot allocation).
    pub fn as_blocks(&self) -> &[PhysicalBlockId] {
        match self {
            Self::Blocks(blocks) => blocks,
            Self::Slot(_) => panic!("KvAllocation::as_blocks called on Slot variant"),
        }
    }

    /// Whether this is slot mode.
    pub fn is_slot(&self) -> bool {
        matches!(self, Self::Slot(_))
    }
}

/// Unified KV resource management trait.
///
/// Different modes have different implementations but expose the same interface
/// to SchedulerEngine. The mode is determined at startup and fixed for the
/// lifetime of the scheduler process.
pub trait KvManager: Send + Sync {
    /// Allocate KV resources for a new sequence that needs `num_tokens` capacity.
    ///
    /// - Slot mode: allocates one slot id.
    /// - Paged mode: allocates ceil(num_tokens / block_size) blocks.
    fn allocate(&mut self, num_tokens: usize) -> Result<KvAllocation>;

    /// Extend allocation when a sequence generates more tokens.
    ///
    /// - Slot mode: no-op (worker handles growth internally).
    /// - Paged mode: allocates additional blocks as needed.
    fn extend(&mut self, alloc: &mut KvAllocation, additional_tokens: usize) -> Result<()>;

    /// Allocate KV resources for a new sequence and optionally reuse a cached prefix.
    /// Default implementation does no prefix reuse.
    fn allocate_with_prefix(&mut self, num_tokens: usize, _tokens: &[i32]) -> Result<(KvAllocation, PrefixMatch)> {
        Ok((self.allocate(num_tokens)?, PrefixMatch::none()))
    }

    /// Allocate exact physical blocks for a worker decode-time block-table extension.
    fn allocate_decode_blocks(&mut self, _num_blocks: usize) -> Result<Vec<PhysicalBlockId>> {
        Err(crate::error::SchedulerError::Internal(
            "decode block grants are only supported in paged KV mode".into(),
        ))
    }

    /// Insert a completed sequence into prefix cache. Default is no-op.
    fn insert_prefix_cache(&mut self, _tokens: &[i32], _alloc: &KvAllocation) {}

    /// Release KV resources for a normally finished sequence. Paged mode may keep
    /// full prompt blocks in the prefix cache instead of immediately freeing them.
    fn free_finished(&mut self, tokens: &[i32], alloc: KvAllocation) {
        self.insert_prefix_cache(tokens, &alloc);
        self.free(alloc);
    }

    /// Release KV resources when a sequence finishes or is preempted.
    fn free(&mut self, alloc: KvAllocation);

    /// Available capacity in tokens (how many more tokens can be allocated).
    fn available_tokens(&self) -> usize;

    /// Total capacity in tokens.
    fn total_capacity_tokens(&self) -> usize;

    /// Mode name for logging.
    fn mode_name(&self) -> &'static str;
}
