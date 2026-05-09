//! Unified KV resource management trait.
//!
//! Abstracts over Slot (non-paged) and Paged modes so the scheduler engine
//! can work with either without knowing the concrete type at compile time.

use crate::cache::traits::PhysicalBlockId;
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

    /// Release KV resources when a sequence finishes or is preempted.
    fn free(&mut self, alloc: KvAllocation);

    /// Available capacity in tokens (how many more tokens can be allocated).
    fn available_tokens(&self) -> usize;

    /// Total capacity in tokens.
    fn total_capacity_tokens(&self) -> usize;

    /// Mode name for logging.
    fn mode_name(&self) -> &'static str;
}
