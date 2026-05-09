//! Per-sequence block table.
//!
//! Maps a sequence's logical block indices to physical block IDs.

use crate::cache::traits::PhysicalBlockId;

/// Per-sequence mapping from logical blocks to physical blocks.
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Logical block index → Physical block ID.
    pub blocks: Vec<PhysicalBlockId>,
    /// Number of tokens filled in the last block (0..block_size).
    pub last_block_fill: usize,
    /// Block size in tokens (copied from allocator config for convenience).
    pub block_size: usize,
}

impl BlockTable {
    /// Create a new empty block table.
    pub fn new(block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            last_block_fill: 0,
            block_size,
        }
    }

    /// Create from a list of pre-allocated blocks.
    pub fn from_blocks(blocks: Vec<PhysicalBlockId>, block_size: usize, last_block_fill: usize) -> Self {
        Self {
            blocks,
            last_block_fill,
            block_size,
        }
    }

    /// Total tokens this block table can hold.
    pub fn capacity_tokens(&self) -> usize {
        self.blocks.len() * self.block_size
    }

    /// Number of tokens currently stored.
    pub fn num_tokens(&self) -> usize {
        if self.blocks.is_empty() {
            0
        } else {
            (self.blocks.len() - 1) * self.block_size + self.last_block_fill
        }
    }

    /// Whether the last block is full (needs a new block for the next token).
    pub fn last_block_full(&self) -> bool {
        !self.blocks.is_empty() && self.last_block_fill == self.block_size
    }

    /// Number of blocks in this table.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Flatten block IDs to u32 array (for wire format).
    pub fn to_flat_ids(&self) -> Vec<u32> {
        self.blocks.iter().map(|b| b.0).collect()
    }
}
