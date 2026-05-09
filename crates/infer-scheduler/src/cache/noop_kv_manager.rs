//! No-op KV manager for Diffusion mode.
//!
//! Diffusion models don't use KV cache on the scheduler side.
//! Cache management (if any) is handled internally by the Worker pipeline.

use crate::cache::kv_manager::{KvAllocation, KvManager};
use crate::error::Result;

/// No-op KV manager — does nothing. Used in Diffusion mode.
pub struct NoopKvManager;

impl NoopKvManager {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoopKvManager {
    fn default() -> Self {
        Self::new()
    }
}

impl KvManager for NoopKvManager {
    fn allocate(&mut self, _num_tokens: usize) -> Result<KvAllocation> {
        Ok(KvAllocation::Slot(0)) // dummy value, not used
    }

    fn extend(&mut self, _alloc: &mut KvAllocation, _additional_tokens: usize) -> Result<()> {
        Ok(())
    }

    fn free(&mut self, _alloc: KvAllocation) {
        // no-op
    }

    fn available_tokens(&self) -> usize {
        usize::MAX // unlimited — no KV constraint
    }

    fn total_capacity_tokens(&self) -> usize {
        usize::MAX
    }

    fn mode_name(&self) -> &'static str {
        "noop (diffusion)"
    }
}
