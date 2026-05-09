//! No-op prefix cache (stub).
//!
//! Always reports cache miss. Used when prefix caching is disabled.

use crate::cache::block_table::BlockTable;
use crate::cache::traits::{CacheStrategy, PhysicalBlockId, PrefixMatch};

/// No-op prefix cache — always misses.
pub struct NoPrefixCache;

impl CacheStrategy for NoPrefixCache {
    fn match_prefix(&self, _tokens: &[i32]) -> PrefixMatch {
        PrefixMatch::none()
    }

    fn insert(&mut self, _tokens: &[i32], _block_table: &BlockTable) {
        // No-op: prefix caching not enabled.
    }

    fn evict_entries(&mut self, _num_blocks_needed: usize) -> Vec<PhysicalBlockId> {
        vec![]
    }

    fn name(&self) -> &'static str {
        "none"
    }
}
