//! RadixTree for prefix matching (stub).
//!
//! When implemented, this provides SGLang-style prefix sharing via a radix/trie
//! structure that indexes KV cache blocks by their token content.

use crate::cache::block_table::BlockTable;
use crate::cache::traits::{CacheStrategy, PhysicalBlockId, PrefixMatch};

/// RadixTree-based prefix cache (stub).
///
/// **Current status: NOT IMPLEMENTED.**
/// Enable with feature flag `radix-tree`.
pub struct RadixTreeCache;

impl RadixTreeCache {
    pub fn new() -> Self {
        Self
    }
}

impl CacheStrategy for RadixTreeCache {
    fn match_prefix(&self, _tokens: &[i32]) -> PrefixMatch {
        // Stub: always miss.
        PrefixMatch::none()
    }

    fn insert(&mut self, _tokens: &[i32], _block_table: &BlockTable) {
        tracing::debug!("RadixTreeCache::insert called but not implemented");
    }

    fn evict_entries(&mut self, _num_blocks_needed: usize) -> Vec<PhysicalBlockId> {
        vec![]
    }

    fn name(&self) -> &'static str {
        "radix-tree (stub)"
    }
}

impl Default for RadixTreeCache {
    fn default() -> Self {
        Self::new()
    }
}
