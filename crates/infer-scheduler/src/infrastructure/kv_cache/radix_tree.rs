//! Radix-tree prefix cache for paged KV blocks.
//!
//! The tree is indexed by full token blocks (`block_size` tokens per edge).
//! Each node that ends at a block boundary owns one physical block id. Matching
//! is therefore block-aligned, which is what Paged KV reuse requires.
//!
//! **Single-threaded ownership**: the engine event loop is single-threaded, so
//! the cache holds its `RadixTreeInner` directly without `Mutex`. `match_prefix`
//! takes `&mut self` because every match touches the LRU clock.

use std::collections::{HashMap, VecDeque};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::infrastructure::kv_cache::block_table::BlockTable;
use crate::infrastructure::kv_cache::traits::{BlockHash, CacheStrategy, PhysicalBlockId, PrefixMatch};

#[derive(Debug)]
struct RadixNode {
    parent: Option<usize>,
    edge_from_parent: Vec<i32>,
    children: HashMap<Vec<i32>, usize>,
    block: Option<PhysicalBlockId>,
    last_block_hash: Option<BlockHash>,
    last_access: u64,
}

#[derive(Debug)]
struct RadixTreeInner {
    block_size: usize,
    nodes: Vec<RadixNode>,
    lru: VecDeque<(usize, u64)>,
    clock: u64,
}

/// RadixTree-based prefix cache with LRU leaf eviction.
///
/// Owns its inner state directly (no `Mutex`). Engine is single-threaded, so
/// shared mutable access is unnecessary.
pub struct RadixTreeCache {
    inner: RadixTreeInner,
}

impl RadixTreeCache {
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "RadixTreeCache block_size must be > 0");
        Self {
            inner: RadixTreeInner {
                block_size,
                nodes: vec![RadixNode {
                    parent: None,
                    edge_from_parent: Vec::new(),
                    children: HashMap::new(),
                    block: None,
                    last_block_hash: None,
                    last_access: 0,
                }],
                lru: VecDeque::new(),
                clock: 0,
            },
        }
    }
}

impl RadixTreeInner {
    fn touch(&mut self, node_id: usize) {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.nodes[node_id].last_access = self.clock;
        self.lru.push_back((node_id, self.clock));
    }

    fn hash_block(block: &[i32]) -> BlockHash {
        let mut hasher = DefaultHasher::new();
        block.hash(&mut hasher);
        BlockHash(hasher.finish())
    }

    fn full_block_count(&self, tokens_len: usize) -> usize {
        tokens_len / self.block_size
    }

    fn remove_leaf_node(&mut self, node_id: usize) -> Option<PhysicalBlockId> {
        if !self.nodes[node_id].children.is_empty() {
            return None;
        }
        let block = self.nodes[node_id].block.take()?;
        if let Some(parent) = self.nodes[node_id].parent {
            let edge = self.nodes[node_id].edge_from_parent.clone();
            self.nodes[parent].children.remove(&edge);
        }
        Some(block)
    }
}

impl CacheStrategy for RadixTreeCache {
    fn match_prefix(&mut self, tokens: &[i32]) -> PrefixMatch {
        let inner = &mut self.inner;
        let full_blocks = inner.full_block_count(tokens.len());
        if full_blocks == 0 {
            return PrefixMatch::none();
        }

        let mut node_id = 0usize;
        let mut cached_blocks = Vec::new();
        let mut last_block_hash = None;

        for block_idx in 0..full_blocks {
            let start = block_idx * inner.block_size;
            let end = start + inner.block_size;
            let edge = &tokens[start..end];
            let Some(&child_id) = inner.nodes[node_id].children.get(edge) else {
                break;
            };
            let Some(block) = inner.nodes[child_id].block else {
                break;
            };
            cached_blocks.push(block);
            last_block_hash = inner.nodes[child_id].last_block_hash;
            inner.touch(child_id);
            node_id = child_id;
        }

        PrefixMatch {
            num_cached_tokens: cached_blocks.len() * inner.block_size,
            cached_blocks,
            last_block_hash,
        }
    }

    fn insert(&mut self, tokens: &[i32], block_table: &BlockTable) {
        let inner = &mut self.inner;
        if block_table.block_size != inner.block_size {
            tracing::warn!(
                "RadixTreeCache insert ignored: block_table block_size {} != cache block_size {}",
                block_table.block_size,
                inner.block_size,
            );
            return;
        }

        let full_blocks = inner.full_block_count(tokens.len()).min(block_table.blocks.len());
        if full_blocks == 0 {
            return;
        }

        let mut node_id = 0usize;
        for block_idx in 0..full_blocks {
            let start = block_idx * inner.block_size;
            let end = start + inner.block_size;
            let edge: Vec<i32> = tokens[start..end].to_vec();
            let child_id = if let Some(&child) = inner.nodes[node_id].children.get(&edge) {
                child
            } else {
                let new_id = inner.nodes.len();
                inner.nodes.push(RadixNode {
                    parent: Some(node_id),
                    edge_from_parent: edge.clone(),
                    children: HashMap::new(),
                    block: None,
                    last_block_hash: None,
                    last_access: 0,
                });
                inner.nodes[node_id].children.insert(edge.clone(), new_id);
                new_id
            };
            inner.nodes[child_id].block = Some(block_table.blocks[block_idx]);
            inner.nodes[child_id].last_block_hash = Some(RadixTreeInner::hash_block(&edge));
            inner.touch(child_id);
            node_id = child_id;
        }
    }

    fn evict_entries(&mut self, num_blocks_needed: usize) -> Vec<PhysicalBlockId> {
        let inner = &mut self.inner;
        let mut evicted = Vec::with_capacity(num_blocks_needed);
        let mut scanned_live_internal = 0usize;

        while evicted.len() < num_blocks_needed {
            let Some((node_id, access)) = inner.lru.pop_front() else {
                break;
            };
            if node_id == 0 || node_id >= inner.nodes.len() {
                continue;
            }
            if inner.nodes[node_id].last_access != access || inner.nodes[node_id].block.is_none() {
                continue;
            }
            if inner.nodes[node_id].children.is_empty() {
                if let Some(block) = inner.remove_leaf_node(node_id) {
                    evicted.push(block);
                    scanned_live_internal = 0;
                }
            } else {
                // Internal prefix node: keep it until its descendants are evicted.
                inner.lru.push_back((node_id, access));
                scanned_live_internal += 1;
                if scanned_live_internal > inner.lru.len() {
                    break;
                }
            }
        }

        evicted
    }

    fn name(&self) -> &'static str {
        "radix-tree"
    }
}

impl Default for RadixTreeCache {
    fn default() -> Self {
        Self::new(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(blocks: &[u32], block_size: usize) -> BlockTable {
        BlockTable::from_blocks(
            blocks.iter().copied().map(PhysicalBlockId).collect(),
            block_size,
            block_size,
        )
    }

    #[test]
    fn matches_longest_block_aligned_prefix() {
        let mut cache = RadixTreeCache::new(4);
        cache.insert(&[1, 2, 3, 4, 5, 6, 7, 8], &table(&[10, 11], 4));

        let hit = cache.match_prefix(&[1, 2, 3, 4, 5, 6, 9, 9]);
        assert_eq!(hit.num_cached_tokens, 4);
        assert_eq!(hit.cached_blocks, vec![PhysicalBlockId(10)]);

        let hit = cache.match_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 99]);
        assert_eq!(hit.num_cached_tokens, 8);
        assert_eq!(hit.cached_blocks, vec![PhysicalBlockId(10), PhysicalBlockId(11)]);
    }

    #[test]
    fn ignores_partial_last_block_on_insert() {
        let mut cache = RadixTreeCache::new(4);
        cache.insert(&[1, 2, 3, 4, 5], &table(&[10, 11], 4));
        let hit = cache.match_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!(hit.num_cached_tokens, 4);
        assert_eq!(hit.cached_blocks, vec![PhysicalBlockId(10)]);
    }

    #[test]
    fn evicts_lru_leaf_nodes() {
        let mut cache = RadixTreeCache::new(2);
        cache.insert(&[1, 1], &table(&[10], 2));
        cache.insert(&[2, 2], &table(&[20], 2));
        // Touch block 10, so block 20 is older.
        let _ = cache.match_prefix(&[1, 1]);

        let evicted = cache.evict_entries(1);
        assert_eq!(evicted, vec![PhysicalBlockId(20)]);
        assert_eq!(cache.match_prefix(&[2, 2]).num_cached_tokens, 0);
        assert_eq!(cache.match_prefix(&[1, 1]).cached_blocks, vec![PhysicalBlockId(10)]);
    }
}
