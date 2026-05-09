//! Core traits for cache management.

/// Unique identifier for a physical KV cache block on the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalBlockId(pub u32);

/// Hash of a block's token content (for prefix deduplication).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(pub u64);

/// Result of a prefix match operation.
#[derive(Debug, Clone)]
pub struct PrefixMatch {
    /// Number of tokens that hit the cache (always block-aligned in paged mode).
    pub num_cached_tokens: usize,
    /// Physical blocks that can be reused (in order).
    pub cached_blocks: Vec<PhysicalBlockId>,
    /// Block hash of the last matched block (for incremental matching).
    pub last_block_hash: Option<BlockHash>,
}

impl PrefixMatch {
    /// No prefix match found.
    pub fn none() -> Self {
        Self {
            num_cached_tokens: 0,
            cached_blocks: vec![],
            last_block_hash: None,
        }
    }
}

/// Manages physical KV cache blocks.
///
/// The allocator tracks which physical blocks are free/in-use/cached.
/// It does NOT touch GPU memory — it manages metadata that tells the
/// worker which block IDs to use.
pub trait BlockAllocator: Send + Sync {
    /// Allocate N blocks. Returns None if insufficient blocks even after eviction.
    fn allocate(&mut self, num_blocks: usize) -> Option<Vec<PhysicalBlockId>>;

    /// Free blocks back to the pool (or mark as cached if prefix-eligible).
    fn free(&mut self, blocks: &[PhysicalBlockId]);

    /// Number of free blocks available without eviction.
    fn num_free_blocks(&self) -> usize;

    /// Total capacity in blocks.
    fn total_blocks(&self) -> usize;

    /// Attempt to evict cached (but unused) blocks to free space.
    /// Returns number of blocks actually freed.
    fn evict(&mut self, num_needed: usize) -> usize;

    /// Check if a block with this content hash exists (prefix cache hit).
    fn lookup_cached(&self, hash: BlockHash) -> Option<PhysicalBlockId>;

    /// Register a block's content hash for future prefix cache lookups.
    fn register_hash(&mut self, block_id: PhysicalBlockId, hash: BlockHash);
}

/// Strategy for prefix cache matching and management.
pub trait CacheStrategy: Send + Sync {
    /// Given a token sequence, find the longest cached prefix.
    fn match_prefix(&self, tokens: &[i32]) -> PrefixMatch;

    /// Register a completed sequence's blocks into the prefix cache.
    fn insert(&mut self, tokens: &[i32], block_table: &super::BlockTable);

    /// Evict entries from the cache to free blocks.
    fn evict_entries(&mut self, num_blocks_needed: usize) -> Vec<PhysicalBlockId>;

    /// Strategy name for logging/metrics.
    fn name(&self) -> &'static str;
}

/// Cache state snapshot — passed to scheduling policy for decisions.
#[derive(Debug, Clone)]
pub struct CacheState {
    /// Number of free blocks (without eviction).
    pub free_blocks: usize,
    /// Total blocks.
    pub total_blocks: usize,
    /// Utilization ratio (0.0 - 1.0).
    pub utilization: f64,
    /// Number of blocks reclaimable via eviction.
    pub evictable_blocks: usize,
}
