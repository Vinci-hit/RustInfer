//! KV cache management module.
//!
//! Provides the `KvManager` trait that unifies Slot (non-paged) and Paged modes,
//! along with block allocation, prefix caching, and eviction strategies.

pub mod traits;
pub mod kv_manager;
pub mod slot_kv_manager;
pub mod paged_kv_manager;
pub mod block_allocator;
pub mod block_table;
pub mod prefix_cache;
pub mod radix_tree;
pub mod evictor;

pub use traits::*;
pub use kv_manager::{KvManager, KvAllocation};
pub use slot_kv_manager::SlotKvManager;
pub use paged_kv_manager::PagedKvManager;
pub use block_allocator::PagedBlockAllocator;
pub use block_table::BlockTable;
pub use prefix_cache::NoPrefixCache;
