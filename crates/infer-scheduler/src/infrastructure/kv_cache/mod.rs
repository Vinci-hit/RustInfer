//! Paged KV cache: physical block algorithms.
//!
//! Infrastructure-layer implementation details consumed by
//! [`crate::domain::kv_cache_pool::PagedKvPool`]:
//!
//! - [`PagedBlockAllocator`] — physical block free/used/cached state.
//! - [`RadixTreeCache`] — token-prefix → block-list reuse table.
//! - [`BlockTable`] — block-aligned view of a sequence's blocks.
//! - [`traits`] — internal allocator/strategy traits used between
//!   the allocator and the radix-tree implementation.

pub mod block_allocator;
pub mod block_table;
pub mod radix_tree;
pub mod traits;

pub use block_allocator::PagedBlockAllocator;
pub use block_table::BlockTable;
pub use radix_tree::RadixTreeCache;
pub use traits::{BlockAllocator, CacheStrategy, PhysicalBlockId, PrefixMatch};
