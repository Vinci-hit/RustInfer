//! Aggregate Root #2: `KvCachePool` trait family.
//!
//! Layout:
//! - `lease.rs`     — `KvLease` RAII guard (P.4 in plan)
//! - `pool_trait.rs` — `KvCachePool` trait
//! - `paged_pool.rs` — `PagedKvPool` wrapping `cache::PagedKvManager`
//! - `noop_pool.rs`  — `NoopKvPool` for diffusion mode
//!
//! Step 11 establishes this surface alongside (not replacing) the
//! legacy `cache::KvManager` family. The `application/` layer split
//! (Step 13+) will swap the engine over.

mod lease;
mod noop_pool;
mod paged_pool;
mod pool_trait;

pub use lease::KvLease;
pub use noop_pool::NoopKvPool;
pub use paged_pool::PagedKvPool;
pub use pool_trait::KvCachePool;
