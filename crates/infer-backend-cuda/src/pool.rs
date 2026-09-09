//! CPU bookkeeping for the CUDA allocation cache.
//!
//! Sizes passed to this module are already rounded to the allocator's alignment.
//! Pointers are opaque allocation identities: this module never dereferences
//! them or calls CUDA. Removing a block from a list does not release its device
//! allocation; the caller records a successful CUDA free separately.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;

/// Allocation-cache counters and current device-memory capacity.
///
/// Byte counts use actual allocation capacities, including any extra capacity
/// in a larger reused block. `reserved_bytes` also includes blocks handed to
/// the caller for release until it records a successful CUDA free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaPoolStats {
    pub allocation_requests: u64,
    pub cache_hits: u64,
    pub larger_reuses: u64,
    pub cuda_malloc_calls: u64,
    pub cold_allocations: u64,
    pub cuda_frees: u64,
    pub allocation_retries: u64,
    pub live_bytes: usize,
    pub pooled_bytes: usize,
    pub reserved_bytes: usize,
    pub peak_live_bytes: usize,
    pub peak_reserved_bytes: usize,
    pub pending_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PoolBlock {
    pub(crate) ptr: *mut c_void,
    pub(crate) capacity: usize,
}

#[derive(Clone, Copy, Debug)]
struct CachedBlock {
    ptr: *mut c_void,
    returned_at: u64,
}

#[derive(Debug, Default)]
pub(crate) struct PoolState {
    free: BTreeMap<usize, Vec<CachedBlock>>,
    return_clock: u64,
    // Exact-size checkouts recover capacity from the caller's rounded size.
    // Only larger checkouts need a pointer-to-capacity entry.
    oversized_live: HashMap<usize, usize>,
    deferred: Vec<PoolBlock>,
    stats: CudaPoolStats,
}

impl PoolState {
    /// Take the smallest cached block with at most 12.5% extra capacity.
    ///
    /// Empty buckets stay allocated across ordinary take/retain cycles so a
    /// fixed-shape workload reuses both the tree node and its pointer vector.
    pub(crate) fn take(&mut self, n: usize) -> Option<PoolBlock> {
        self.stats.allocation_requests += 1;
        let upper = n.saturating_add(n / 8);
        let (&capacity, blocks) = self
            .free
            .range_mut(n..=upper)
            .find(|(_, blocks)| !blocks.is_empty())?;
        let ptr = blocks.pop().expect("selected a non-empty pool bucket").ptr;

        debug_assert!(self.stats.pooled_bytes >= capacity);
        self.stats.pooled_bytes -= capacity;
        self.stats.live_bytes += capacity;
        self.stats.peak_live_bytes = self.stats.peak_live_bytes.max(self.stats.live_bytes);
        self.stats.cache_hits += 1;
        if capacity != n {
            let previous = self.oversized_live.insert(ptr as usize, capacity);
            debug_assert!(previous.is_none(), "pool block is already checked out");
            self.stats.larger_reuses += 1;
        } else {
            debug_assert!(!self.oversized_live.contains_key(&(ptr as usize)));
        }
        Some(PoolBlock { ptr, capacity })
    }

    pub(crate) fn note_malloc_attempt(&mut self) {
        self.stats.cuda_malloc_calls += 1;
    }

    /// Record a successful allocation of exactly the rounded request size.
    pub(crate) fn record_cold_allocation(&mut self, n: usize) {
        self.stats.cold_allocations += 1;
        self.stats.live_bytes += n;
        self.stats.reserved_bytes += n;
        self.stats.peak_live_bytes = self.stats.peak_live_bytes.max(self.stats.live_bytes);
        self.stats.peak_reserved_bytes = self
            .stats
            .peak_reserved_bytes
            .max(self.stats.reserved_bytes);
    }

    /// End a checkout, recovering the allocation's original capacity.
    ///
    /// `n` is the rounded size of this checkout, which can be smaller than its
    /// allocation. The returned block still belongs to the caller until it is
    /// retained, deferred, or successfully freed.
    pub(crate) fn release(&mut self, ptr: *mut c_void, n: usize) -> PoolBlock {
        let capacity = self.oversized_live.remove(&(ptr as usize)).unwrap_or(n);
        debug_assert!(capacity >= n);
        debug_assert!(self.stats.live_bytes >= capacity);
        self.stats.live_bytes -= capacity;
        PoolBlock { ptr, capacity }
    }

    /// Retain a released block if its actual capacity fits the cache budget.
    pub(crate) fn retain(&mut self, block: PoolBlock, limit: usize) -> Result<(), PoolBlock> {
        if self.stats.pooled_bytes > limit || block.capacity > limit - self.stats.pooled_bytes {
            return Err(block);
        }
        let blocks = self.free.entry(block.capacity).or_default();
        debug_assert!(
            !blocks.iter().any(|cached| cached.ptr == block.ptr),
            "double return to CUDA pool"
        );
        self.return_clock += 1;
        blocks.push(CachedBlock {
            ptr: block.ptr,
            returned_at: self.return_clock,
        });
        self.stats.pooled_bytes += block.capacity;
        Ok(())
    }

    /// Retain a returned block, evicting the oldest idle blocks if necessary.
    ///
    /// A block larger than the entire budget is returned directly without
    /// disturbing the cache. Ordinary returns avoid an eviction scan and any
    /// allocation for the returned vector. Under pressure, each bucket's first
    /// block is its oldest: returns append and cache hits pop from the tail.
    /// The caller frees the returned allocations outside the pool lock, then
    /// records those successful frees separately.
    pub(crate) fn retain_or_evict(&mut self, block: PoolBlock, limit: usize) -> Vec<PoolBlock> {
        let block = match self.retain(block, limit) {
            Ok(()) => return Vec::new(),
            Err(block) => block,
        };
        if block.capacity > limit {
            return vec![block];
        }

        let target = limit - block.capacity;
        let mut evicted = Vec::new();
        while self.stats.pooled_bytes > target {
            let capacity = self
                .free
                .iter()
                .filter_map(|(&capacity, blocks)| {
                    blocks.first().map(|oldest| (capacity, oldest.returned_at))
                })
                .min_by_key(|&(_, returned_at)| returned_at)
                .map(|(capacity, _)| capacity)
                .expect("pooled bytes require an idle allocation");
            let oldest = self
                .free
                .get_mut(&capacity)
                .expect("selected pool bucket exists")
                .remove(0);
            debug_assert!(self.stats.pooled_bytes >= capacity);
            self.stats.pooled_bytes -= capacity;
            evicted.push(PoolBlock {
                ptr: oldest.ptr,
                capacity,
            });
        }
        self.retain(block, limit)
            .expect("eviction made room for the returned allocation");
        evicted
    }

    /// Keep a released block pending until the caller can legally free it.
    pub(crate) fn defer(&mut self, block: PoolBlock) {
        self.stats.pending_bytes += block.capacity;
        self.deferred.push(block);
    }

    /// Transfer pending blocks to the caller without recording device frees.
    pub(crate) fn take_deferred(&mut self) -> Vec<PoolBlock> {
        let blocks = std::mem::take(&mut self.deferred);
        let bytes: usize = blocks.iter().map(|block| block.capacity).sum();
        debug_assert_eq!(self.stats.pending_bytes, bytes);
        self.stats.pending_bytes -= bytes;
        blocks
    }

    /// Evict largest idle blocks first until pooled capacity is at most target.
    ///
    /// Live and deferred blocks are untouched. Reserved capacity remains
    /// unchanged until the caller records each successful device free.
    pub(crate) fn drain_to(&mut self, target: usize) -> Vec<PoolBlock> {
        let mut drained = Vec::new();
        while self.stats.pooled_bytes > target {
            let mut entry = self
                .free
                .last_entry()
                .expect("pooled bytes require at least one pool bucket");
            let capacity = *entry.key();
            let Some(cached) = entry.get_mut().pop() else {
                entry.remove();
                continue;
            };
            if entry.get().is_empty() {
                entry.remove();
            }
            debug_assert!(self.stats.pooled_bytes >= capacity);
            self.stats.pooled_bytes -= capacity;
            drained.push(PoolBlock {
                ptr: cached.ptr,
                capacity,
            });
        }
        // Explicit trim is also the opportunity to discard unused size classes.
        self.free.retain(|_, blocks| !blocks.is_empty());
        drained
    }

    /// Record a successful CUDA free of a previously released/evicted block.
    pub(crate) fn record_free(&mut self, capacity: usize) {
        debug_assert!(self.stats.reserved_bytes >= capacity);
        self.stats.reserved_bytes -= capacity;
        self.stats.cuda_frees += 1;
    }

    pub(crate) fn note_retry(&mut self) {
        self.stats.allocation_retries += 1;
    }

    pub(crate) fn stats(&self) -> CudaPoolStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    // These integer-derived pointers are only opaque identities. Neither the
    // pool nor these tests dereference them or pass them to a device API.
    fn checkout(pool: &mut PoolState, n: usize, identity: usize) -> PoolBlock {
        pool.take(n).unwrap_or_else(|| {
            pool.note_malloc_attempt();
            pool.record_cold_allocation(n);
            PoolBlock {
                ptr: identity as *mut c_void,
                capacity: n,
            }
        })
    }

    fn return_block(pool: &mut PoolState, block: PoolBlock, request: usize) {
        let released = pool.release(block.ptr, request);
        assert_eq!(released.capacity, block.capacity);
        pool.retain(released, usize::MAX).unwrap();
    }

    fn seeded_pool(capacities: &[usize]) -> PoolState {
        let mut pool = PoolState::default();
        let blocks: Vec<_> = capacities
            .iter()
            .enumerate()
            .map(|(i, &n)| checkout(&mut pool, n, i + 1))
            .collect();
        // All seed allocations coexist before any returns, including equal
        // capacities, so each has a distinct identity.
        for block in blocks {
            return_block(&mut pool, block, block.capacity);
        }
        pool
    }

    #[test]
    fn larger_reuse_restores_capacity_across_smaller_and_exact_checkouts() {
        let mut pool = seeded_pool(&[2304]);
        let original = pool.free[&2304][0].ptr;
        for _ in 0..64 {
            let smaller = pool.take(2048).unwrap();
            assert_eq!(smaller.ptr, original);
            assert_eq!(smaller.capacity, 2304);
            assert_eq!(pool.stats().live_bytes, 2304);
            assert_eq!(pool.stats().reserved_bytes, 2304);
            assert_eq!(pool.oversized_live.len(), 1);
            return_block(&mut pool, smaller, 2048);
            assert!(pool.oversized_live.is_empty());

            let exact = pool.take(2304).unwrap();
            assert_eq!(exact.ptr, original);
            assert!(pool.oversized_live.is_empty());
            return_block(&mut pool, exact, 2304);
        }
        let stats = pool.stats();
        assert_eq!(stats.allocation_requests, 129);
        assert_eq!(stats.cache_hits, 128);
        assert_eq!(stats.larger_reuses, 64);
        assert_eq!(stats.cold_allocations, 1);
        assert_eq!(stats.live_bytes, 0);
        assert_eq!(stats.pooled_bytes, 2304);
        assert_eq!(stats.reserved_bytes, 2304);
    }

    #[test]
    fn exact_then_smallest_fit_skips_empty_buckets_and_includes_upper_bound() {
        let mut pool = seeded_pool(&[4608, 4096, 4352, 4864]);
        let exact = pool.take(4096).unwrap();
        let smallest_larger = pool.take(4096).unwrap();
        let upper_bound = pool.take(4096).unwrap();
        assert_eq!(exact.capacity, 4096);
        assert_eq!(smallest_larger.capacity, 4352);
        assert_eq!(upper_bound.capacity, 4608);
        assert!(pool.free[&4096].is_empty());
        assert!(pool.free[&4352].is_empty());
        assert!(pool.free[&4608].is_empty());
        // 4864 is present but above the inclusive 4096 + 4096/8 limit.
        assert!(pool.take(4096).is_none());
        assert_eq!(pool.stats().pooled_bytes, 4864);
        assert_eq!(pool.stats().larger_reuses, 2);
        assert!(
            pool.take(5120).is_none(),
            "a smaller block cannot satisfy an allocation"
        );
    }

    #[test]
    fn concurrent_checkouts_have_distinct_identities_and_return_independently() {
        let mut pool = seeded_pool(&[4096, 4096, 4096]);
        let blocks: Vec<_> = (0..3).map(|_| pool.take(3840).unwrap()).collect();
        let identities: HashSet<_> = blocks.iter().map(|block| block.ptr as usize).collect();
        assert_eq!(identities.len(), 3);
        assert_eq!(pool.oversized_live.len(), 3);
        assert!(pool.take(3840).is_none());
        return_block(&mut pool, blocks[1], 3840);
        let recycled = pool.take(3840).unwrap();
        assert_eq!(recycled.ptr, blocks[1].ptr);
        assert_ne!(recycled.ptr, blocks[0].ptr);
        assert_ne!(recycled.ptr, blocks[2].ptr);
        assert_eq!(pool.stats().live_bytes, 3 * 4096);
        assert_eq!(pool.stats().pooled_bytes, 0);
    }

    #[test]
    fn retention_limit_uses_actual_capacity_and_frees_are_recorded_separately() {
        let mut pool = seeded_pool(&[1024, 2304]);
        let exact = pool.take(1024).unwrap();
        let oversized = pool.take(2048).unwrap();
        let released = pool.release(exact.ptr, 1024);
        pool.retain(released, 3072).unwrap();
        let released = pool.release(oversized.ptr, 2048);
        // Logical sizes would fit (1024 + 2048); actual capacity does not.
        let rejected = pool.retain(released, 3072).unwrap_err();
        assert_eq!(rejected.capacity, 2304);
        assert_eq!(pool.stats().live_bytes, 0);
        assert_eq!(pool.stats().pooled_bytes, 1024);
        assert_eq!(pool.stats().reserved_bytes, 3328);
        assert_eq!(pool.stats().cuda_frees, 0);
        pool.record_free(rejected.capacity);
        assert_eq!(pool.stats().reserved_bytes, 1024);
        assert_eq!(pool.stats().cuda_frees, 1);
    }

    #[test]
    fn lowering_retention_below_existing_idle_bytes_rejects_without_underflow() {
        let mut pool = seeded_pool(&[1024, 2048]);
        let block = pool.take(1024).unwrap();
        let released = pool.release(block.ptr, 1024);
        assert!(pool.retain(released, 512).is_err());
        assert_eq!(pool.stats().pooled_bytes, 2048);
        assert_eq!(pool.stats().reserved_bytes, 3072);
    }

    #[test]
    fn retention_pressure_evicts_oldest_return_instead_of_smallest_size() {
        let mut pool = seeded_pool(&[4096, 1024, 2048]);
        let oldest = pool.free[&4096][0].ptr;
        let incoming = checkout(&mut pool, 3072, 4);
        let released = pool.release(incoming.ptr, 3072);
        let evicted = pool.retain_or_evict(released, 8192);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].ptr, oldest);
        assert_eq!(evicted[0].capacity, 4096);
        assert_eq!(pool.free[&1024].len(), 1);
        assert_eq!(pool.free[&2048].len(), 1);
        assert_eq!(pool.free[&3072][0].ptr, incoming.ptr);
        assert_eq!(pool.stats().pooled_bytes, 6144);
        assert_eq!(pool.stats().reserved_bytes, 10240);
        assert_eq!(pool.stats().cuda_frees, 0);
        pool.record_free(evicted[0].capacity);
        assert_eq!(pool.stats().reserved_bytes, 6144);
        assert_eq!(pool.stats().cuda_frees, 1);
    }

    #[test]
    fn a_new_return_refreshes_age_and_can_evict_multiple_older_buckets() {
        let mut pool = seeded_pool(&[4096, 1024, 2048]);
        let refreshed = pool.take(4096).unwrap();
        return_block(&mut pool, refreshed, 4096);
        let incoming = checkout(&mut pool, 3072, 4);
        let released = pool.release(incoming.ptr, 3072);
        let evicted = pool.retain_or_evict(released, 8192);
        assert_eq!(
            evicted.iter().map(|b| b.capacity).collect::<Vec<_>>(),
            [1024, 2048]
        );
        assert_eq!(pool.free[&4096][0].ptr, refreshed.ptr);
        assert_eq!(pool.free[&3072][0].ptr, incoming.ptr);
        assert_eq!(pool.stats().pooled_bytes, 7168);
        assert_eq!(pool.return_clock, 5);
    }

    #[test]
    fn equal_size_checkouts_are_lifo_but_pressure_evicts_oldest_in_bucket() {
        let mut pool = seeded_pool(&[2048, 1024, 2048]);
        let oldest = pool.free[&2048][0].ptr;
        let newest = pool.free[&2048][1].ptr;
        let checked_out = pool.take(2048).unwrap();
        assert_eq!(checked_out.ptr, newest);
        return_block(&mut pool, checked_out, 2048);
        let incoming = checkout(&mut pool, 1536, 4);
        let released = pool.release(incoming.ptr, 1536);
        let evicted = pool.retain_or_evict(released, 5120);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].ptr, oldest);
        assert_eq!(pool.free[&2048].len(), 1);
        assert_eq!(pool.take(2048).unwrap().ptr, newest);
    }

    #[test]
    fn lru_eviction_preserves_live_oversized_and_deferred_allocations() {
        let mut pool = seeded_pool(&[4096, 8192, 2048]);
        let live = pool.take(3840).unwrap();
        let pending = pool.take(8192).unwrap();
        let released = pool.release(pending.ptr, 8192);
        pool.defer(released);
        let incoming = checkout(&mut pool, 3072, 4);
        let released = pool.release(incoming.ptr, 3072);
        let evicted = pool.retain_or_evict(released, 4096);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].capacity, 2048);
        assert_ne!(evicted[0].ptr, live.ptr);
        assert_ne!(evicted[0].ptr, pending.ptr);
        assert_eq!(pool.oversized_live[&(live.ptr as usize)], 4096);
        assert_eq!(pool.deferred[0].ptr, pending.ptr);
        assert_eq!(pool.stats().live_bytes, 4096);
        assert_eq!(pool.stats().pending_bytes, 8192);
        assert_eq!(pool.stats().pooled_bytes, 3072);
        assert_eq!(pool.stats().reserved_bytes, 17408);
        pool.record_free(evicted[0].capacity);
        assert_eq!(pool.stats().reserved_bytes, 15360);
        assert_eq!(pool.release(live.ptr, 3840).capacity, 4096);
    }

    #[test]
    fn a_block_larger_than_the_budget_does_not_evict_the_hot_cache() {
        let mut pool = seeded_pool(&[1024, 2048]);
        let first = pool.free[&1024][0].ptr;
        let second = pool.free[&2048][0].ptr;
        let previous_clock = pool.return_clock;
        let incoming = checkout(&mut pool, 8192, 3);
        let released = pool.release(incoming.ptr, 8192);
        let rejected = pool.retain_or_evict(released, 4096);
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].ptr, incoming.ptr);
        assert_eq!(rejected[0].capacity, 8192);
        assert_eq!(pool.return_clock, previous_clock);
        assert_eq!(pool.stats().pooled_bytes, 3072);
        assert_eq!(pool.stats().reserved_bytes, 11264);
        pool.record_free(rejected[0].capacity);
        assert_eq!(pool.stats().reserved_bytes, 3072);
        assert_eq!(pool.take(1024).unwrap().ptr, first);
        assert_eq!(pool.take(2048).unwrap().ptr, second);
    }

    #[test]
    fn partial_trim_evicts_largest_idle_blocks_and_protects_live_allocations() {
        let mut pool = seeded_pool(&[1024, 2048, 4096, 8192]);
        let live = pool.take(1024).unwrap();
        let removed = pool.drain_to(6144);
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].capacity, 8192);
        assert_eq!(pool.stats().live_bytes, 1024);
        assert_eq!(pool.stats().pooled_bytes, 6144);
        assert_eq!(pool.stats().reserved_bytes, 15360);
        pool.record_free(removed[0].capacity);
        assert_eq!(pool.stats().reserved_bytes, 7168);

        let removed = pool.drain_to(0);
        assert_eq!(
            removed.iter().map(|b| b.capacity).collect::<Vec<_>>(),
            [4096, 2048]
        );
        for block in removed {
            assert_ne!(block.ptr, live.ptr);
            pool.record_free(block.capacity);
        }
        assert_eq!(pool.stats().reserved_bytes, 1024);
        assert_eq!(pool.stats().peak_reserved_bytes, 15360);
        assert!(pool.free.is_empty());
        return_block(&mut pool, live, 1024);
        assert_eq!(pool.take(1024).unwrap().ptr, live.ptr);
    }

    #[test]
    fn trim_skips_an_empty_largest_bucket() {
        let mut pool = seeded_pool(&[1024, 2048]);
        let live = pool.take(2048).unwrap();
        assert!(pool.free[&2048].is_empty());
        let removed = pool.drain_to(0);
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].capacity, 1024);
        assert_ne!(removed[0].ptr, live.ptr);
        assert!(pool.free.is_empty());
        assert_eq!(pool.stats().live_bytes, 2048);
        assert_eq!(pool.stats().reserved_bytes, 3072);
    }

    #[test]
    fn deferred_blocks_stay_reserved_until_successful_free_and_are_not_reused() {
        let mut pool = seeded_pool(&[1024, 2304]);
        let first = pool.take(1024).unwrap();
        let second = pool.take(2048).unwrap();
        let released = pool.release(first.ptr, 1024);
        pool.defer(released);
        let released = pool.release(second.ptr, 2048);
        pool.defer(released);
        assert_eq!(pool.stats().live_bytes, 0);
        assert_eq!(pool.stats().pooled_bytes, 0);
        assert_eq!(pool.stats().pending_bytes, 3328);
        assert_eq!(pool.stats().reserved_bytes, 3328);
        assert!(pool.take(1024).is_none());
        assert!(pool.drain_to(0).is_empty());
        assert_eq!(pool.stats().pending_bytes, 3328);

        let pending = pool.take_deferred();
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].ptr, first.ptr);
        assert_eq!(pending[1].ptr, second.ptr);
        assert_eq!(pending[1].capacity, 2304);
        assert_eq!(pool.stats().pending_bytes, 0);
        assert_eq!(pool.stats().reserved_bytes, 3328);
        assert!(pool.take_deferred().is_empty());
        // A caller can re-defer a block if it could not free it yet.
        pool.defer(pending[1]);
        pool.record_free(pending[0].capacity);
        assert_eq!(pool.stats().reserved_bytes, 2304);
        assert_eq!(pool.stats().pending_bytes, 2304);
        let retried = pool.take_deferred();
        pool.record_free(retried[0].capacity);
        assert_eq!(pool.stats().reserved_bytes, 0);
        assert_eq!(pool.stats().pending_bytes, 0);
        assert_eq!(pool.stats().cuda_frees, 2);
    }

    #[test]
    fn fixed_shape_reuses_bucket_and_pointer_vector_allocation() {
        let mut pool = seeded_pool(&[4096]);
        let vector_address = pool.free[&4096].as_ptr();
        let vector_capacity = pool.free[&4096].capacity();
        for _ in 0..100 {
            let block = pool.take(4096).unwrap();
            assert!(pool.free[&4096].is_empty());
            assert_eq!(pool.free[&4096].as_ptr(), vector_address);
            assert_eq!(pool.free[&4096].capacity(), vector_capacity);
            assert!(pool.oversized_live.is_empty());
            let released = pool.release(block.ptr, 4096);
            let evicted = pool.retain_or_evict(released, 4096);
            assert!(evicted.is_empty());
            assert_eq!(evicted.capacity(), 0);
            assert_eq!(pool.free[&4096].as_ptr(), vector_address);
        }
    }

    fn growing_shape_step(
        pool: &mut PoolState,
        next_identity: &mut usize,
        tokens: usize,
        limit: usize,
    ) {
        // Three different scratch buffers coexist for one operation. Their
        // roles retain different size scales while all grow with token count.
        let requests = [tokens * 256, tokens * 2048, tokens * 4096];
        let blocks = requests.map(|n| {
            let block = checkout(pool, n, *next_identity);
            *next_identity += 1;
            block
        });
        for (block, request) in blocks.into_iter().zip(requests) {
            let released = pool.release(block.ptr, request);
            for evicted in pool.retain_or_evict(released, limit) {
                pool.record_free(evicted.capacity);
            }
        }
        assert_eq!(pool.stats().live_bytes, 0);
        assert!(pool.stats().pooled_bytes <= limit);
        assert_eq!(pool.stats().pooled_bytes, pool.stats().reserved_bytes);
    }

    #[test]
    fn growing_three_buffer_trace_adapts_to_the_largest_working_set() {
        let mut pool = PoolState::default();
        let mut next_identity = 1;
        // Only the largest operation's three allocations fit, rather than all
        // 17 shapes. Its capacities are within 12.5% of the smallest request.
        let limit = 144 * (256 + 2048 + 4096);
        for tokens in 128..=144 {
            growing_shape_step(&mut pool, &mut next_identity, tokens, limit);
        }
        let warm = pool.stats();
        assert_eq!(warm.cold_allocations, 17 * 3);
        assert_eq!(warm.pooled_bytes, limit);
        assert!(
            warm.cuda_frees > 0,
            "growing shapes must evict obsolete buffers"
        );
        for _ in 0..5 {
            for tokens in 128..=144 {
                growing_shape_step(&mut pool, &mut next_identity, tokens, limit);
            }
        }
        let steady = pool.stats();
        assert_eq!(steady.cold_allocations, warm.cold_allocations);
        assert_eq!(steady.cuda_frees, warm.cuda_frees);
        assert_eq!(steady.cache_hits - warm.cache_hits, 5 * 17 * 3);
        assert_eq!(steady.reserved_bytes, limit);
    }

    fn allocation_trace(dynamic: bool) -> CudaPoolStats {
        let mut pool = PoolState::default();
        let large = [4096, 8192, 16384];
        let small = [3840, 7680, 15360];
        for iteration in 0..100 {
            let requests = if dynamic && iteration % 2 == 1 {
                small
            } else {
                large
            };
            let blocks: Vec<_> = requests
                .iter()
                .enumerate()
                .map(|(index, &n)| checkout(&mut pool, n, iteration * 3 + index + 1))
                .collect();
            assert_eq!(pool.stats().live_bytes, 28672);
            for (block, request) in blocks.into_iter().zip(requests) {
                return_block(&mut pool, block, request);
            }
        }
        pool.stats()
    }

    #[test]
    fn fixed_and_dynamic_traces_have_no_cold_allocations_after_warmup() {
        let expected = CudaPoolStats {
            allocation_requests: 300,
            cache_hits: 297,
            cuda_malloc_calls: 3,
            cold_allocations: 3,
            pooled_bytes: 28672,
            reserved_bytes: 28672,
            peak_live_bytes: 28672,
            peak_reserved_bytes: 28672,
            ..CudaPoolStats::default()
        };
        assert_eq!(allocation_trace(false), expected);
        assert_eq!(
            allocation_trace(true),
            CudaPoolStats {
                larger_reuses: 150,
                ..expected
            }
        );
    }

    #[test]
    fn best_fit_upper_bound_saturates_without_wrapping() {
        let capacity = usize::MAX & !255;
        let request = (usize::MAX / 16 * 15) & !255;
        assert!(request.checked_add(request / 8).is_none());
        let mut pool = seeded_pool(&[capacity]);
        let block = pool.take(request).unwrap();
        assert_eq!(block.capacity, capacity);
        assert_eq!(pool.stats().live_bytes, capacity);
        assert_eq!(pool.stats().reserved_bytes, capacity);
        return_block(&mut pool, block, request);
        assert_eq!(pool.stats().pooled_bytes, capacity);
    }

    #[test]
    fn failed_malloc_attempt_and_retry_do_not_reserve_capacity() {
        let mut pool = PoolState::default();
        assert!(pool.take(4096).is_none());
        pool.note_malloc_attempt();
        pool.note_retry();
        assert_eq!(pool.stats().reserved_bytes, 0);
        assert_eq!(pool.stats().live_bytes, 0);
        assert_eq!(pool.stats().cold_allocations, 0);
        pool.note_malloc_attempt();
        pool.record_cold_allocation(4096);
        assert_eq!(pool.stats().allocation_requests, 1);
        assert_eq!(pool.stats().cuda_malloc_calls, 2);
        assert_eq!(pool.stats().allocation_retries, 1);
        assert_eq!(pool.stats().cold_allocations, 1);
        assert_eq!(pool.stats().reserved_bytes, 4096);
        assert_eq!(pool.stats().live_bytes, 4096);
    }
}
