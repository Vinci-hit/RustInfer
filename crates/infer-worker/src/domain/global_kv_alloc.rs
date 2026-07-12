//! `GlobalKvAllocator` — sorted free-list over `[0, total)` token-slot indices.
//!
//! ## Design
//!
//! With `block_size = 1` the kernel does pure gather:
//! `block_table[seq][i] → arbitrary global index`. Whether the i-th slot
//! lives next to the (i+1)-th in the global pool is **architecturally
//! irrelevant**. Two non-contiguous indices `[0, 100]` produce the same
//! gather pattern (one cache line per token) as two contiguous `[0, 1]`.
//!
//! Allocator semantics:
//! - **alloc**: bump `head` over `free[head..]`, returning `n` indices in
//!   ascending order — O(n).
//! - **free**: drop the consumed prefix, append the freed indices, sort the
//!   tail. The user-facing contract requires that after every `free()` the
//!   pool is fully merged and sorted, so the next `alloc` always returns
//!   the smallest available indices. O(N log N) per `free()`, but freed
//!   batches are typically small.
//!
//! ## Invariant
//!
//! `free[head..]` holds every currently-free index exactly once, in
//! ascending order. `head == 0` after every `free()` call.
//!
//! ## Determinism
//!
//! Sort is unstable, but the input set is a unique collection of indices —
//! every allocator instance with the same op history sees the same set,
//! hence the same sorted result. Required for future TP/PP rank
//! consistency.

/// Returned when alloc cannot satisfy the request even after merging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocFull {
    pub need: u32,
    /// Slots immediately bump-able without merging.
    pub available: u32,
    /// Total free slots including not-yet-merged frees at the tail.
    pub total_free: u32,
}

impl std::fmt::Display for AllocFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GlobalKvAllocator full: need={} available={} total_free={}",
            self.need, self.available, self.total_free
        )
    }
}

impl std::error::Error for AllocFull {}

/// Bump allocator over `[0, total)`.
#[derive(Debug)]
pub struct GlobalKvAllocator {
    total: u32,
    /// Free slot pool. Bump head moves through `free[..]` left-to-right.
    /// Indices `< head` were allocated; indices `>= head` are available.
    /// `free()` calls push freed indices to the tail (they sit "behind"
    /// any not-yet-bump-allocated indices). `merge_and_sort` drops the
    /// consumed prefix and re-sorts to expose them.
    free: Vec<u32>,
    /// Bump pointer. Invariant: `head <= free.len()`.
    head: usize,
    /// Indices released by completed requests (not yet recycled into the
    /// free pool). These sit here until the next allocation attempt fails
    /// and triggers `recycle()`, which drains them into `free` and sorts.
    /// Used in real-time recycling mode (prefix caching disabled).
    released: Vec<u32>,
}

impl GlobalKvAllocator {
    /// Build an allocator covering `[0, total)`. All space is initially free.
    pub fn new(total: u32) -> Self {
        let mut free: Vec<u32> = Vec::with_capacity(total as usize);
        for i in 0..total {
            free.push(i);
        }
        Self {
            total,
            free,
            head: 0,
            released: Vec::new(),
        }
    }

    pub fn total(&self) -> u32 {
        self.total
    }

    /// Slots the next `alloc_indices` can take immediately without merging.
    pub fn available(&self) -> u32 {
        (self.free.len() - self.head) as u32
    }

    /// Total free slots (bump-available + released holding list). Equal to
    /// `total - outstanding`.
    pub fn total_free(&self) -> u32 {
        // `free[head..]` contains immediately allocable slots; `released`
        // holds slots returned by completed sequences but not yet recycled
        // into the free pool. Both are logically free from the budget's
        // perspective.
        (self.free.len() - self.head) as u32 + self.released.len() as u32
    }

    /// Outstanding = total − total_free.
    pub fn outstanding(&self) -> u32 {
        let total_free = self.total_free();
        if total_free > self.total {
            tracing::warn!(
                "[kv-alloc] invariant violation: total_free={} > total={}",
                total_free,
                self.total
            );
        }
        self.total.saturating_sub(total_free)
    }

    /// Allocate `n` indices.
    ///
    /// Fast path: bump from `head`. If that fails and there are released
    /// blocks, `recycle()` is called to drain them into the free pool,
    /// followed by a retry. The slow merge-then-retry path remains for
    /// paranoia but is unreachable in practice — `free()` already sorts
    /// the pool, so `free[head..]` is always a contiguous run of every
    /// free index in ascending order. Returns `AllocFull` only on true
    /// OOM (`total_free < n`).
    pub fn alloc_indices(&mut self, n: u32) -> Result<Vec<u32>, AllocFull> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let n_usize = n as usize;

        // Fast path: enough free at the head.
        if self.head + n_usize <= self.free.len() {
            let out = self.free[self.head..self.head + n_usize].to_vec();
            self.head += n_usize;
            return Ok(out);
        }

        // Try recycling released blocks before giving up.
        if !self.released.is_empty() {
            self.recycle();
            if self.head + n_usize <= self.free.len() {
                let out = self.free[self.head..self.head + n_usize].to_vec();
                self.head += n_usize;
                return Ok(out);
            }
        }

        // Slow path: after free()-time sort this is unreachable in
        // practice; kept for paranoia. If the fast path failed and yet
        // `total_free >= n`, something has bypassed the `free()` sort —
        // we still attempt to recover.
        if self.total_free() >= n {
            self.merge_and_sort();
            debug_assert!(self.free.len() >= n_usize);
            debug_assert_eq!(self.head, 0);
            let out = self.free[..n_usize].to_vec();
            self.head = n_usize;
            return Ok(out);
        }

        Err(AllocFull {
            need: n,
            available: self.available(),
            total_free: self.total_free(),
        })
    }

    /// Return freed indices to the pool. Drops the already-allocated
    /// prefix, sorts only the returned batch, then merges it with the
    /// already-sorted free pool. This keeps allocations deterministic
    /// without re-sorting the entire free pool on every completion.
    ///
    /// Caller is trusted to free only previously-allocated indices.
    /// Debug builds verify each input is `< total`; release builds skip
    /// the check.
    pub fn free(&mut self, indices: &[u32]) {
        if indices.is_empty() {
            return;
        }
        // P0: Replace O(N) `drain(..head)` memmove with O(1) pointer
        // copy-down only when head is a significant fraction of the vec.
        // For small head values the drain was already fast; for large ones
        // we swap to a truncation that avoids the memmove.
        self.compact_head();
        let returned = self.sanitize_returned_indices(indices, "free");
        if returned.is_empty() {
            return;
        }
        self.merge_sorted_returned(&returned);
    }

    fn merge_sorted_returned(&mut self, returned: &[u32]) {
        if returned.is_empty() {
            return;
        }

        // P0: Merge in-place when possible, avoiding a second Vec
        // allocation. We reserve exactly enough, then merge backwards
        // from the tail so we never overwrite an unread source element.
        let old_len = self.free.len();
        let new_len = old_len + returned.len();
        self.free.resize(new_len, 0);

        let mut i = old_len; // read cursor in old free (past-the-end)
        let mut j = returned.len(); // read cursor in returned (past-the-end)
        let mut w = new_len; // write cursor (past-the-end)

        while i > 0 && j > 0 {
            w -= 1;
            if self.free[i - 1] >= returned[j - 1] {
                self.free[w] = self.free[i - 1];
                i -= 1;
            } else {
                self.free[w] = returned[j - 1];
                j -= 1;
            }
        }
        // Only one of these loops will execute.
        while j > 0 {
            w -= 1;
            j -= 1;
            self.free[w] = returned[j];
        }
        // If i > 0, those elements are already in place at the front.
        debug_assert_eq!(w, i);
    }

    /// Filter out-of-range, already-free, and already-released indices.
    /// A double-freed index silently duplicates a slot in the pool and the
    /// next alloc hands the same KV slot to two sequences — validation runs
    /// in release builds too. Cost per index: O(log n) `binary_search` on
    /// the sorted free pool plus O(m) over the (in practice empty)
    /// `released` list — negligible next to the O(n) merge in `free()`.
    fn sanitize_returned_indices(&self, indices: &[u32], op: &str) -> Vec<u32> {
        let mut returned = indices.to_vec();
        returned.sort_unstable();
        returned.dedup();

        let mut valid = Vec::with_capacity(returned.len());
        for idx in returned {
            if idx >= self.total {
                tracing::warn!(
                    "[kv-alloc] ignoring {} index out of range: idx={} total={}",
                    op,
                    idx,
                    self.total
                );
                continue;
            }
            if self.free[self.head..].binary_search(&idx).is_ok() {
                tracing::warn!(
                    "[kv-alloc] ignoring {} index that is already free: idx={}",
                    op,
                    idx
                );
                continue;
            }
            if self.released.contains(&idx) {
                tracing::warn!(
                    "[kv-alloc] ignoring {} index that is already released: idx={}",
                    op,
                    idx
                );
                continue;
            }
            valid.push(idx);
        }
        valid
    }

    /// Move indices to the released holding list (real-time recycling mode).
    ///
    /// Unlike `free()`, released indices are NOT immediately merged into the
    /// free pool. They stay in a separate holding list until the next
    /// allocation attempt fails, at which point `recycle()` drains them
    /// into the free pool and sorts.
    ///
    /// Caller is trusted to release only previously-allocated indices.
    pub fn release(&mut self, indices: &[u32]) {
        if indices.is_empty() {
            return;
        }
        let returned = self.sanitize_returned_indices(indices, "release");
        self.released.extend_from_slice(&returned);
    }

    /// Drain the released holding list into the free pool, drop any
    /// consumed prefix, and sort so the pool is ready for the next
    /// `alloc_indices`. Returns the number of indices recycled.
    pub fn recycle(&mut self) -> usize {
        if self.released.is_empty() {
            return 0;
        }
        let n = self.released.len();
        self.compact_head();
        self.free.append(&mut self.released);
        self.free.sort_unstable();
        n
    }

    /// Number of indices currently held in the released list (not yet recycled).
    pub fn released_len(&self) -> usize {
        self.released.len()
    }

    /// Release a finished/evicted sequence's block table back to the pool.
    ///
    /// With prefix caching enabled the slots stay pinned by the
    /// scheduler-side RadixTree, so this is a no-op. With it disabled the
    /// slots are returned via `free()` — merged into the free pool
    /// **immediately** so the very next `alloc_indices` (e.g. the same decode
    /// step admitting a new row) can reuse them. Empty tables are ignored.
    ///
    /// This is the single choke point for the "free a sequence's KV" decision
    /// that was previously copy-pasted across serve_loop / kv_relief /
    /// worker_scheduler.
    ///
    /// Eager merge (A1): the older `release()` path parked slots in a holding
    /// list that was only drained on the next allocation *failure*. Under
    /// continuous batching that starved the decode batch — completed
    /// sequences freed slots that stayed invisible while `alloc_indices`
    /// succeeded via bump, so the batch filled below capacity. `free()` makes
    /// freed slots allocatable on the spot.
    pub fn release_owned(&mut self, block_table: &[u32], enable_prefix_caching: bool) {
        if enable_prefix_caching || block_table.is_empty() {
            return;
        }
        self.free(block_table);
    }

    /// Drop the consumed prefix and sort the remaining free indices.
    /// Pool is fully sorted after every `free()` call; this method stays
    /// public for explicit "compact while idle" patterns and tests but
    /// in steady state it's a no-op.
    pub fn merge_and_sort(&mut self) {
        self.compact_head();
        self.free.sort_unstable();
    }

    /// P0: Efficient head compaction. When `head` is past the midpoint,
    /// `copy_within` + `truncate` is cheaper than `drain`'s element-by-element
    /// shift. For small heads, `drain` is fine; for large heads we avoid the
    /// O(N) memmove by using the already-O(N) `copy_within` that memcpy's in
    /// one shot. Either way, head resets to 0.
    fn compact_head(&mut self) {
        if self.head == 0 {
            return;
        }
        let remaining = self.free.len() - self.head;
        if remaining == 0 {
            self.free.clear();
        } else {
            self.free.copy_within(self.head.., 0);
            self.free.truncate(remaining);
        }
        self.head = 0;
    }

    /// Bump head position (test/debug).
    pub fn head(&self) -> usize {
        self.head
    }

    /// Snapshot of currently allocatable indices in bump order.
    /// Test-only; allocates a copy.
    pub fn free_snapshot(&self) -> Vec<u32> {
        self.free[self.head..].to_vec()
    }

    /// Allocate `n` indices as a [`KvLease`] that must be explicitly committed
    /// or released. Prefer this over [`Self::alloc_indices`] anywhere the slots
    /// live across function/step boundaries before finding an owner.
    pub fn lease(&mut self, n: u32) -> Result<KvLease, AllocFull> {
        self.alloc_indices(n).map(|slots| KvLease { slots })
    }
}

/// A batch of KV slot indices checked out of [`GlobalKvAllocator`] that MUST
/// be explicitly consumed — committed into an owning block table
/// ([`KvLease::commit`]) or returned to the pool ([`KvLease::release`]).
///
/// The decode/fused pipelines hold slots in "outstanding-but-unowned" windows
/// (speculative next-step reservations, in-flight step slots, prefill base
/// slots) whose reclamation used to be enforced by call-ordering comments —
/// the seam the historical double-free/leak incidents grew from. A lease makes
/// the obligation structural: dropping a non-empty lease panics in debug
/// builds and logs an error in release builds instead of silently shrinking
/// the pool.
#[must_use = "a KvLease must be committed or released, or its slots leak"]
#[derive(Debug, Default, PartialEq, Eq)]
pub struct KvLease {
    slots: Vec<u32>,
}

impl KvLease {
    pub fn empty() -> Self {
        Self { slots: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.slots
    }

    /// Move the lease out of a struct field, leaving an empty lease behind.
    pub fn take(&mut self) -> KvLease {
        KvLease {
            slots: std::mem::take(&mut self.slots),
        }
    }

    /// Consume the lease: ownership of the slots has been transferred to a
    /// live owner (a sequence block table / the prefilling map). The returned
    /// indices are for the caller's bookkeeping; dropping them is fine.
    pub fn commit(mut self) -> Vec<u32> {
        std::mem::take(&mut self.slots)
    }

    /// Consume the lease by returning every slot to the pool. No-op if empty.
    pub fn release(mut self, alloc: &mut GlobalKvAllocator) {
        let slots = std::mem::take(&mut self.slots);
        if !slots.is_empty() {
            alloc.free(&slots);
        }
    }

    /// Keep the first `keep` slots, returning the surplus tail to the pool.
    pub fn shrink_to(&mut self, keep: usize, alloc: &mut GlobalKvAllocator) {
        if self.slots.len() > keep {
            let surplus = self.slots.split_off(keep);
            alloc.free(&surplus);
        }
    }
}

impl Drop for KvLease {
    fn drop(&mut self) {
        if !self.slots.is_empty() {
            debug_assert!(
                false,
                "KvLease dropped with {} unconsumed slots — commit() or release() it",
                self.slots.len()
            );
            tracing::error!(
                slots = self.slots.len(),
                "KvLease dropped without commit/release; KV pool leaked"
            );
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn new_pool_starts_full() {
        let a = GlobalKvAllocator::new(64);
        assert_eq!(a.total(), 64);
        assert_eq!(a.outstanding(), 0);
        assert_eq!(a.total_free(), 64);
        assert_eq!(a.available(), 64);
    }

    #[test]
    fn alloc_bump_returns_ascending_indices_initially() {
        let mut a = GlobalKvAllocator::new(100);
        let v = a.alloc_indices(5).unwrap();
        assert_eq!(v, vec![0, 1, 2, 3, 4]);
        assert_eq!(a.outstanding(), 5);
        let v = a.alloc_indices(3).unwrap();
        assert_eq!(v, vec![5, 6, 7]);
        assert_eq!(a.outstanding(), 8);
    }

    #[test]
    fn alloc_advances_head_no_other_state_change() {
        let mut a = GlobalKvAllocator::new(10);
        let snap_before = a.free_snapshot();
        a.alloc_indices(3).unwrap();
        // After 3-slot bump, the snapshot is just the tail.
        assert_eq!(a.free_snapshot(), snap_before[3..]);
    }

    #[test]
    fn free_sorts_immediately_keeps_head_zero() {
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_indices(5).unwrap(); // [0..5), head=5
        a.free(&[3, 1, 4]);
        // free() drains the consumed prefix and re-sorts: free pool now is
        // [1, 3, 4, 5, 6, 7, 8, 9] with head reset to 0.
        assert_eq!(a.head(), 0, "free() resets head to zero");
        assert_eq!(a.outstanding(), 2, "5 alloc'd − 3 freed = 2 outstanding");
        assert_eq!(a.free_snapshot(), vec![1, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn free_ignores_invalid_and_already_free_indices() {
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_indices(5).unwrap(); // allocated [0..5), free [5..10)

        a.free(&[1, 1, 8, 99]);

        assert_eq!(a.outstanding(), 4);
        assert_eq!(a.free_snapshot(), vec![1, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn release_ignores_invalid_and_already_free_indices() {
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_indices(5).unwrap(); // allocated [0..5), free [5..10)

        a.release(&[1, 1, 8, 99]);

        assert_eq!(a.released_len(), 1);
        assert_eq!(a.recycle(), 1);
        assert_eq!(a.outstanding(), 4);
        assert_eq!(a.free_snapshot(), vec![1, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn alloc_after_free_returns_smallest_available() {
        // Under the new "sort on free" contract, alloc always returns the
        // smallest currently-free indices in ascending order — no lazy
        // tail-sit-then-merge.
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_indices(5).unwrap(); // [0,1,2,3,4], head=5
        a.free(&[0, 1, 2]);
        // Pool is now sorted to [0, 1, 2, 5, 6, 7, 8, 9], head=0.
        let v = a.alloc_indices(3).unwrap();
        assert_eq!(
            v,
            vec![0, 1, 2],
            "next alloc takes the smallest free indices"
        );
        assert_eq!(a.head(), 3);
        assert_eq!(a.outstanding(), 5);
    }

    #[test]
    fn fragmented_free_alloc_recovers_all_smallest() {
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_indices(10).unwrap();
        // Free a non-monotone batch: free() must produce a sorted pool.
        a.free(&[7, 1, 5, 3]);
        // Pool sorted to [1, 3, 5, 7], head=0.
        assert_eq!(a.free_snapshot(), vec![1, 3, 5, 7]);
        let v = a.alloc_indices(4).unwrap();
        assert_eq!(v, vec![1, 3, 5, 7]);
        assert_eq!(a.outstanding(), 10);
    }

    #[test]
    fn alloc_full_only_on_true_oom() {
        let mut a = GlobalKvAllocator::new(5);
        let _ = a.alloc_indices(5).unwrap();
        let err = a.alloc_indices(1).unwrap_err();
        assert_eq!(err.need, 1);
        assert_eq!(err.available, 0);
        assert_eq!(err.total_free, 0);
    }

    #[test]
    fn alloc_full_when_total_free_below_request() {
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_indices(10).unwrap();
        a.free(&[0, 1]); // total_free = 2
        let err = a.alloc_indices(3).unwrap_err();
        assert_eq!(err.need, 3);
        assert_eq!(err.total_free, 2);
    }

    #[test]
    fn alloc_zero_returns_empty() {
        let mut a = GlobalKvAllocator::new(10);
        let v = a.alloc_indices(0).unwrap();
        assert!(v.is_empty());
        assert_eq!(a.head(), 0);
    }

    #[test]
    fn merge_and_sort_drops_consumed_prefix_and_sorts_rest() {
        // In normal use, free()-time sorting has already reset head to 0.
        // Drive head forward first so this test covers the merge path too.
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_indices(7).unwrap(); // head=7
        // Use the public API; free() sorts these entries into place.
        a.free(&[5, 2, 0]);
        // Already sorted by free(); merge_and_sort is a no-op.
        a.merge_and_sort();
        assert_eq!(a.head(), 0);
        assert_eq!(a.free_snapshot(), vec![0, 2, 5, 7, 8, 9]);
    }

    #[test]
    fn merge_idempotent_when_head_zero_and_already_sorted() {
        let mut a = GlobalKvAllocator::new(10);
        let snap_before = a.free_snapshot();
        a.merge_and_sort();
        assert_eq!(a.free_snapshot(), snap_before);
        assert_eq!(a.head(), 0);
    }

    #[test]
    fn fuzz_no_duplicates_no_loss_no_panic() {
        // Deterministic xorshift RNG — avoid pulling proptest just for this.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let total: u32 = 256;
        let mut a = GlobalKvAllocator::new(total);
        let mut outstanding: Vec<Vec<u32>> = Vec::new();
        let mut union: HashSet<u32> = HashSet::new();

        for _ in 0..1_000 {
            let coin = (rng() % 100) as u32;
            if coin < 60 || outstanding.is_empty() {
                let n = (rng() % 20) as u32 + 1;
                if let Ok(v) = a.alloc_indices(n) {
                    for &idx in &v {
                        assert!(idx < total);
                        assert!(union.insert(idx), "duplicate {}", idx);
                    }
                    outstanding.push(v);
                }
            } else {
                let i = (rng() as usize) % outstanding.len();
                let v = outstanding.swap_remove(i);
                for &idx in &v {
                    assert!(union.remove(&idx));
                }
                a.free(&v);
            }
            // Invariants every iteration.
            assert_eq!(a.outstanding() as usize, union.len());
            assert_eq!(a.outstanding() + a.total_free(), total);
        }

        // Free everything; allocator returns to empty.
        for v in outstanding.drain(..) {
            for &idx in &v {
                assert!(union.remove(&idx));
            }
            a.free(&v);
        }
        assert_eq!(a.outstanding(), 0);
        assert_eq!(a.total_free(), total);
    }

    #[test]
    fn deterministic_replay() {
        // Two allocators given the same op trace must produce identical
        // outputs. Required for TP/PP rank consistency.
        let total: u32 = 64;
        let ops: &[(bool, u32)] = &[
            (true, 10), // alloc
            (true, 8),
            (false, 1), // free the second alloc
            (true, 12),
            (true, 30), // forces merge_and_sort
            (true, 4),
        ];
        let mut a = GlobalKvAllocator::new(total);
        let mut b = GlobalKvAllocator::new(total);
        let mut a_history: Vec<Vec<u32>> = vec![];
        let mut b_history: Vec<Vec<u32>> = vec![];
        for &(is_alloc, n) in ops {
            if is_alloc {
                a_history.push(a.alloc_indices(n).unwrap());
                b_history.push(b.alloc_indices(n).unwrap());
            } else {
                a.free(&a_history[1]);
                b.free(&b_history[1]);
            }
        }
        assert_eq!(a_history, b_history);
        assert_eq!(a.outstanding(), b.outstanding());
        assert_eq!(a.head(), b.head());
        assert_eq!(a.free_snapshot(), b.free_snapshot());
    }

    #[test]
    fn large_pool_alloc_fragmentation_resilience() {
        // Replay the failure scenario from Phase 7B-1 bench: allocate
        // many small ranges, free non-adjacent ones, then ask for a
        // larger contiguous-feeling allocation. New design: total_free
        // is the only ceiling; non-contiguous is a non-issue.
        let mut a = GlobalKvAllocator::new(100);
        let _ = a.alloc_indices(100).unwrap(); // exhaust
        // Free 4 disjoint windows.
        a.free(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        a.free(&[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]);
        a.free(&[40, 41, 42, 43, 44, 45, 46, 47, 48, 49]);
        a.free(&[60, 61, 62, 63, 64, 65, 66, 67, 68, 69]);
        // total_free = 40, asking for 25 (would have failed on the old
        // alloc_segment with largest_free=10).
        let v = a.alloc_indices(25).unwrap();
        assert_eq!(v.len(), 25);
        // No duplicates in the returned list.
        let s: HashSet<u32> = v.iter().copied().collect();
        assert_eq!(s.len(), 25);
        assert_eq!(a.outstanding(), 100 - 40 + 25);
    }
}
