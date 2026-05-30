//! `GlobalKvAllocator` — token-level bump allocator for the worker-owned
//! global KV cache.
//!
//! The KV cache pool is a single contiguous tensor `[total_tokens, kv_dim]`
//! per layer; this allocator hands out *segments* of that flat space. Each
//! segment is a half-open range `[base, base+n)` of global token indices,
//! which the scatter / attention kernels consume verbatim as `block_tables`
//! when `block_size == 1`.
//!
//! ## Why segments instead of per-token slots
//!
//! The scheduler reserves token-count budget up front (`KvBudget::try_reserve`).
//! At step time the worker knows exactly how many new tokens will be written
//! across the entire batch, and asks for **one** contiguous `[base, base+N)`
//! range. Splitting that range across the batch's sequences is then pure
//! arithmetic. This collapses what was a per-sequence allocation problem into
//! a single first-fit lookup.
//!
//! ## Free-range representation
//!
//! `free_ranges: BTreeMap<base, len>`:
//! - sorted by base for deterministic first-fit
//! - adjacent ranges are merged on `free` so the map stays compact
//! - empty pool ⇒ single entry `(0, total)`
//! - fully allocated ⇒ empty map
//!
//! ## Determinism
//!
//! `alloc_segment` is strictly first-fit on the lowest-base free range that
//! fits. No randomness, no thread-local state. Two allocators with the same
//! `total` and the same operation history will produce identical bases —
//! required for future TP/PP rank consistency.
//!
//! ## Failure mode
//!
//! `alloc_segment(n) -> Result<u32, AllocFull>`. The scheduler is supposed to
//! have reserved capacity through `KvBudget` before sending the step, so this
//! should be unreachable on the happy path. Returning `Err` instead of
//! panicking lets the worker fail-fast and report `step_alloc_fail_total` to
//! the scheduler for re-sync.

use std::collections::BTreeMap;

/// Returned when no free range is large enough for the requested segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocFull {
    pub need: u32,
    pub largest_free: u32,
    pub total_free: u32,
}

impl std::fmt::Display for AllocFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GlobalKvAllocator full: need={} largest_free={} total_free={}",
            self.need, self.largest_free, self.total_free
        )
    }
}

impl std::error::Error for AllocFull {}

/// Token-level bump allocator over `[0, total)`.
#[derive(Debug)]
pub struct GlobalKvAllocator {
    total: u32,
    /// Maps `base → len`. Invariant: ranges are non-empty, non-overlapping,
    /// and never adjacent (always coalesced after `free`).
    free_ranges: BTreeMap<u32, u32>,
}

impl GlobalKvAllocator {
    /// Build an allocator covering `[0, total)`. All space is initially free.
    pub fn new(total: u32) -> Self {
        let mut free_ranges = BTreeMap::new();
        if total > 0 {
            free_ranges.insert(0, total);
        }
        Self { total, free_ranges }
    }

    pub fn total(&self) -> u32 {
        self.total
    }

    /// Tokens currently outstanding (allocated, not yet freed).
    pub fn outstanding(&self) -> u32 {
        let free: u32 = self.free_ranges.values().copied().sum();
        self.total - free
    }

    /// Total free tokens across all free ranges.
    pub fn total_free(&self) -> u32 {
        self.free_ranges.values().copied().sum()
    }

    /// Number of free range fragments. Used as a fragmentation metric.
    pub fn free_ranges_count(&self) -> usize {
        self.free_ranges.len()
    }

    /// Snapshot of `(base, len)` pairs in ascending base order. Test/debug only.
    pub fn free_ranges_snapshot(&self) -> Vec<(u32, u32)> {
        self.free_ranges.iter().map(|(&b, &l)| (b, l)).collect()
    }

    /// Allocate one contiguous segment of `n` tokens. Returns the base of the
    /// segment; the segment is `[base, base+n)`.
    ///
    /// First-fit on the lowest-base free range that fits. Deterministic.
    /// **Strict**: fails if no single free range is large enough, even when
    /// `total_free >= n`. Use [`alloc_indices`] when caller can accept a
    /// non-contiguous result (the common case for the worker, where
    /// per-token `block_table[seq][i]` lookups don't require contiguity).
    pub fn alloc_segment(&mut self, n: u32) -> Result<u32, AllocFull> {
        if n == 0 {
            // Zero-sized allocation: harmless, but pick the lowest base for
            // deterministic behavior. If totally empty, base is 0 by convention.
            return Ok(self.free_ranges.keys().next().copied().unwrap_or(0));
        }
        // First-fit on ascending base order.
        let (base, len) = match self
            .free_ranges
            .iter()
            .find(|&(_, &len)| len >= n)
            .map(|(&b, &l)| (b, l))
        {
            Some(pair) => pair,
            None => {
                let largest_free = self.free_ranges.values().copied().max().unwrap_or(0);
                let total_free = self.total_free();
                return Err(AllocFull {
                    need: n,
                    largest_free,
                    total_free,
                });
            }
        };
        // Carve the front of this range.
        self.free_ranges.remove(&base);
        if len > n {
            self.free_ranges.insert(base + n, len - n);
        }
        Ok(base)
    }

    /// Allocate `n` indices, possibly drawn from multiple free ranges.
    /// Returns the indices in ascending address order. Fails only when
    /// `total_free < n`.
    ///
    /// This is the right choice when the caller writes the resulting list
    /// into a per-token `block_table` — those entries never need to be
    /// contiguous because `block_table[seq][i]` is looked up independently
    /// for each token. By relaxing contiguity we eliminate the
    /// fragmentation-induced `AllocFull` failures observed under heavy
    /// burst load (where `total_free` was 43k but largest single range was
    /// only 33 slots).
    ///
    /// Strategy: walk free ranges in ascending order, take the first
    /// `n` slots across as many ranges as needed. Always carves from the
    /// front of each consumed range.
    pub fn alloc_indices(&mut self, n: u32) -> Result<Vec<u32>, AllocFull> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let total_free = self.total_free();
        if total_free < n {
            let largest_free = self.free_ranges.values().copied().max().unwrap_or(0);
            return Err(AllocFull {
                need: n,
                largest_free,
                total_free,
            });
        }
        let mut out: Vec<u32> = Vec::with_capacity(n as usize);
        let mut remaining = n;
        // Take from the lowest-base ranges first. We collect the bases up
        // front (immutable iter) then mutate the map below.
        let to_take: Vec<(u32, u32)> = self
            .free_ranges
            .iter()
            .map(|(&b, &l)| (b, l))
            .collect();
        for (base, len) in to_take {
            if remaining == 0 {
                break;
            }
            let take = remaining.min(len);
            for k in 0..take {
                out.push(base + k);
            }
            self.free_ranges.remove(&base);
            if len > take {
                self.free_ranges.insert(base + take, len - take);
            }
            remaining -= take;
        }
        debug_assert_eq!(remaining, 0);
        Ok(out)
    }

    /// Free a list of token indices. The list does NOT need to be sorted or
    /// contiguous; this routine sorts, dedups, and merges adjacent runs into
    /// segments before splicing them back into `free_ranges`.
    ///
    /// Panics in debug builds if any index is out of range or already free.
    pub fn free(&mut self, indices: &[u32]) {
        if indices.is_empty() {
            return;
        }
        let mut sorted: Vec<u32> = indices.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        // Compress sorted indices into runs `[(base, len), ...]`.
        let mut runs: Vec<(u32, u32)> = Vec::with_capacity(sorted.len());
        let mut i = 0usize;
        while i < sorted.len() {
            let base = sorted[i];
            let mut len = 1u32;
            let mut j = i + 1;
            while j < sorted.len() && sorted[j] == base + len {
                len += 1;
                j += 1;
            }
            debug_assert!(
                base < self.total && base + len <= self.total,
                "free index out of range: base={} len={} total={}",
                base,
                len,
                self.total
            );
            runs.push((base, len));
            i = j;
        }

        // Splice each run into free_ranges, merging neighbors.
        for (base, len) in runs {
            self.insert_and_coalesce(base, len);
        }
    }

    /// Free a half-open contiguous range `[base, base+len)` directly. Avoids
    /// the sort+dedup pass when the caller already knows the indices form a
    /// segment (the common case after a step).
    pub fn free_segment(&mut self, base: u32, len: u32) {
        if len == 0 {
            return;
        }
        debug_assert!(
            base < self.total && base.checked_add(len).is_some_and(|e| e <= self.total),
            "free_segment out of range: base={} len={} total={}",
            base,
            len,
            self.total
        );
        self.insert_and_coalesce(base, len);
    }

    /// Insert `[base, base+len)` into `free_ranges` and coalesce with any
    /// adjacent ranges. Assumes the new range does not overlap an existing
    /// free range (debug-asserted).
    fn insert_and_coalesce(&mut self, mut base: u32, mut len: u32) {
        // Coalesce with predecessor if adjacent.
        if let Some((&p_base, &p_len)) = self.free_ranges.range(..base).next_back() {
            debug_assert!(
                p_base + p_len <= base,
                "free range overlap: predecessor ({},{}) vs new ({},{})",
                p_base,
                p_len,
                base,
                len
            );
            if p_base + p_len == base {
                self.free_ranges.remove(&p_base);
                base = p_base;
                len += p_len;
            }
        }
        // Coalesce with successor if adjacent.
        if let Some((&s_base, &s_len)) = self.free_ranges.range(base..).next() {
            debug_assert!(
                base + len <= s_base,
                "free range overlap: new ({},{}) vs successor ({},{})",
                base,
                len,
                s_base,
                s_len
            );
            if base + len == s_base {
                self.free_ranges.remove(&s_base);
                len += s_len;
            }
        }
        self.free_ranges.insert(base, len);
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn new_pool_has_single_full_range() {
        let a = GlobalKvAllocator::new(64);
        assert_eq!(a.outstanding(), 0);
        assert_eq!(a.total_free(), 64);
        assert_eq!(a.free_ranges_snapshot(), vec![(0, 64)]);
    }

    #[test]
    fn sequential_alloc_returns_contiguous_bases() {
        let mut a = GlobalKvAllocator::new(100);
        assert_eq!(a.alloc_segment(10).unwrap(), 0);
        assert_eq!(a.alloc_segment(20).unwrap(), 10);
        assert_eq!(a.alloc_segment(5).unwrap(), 30);
        assert_eq!(a.outstanding(), 35);
        assert_eq!(a.free_ranges_snapshot(), vec![(35, 65)]);
    }

    #[test]
    fn alloc_full_returns_err() {
        let mut a = GlobalKvAllocator::new(10);
        a.alloc_segment(10).unwrap();
        let err = a.alloc_segment(1).unwrap_err();
        assert_eq!(err.need, 1);
        assert_eq!(err.largest_free, 0);
        assert_eq!(err.total_free, 0);
    }

    #[test]
    fn alloc_too_large_when_partial_free() {
        let mut a = GlobalKvAllocator::new(10);
        a.alloc_segment(7).unwrap();
        let err = a.alloc_segment(5).unwrap_err();
        assert_eq!(err.need, 5);
        assert_eq!(err.largest_free, 3);
    }

    #[test]
    fn free_after_alloc_first_fit_fills_lowest_hole() {
        let mut a = GlobalKvAllocator::new(100);
        let _ = a.alloc_segment(10).unwrap(); // [0..10)
        let _ = a.alloc_segment(10).unwrap(); // [10..20)
        let c = a.alloc_segment(10).unwrap(); //  [20..30)
        // Free middle.
        a.free_segment(10, 10);
        // Next alloc 5 → fills the hole at 10, not 30.
        assert_eq!(a.alloc_segment(5).unwrap(), 10);
        assert_eq!(a.alloc_segment(5).unwrap(), 15);
        // Hole is exactly filled now; next alloc must come from tail.
        assert_eq!(a.alloc_segment(1).unwrap(), c + 10);
    }

    #[test]
    fn free_coalesces_with_predecessor_and_successor() {
        let mut a = GlobalKvAllocator::new(100);
        a.alloc_segment(20).unwrap(); // [0..20)
        a.alloc_segment(20).unwrap(); // [20..40)
        a.alloc_segment(20).unwrap(); // [40..60)
        // free both edges first → leaves [20..40) allocated, holes (0,20) and (40,60).
        a.free_segment(0, 20);
        a.free_segment(40, 20);
        assert_eq!(
            a.free_ranges_snapshot(),
            vec![(0, 20), (40, 60)]
        );
        // free middle → must coalesce all three into a single (0, 100).
        a.free_segment(20, 20);
        assert_eq!(a.free_ranges_snapshot(), vec![(0, 100)]);
        assert_eq!(a.outstanding(), 0);
    }

    #[test]
    fn free_with_unsorted_scattered_indices_merges_runs() {
        let mut a = GlobalKvAllocator::new(64);
        a.alloc_segment(64).unwrap();
        // Free a scattered set: 5, 6, 7, 0, 1, 30, 31.
        a.free(&[5, 6, 7, 0, 1, 30, 31]);
        // After dedup+sort+run-compression: (0,2) (5,3) (30,2). None merge with each other.
        assert_eq!(a.free_ranges_snapshot(), vec![(0, 2), (5, 3), (30, 2)]);
    }

    #[test]
    fn free_dedup_handles_repeated_indices() {
        let mut a = GlobalKvAllocator::new(16);
        a.alloc_segment(16).unwrap();
        a.free(&[3, 3, 4, 4, 4, 5]);
        assert_eq!(a.free_ranges_snapshot(), vec![(3, 3)]);
    }

    #[test]
    fn alloc_zero_is_noop_and_returns_a_valid_base() {
        let mut a = GlobalKvAllocator::new(8);
        let b = a.alloc_segment(0).unwrap();
        assert!(b <= a.total());
        assert_eq!(a.outstanding(), 0);
    }

    #[test]
    fn fuzz_random_alloc_free_invariants() {
        // Deterministic xorshift-style RNG (avoid pulling proptest dep just for this).
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let total: u32 = 256;
        let mut a = GlobalKvAllocator::new(total);
        // Track every outstanding (base, len) so we can free correctly and
        // assert no index is ever returned twice.
        let mut outstanding: Vec<(u32, u32)> = Vec::new();
        let mut union: HashSet<u32> = HashSet::new();

        for _ in 0..500 {
            let coin = (rng() % 100) as u32;
            if coin < 60 || outstanding.is_empty() {
                // alloc
                let n = (rng() % 20) as u32 + 1;
                if let Ok(base) = a.alloc_segment(n) {
                    // Every index in [base, base+n) must be unique system-wide.
                    for i in base..base + n {
                        assert!(union.insert(i), "duplicate index {} returned", i);
                    }
                    outstanding.push((base, n));
                }
            } else {
                // free a random outstanding segment
                let idx = (rng() as usize) % outstanding.len();
                let (base, n) = outstanding.swap_remove(idx);
                for i in base..base + n {
                    assert!(union.remove(&i), "freed index {} was not outstanding", i);
                }
                a.free_segment(base, n);
            }
            // Invariant: outstanding count matches the union set size.
            assert_eq!(a.outstanding() as usize, union.len());
            // Invariant: total = outstanding + total_free.
            assert_eq!(a.outstanding() + a.total_free(), total);
        }

        // Free everything and check final state coalesces back to [0, total).
        for (base, n) in outstanding.drain(..) {
            a.free_segment(base, n);
        }
        assert_eq!(a.outstanding(), 0);
        assert_eq!(a.free_ranges_snapshot(), vec![(0, total)]);
    }

    #[test]
    fn alloc_indices_falls_back_to_multi_range_when_fragmented() {
        let mut a = GlobalKvAllocator::new(100);
        // Carve out 4 disjoint 10-slot holes by allocating then freeing.
        a.alloc_segment(20).unwrap(); // [0..20)
        a.alloc_segment(20).unwrap(); // [20..40)
        a.alloc_segment(20).unwrap(); // [40..60)
        a.alloc_segment(20).unwrap(); // [60..80)
        a.alloc_segment(20).unwrap(); // [80..100)
        // Free non-adjacent so they can't coalesce.
        a.free_segment(0, 10);   // hole 1
        a.free_segment(20, 10);  // hole 2
        a.free_segment(40, 10);  // hole 3
        a.free_segment(60, 10);  // hole 4
        // total_free = 40 in 4 fragments of 10.
        assert_eq!(a.total_free(), 40);
        // alloc_segment(20) MUST fail (no single range ≥ 20).
        assert!(a.alloc_segment(20).is_err());
        // alloc_indices(20) MUST succeed by drawing from multiple ranges.
        let v = a.alloc_indices(20).unwrap();
        assert_eq!(v.len(), 20);
        // Ascending order, no duplicates.
        for i in 1..v.len() {
            assert!(v[i] > v[i - 1]);
        }
        // Total free now = 40 - 20 = 20.
        assert_eq!(a.total_free(), 20);
    }

    #[test]
    fn alloc_indices_fails_only_when_total_free_insufficient() {
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_segment(8).unwrap();
        // 2 free, asking for 3 must fail.
        let err = a.alloc_indices(3).unwrap_err();
        assert_eq!(err.need, 3);
        assert_eq!(err.total_free, 2);
        // 2 must succeed.
        let v = a.alloc_indices(2).unwrap();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn alloc_indices_zero_returns_empty() {
        let mut a = GlobalKvAllocator::new(10);
        let v = a.alloc_indices(0).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn allocator_is_deterministic_across_replays() {
        // Two allocators given the same operations must produce identical bases.
        let mut a = GlobalKvAllocator::new(128);
        let mut b = GlobalKvAllocator::new(128);
        let ops: &[(bool, u32)] = &[
            (true, 10),  // alloc
            (true, 20),
            (true, 5),
            (false, 1),  // free index 1 (the second alloc, base=10, len=20) — re-derived below
        ];
        let mut a_bases = vec![];
        let mut b_bases = vec![];
        for &(is_alloc, n) in ops {
            if is_alloc {
                a_bases.push(a.alloc_segment(n).unwrap());
                b_bases.push(b.alloc_segment(n).unwrap());
            } else {
                // free the second allocation in both
                a.free_segment(a_bases[1], 20);
                b.free_segment(b_bases[1], 20);
            }
        }
        assert_eq!(a_bases, b_bases);
        assert_eq!(a.free_ranges_snapshot(), b.free_ranges_snapshot());
    }
}
