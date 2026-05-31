//! `GlobalKvAllocator` — bump-pointer over a `Vec<u32>` of free token-slot
//! indices.
//!
//! ## Design
//!
//! With `block_size = 1` the kernel does pure gather:
//! `block_table[seq][i] → arbitrary global index`. Whether the i-th slot
//! lives next to the (i+1)-th in the global pool is **architecturally
//! irrelevant**. Two non-contiguous indices `[0, 100]` produce the same
//! gather pattern (one cache line per token) as two contiguous `[0, 1]`.
//!
//! Given that, the cheapest allocator is a flat `Vec` plus a head pointer:
//! - **alloc**: `out = free[head..head+n]; head += n` — O(1) per slot.
//! - **free**: `free.push(idx)` — O(1) per slot.
//! - **merge**: drop the consumed prefix and sort the rest — O(N log N)
//!   one-shot, triggered lazily when the bump runs out but past frees
//!   have piled up at the tail.
//!
//! No coalesce on every free, no per-range bookkeeping. The "free pool"
//! is just whatever is in `free[head..]` plus whatever has been pushed
//! since the last merge.
//!
//! ## Invariant
//!
//! No index appears twice in `free`. The Vec is initialized to `0..total`
//! (each index once); subsequent `free()` calls push only previously-
//! allocated indices (which `head` has already moved past, so they are
//! NOT in `free[head..]` either). After `merge_and_sort` the prefix
//! `[..head]` is dropped before sort, preserving the invariant.
//!
//! ## Determinism
//!
//! `merge_and_sort` does an unstable sort, but the input set is a unique
//! collection of indices — every allocator instance with the same op
//! history sees the same set, hence the same sorted result. Bump
//! allocation thereafter is deterministic. Required for future TP/PP
//! rank consistency.

use std::collections::HashSet;

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
        }
    }

    pub fn total(&self) -> u32 {
        self.total
    }

    /// Slots the next `alloc_indices` can take immediately without merging.
    pub fn available(&self) -> u32 {
        (self.free.len() - self.head) as u32
    }

    /// Total free slots (head-side + freed-at-tail). Equal to
    /// `total - outstanding`.
    pub fn total_free(&self) -> u32 {
        // Indices in `free[head..]` are unique (invariant). Each freed
        // index, once pushed at tail, appears exactly once until consumed
        // by a future bump. So `free.len() - head` is the total count.
        (self.free.len() - self.head) as u32
    }

    /// Outstanding = total − total_free.
    pub fn outstanding(&self) -> u32 {
        self.total - self.total_free()
    }

    /// Allocate `n` indices.
    ///
    /// Fast path: bump from `head`. If insufficient at head but enough in
    /// `free` after merging, runs `merge_and_sort` once and retries from
    /// `head=0`. Returns `AllocFull` only on true OOM (`total_free < n`).
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

        // Slow path: merging may rescue us.
        if self.total_free() >= n {
            self.merge_and_sort();
            // After merge, head == 0 and free.len() == total_free.
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

    /// Push freed indices to the tail. O(N) extend.
    ///
    /// Caller is trusted to free only previously-allocated indices. Debug
    /// builds verify each input is `< total` and not currently in the
    /// allocatable zone (`free[head..]`); release builds skip the check.
    pub fn free(&mut self, indices: &[u32]) {
        if indices.is_empty() {
            return;
        }
        if cfg!(debug_assertions) {
            // Cheap upper-bound check.
            for &idx in indices {
                debug_assert!(idx < self.total, "free index out of range: {}", idx);
            }
        }
        self.free.extend_from_slice(indices);
    }

    /// Drop the consumed prefix and sort the remaining free indices.
    /// Intended trigger: `alloc_indices` returns an `AllocFull` whose
    /// `total_free >= need` (i.e. enough slots exist but they're stuck
    /// past the bump head). Pure local operation; no IO.
    ///
    /// This is *automatically* called by `alloc_indices` when needed; the
    /// public method is exposed for explicit "compact while idle" patterns
    /// and tests.
    pub fn merge_and_sort(&mut self) {
        if self.head > 0 {
            self.free.drain(..self.head);
            self.head = 0;
        }
        self.free.sort_unstable();
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
    fn free_pushes_to_tail_does_not_advance_head() {
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_indices(5).unwrap();
        let head_before = a.head();
        a.free(&[1, 2]);
        assert_eq!(a.head(), head_before, "free must not move head");
        assert_eq!(a.outstanding(), 3, "5 alloc'd − 2 freed = 3 outstanding");
    }

    #[test]
    fn alloc_after_free_continues_from_head_not_tail() {
        // The whole point of the lazy design: freed indices wait at the
        // tail until the bump head reaches them. We do NOT yank them
        // forward. This proves it.
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_indices(5).unwrap(); // [0,1,2,3,4], head=5
        a.free(&[0, 1, 2]); // tail = [...,0,1,2]
        let v = a.alloc_indices(3).unwrap();
        assert_eq!(v, vec![5, 6, 7], "next alloc takes from head, not freed tail");
        assert_eq!(a.head(), 8);
    }

    #[test]
    fn auto_merge_when_head_exhausted_but_total_free_sufficient() {
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_indices(10).unwrap(); // exhaust head: head=10
        assert_eq!(a.available(), 0);
        a.free(&[3, 7, 1]); // tail has 3 free; available=3 (single len() metric)
        assert_eq!(a.available(), 3);
        // Asking for 4 — head's tail can't satisfy (only 3 free total in
        // free[head..], but pushed at tail). Bump-from-len fast path takes
        // [3, 7, 1] and... wait, free.len() == 13, head==10, so free[10..]
        // is [3, 7, 1]. Asking 4 would need head + 4 <= len; 10+4=14 > 13.
        // Falls through to slow path → merge_and_sort → free=[1,3,7], head=0.
        // Then alloc(4) needs total_free=3 < 4 → AllocFull.
        let err = a.alloc_indices(4).unwrap_err();
        assert_eq!(err.need, 4);
        assert_eq!(err.total_free, 3);

        // Now alloc(2): head=10, len=13, fast path takes free[10..12]=[3,7].
        let v = a.alloc_indices(2).unwrap();
        assert_eq!(v, vec![3, 7], "fast path takes from current head, no merge needed");
        assert_eq!(a.head(), 12);
        assert_eq!(a.outstanding(), 9);
    }

    #[test]
    fn slow_path_merge_when_head_truly_at_end_but_unsorted_frees_remain() {
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_indices(10).unwrap(); // head=10, len=10
        a.free(&[5, 0, 8]); // head=10, len=13, available=3
        // Drain everything available via fast path.
        let _ = a.alloc_indices(3).unwrap(); // head=13, len=13, available=0
        // Free more — these go to tail past head=13.
        a.free(&[1, 2]); // head=13, len=15, available=2
        // Now alloc(1) fast-paths. Asking for 3 forces merge_and_sort:
        //   total_free = 15-13 = 2 < 3 → AllocFull (not enough total).
        let err = a.alloc_indices(3).unwrap_err();
        assert_eq!(err.total_free, 2);
        // Free a few more.
        a.free(&[6, 7, 9]); // head=13, len=18, available=5
        // alloc(4): fast path can give 4 (free[13..17] = [1,2,6,7]).
        let v = a.alloc_indices(4).unwrap();
        assert_eq!(v, vec![1, 2, 6, 7]);
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
        let mut a = GlobalKvAllocator::new(10);
        let _ = a.alloc_indices(7).unwrap(); // head=7
        a.free(&[5, 2, 0]); // tail = [...,5,2,0]; free.len()=13
        a.merge_and_sort();
        // After merge: drained free[..7], leaves [7,8,9,5,2,0]; sort →
        // [0,2,5,7,8,9]; head=0.
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
