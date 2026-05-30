//! `KvBudget` — host-side authority on worker KV-pool token-slot capacity.
//!
//! Phase 6 lands this type alongside `RadixTree` (Phase 5) and the new
//! `admission` System (Phase 6). The scheduler's main loop owns one
//! `KvBudget` instance whose `outstanding` count is updated by:
//!
//! - **`+= len`** when a `StepOutput.assigned_indices` arrives — each entry
//!   represents `len` new global slots the worker just consumed for that seq.
//! - **`-= len`** when the scheduler evicts a chain from `RadixTree` and
//!   sends `FreeKvIndices` back to the worker. The `outstanding` decrement
//!   must happen *together with* the message send so the budget never
//!   double-counts.
//!
//! Admission consults `try_reserve(projected)` before sending the next
//! batch. On `Err(KvBudgetFull)` the engine kicks off the eviction →
//! preemption → defer cascade described in plan §6/§7.
//!
//! ## Why a separate type, not just a `u32` field on the engine
//!
//! - The reservation logic is pure: mockable from tests without spinning up
//!   the whole engine.
//! - It defends a *single* invariant (`0 ≤ outstanding ≤ capacity`) which we
//!   want to be unmissable. Hard-typing the operation makes integer
//!   underflow detectable in `release`.
//! - Phase 8 (TP/PP) will likely change this from a single counter to a
//!   per-rank fan-out; pinning the boundary now keeps the change local.

use std::fmt;

/// Returned when `try_reserve(n)` cannot proceed because the worker pool
/// would overflow. `need` is the requested reservation; `outstanding` and
/// `capacity` reflect the pre-call state so admission has enough info to
/// compute the eviction target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvBudgetFull {
    pub need: u32,
    pub outstanding: u32,
    pub capacity: u32,
}

impl fmt::Display for KvBudgetFull {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KvBudget full: need={} outstanding={} capacity={} headroom={}",
            self.need,
            self.outstanding,
            self.capacity,
            self.capacity.saturating_sub(self.outstanding),
        )
    }
}

impl std::error::Error for KvBudgetFull {}

#[derive(Debug)]
pub struct KvBudget {
    outstanding: u32,
    capacity: u32,
}

impl KvBudget {
    pub fn new(capacity: u32) -> Self {
        Self {
            outstanding: 0,
            capacity,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn outstanding(&self) -> u32 {
        self.outstanding
    }

    pub fn headroom(&self) -> u32 {
        self.capacity.saturating_sub(self.outstanding)
    }

    /// True when `outstanding > capacity * threshold` (in basis points,
    /// i.e. 9_500 = 95.00 %). Used by admission to decide when to run a
    /// proactive evict before reservation pressure actually hits the
    /// ceiling.
    pub fn over_high_water(&self, threshold_bps: u32) -> bool {
        // Use u64 to avoid overflow when capacity * threshold is large.
        let lhs = (self.outstanding as u64) * 10_000u64;
        let rhs = (self.capacity as u64) * (threshold_bps as u64);
        lhs > rhs
    }

    /// Reserve `n` slots if available. On success, `outstanding += n`. On
    /// failure, returns `Err(KvBudgetFull)` and leaves `outstanding`
    /// unchanged.
    pub fn try_reserve(&mut self, n: u32) -> Result<(), KvBudgetFull> {
        let new_outstanding = match self.outstanding.checked_add(n) {
            Some(v) if v <= self.capacity => v,
            _ => {
                return Err(KvBudgetFull {
                    need: n,
                    outstanding: self.outstanding,
                    capacity: self.capacity,
                });
            }
        };
        self.outstanding = new_outstanding;
        Ok(())
    }

    /// Release `n` slots. Saturating subtraction; we still log a warning
    /// in debug builds because a release that would underflow indicates
    /// counter drift between scheduler and worker.
    pub fn release(&mut self, n: u32) {
        debug_assert!(
            self.outstanding >= n,
            "KvBudget release({}) underflows outstanding={}",
            n,
            self.outstanding,
        );
        self.outstanding = self.outstanding.saturating_sub(n);
    }

    /// Re-set capacity (e.g. after a worker-side resize). `outstanding` is
    /// preserved; if it now exceeds the new capacity, the next
    /// `try_reserve` will fail and admission must drain.
    pub fn set_capacity(&mut self, capacity: u32) {
        self.capacity = capacity;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_budget_has_full_headroom() {
        let b = KvBudget::new(100);
        assert_eq!(b.outstanding(), 0);
        assert_eq!(b.capacity(), 100);
        assert_eq!(b.headroom(), 100);
    }

    #[test]
    fn try_reserve_succeeds_within_capacity() {
        let mut b = KvBudget::new(100);
        assert!(b.try_reserve(40).is_ok());
        assert_eq!(b.outstanding(), 40);
        assert!(b.try_reserve(60).is_ok());
        assert_eq!(b.outstanding(), 100);
        assert_eq!(b.headroom(), 0);
    }

    #[test]
    fn try_reserve_fails_when_over_capacity() {
        let mut b = KvBudget::new(100);
        b.try_reserve(80).unwrap();
        let err = b.try_reserve(30).unwrap_err();
        // Failure leaves outstanding unchanged.
        assert_eq!(b.outstanding(), 80);
        assert_eq!(err.need, 30);
        assert_eq!(err.outstanding, 80);
        assert_eq!(err.capacity, 100);
    }

    #[test]
    fn release_reduces_outstanding() {
        let mut b = KvBudget::new(100);
        b.try_reserve(60).unwrap();
        b.release(40);
        assert_eq!(b.outstanding(), 20);
    }

    #[test]
    fn release_saturates_at_zero_in_release_builds() {
        // We cannot test the debug_assert without panic=abort; ensure
        // saturating semantics in release builds.
        let mut b = KvBudget::new(100);
        b.try_reserve(10).unwrap();
        // Suppress the debug assertion by building with release tests would
        // be cleaner; here we just confirm the saturating_sub doesn't wrap.
        // (In debug it panics — that's intentional.)
        if !cfg!(debug_assertions) {
            b.release(50);
            assert_eq!(b.outstanding(), 0);
        }
    }

    #[test]
    fn over_high_water_thresholds() {
        let mut b = KvBudget::new(100);
        assert!(!b.over_high_water(9_500));
        b.try_reserve(95).unwrap();
        // 95/100 = 95.00 %; threshold 95.00 % is NOT exceeded (strict >).
        assert!(!b.over_high_water(9_500));
        b.try_reserve(1).unwrap();
        assert!(b.over_high_water(9_500));
    }

    #[test]
    fn try_reserve_zero_is_noop() {
        let mut b = KvBudget::new(100);
        assert!(b.try_reserve(0).is_ok());
        assert_eq!(b.outstanding(), 0);
    }

    #[test]
    fn try_reserve_does_not_overflow_on_huge_n() {
        let mut b = KvBudget::new(u32::MAX);
        b.try_reserve(u32::MAX - 10).unwrap();
        // Asking for 100 more must fail cleanly (saturating_add caps at u32::MAX).
        let err = b.try_reserve(100).unwrap_err();
        assert_eq!(err.need, 100);
    }

    #[test]
    fn set_capacity_preserves_outstanding() {
        let mut b = KvBudget::new(100);
        b.try_reserve(80).unwrap();
        b.set_capacity(50);
        // Outstanding remained 80; new capacity is 50 — headroom is 0
        // (saturating). The next try_reserve cannot succeed until release.
        assert_eq!(b.outstanding(), 80);
        assert_eq!(b.headroom(), 0);
        assert!(b.try_reserve(1).is_err());
    }
}
