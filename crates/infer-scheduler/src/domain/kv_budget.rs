//! `KvBudget` — host-side authority on worker KV-pool token-slot capacity.
//!
//! The scheduler's main loop owns one `KvBudget` instance whose
//! `outstanding` count is updated by:
//!
//! - **`+= len`** when a `StepOutput.assigned_indices` arrives — each entry
//!   represents `len` new global slots the worker just consumed for that seq.
//! - **`-= len`** when the scheduler evicts a chain from `RadixTree` and
//!   sends `FreeKvIndices` back to the worker. The `outstanding` decrement
//!   must happen *together with* the message send so the budget never
//!   double-counts.
//!
//! Admission consults `headroom()` before sending the next batch and
//! drives the eviction → preemption → defer cascade until
//! `headroom() >= projected`. Reservation against the budget happens
//! when the worker reports the slots back, not at admission time.
//!
//! ## Why a separate type, not just a `u32` field on the engine
//!
//! - The reservation logic is pure: mockable from tests without spinning up
//!   the whole engine.
//! - It defends a *single* invariant (`0 ≤ outstanding ≤ capacity`) which we
//!   want to be unmissable. Hard-typing the operation makes integer
//!   underflow detectable in `release`.
//! - A future TP/PP variant could swap the single counter for a
//!   per-rank fan-out without disturbing call sites.

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
    /// KV slots that have been admitted into an in-flight prefill batch but
    /// whose `assigned_indices` the worker has **not yet reported back**.
    ///
    /// `outstanding` is only bumped when the worker confirms real slot usage
    /// (reserve-on-report). Without an admission-time reservation, two
    /// prefill batches scheduled back-to-back would both read the same
    /// pre-report `headroom()` and over-commit the worker pool — the bug the
    /// old `!has_inflight_prefill()` gate worked around by serializing
    /// prefills (which inflated TTFT at low QPS).
    ///
    /// We instead reserve *projected* slots here at admission time and net
    /// them out when the matching report arrives, so `headroom()` is always
    /// pressure-accurate and prefills no longer need to be serialized.
    pending_prefill: u32,
}

impl KvBudget {
    pub fn new(capacity: u32) -> Self {
        Self {
            outstanding: 0,
            capacity,
            pending_prefill: 0,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn outstanding(&self) -> u32 {
        self.outstanding
    }

    /// Slots admitted into in-flight prefill batches but not yet reported.
    pub fn pending_prefill(&self) -> u32 {
        self.pending_prefill
    }

    /// Free slots available for the next admission decision.
    ///
    /// Subtracts both confirmed (`outstanding`) and admitted-but-unreported
    /// (`pending_prefill`) usage, so concurrent prefill scheduling cannot
    /// over-commit the worker pool.
    pub fn headroom(&self) -> u32 {
        self.capacity
            .saturating_sub(self.outstanding)
            .saturating_sub(self.pending_prefill)
    }

    /// Set the projected in-flight prefill footprint (slots dispatched to the
    /// worker but not yet reported via `assigned_indices`).
    ///
    /// Recomputed from live session state at the head of every scheduling
    /// iteration, so it self-heals: sequences cancelled / preempted / failed
    /// before their ack simply drop out of the recomputation and free their
    /// pending pressure automatically — there is no counter to leak.
    pub fn set_pending_prefill(&mut self, tokens: u32) {
        self.pending_prefill = tokens;
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

    /// Release `n` slots.
    ///
    /// Saturates instead of underflowing, but logs in all builds because a
    /// release larger than `outstanding` means scheduler/worker budget drift.
    pub fn release(&mut self, n: u32) {
        if n > self.outstanding {
            tracing::warn!(
                requested = n,
                outstanding = self.outstanding,
                capacity = self.capacity,
                "KvBudget release exceeds outstanding; clamping"
            );
            debug_assert!(
                self.outstanding >= n,
                "KvBudget release({}) underflows outstanding={}",
                n,
                self.outstanding,
            );
        }
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

    #[test]
    fn pending_prefill_reduces_headroom() {
        let mut b = KvBudget::new(100);
        b.set_pending_prefill(30);
        // 30 slots are admitted-but-unreported: headroom must exclude them.
        assert_eq!(b.pending_prefill(), 30);
        assert_eq!(b.outstanding(), 0);
        assert_eq!(b.headroom(), 70);
    }

    #[test]
    fn pending_and_outstanding_both_subtract_from_headroom() {
        let mut b = KvBudget::new(100);
        b.try_reserve(40).unwrap(); // worker-confirmed
        b.set_pending_prefill(25); // in-flight prefill
        assert_eq!(b.headroom(), 35);
    }

    #[test]
    fn report_settles_pending_via_recompute() {
        // Models the scheduler flow: dispatch sets pending; the worker
        // report books outstanding; the next iteration recomputes pending
        // (segment now acked → 0). Net headroom is unchanged across the
        // hand-off, so no over-commit window opens.
        let mut b = KvBudget::new(100);
        b.set_pending_prefill(20); // batch dispatched, 20 slots in flight
        assert_eq!(b.headroom(), 80);

        b.try_reserve(20).unwrap(); // worker reports 20 real slots
        // Next scheduling iteration recomputes pending from live state; the
        // segment is acked so it no longer contributes.
        b.set_pending_prefill(0);
        assert_eq!(b.outstanding(), 20);
        assert_eq!(b.headroom(), 80);
    }

    #[test]
    fn pending_self_heals_to_zero_when_inflight_drains() {
        let mut b = KvBudget::new(100);
        b.set_pending_prefill(50);
        assert_eq!(b.headroom(), 50);
        // Sequences cancelled/failed before ack → recompute yields 0.
        b.set_pending_prefill(0);
        assert_eq!(b.headroom(), 100);
    }

    #[test]
    fn pending_over_capacity_saturates_headroom_to_zero() {
        let mut b = KvBudget::new(100);
        b.try_reserve(70).unwrap();
        b.set_pending_prefill(50); // 70 + 50 > 100
        assert_eq!(b.headroom(), 0);
    }
}
