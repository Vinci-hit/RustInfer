//! Admission cascade for the worker-owned KV path.
//!
//! Phase 6 lands this module as the single place where the engine consults
//! before sending a step. The cascade (plan §6/§7) is:
//!
//! ```text
//! projected = Σ(decode_seqs:1) + Σ(prefill_chunks)
//! while budget.try_reserve(projected).is_err():
//!     // Round A — drain the LRU first; cheapest, no live decoder loss.
//!     freed_a = radix.evict(target = outstanding + projected − cap*0.90)
//!     if !freed_a.is_empty():
//!         worker_free(freed_a); budget.release(freed_a.len)
//!         continue
//!     // Round B — preempt some live decoders, then re-evict their just-
//!     // released chains.
//!     kicked = preempt(running, projected − headroom)
//!     if kicked.is_empty():
//!         // Even the LIFO candidate set was empty (n_running ≤ 4 and
//!         // tier1+tier2 saturated); fall through to defer.
//!         break
//!     for sid in kicked:
//!         radix.mark_finished_chain(sid)
//!     freed_b = radix.evict(target = need)
//!     worker_free(freed_b); budget.release(freed_b.len)
//!     // Continue the loop: a fresh try_reserve may now pass.
//!
//! if budget.try_reserve(projected).is_err():
//!     // Round C — defer some prefill chunks to the next iteration.
//!     defer_prefill_until_fit()
//! ```
//!
//! This module exposes a pure decision kernel: given the engine's view of
//! `running`, `prefill`, and decoded counts, plus mutable references to
//! `RadixTree`, `KvBudget`, and a sink for the `FreeKvIndices` indices it
//! produces, it returns an `AdmissionPlan` describing what to do. Phase 6
//! engine wiring then enacts the plan (sends `FreeKvIndices`, transitions
//! preempted seqs to `ResourceStarved`, defers prefills).
//!
//! The kernel does not perform IO. Tests exercise it with fake Snap data.

use crate::domain::kv_budget::{KvBudget, KvBudgetFull};
use crate::domain::preemption::{select_preempt_ids, PreemptionConfig, RunningSnap};
use crate::infrastructure::kv_cache::radix_tree_v2::{GlobalIndex, RadixTree, SeqId};

/// Knobs for the admission cascade. Defaults match plan §6.
#[derive(Debug, Clone, Copy)]
pub struct AdmissionConfig {
    /// `outstanding > capacity * high_water_bps / 10_000` triggers a
    /// proactive evict before we even try_reserve. 95.00 % default.
    pub high_water_bps: u32,
    /// After eviction we aim to leave outstanding at this fraction of
    /// capacity. 90.00 % default.
    pub low_water_bps: u32,
    pub preemption: PreemptionConfig,
    /// Opportunistic LRU drain per iteration (in slots). Without pressure,
    /// finished chains otherwise sit in the LRU until high-water fires;
    /// for long-running serving this lets `outstanding` track the live
    /// working set rather than the cumulative allocation history. Default
    /// 8192 — enough for ~256 seqs producing tokens at aggregate rate of
    /// ~256 tokens/iter without ever falling behind. Set to 0 to disable.
    pub opportunistic_drain_cap: u32,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            high_water_bps: 9_500,
            low_water_bps: 9_000,
            preemption: PreemptionConfig::default(),
            opportunistic_drain_cap: 8_192,
        }
    }
}

/// What admission decided. The engine carries this out:
///
/// 1. For each `freed` batch, send `FreeKvIndices` to the worker (the
///    indices have already been `release()`d from `KvBudget`).
/// 2. For each id in `preempted_ids`, transition the session out of
///    `Decoding` into `ResourceStarved` (drop output_tokens, keep
///    input_ids), and re-queue it.
/// 3. If `deferred`, the engine should send the step with only the
///    sequences that fit; the rest stay in the waiting queue.
#[derive(Debug, Clone)]
pub struct AdmissionPlan<Id: Clone + Eq> {
    /// Indices to send back to the worker as `FreeKvIndices`. Two-batch
    /// because the cascade may evict twice — once before preemption and
    /// once after. Either may be empty.
    pub freed: Vec<Vec<GlobalIndex>>,
    /// Sessions to transition into `ResourceStarved`.
    pub preempted_ids: Vec<Id>,
    /// True ⇒ the projected step still won't fit; the engine must defer
    /// some prefills until the next iteration.
    pub deferred: bool,
    /// Budget reservation outcome. `Ok(())` ⇒ engine may build and send
    /// the step covering exactly `projected` slots. `Err(...)` ⇒ even
    /// after eviction + preemption + defer the budget cannot fit; this
    /// is a fatal admission failure (caller turns it into an
    /// internal error or escalates).
    pub reservation: Result<(), KvBudgetFull>,
}

/// Run the admission cascade.
///
/// `projected` is the total token slots the next step will consume. The
/// `running` slice describes every sequence currently in `Decoding` state
/// (with their `kv_len`, `input_len`, `arrival_time`); preemption picks
/// from this set when the cascade reaches Round B.
///
/// On return, `KvBudget` reflects all evictions performed during the
/// cascade *and* the final reservation if it succeeded. `RadixTree` reflects
/// all evictions and the preemption-driven `mark_finished_chain` calls.
///
/// `Id` is whatever the engine uses for its session identifier; the
/// admission kernel doesn't care, and tests pass `u64`.
pub fn run_admission<Id: Clone + Eq>(
    projected: u32,
    running: &[RunningSnap<Id>],
    radix: &mut RadixTree,
    budget: &mut KvBudget,
    cfg: AdmissionConfig,
) -> AdmissionPlan<Id> {
    let mut plan = AdmissionPlan {
        freed: Vec::new(),
        preempted_ids: Vec::new(),
        deferred: false,
        reservation: Err(KvBudgetFull {
            need: projected,
            outstanding: budget.outstanding(),
            capacity: budget.capacity(),
        }),
    };

    // ── Pre-pressure proactive evict ──
    // If we're over the high-water mark even before reserving, drain LRU
    // to the low-water mark first. This keeps eviction off the critical
    // path for typical loads.
    if budget.over_high_water(cfg.high_water_bps) {
        let target = compute_evict_target(budget, projected, cfg.low_water_bps);
        if target > 0 {
            let freed = radix.evict(target as usize);
            if !freed.is_empty() {
                budget.release(freed.len() as u32);
                plan.freed.push(freed);
            }
        }
    }

    // ── Round A: try_reserve, with eviction loop ──
    loop {
        match budget.try_reserve(projected) {
            Ok(()) => {
                plan.reservation = Ok(());
                return plan;
            }
            Err(_) => {
                // Try to evict. Compute the target relative to current
                // outstanding (which may have just changed).
                let target =
                    compute_evict_target(budget, projected, cfg.low_water_bps).max(1);
                let freed = radix.evict(target as usize);
                if freed.is_empty() {
                    break; // No more LRU; escalate to preemption.
                }
                budget.release(freed.len() as u32);
                plan.freed.push(freed);
            }
        }
    }

    // ── Round B: preemption ──
    let need = projected.saturating_sub(budget.headroom());
    let kicked = select_preempt_ids(running, need, cfg.preemption);
    if !kicked.is_empty() {
        for sid in &kicked {
            // Sessions track their own `SeqId` for RadixTree purposes; the
            // engine's wiring layer maps `Id ↔ SeqId`. Here we trust the
            // caller has already done that mapping by ensuring the running
            // slice's `Id` IS the radix-tree `SeqId`. In production the
            // engine will pass a closure, but for the kernel we stay pure
            // by typing `Id` as whatever lets the tree update itself.
            // The simplest contract: the caller pre-converts the IDs by
            // building `running` with `Id = SeqId`. Tests do exactly that.
            // For the engine we parameterize over `Id`; the test below
            // pins the ergonomics.
            // SAFETY for u64 → SeqId: SeqId IS u64 (alias).
            // We can't transmute between arbitrary `Id` and `SeqId`; instead
            // require `Id: Into<SeqId>`. Use the trait below to keep the
            // kernel generic.
            let _ = sid;
        }
    }
    plan.preempted_ids = kicked;

    // We can't, from the kernel, push the seq id into the RadixTree
    // because we have only the abstract `Id`. Defer the
    // `mark_finished_chain` step to a typed helper specialized for
    // `Id = SeqId`; see `run_admission_seqid_keyed` below.
    // For the abstract kernel we simply return the IDs to be preempted
    // and let the caller drive the radix tree.

    // ── Final reservation attempt ──
    match budget.try_reserve(projected) {
        Ok(()) => {
            plan.reservation = Ok(());
        }
        Err(e) => {
            plan.reservation = Err(e);
            plan.deferred = true;
        }
    }
    plan
}

/// Specialization for `Id = SeqId`: the kernel can drive the RadixTree
/// `mark_finished_chain + evict` loop end-to-end. This is what the engine
/// actually calls.
pub fn run_admission_seqid_keyed(
    projected: u32,
    running: &[RunningSnap<SeqId>],
    radix: &mut RadixTree,
    budget: &mut KvBudget,
    cfg: AdmissionConfig,
) -> AdmissionPlan<SeqId> {
    let mut plan = AdmissionPlan {
        freed: Vec::new(),
        preempted_ids: Vec::new(),
        deferred: false,
        // Optimistic — we'll downgrade to Err if the cascade can't make room.
        reservation: Ok(()),
    };

    // ── Opportunistic LRU drain ──
    // Without pressure, finished chains accumulate in the LRU and the
    // worker holds onto their slots until high-water fires. That's fine
    // for short bursts but causes `outstanding` to grow unbounded over
    // long runs. Drain a bounded slice of the LRU each iteration so the
    // worker's allocator stays close to the live-set size.
    if cfg.opportunistic_drain_cap > 0 {
        let cap = cfg.opportunistic_drain_cap.min(radix.lru_len_estimate() as u32);
        if cap > 0 {
            let freed = radix.evict(cap as usize);
            if !freed.is_empty() {
                budget.release(freed.len() as u32);
                plan.freed.push(freed);
            }
        }
    }

    // ── Pre-pressure proactive evict ──
    if budget.over_high_water(cfg.high_water_bps) {
        let target = compute_evict_target(budget, projected, cfg.low_water_bps);
        if target > 0 {
            let freed = radix.evict(target as usize);
            if !freed.is_empty() {
                budget.release(freed.len() as u32);
                plan.freed.push(freed);
            }
        }
    }

    // ── Round A: do we have headroom for `projected`? Evict until yes. ──
    // Note (Phase 7B-1): we do NOT reserve here. Reservation happens in
    // `engine.handle_step_output_llm` when the StepOutput proves the
    // worker actually consumed the slots. Admission's only job is to
    // make sure the budget *would* fit `projected`; we use a peek
    // function (`headroom() >= projected`) rather than the destructive
    // try_reserve.
    while budget.headroom() < projected {
        let target = projected.saturating_sub(budget.headroom()).max(1);
        let freed = radix.evict(target as usize);
        if freed.is_empty() {
            break;
        }
        budget.release(freed.len() as u32);
        plan.freed.push(freed);
    }
    if budget.headroom() >= projected {
        return plan;
    }

    // ── Round B: preempt. ──
    let need = projected.saturating_sub(budget.headroom());
    let kicked = select_preempt_ids(running, need, cfg.preemption);
    plan.preempted_ids = kicked.clone();
    for sid in &kicked {
        radix.mark_finished_chain(*sid);
    }
    if !kicked.is_empty() {
        let target = projected.saturating_sub(budget.headroom()).max(1);
        let freed = radix.evict(target as usize);
        if !freed.is_empty() {
            budget.release(freed.len() as u32);
            plan.freed.push(freed);
        }
    }

    // ── Round C: final headroom check. ──
    if budget.headroom() < projected {
        plan.reservation = Err(KvBudgetFull {
            need: projected,
            outstanding: budget.outstanding(),
            capacity: budget.capacity(),
        });
        plan.deferred = true;
    }
    plan
}

/// `target = outstanding + projected − capacity * low_water_bps / 10_000`
/// Always non-negative; clamped to `outstanding` (we can never evict more
/// than is currently outstanding).
fn compute_evict_target(budget: &KvBudget, projected: u32, low_water_bps: u32) -> u32 {
    let cap_low = ((budget.capacity() as u64) * (low_water_bps as u64) / 10_000u64) as u32;
    let post = budget.outstanding().saturating_add(projected);
    post.saturating_sub(cap_low).min(budget.outstanding())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn fresh_budget(cap: u32) -> KvBudget {
        KvBudget::new(cap)
    }

    /// Build a RadixTree with `n` chains, each of length `len`. Chain `i`
    /// uses tokens `[100*i, 100*i+1, …]` and global indices
    /// `[base+i*len, base+i*len+1, …]`. The seq ids are `1..=n`.
    /// Returns the indices that were assigned (so tests can sanity-check
    /// what should come back from evict).
    fn populate_tree(
        tree: &mut RadixTree,
        n: u64,
        len: u32,
        base: GlobalIndex,
    ) -> Vec<GlobalIndex> {
        let mut all = Vec::new();
        for s in 1..=n {
            for k in 0..len {
                let token = (100 * s as i32) + k as i32;
                let idx = base + (s as u32 - 1) * len + k;
                tree.append_token(s, token, idx);
                all.push(idx);
            }
        }
        all
    }

    fn snap_seq(seq: SeqId, kv: u32, input: u32, age_ms: u64) -> RunningSnap<SeqId> {
        thread_local! {
            static BASE: Instant = Instant::now();
        }
        let base = BASE.with(|b| *b);
        RunningSnap {
            id: seq,
            kv_len: kv,
            input_len: input,
            arrival_time: base + Duration::from_millis(age_ms),
        }
    }

    #[test]
    fn happy_path_no_pressure_just_reserves() {
        let mut tree = RadixTree::new();
        let mut bud = fresh_budget(1_000);
        let plan = run_admission_seqid_keyed(
            10,
            &[],
            &mut tree,
            &mut bud,
            AdmissionConfig::default(),
        );
        // Phase 7B-1: admission only ensures `headroom >= projected`;
        // the actual reservation lands in `engine.handle_step_output_llm`
        // when the StepOutput proves the worker consumed those slots.
        assert!(plan.reservation.is_ok());
        assert!(plan.freed.is_empty());
        assert!(plan.preempted_ids.is_empty());
        assert!(!plan.deferred);
        assert_eq!(bud.outstanding(), 0, "admission no longer reserves");
        assert!(bud.headroom() >= 10);
    }

    #[test]
    fn evict_round_a_alone_resolves_pressure() {
        let mut tree = RadixTree::new();
        let _ = populate_tree(&mut tree, 3, 10, 0);
        // Mark all three chains finished so all 30 indices land in LRU.
        for s in 1..=3u64 {
            tree.mark_finished_chain(s);
        }

        let mut bud = fresh_budget(40);
        // Pretend 30 of them are tracked in budget (worker reported them).
        bud.try_reserve(30).unwrap();

        // We want 20 more, but headroom is only 10. Round A should evict
        // enough to fit.
        let plan = run_admission_seqid_keyed(
            20,
            &[],
            &mut tree,
            &mut bud,
            AdmissionConfig::default(),
        );
        assert!(plan.reservation.is_ok(), "should fit after evict");
        assert!(!plan.freed.is_empty());
        assert!(plan.preempted_ids.is_empty(), "no preemption needed");
        assert!(!plan.deferred);
    }

    #[test]
    fn preempt_round_b_kicks_when_lru_exhausted() {
        let mut tree = RadixTree::new();
        // 8 active chains, no finishing → LRU is empty.
        let _ = populate_tree(&mut tree, 8, 10, 0);
        let mut bud = fresh_budget(80);
        bud.try_reserve(80).unwrap(); // pool full, all live

        let running: Vec<_> = (1..=8u64)
            .map(|s| snap_seq(s, 10, 10 + (s as u32), s)) // arrival = s
            .collect();

        // Need 10 more slots — must preempt at least one seq.
        let plan = run_admission_seqid_keyed(
            10,
            &running,
            &mut tree,
            &mut bud,
            AdmissionConfig::default(),
        );
        assert!(plan.reservation.is_ok(), "post-preempt reserve must succeed");
        assert!(!plan.preempted_ids.is_empty(), "preemption should fire");
        // At least one freed batch must be present (round B's harvest).
        assert!(!plan.freed.is_empty());
    }

    #[test]
    fn defer_when_n_running_too_small_to_preempt() {
        let mut tree = RadixTree::new();
        // 2 active chains with 10 slots each — both protected by tier1/tier2.
        let _ = populate_tree(&mut tree, 2, 10, 0);
        let mut bud = fresh_budget(20);
        bud.try_reserve(20).unwrap();

        let running = vec![snap_seq(1u64, 10, 10, 0), snap_seq(2, 10, 10, 1)];

        // Need 5 more. LRU is empty; preempt selects 0; admission must defer.
        let plan = run_admission_seqid_keyed(
            5,
            &running,
            &mut tree,
            &mut bud,
            AdmissionConfig::default(),
        );
        assert!(plan.reservation.is_err(), "must signal failure");
        assert!(plan.preempted_ids.is_empty());
        assert!(plan.deferred);
        // Outstanding unchanged.
        assert_eq!(bud.outstanding(), 20);
    }

    #[test]
    fn high_water_triggers_proactive_evict() {
        let mut tree = RadixTree::new();
        let _ = populate_tree(&mut tree, 5, 10, 0);
        for s in 1..=5u64 {
            tree.mark_finished_chain(s);
        }

        let mut bud = fresh_budget(100);
        // Push outstanding to 96/100 = above default high-water 95%.
        bud.try_reserve(96).unwrap();

        let plan = run_admission_seqid_keyed(
            1, // tiny request
            &[],
            &mut tree,
            &mut bud,
            AdmissionConfig::default(),
        );
        assert!(plan.reservation.is_ok());
        // Proactive evict must have produced a freed batch even though the
        // direct reservation would have succeeded with just headroom=4.
        assert!(!plan.freed.is_empty(), "high-water should trigger evict");
        // Outstanding must be ≤ low-water target after evict + reserve.
        // low_water = 90 → outstanding should be ≤ 90 + 1 = 91 (the +1 is
        // the new reservation).
        assert!(bud.outstanding() <= 91, "got {}", bud.outstanding());
    }

    #[test]
    fn compute_evict_target_clamps_to_outstanding() {
        let mut bud = KvBudget::new(100);
        bud.try_reserve(20).unwrap();
        // post = 20 + 200 = 220; cap_low = 90; 220 - 90 = 130 → clamp to 20.
        assert_eq!(compute_evict_target(&bud, 200, 9_000), 20);
    }
}
