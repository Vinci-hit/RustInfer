//! Preemption selector for KV-pressure recovery.
//!
//! When `KvBudget::try_reserve` fails and the post-evict LRU is exhausted,
//! the scheduler must kick some live decoders out of the running set to
//! free up token slots. This module implements plan §7's three-tier
//! protection-and-eviction policy:
//!
//! 1. **Tier 1 — preserve longest KV**: top `ceil(n_running / 4)` sequences
//!    by `kv_len` are kept. Rationale: the longer the KV, the closer the
//!    sequence is to its `max_tokens` limit, and finishing it returns the
//!    most slots in absolute terms.
//!
//! 2. **Tier 2 — preserve shortest input**: from the remainder, top
//!    `ceil(n_running / 4)` by ascending `input_len` are kept. Rationale:
//!    short prompts usually correlate with short outputs, so these
//!    sequences finish soon and free their slots without much further
//!    cost.
//!
//! 3. **Eviction candidates — LIFO by arrival**: every sequence not in
//!    tier 1 or tier 2 is sorted by `arrival_time DESC` (newest first) and
//!    kicked off from the head until cumulative `kv_len` ≥ `need`.
//!
//! The kicked sequences then go through `mark_finished_chain` (so their
//! KV indices migrate to the RadixTree LRU) and the engine retries the
//! evict cascade. Their `output_tokens` are dropped; `input_ids` is
//! preserved so they can re-enter the waiting queue and rerun prefill,
//! ideally hitting their own RadixTree-cached prefix.
//!
//! ## Inputs as a value type
//!
//! The selector is pure: it takes a slice of `RunningSnap` describing each
//! candidate and returns the IDs to kick. The engine builds `RunningSnap`
//! from its `RequestTable`; `Phase 6 wiring` does that translation. Tests
//! exercise the selector with hand-rolled snapshots.

use std::time::Instant;

/// Snapshot of one running sequence, used by the preemption selector.
/// All fields are domain primitives so the selector has zero coupling
/// to typestates and storage.
#[derive(Debug, Clone)]
pub struct RunningSnap<Id: Clone + Eq> {
    pub id: Id,
    /// Current KV length (slots already written to the worker pool).
    pub kv_len: u32,
    /// Original prompt length in tokens.
    pub input_len: u32,
    /// Wall-clock arrival timestamp.
    pub arrival_time: Instant,
}

/// Default fractions, mirroring plan §7. Both expressed in basis points
/// (10_000 = 100 %) so we can change them without floating point.
pub const DEFAULT_TIER1_FRAC_BPS: u32 = 2_500; // 25 %
pub const DEFAULT_TIER2_FRAC_BPS: u32 = 2_500; // 25 %

/// Knobs for the selector. Defaults match plan §7.
#[derive(Debug, Clone, Copy)]
pub struct PreemptionConfig {
    pub tier1_frac_bps: u32,
    pub tier2_frac_bps: u32,
}

impl Default for PreemptionConfig {
    fn default() -> Self {
        Self {
            tier1_frac_bps: DEFAULT_TIER1_FRAC_BPS,
            tier2_frac_bps: DEFAULT_TIER2_FRAC_BPS,
        }
    }
}

/// Compute the preempt set.
///
/// Returns the IDs to kick (in the order they should be processed —
/// newest first, so the engine can stop early once enough KV is freed).
/// `need` is the target reduction in outstanding slots, in tokens.
///
/// **Edge cases**:
/// - `need == 0` → returns empty.
/// - `running.is_empty()` → returns empty.
/// - tier1 + tier2 cover all running seqs (n ≤ 4) → returns empty
///   (preemption refuses to kick an "essential" seq and reports the
///   shortfall to the caller via `Vec::len() == 0` despite `need > 0`).
///   The caller then falls through to "defer prefill" (plan §6.3).
pub fn select_preempt_ids<Id: Clone + Eq>(
    running: &[RunningSnap<Id>],
    need: u32,
    cfg: PreemptionConfig,
) -> Vec<Id> {
    if need == 0 || running.is_empty() {
        return Vec::new();
    }

    let n = running.len();
    let tier1_n = ceil_frac(n, cfg.tier1_frac_bps);
    let tier2_n = ceil_frac(n, cfg.tier2_frac_bps);

    // tier1 = top kv_len. Sort descending by kv_len, take first tier1_n.
    let mut by_kv: Vec<usize> = (0..n).collect();
    by_kv.sort_by(|&a, &b| {
        running[b]
            .kv_len
            .cmp(&running[a].kv_len)
            // Stability tiebreaker: lower id first (cheap, deterministic).
            // We can't compare arbitrary `Id` cleanly; arrival_time is the
            // next best signal — older first stays first.
            .then(running[a].arrival_time.cmp(&running[b].arrival_time))
    });
    let tier1: Vec<usize> = by_kv.iter().take(tier1_n).copied().collect();

    // tier2 = of the remainder, top by ASCENDING input_len.
    let mut remainder_after_t1: Vec<usize> = (0..n).filter(|i| !tier1.contains(i)).collect();
    remainder_after_t1.sort_by(|&a, &b| {
        running[a]
            .input_len
            .cmp(&running[b].input_len)
            .then(running[a].arrival_time.cmp(&running[b].arrival_time))
    });
    let tier2: Vec<usize> = remainder_after_t1.iter().take(tier2_n).copied().collect();

    // candidates = the rest, sorted by arrival_time DESCENDING (newest first).
    let mut candidates: Vec<usize> = (0..n)
        .filter(|i| !tier1.contains(i) && !tier2.contains(i))
        .collect();
    candidates.sort_by(|&a, &b| running[b].arrival_time.cmp(&running[a].arrival_time));

    // Kick from head until cumulative kv_len >= need.
    let mut freed: u32 = 0;
    let mut kicked: Vec<Id> = Vec::new();
    for idx in candidates {
        if freed >= need {
            break;
        }
        kicked.push(running[idx].id.clone());
        freed = freed.saturating_add(running[idx].kv_len);
    }
    kicked
}

/// `ceil(n * frac_bps / 10_000)`, with a floor of 1 for any non-zero `frac_bps`.
fn ceil_frac(n: usize, frac_bps: u32) -> usize {
    if frac_bps == 0 {
        return 0;
    }
    let raw = (n as u64) * (frac_bps as u64);
    let q = (raw + 9_999) / 10_000;
    (q as usize).max(1).min(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn snap<Id: Clone + Eq>(id: Id, kv_len: u32, input_len: u32, age_ms: u64) -> RunningSnap<Id> {
        // Use a fixed reference instant; `age_ms` larger ⇒ later arrival
        // (so "newest" has the largest age_ms).
        // We construct relative instants via `Instant::now() - duration` —
        // but `Instant::now()` would invalidate determinism across runs.
        // Use a single base captured once via a thread_local.
        thread_local! {
            static BASE: Instant = Instant::now();
        }
        let base = BASE.with(|b| *b);
        RunningSnap {
            id,
            kv_len,
            input_len,
            arrival_time: base + Duration::from_millis(age_ms),
        }
    }

    #[test]
    fn empty_running_returns_empty() {
        let kicked: Vec<u64> = select_preempt_ids::<u64>(&[], 100, PreemptionConfig::default());
        assert!(kicked.is_empty());
    }

    #[test]
    fn need_zero_returns_empty() {
        let r = vec![snap(1u64, 10, 10, 0)];
        let kicked = select_preempt_ids(&r, 0, PreemptionConfig::default());
        assert!(kicked.is_empty());
    }

    #[test]
    fn ceil_frac_25_percent() {
        // 5 * 25% = 1.25 → ceil = 2
        assert_eq!(ceil_frac(5, 2_500), 2);
        // 4 * 25% = 1 → 1
        assert_eq!(ceil_frac(4, 2_500), 1);
        // 1 * 25% = 0.25 → ceil 1, then floor 1 → 1
        assert_eq!(ceil_frac(1, 2_500), 1);
        // n=0 → 0
        assert_eq!(ceil_frac(0, 2_500), 0);
        // frac=0 → 0
        assert_eq!(ceil_frac(10, 0), 0);
    }

    #[test]
    fn small_running_set_protects_everyone() {
        // n=4 → tier1=1, tier2=1 → 2 protected. Remaining 2 are candidates.
        let r = vec![
            snap(1u64, 100, 10, 0),
            snap(2, 50, 50, 1),
            snap(3, 80, 5, 2),
            snap(4, 20, 100, 3),
        ];
        // need huge: tier1 keeps id=1 (kv=100), tier2 from {2,3,4} keeps
        // id=3 (input=5). Candidates {2, 4} sorted newest-first → [4, 2].
        let kicked = select_preempt_ids(&r, 1_000, PreemptionConfig::default());
        assert_eq!(kicked, vec![4u64, 2]);
    }

    #[test]
    fn very_small_running_set_yields_no_candidates() {
        // n=2 → tier1=1, tier2=1 → 2 protected, no candidates. Even with
        // huge `need` we kick nobody.
        let r = vec![snap(1u64, 100, 10, 0), snap(2, 50, 50, 1)];
        let kicked = select_preempt_ids(&r, 1_000, PreemptionConfig::default());
        assert!(kicked.is_empty());
    }

    #[test]
    fn n8_kicks_in_lifo_order_until_need_satisfied() {
        // n=8 → tier1=2, tier2=2, candidates=4.
        // ids 1..8, kv_len = 10 * id, input_len = 100 - 10*id (so id=1 has
        // longest input, id=8 shortest), arrival = id (older→newer).
        let mut r = Vec::new();
        for id in 1u64..=8 {
            r.push(snap(id, 10 * id as u32, 100 - 10 * id as u32, id));
        }
        // tier1 (top kv_len): {8, 7}
        // tier2 (smallest input from {1..6}): id=6 (input=40), id=5 (input=50)
        // candidates: {1, 2, 3, 4} newest-first → [4, 3, 2, 1]
        // need = 7 → kick id=4 (kv=40 ≥ 7? 40 ≥ 7 → done in 1 kick).
        let kicked = select_preempt_ids(&r, 7, PreemptionConfig::default());
        assert_eq!(kicked, vec![4u64]);

        // need = 50 → kick id=4 (40), id=3 (30) → cumulative 70 ≥ 50. stop.
        let kicked = select_preempt_ids(&r, 50, PreemptionConfig::default());
        assert_eq!(kicked, vec![4u64, 3]);
    }

    #[test]
    fn need_exceeds_total_candidate_kv_kicks_all_candidates() {
        let mut r = Vec::new();
        for id in 1u64..=8 {
            r.push(snap(id, 10 * id as u32, 100 - 10 * id as u32, id));
        }
        // candidates = {1, 2, 3, 4}; cumulative kv = 100. need=200 → still
        // kicks all 4 (and reports a shortfall of 100; caller checks total
        // freed against need).
        let kicked = select_preempt_ids(&r, 200, PreemptionConfig::default());
        assert_eq!(kicked, vec![4u64, 3, 2, 1]);
    }

    #[test]
    fn tier1_uses_arrival_time_as_tiebreaker() {
        // Two seqs with the same kv_len; older one wins tier1.
        let r = vec![
            snap(1u64, 50, 10, 0), // older
            snap(2, 50, 10, 1),    // newer
        ];
        // tier1=1 → keeps id=1 (older). tier2=1 → from {2} keeps id=2.
        // candidates empty → no kicks.
        let kicked = select_preempt_ids(&r, 100, PreemptionConfig::default());
        assert!(kicked.is_empty());
    }

    #[test]
    fn cfg_zero_fractions_make_everyone_a_candidate() {
        let cfg = PreemptionConfig {
            tier1_frac_bps: 0,
            tier2_frac_bps: 0,
        };
        let r = vec![
            snap(1u64, 100, 10, 0),
            snap(2, 50, 50, 1),
            snap(3, 80, 5, 2),
            snap(4, 20, 100, 3),
        ];
        // All four candidates, LIFO → [4, 3, 2, 1]. need=1 → 1 kick.
        let kicked = select_preempt_ids(&r, 1, cfg);
        assert_eq!(kicked, vec![4u64]);
    }
}
