use std::time::{Duration, Instant};

use infer_protocol::control_envelope::RequestId;
use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
use infer_protocol::worker_to_scheduler_control::WorkerControlMessage;

use crate::application::worker_state::{ActiveSeq, ActiveSeqMap, PrefillSeq, PrefillSeqMap};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::infrastructure::transport::control_pump::ControlPump;

/// Control-plane operations the KV-relief loop depends on. Abstracted into a
/// trait (H6) so the relief state machine can be unit-tested against a
/// scripted mock instead of a live ZMQ `ControlPump`.
pub trait ReliefControl {
    fn try_recv(
        &self,
        timeout_ms: i64,
    ) -> Result<Option<(SchedulerControlMessage, RequestId)>, String>;
    fn send(&self, msg: WorkerControlMessage, request_id: RequestId) -> Result<(), String>;
    fn send_alloc_failed(&self, shortfall: u32, round: u8) -> Result<(), String>;
}

impl ReliefControl for ControlPump {
    fn try_recv(
        &self,
        timeout_ms: i64,
    ) -> Result<Option<(SchedulerControlMessage, RequestId)>, String> {
        ControlPump::try_recv(self, timeout_ms)
    }
    fn send(&self, msg: WorkerControlMessage, request_id: RequestId) -> Result<(), String> {
        ControlPump::send(self, msg, request_id)
    }
    fn send_alloc_failed(&self, shortfall: u32, round: u8) -> Result<(), String> {
        ControlPump::send_alloc_failed(self, shortfall, round)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReliefWaitOutcome {
    Satisfied,
    TimedOut,
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocWithReliefOutcome {
    Allocated(Vec<u32>),
    Unavailable,
    Shutdown,
}

/// Block-poll the control plane until the scheduler ships KV relief
/// (either a `FreeKvIndices` or a `Preempt`) or `wait_ms` elapses.
///
/// Other control messages that arrive during the wait (Cancel,
/// Shutdown, Ping) are handled inline because ZMQ has no peek API.
pub fn wait_for_relief<C: ReliefControl>(
    control: &C,
    kv_allocator: &mut GlobalKvAllocator,
    active: &mut ActiveSeqMap,
    prefilling: &mut PrefillSeqMap,
    needed_slots: u32,
    wait_ms: i64,
    enable_prefix_caching: bool,
    shrink_to_active: bool,
) -> ReliefWaitOutcome {
    let deadline = Instant::now() + Duration::from_millis(wait_ms.max(0) as u64);
    while Instant::now() < deadline {
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .as_millis() as i64;
        let poll_ms = remaining.min(50).max(1);
        match control.try_recv(poll_ms) {
            Ok(Some((SchedulerControlMessage::FreeKvIndices(f), _))) => {
                if !f.indices.is_empty() {
                    tracing::warn!(
                        "[serve] relief received: FreeKvIndices count={}",
                        f.indices.len()
                    );
                    kv_allocator.free(&f.indices);
                } else {
                    tracing::warn!("[serve] relief received: FreeKvIndices count=0");
                }
                if relief_satisfies_request(kv_allocator, active, needed_slots, shrink_to_active) {
                    return ReliefWaitOutcome::Satisfied;
                }
                log_partial_relief(kv_allocator, needed_slots);
            }
            Ok(Some((SchedulerControlMessage::Preempt(p), _))) => {
                tracing::warn!(
                    "[serve] relief received: Preempt victims={} free_indices={}",
                    p.sequence_ids.len(),
                    p.free_indices.len()
                );
                for sid in &p.sequence_ids {
                    if let Some(entry) = active.remove(sid) {
                        release_removed_active(entry, kv_allocator, enable_prefix_caching);
                    }
                    if let Some(entry) = prefilling.remove(sid) {
                        release_removed_prefill(entry, kv_allocator, enable_prefix_caching);
                    }
                }
                if !p.free_indices.is_empty() {
                    kv_allocator.free(&p.free_indices);
                }
                if relief_satisfies_request(kv_allocator, active, needed_slots, shrink_to_active) {
                    return ReliefWaitOutcome::Satisfied;
                }
                log_partial_relief(kv_allocator, needed_slots);
            }
            Ok(Some((SchedulerControlMessage::Shutdown, _))) => {
                tracing::warn!("[serve] Shutdown received during wait_for_relief");
                return ReliefWaitOutcome::Shutdown;
            }
            Ok(Some((SchedulerControlMessage::Cancel(c), _))) => {
                let mut cancelled = false;
                if let Some(removed) = active.remove(&c.sequence_id) {
                    release_removed_active(removed, kv_allocator, enable_prefix_caching);
                    cancelled = true;
                }
                if let Some(removed) = prefilling.remove(&c.sequence_id) {
                    release_removed_prefill(removed, kv_allocator, enable_prefix_caching);
                    cancelled = true;
                }
                if cancelled {
                    tracing::warn!(
                        "[serve] cancelled seq {} (during relief wait)",
                        c.sequence_id
                    );
                }
                if relief_satisfies_request(kv_allocator, active, needed_slots, shrink_to_active) {
                    return ReliefWaitOutcome::Satisfied;
                }
            }
            Ok(Some((SchedulerControlMessage::Ping, req_id))) => {
                if let Err(e) = control.send(WorkerControlMessage::Pong, req_id) {
                    tracing::warn!("[serve] failed to send Pong during relief wait: {}", e);
                    return ReliefWaitOutcome::TimedOut;
                }
            }
            Ok(Some(_)) => continue,
            _ => continue,
        }
    }
    ReliefWaitOutcome::TimedOut
}

/// Try to allocate `n_initial` slots. On `AllocFull`, signal the
/// scheduler with `AllocFailed{round=0}`, block-poll for relief,
/// retry, then escalate once to round 1.
pub fn alloc_with_relief<C: ReliefControl>(
    kv_allocator: &mut GlobalKvAllocator,
    control: &C,
    active: &mut ActiveSeqMap,
    prefilling: &mut PrefillSeqMap,
    n_initial: u32,
    enable_prefix_caching: bool,
    shrink_to_active: bool,
) -> AllocWithReliefOutcome {
    const RELIEF_TIMEOUT_MS: i64 = 500;
    let mut round: u8 = 0;
    let mut n = n_initial;
    let mut retried_after_round1_relief = false;
    loop {
        if shrink_to_active {
            n = n.min(active.len() as u32);
        }
        if n == 0 {
            return AllocWithReliefOutcome::Allocated(Vec::new());
        }
        match kv_allocator.alloc_indices(n) {
            Ok(v) => return AllocWithReliefOutcome::Allocated(v),
            Err(e) => {
                if retried_after_round1_relief {
                    tracing::warn!(
                        "[serve] alloc({}) still failing after round=1 relief -- giving up",
                        n
                    );
                    return AllocWithReliefOutcome::Unavailable;
                }
                tracing::warn!(
                    "[serve] alloc({}) failed at round={}: {} -- requesting relief",
                    n, round, e
                );
                if let Err(send_err) = control.send_alloc_failed(n, round) {
                    tracing::warn!("[serve] failed to send AllocFailed: {}", send_err);
                    return AllocWithReliefOutcome::Unavailable;
                }
                let relief = wait_for_relief(
                    control,
                    kv_allocator,
                    active,
                    prefilling,
                    n,
                    RELIEF_TIMEOUT_MS,
                    enable_prefix_caching,
                    shrink_to_active,
                );
                if shrink_to_active {
                    let active_now = active.len() as u32;
                    if active_now < n {
                        n = active_now;
                        continue;
                    }
                }
                if matches!(relief, ReliefWaitOutcome::Shutdown) {
                    return AllocWithReliefOutcome::Shutdown;
                }
                if !matches!(relief, ReliefWaitOutcome::Satisfied) {
                    tracing::warn!(
                        "[serve] relief timed out at round={} (still need {} slots)",
                        round, n
                    );
                    if round == 0 {
                        round = 1;
                        continue;
                    }
                    return AllocWithReliefOutcome::Unavailable;
                }
                if round == 0 {
                    round = 1;
                } else {
                    retried_after_round1_relief = true;
                }
            }
        }
    }
}

fn relief_satisfies_request(
    kv_allocator: &GlobalKvAllocator,
    active: &ActiveSeqMap,
    needed_slots: u32,
    shrink_to_active: bool,
) -> bool {
    if needed_slots == 0 {
        return true;
    }
    if shrink_to_active && (active.len() as u32) < needed_slots {
        return true;
    }
    kv_allocator.total_free() >= needed_slots
}

fn log_partial_relief(kv_allocator: &GlobalKvAllocator, needed_slots: u32) {
    tracing::warn!(
        "[serve] relief partial: need={} available={} total_free={} -- continuing wait",
        needed_slots,
        kv_allocator.available(),
        kv_allocator.total_free(),
    );
}

fn release_removed_active(
    removed: ActiveSeq,
    kv_allocator: &mut GlobalKvAllocator,
    enable_prefix_caching: bool,
) {
    if removed.block_table.is_empty() {
        return;
    }
    if !enable_prefix_caching {
        kv_allocator.release(&removed.block_table);
    }
}

fn release_removed_prefill(
    removed: PrefillSeq,
    kv_allocator: &mut GlobalKvAllocator,
    enable_prefix_caching: bool,
) {
    if removed.block_table.is_empty() {
        return;
    }
    if !enable_prefix_caching {
        kv_allocator.release(&removed.block_table);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::worker_state::ActiveSeq;
    use infer_protocol::scheduler_to_worker_control::FreeKvIndices;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    /// Scripted `ReliefControl` mock. `recv_script` is consumed front-to-back;
    /// once exhausted it returns `Ok(None)` (the "no message" poll result).
    struct MockControl {
        recv_script: RefCell<VecDeque<Option<(SchedulerControlMessage, RequestId)>>>,
        alloc_failed: RefCell<Vec<(u32, u8)>>,
        sent: RefCell<Vec<WorkerControlMessage>>,
        fail_alloc_failed: bool,
    }

    impl MockControl {
        fn new(script: Vec<Option<(SchedulerControlMessage, RequestId)>>) -> Self {
            Self {
                recv_script: RefCell::new(script.into_iter().collect()),
                alloc_failed: RefCell::new(Vec::new()),
                sent: RefCell::new(Vec::new()),
                fail_alloc_failed: false,
            }
        }
    }

    impl ReliefControl for MockControl {
        fn try_recv(
            &self,
            _timeout_ms: i64,
        ) -> Result<Option<(SchedulerControlMessage, RequestId)>, String> {
            match self.recv_script.borrow_mut().pop_front() {
                Some(item) => Ok(item),
                None => Ok(None),
            }
        }
        fn send(&self, msg: WorkerControlMessage, _req: RequestId) -> Result<(), String> {
            self.sent.borrow_mut().push(msg);
            Ok(())
        }
        fn send_alloc_failed(&self, shortfall: u32, round: u8) -> Result<(), String> {
            self.alloc_failed.borrow_mut().push((shortfall, round));
            if self.fail_alloc_failed {
                Err("mock control plane down".into())
            } else {
                Ok(())
            }
        }
    }

    fn free_kv(indices: Vec<u32>) -> Option<(SchedulerControlMessage, RequestId)> {
        Some((
            SchedulerControlMessage::FreeKvIndices(FreeKvIndices {
                model_instance_id: String::new(),
                indices,
            }),
            RequestId::NONE,
        ))
    }

    fn active_seq(block_table: Vec<u32>) -> ActiveSeq {
        ActiveSeq {
            last_token: 0,
            kv_len: block_table.len(),
            block_table,
            max_tokens: 16,
            generated_count: 1,
            ignore_eos: false,
        }
    }

    // ── relief_satisfies_request (pure) ──────────────────────────────

    #[test]
    fn satisfies_when_zero_needed() {
        let kv = GlobalKvAllocator::new(4);
        let active = ActiveSeqMap::new();
        assert!(relief_satisfies_request(&kv, &active, 0, false));
    }

    #[test]
    fn satisfies_when_shrink_to_active_below_need() {
        let mut kv = GlobalKvAllocator::new(4);
        let _ = kv.alloc_indices(4).unwrap(); // exhausted
        let active = ActiveSeqMap::new(); // 0 active < needed
        assert!(relief_satisfies_request(&kv, &active, 2, true));
        // Without shrink, an exhausted pool does NOT satisfy.
        assert!(!relief_satisfies_request(&kv, &active, 2, false));
    }

    #[test]
    fn satisfies_when_total_free_meets_need() {
        let kv = GlobalKvAllocator::new(4);
        let active = ActiveSeqMap::new();
        assert!(relief_satisfies_request(&kv, &active, 4, false));
        // total=4 free, asking 5 → not satisfied.
        assert!(!relief_satisfies_request(&kv, &active, 5, false));
    }

    // ── wait_for_relief ──────────────────────────────────────────────

    #[test]
    fn wait_returns_satisfied_on_freeing_relief() {
        let mut kv = GlobalKvAllocator::new(2);
        let _ = kv.alloc_indices(2).unwrap(); // free=0
        let mut active = ActiveSeqMap::new();
        let mut prefilling = PrefillSeqMap::new();
        let ctl = MockControl::new(vec![free_kv(vec![0, 1])]);
        let out = wait_for_relief(&ctl, &mut kv, &mut active, &mut prefilling, 2, 50, false, false);
        assert_eq!(out, ReliefWaitOutcome::Satisfied);
        assert_eq!(kv.total_free(), 2);
    }

    #[test]
    fn wait_returns_shutdown() {
        let mut kv = GlobalKvAllocator::new(2);
        let mut active = ActiveSeqMap::new();
        let mut prefilling = PrefillSeqMap::new();
        let ctl = MockControl::new(vec![Some((SchedulerControlMessage::Shutdown, RequestId::NONE))]);
        let out = wait_for_relief(&ctl, &mut kv, &mut active, &mut prefilling, 2, 50, false, false);
        assert_eq!(out, ReliefWaitOutcome::Shutdown);
    }

    #[test]
    fn wait_times_out_when_no_relief() {
        let mut kv = GlobalKvAllocator::new(2);
        let _ = kv.alloc_indices(2).unwrap();
        let mut active = ActiveSeqMap::new();
        let mut prefilling = PrefillSeqMap::new();
        let ctl = MockControl::new(vec![]); // always Ok(None)
        let out = wait_for_relief(&ctl, &mut kv, &mut active, &mut prefilling, 2, 5, false, false);
        assert_eq!(out, ReliefWaitOutcome::TimedOut);
    }

    #[test]
    fn wait_partial_then_satisfied() {
        let mut kv = GlobalKvAllocator::new(4);
        let _ = kv.alloc_indices(4).unwrap(); // free=0
        let mut active = ActiveSeqMap::new();
        let mut prefilling = PrefillSeqMap::new();
        // First relief frees 1 (need 3 → partial), second frees 2 more → satisfied.
        let ctl = MockControl::new(vec![free_kv(vec![0]), free_kv(vec![1, 2])]);
        let out = wait_for_relief(&ctl, &mut kv, &mut active, &mut prefilling, 3, 100, false, false);
        assert_eq!(out, ReliefWaitOutcome::Satisfied);
        assert_eq!(kv.total_free(), 3);
    }

    // ── alloc_with_relief ────────────────────────────────────────────

    #[test]
    fn alloc_succeeds_immediately_without_relief() {
        let mut kv = GlobalKvAllocator::new(4);
        let mut active = ActiveSeqMap::new();
        let mut prefilling = PrefillSeqMap::new();
        let ctl = MockControl::new(vec![]);
        let out = alloc_with_relief(&mut kv, &ctl, &mut active, &mut prefilling, 2, false, false);
        assert_eq!(out, AllocWithReliefOutcome::Allocated(vec![0, 1]));
        assert!(ctl.alloc_failed.borrow().is_empty(), "no relief should be requested");
    }

    #[test]
    fn alloc_shrink_to_active_zero_returns_empty() {
        let mut kv = GlobalKvAllocator::new(4);
        let _ = kv.alloc_indices(4).unwrap(); // exhausted
        let mut active = ActiveSeqMap::new(); // 0 active
        let mut prefilling = PrefillSeqMap::new();
        let ctl = MockControl::new(vec![]);
        // shrink_to_active clamps n to active.len()==0 → Allocated(empty), no relief.
        let out = alloc_with_relief(&mut kv, &ctl, &mut active, &mut prefilling, 3, false, true);
        assert_eq!(out, AllocWithReliefOutcome::Allocated(Vec::new()));
        assert!(ctl.alloc_failed.borrow().is_empty());
    }

    #[test]
    fn alloc_unavailable_when_alloc_failed_send_errors() {
        let mut kv = GlobalKvAllocator::new(2);
        let _ = kv.alloc_indices(2).unwrap(); // exhausted
        let mut active = ActiveSeqMap::new();
        let mut prefilling = PrefillSeqMap::new();
        let mut ctl = MockControl::new(vec![]);
        ctl.fail_alloc_failed = true;
        let out = alloc_with_relief(&mut kv, &ctl, &mut active, &mut prefilling, 2, false, false);
        assert_eq!(out, AllocWithReliefOutcome::Unavailable);
        assert_eq!(ctl.alloc_failed.borrow().len(), 1, "AllocFailed was attempted once");
    }

    #[test]
    fn alloc_succeeds_after_relief_frees_slots() {
        let mut kv = GlobalKvAllocator::new(2);
        let _ = kv.alloc_indices(2).unwrap(); // exhausted, outstanding=[0,1]
        let mut active = ActiveSeqMap::new();
        let mut prefilling = PrefillSeqMap::new();
        // round0: AllocFailed → relief frees [0,1] → retry allocates.
        let ctl = MockControl::new(vec![free_kv(vec![0, 1])]);
        let out = alloc_with_relief(&mut kv, &ctl, &mut active, &mut prefilling, 2, false, false);
        assert_eq!(out, AllocWithReliefOutcome::Allocated(vec![0, 1]));
        assert_eq!(ctl.alloc_failed.borrow().as_slice(), &[(2, 0)]);
    }

    #[test]
    fn alloc_with_relief_preempt_frees_and_removes_victim() {
        let mut kv = GlobalKvAllocator::new(2);
        let _ = kv.alloc_indices(2).unwrap(); // exhausted
        let mut active = ActiveSeqMap::new();
        active.insert(7, active_seq(vec![0, 1]));
        let mut prefilling = PrefillSeqMap::new();
        // Preempt victim 7 and hand back its slots.
        let preempt = Some((
            SchedulerControlMessage::Preempt(infer_protocol::scheduler_to_worker_control::Preempt {
                model_instance_id: String::new(),
                sequence_ids: vec![7],
                free_indices: vec![0, 1],
            }),
            RequestId::NONE,
        ));
        let ctl = MockControl::new(vec![preempt]);
        let out = alloc_with_relief(&mut kv, &ctl, &mut active, &mut prefilling, 2, false, false);
        assert_eq!(out, AllocWithReliefOutcome::Allocated(vec![0, 1]));
        assert!(!active.contains_key(&7), "preempted victim removed from active");
    }
}
