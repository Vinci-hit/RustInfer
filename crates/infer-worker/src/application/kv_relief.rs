use std::time::{Duration, Instant};

use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
use infer_protocol::worker_to_scheduler_control::WorkerControlMessage;

use crate::application::worker_state::{ActiveSeq, ActiveSeqMap};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::infrastructure::transport::control_pump::ControlPump;

/// Block-poll the control plane until the scheduler ships KV relief
/// (either a `FreeKvIndices` or a `Preempt`) or `wait_ms` elapses.
///
/// Other control messages that arrive during the wait (Cancel,
/// Shutdown, Ping) are handled inline because ZMQ has no peek API.
pub fn wait_for_relief(
    control: &ControlPump,
    kv_allocator: &mut GlobalKvAllocator,
    active: &mut ActiveSeqMap,
    wait_ms: i64,
    enable_prefix_caching: bool,
) -> bool {
    let deadline = Instant::now() + Duration::from_millis(wait_ms.max(0) as u64);
    while Instant::now() < deadline {
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .as_millis() as i64;
        let poll_ms = remaining.min(50).max(1);
        match control.try_recv(poll_ms) {
            Ok(Some((SchedulerControlMessage::FreeKvIndices(f), _))) => {
                if !f.indices.is_empty() {
                    kv_allocator.free(&f.indices);
                }
                return true;
            }
            Ok(Some((SchedulerControlMessage::Preempt(p), _))) => {
                for sid in &p.sequence_ids {
                    if let Some(entry) = active.remove(sid) {
                        release_removed(entry, kv_allocator, enable_prefix_caching);
                    }
                }
                if !p.free_indices.is_empty() {
                    kv_allocator.free(&p.free_indices);
                }
                return true;
            }
            Ok(Some((SchedulerControlMessage::Shutdown, _))) => {
                eprintln!("[serve] Shutdown received during wait_for_relief");
                std::process::exit(0);
            }
            Ok(Some((SchedulerControlMessage::Cancel(c), _))) => {
                if let Some(removed) = active.remove(&c.sequence_id) {
                    release_removed(removed, kv_allocator, enable_prefix_caching);
                    eprintln!(
                        "[serve] cancelled seq {} (during relief wait)",
                        c.sequence_id
                    );
                }
            }
            Ok(Some((SchedulerControlMessage::Ping, req_id))) => {
                let _ = control.send(WorkerControlMessage::Pong, req_id);
            }
            Ok(Some(_)) => continue,
            _ => continue,
        }
    }
    false
}

/// Try to allocate `n_initial` slots. On `AllocFull`, signal the
/// scheduler with `AllocFailed{round=0}`, block-poll for relief,
/// retry, then escalate once to round 1.
pub fn alloc_with_relief(
    kv_allocator: &mut GlobalKvAllocator,
    control: &ControlPump,
    active: &mut ActiveSeqMap,
    n_initial: u32,
    enable_prefix_caching: bool,
    shrink_to_active: bool,
) -> Option<Vec<u32>> {
    const RELIEF_TIMEOUT_MS: i64 = 500;
    let mut round: u8 = 0;
    let mut n = n_initial;
    let mut retried_after_round1_relief = false;
    loop {
        if shrink_to_active {
            n = n.min(active.len() as u32);
        }
        if n == 0 {
            return Some(Vec::new());
        }
        match kv_allocator.alloc_indices(n) {
            Ok(v) => return Some(v),
            Err(e) => {
                if retried_after_round1_relief {
                    eprintln!(
                        "[serve] alloc({}) still failing after round=1 relief -- giving up",
                        n
                    );
                    return None;
                }
                eprintln!(
                    "[serve] alloc({}) failed at round={}: {} -- requesting relief",
                    n, round, e
                );
                let _ = control.send_alloc_failed(n, round);
                let relieved = wait_for_relief(
                    control,
                    kv_allocator,
                    active,
                    RELIEF_TIMEOUT_MS,
                    enable_prefix_caching,
                );
                if shrink_to_active {
                    let active_now = active.len() as u32;
                    if active_now < n {
                        n = active_now;
                        continue;
                    }
                }
                if !relieved {
                    eprintln!(
                        "[serve] relief timed out at round={} (still need {} slots)",
                        round, n
                    );
                    if round == 0 {
                        round = 1;
                        continue;
                    }
                    return None;
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

fn release_removed(
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
