use std::time::{Duration, Instant};

use half::bf16;

use infer_protocol::control_envelope::RequestId;
use infer_protocol::scheduler_to_worker_control::{DrainMode, LoadModel, SchedulerControlMessage};
use infer_protocol::scheduler_to_worker_data::{BatchCommand, PrefillBatchCmd};
use infer_protocol::worker_to_scheduler_control::{
    CancelAck, DrainAck, UnloadAck, WorkerCapacity, WorkerControlMessage,
};
use infer_protocol::worker_to_scheduler_data::DiffusionBatchOutput;

use crate::application::decode_engine::DecodeEngine;
use crate::application::runtime::Runtime;
use crate::application::sampler_stack::GreedySampler;
use crate::application::worker_scheduler::{PrefillCtx, handle_prefill};
use crate::application::worker_state::{ActiveSeqMap, PrefillSeqMap};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::domain::model::DecoderModel;
use crate::domain::ports::OpError;
use crate::infrastructure::cuda::{Cuda, CudaScope, device_utils};
use crate::infrastructure::transport::control_pump::ControlPump;
use crate::infrastructure::transport::data_pump::DataPump;
use crate::models::loader::LoadConfig;

/// Upper bound on how many consecutive prefill rounds the serve loop drains
/// before forcing a decode step. Caps prefill starvation of decode under a
/// sustained prefill backlog (extracted magic number, refactor #16).
const MAX_CONSECUTIVE_PREFILL_ROUNDS: usize = 16;

// cudaProfiler API for precise nsys capture.
unsafe extern "C" {
    fn cudaProfilerStart() -> u32;
    fn cudaProfilerStop() -> u32;
}

/// P2: Bundles the mutable worker state that `drain_control` and its helpers
/// pass around, eliminating 6 repeated parameters across every call site.
struct WorkerCtx<'a> {
    active: &'a mut ActiveSeqMap,
    prefilling: &'a mut PrefillSeqMap,
    decode_engine: &'a mut DecodeEngine,
    kv_allocator: &'a mut GlobalKvAllocator,
    enable_prefix_caching: bool,
}

/// Bundle of bootstrap parameters that do not depend on the concrete model type.
pub struct Bootstrap<'a> {
    pub load: &'a LoadModel,
    pub cuda: &'a Cuda,
    pub load_cfg: &'a LoadConfig,
    pub max_seq_len: usize,
    pub block_size: usize,
    pub num_blocks_override: usize,
    pub server_heartbeat_ms: Option<u64>,
    /// Resolved from config.json, not `load.model_type`.
    pub model_type: String,
    /// CUDA graph capture sizes for decode batches.
    pub capture_sizes: Vec<usize>,
}

pub fn run_with_model<M>(
    control: &ControlPump,
    data: &DataPump,
    model: M,
    bs: Bootstrap<'_>,
    eos_ids: &[i32],
    profile_cuda_steps: Option<u32>,
) -> Result<(), String>
where
    M: DecoderModel<bf16, Cuda>,
{
    let _ = bs.load_cfg;
    if bs.block_size != 1 {
        return Err(format!(
            "worker-owned KV allocator requires block_size=1, got {}",
            bs.block_size
        ));
    }
    let max_blocks_per_seq = bs.max_seq_len.div_ceil(bs.block_size);
    let model_dims = model.dims();

    let num_blocks = if bs.num_blocks_override != 0 {
        tracing::info!(
            "[bootstrap] num_blocks override from CLI: {} (skipping GPU mem probe)",
            bs.num_blocks_override,
        );
        bs.num_blocks_override
    } else {
        let bytes_per_block = model_dims.num_layers
            * 2
            * bs.block_size
            * model_dims.kv_dim
            * std::mem::size_of::<bf16>();
        let fraction = bs.load.kv_cache_memory_fraction.unwrap_or(0.9).clamp(
            crate::application::tuning::KV_MEM_FRACTION_MIN,
            crate::application::tuning::KV_MEM_FRACTION_MAX,
        );
        let (free, total) =
            device_utils::mem_get_info().map_err(|e| format!("cudaMemGetInfo: {:?}", e))?;
        let budget = (free as f64 * fraction as f64) as usize;
        let raw = budget / bytes_per_block.max(1);
        // Safety reserve (M8): the `free` probe is taken before the forward
        // activation workspace and the decode CUDA-graph capture pool are
        // allocated (both happen in `Runtime::new` / `prime_graphs`
        // *after* this point). `fraction` (default 0.9) is the primary
        // headroom for them; on top we keep an explicit reserve so rounding +
        // allocator fragmentation never pushes runtime peak past the probe:
        //   - 1 block is the graph scratch block (see `pool_blocks` below),
        //   - plus 0.5% of the raw block count (min 1) as fragmentation slack.
        // OOM on this path is otherwise swallowed by the kernel error layer,
        // so we err conservative here.
        let reserve = 1 + (raw / 200).max(1);
        let derived = raw.saturating_sub(reserve).max(1);
        tracing::info!(
            "[bootstrap] KV mem probe: free={:.2}GiB total={:.2}GiB fraction={} bytes/block={} -> num_blocks={} (~{:.2}GiB KV pool)",
            free as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64,
            fraction,
            bytes_per_block,
            derived,
            (derived * bytes_per_block) as f64 / (1u64 << 30) as f64,
        );
        derived
    };

    let pool_blocks = num_blocks + 1;
    tracing::info!(
        "[bootstrap] paged KV pool: block_size={} num_blocks={} pool_blocks={} (last block reserved as graph scratch) max_blocks_per_seq={}",
        bs.block_size,
        num_blocks,
        pool_blocks,
        max_blocks_per_seq,
    );

    let cap_num_tokens = bs.load.max_batch_tokens;
    let cap_batch = bs.load.max_batch_seqs;
    let mut runner: Runtime<bf16, Cuda, M> = Runtime::new(
        model,
        CudaScope::new(bs.cuda.clone()),
        Box::new(GreedySampler),
        pool_blocks,
        bs.block_size,
        max_blocks_per_seq,
        bs.max_seq_len,
        cap_num_tokens,
        cap_batch,
        bs.capture_sizes.clone(),
    )
    .map_err(|e| format!("Runtime::new: {:?}", e))?;

    // Install the decode CUDA-graph runner. On CUDA (arena reserved) this
    // enables real capture-on-first-hit / replay for decode-only batches whose
    // size matches a capture slot; on backends without graph support it is a
    // no-op and decode stays eager.
    if let Err(e) = runner.prime_graphs() {
        tracing::info!(
            "[bootstrap] graph priming skipped, eager decode: {:?}",
            e
        );
    }

    let max_total_kv_tokens = num_blocks * bs.block_size;
    control.send_ready(
        bs.load.model_instance_id.clone(),
        bs.load.model_path.clone(),
        bs.model_type.clone(),
        WorkerCapacity {
            max_batch_tokens: bs.load.max_batch_tokens,
            max_batch_seqs: bs.load.max_batch_seqs,
            max_running_requests: bs.load.max_batch_seqs,
            max_total_kv_tokens: Some(max_total_kv_tokens),
            free_mem_before_load_gb: None,
            free_mem_after_load_gb: None,
            weight_mem_usage_gb: None,
            workspace_mem_usage_gb: None,
            graph_mem_usage_gb: None,
        },
    )?;
    tracing::info!(
        "[bootstrap] Ready sent. max_total_kv_tokens={}. Entering serve loop...",
        max_total_kv_tokens,
    );

    let hb_ms = bs.server_heartbeat_ms.unwrap_or(1000).max(200);
    let heartbeat_interval = Duration::from_millis(hb_ms);
    tracing::info!("[serve] heartbeat interval = {:?}", heartbeat_interval);
    let mut last_heartbeat = Instant::now();
    let mut active = ActiveSeqMap::new();
    let mut prefilling = PrefillSeqMap::new();
    let mut kv_allocator = GlobalKvAllocator::new(num_blocks as u32);
    tracing::info!(
        "[serve] worker-owned KV allocator: total={} (block_size={})",
        num_blocks,
        bs.block_size,
    );
    let enable_prefix_caching = bs.load.enable_prefix_caching;
    tracing::info!(
        "[serve] prefix caching: {}",
        if enable_prefix_caching {
            "enabled (RadixTree)"
        } else {
            "disabled (real-time recycling)"
        },
    );
    let mut profile_step_count: u32 = 0;
    let mut profile_started = false;
    let mut decode_engine = DecodeEngine::new();

    loop {
        let mut ctx = WorkerCtx {
            active: &mut active,
            prefilling: &mut prefilling,
            decode_engine: &mut decode_engine,
            kv_allocator: &mut kv_allocator,
            enable_prefix_caching,
        };
        if drain_control(control, &mut ctx) {
            return Ok(());
        }
        // Drop ctx to release the mutable borrows for the rest of the loop body.
        drop(ctx);

        // ── Wait policy: zero-latency event-driven scheduling ──
        //
        // We multiplex the data PULL and control DEALER sockets through a
        // single `zmq::poll`, so *any* arriving message wakes us up
        // immediately — no per-socket polling phase that could stall a
        // freshly-arrived prefill behind a 500ms idle window.
        //
        // Timeout selection:
        //   * active decode in flight  → 0   : never sleep; the decode step
        //     below is the loop's natural clock. We only peek for new work.
        //   * fully idle               → block until the next heartbeat
        //     deadline, but get woken the instant a socket becomes readable.
        //
        // This removes the old `idle_wait_ms = heartbeat/2` polling window
        // that dominated TTFT at low QPS.
        let mut pending_prefills = drain_data(data);
        if pending_prefills.is_empty() && active.is_empty() && !decode_engine.has_pending() {
            maybe_heartbeat(
                control,
                active.len() + prefilling.len(),
                &mut last_heartbeat,
                heartbeat_interval,
                &kv_allocator,
            );

            // Block until a socket is readable or the next heartbeat is due.
            // `-1`-style infinite waits are avoided so heartbeats still fire
            // during long idle periods; the cap is the remaining time to the
            // next heartbeat (min 1ms to avoid a busy 0-timeout spin).
            let until_next_hb = heartbeat_interval
                .saturating_sub(last_heartbeat.elapsed())
                .as_millis() as i64;
            let idle_timeout_ms = until_next_hb.max(1);

            let mut items = [
                data.recv_socket().as_poll_item(zmq::POLLIN),
                control.recv_socket().as_poll_item(zmq::POLLIN),
            ];
            match zmq::poll(&mut items, idle_timeout_ms) {
                Ok(_) => {}
                Err(zmq::Error::EINTR) => continue,
                Err(e) => {
                    tracing::info!("[serve] zmq::poll error: {:?}", e);
                    continue;
                }
            }
            // Snapshot readiness and release the `PollItem` borrows on the
            // sockets before re-entering `drain_data` / `drain_control`.
            let data_ready = items[0].is_readable();
            let control_ready = items[1].is_readable();
            drop(items);

            // Data plane ready → pull every queued prefill in one go.
            if data_ready {
                pending_prefills.extend(drain_data(data));
            }
            // Control plane ready → handle it now (may be a cancel/shutdown
            // that voids the prefill we are about to run).
            if control_ready {
                let mut ctx = WorkerCtx {
                    active: &mut active,
                    prefilling: &mut prefilling,
                    decode_engine: &mut decode_engine,
                    kv_allocator: &mut kv_allocator,
                    enable_prefix_caching,
                };
                if drain_control(control, &mut ctx) {
                    return Ok(());
                }
            }

            if pending_prefills.is_empty() && active.is_empty() && !decode_engine.has_pending() {
                continue;
            }
        }

        // ── Prefill first, then decode ──
        //
        // New prefills are admitted *before* the decode step so the first
        // token is produced on this very iteration instead of waiting for a
        // full decode round-trip. `handle_prefill` emits the first token
        // directly (FinishPrefillAndStartDecode), so TTFT no longer eats an
        // extra decode-step latency.
        //
        // Decode is intentionally not interleaved between prefills in the
        // same drained backlog. Interleaving makes identical burst requests
        // enter decode at different batch shapes (1, 2, 3, ...), which can
        // expose shape-dependent kernel/numeric differences and permanently
        // diverge greedy outputs. Drain the currently available prefills first
        // so a short burst starts decode as one cohort.
        // A graph-eligible prefill writes buffer A on the compute stream; if a
        // decode step is still in flight, order that write after the pending
        // step's copy-out read of A (enqueue-only ev_out wait).
        if !pending_prefills.is_empty() && decode_engine.has_pending() {
            if let Err(e) = runner.guard_buffer_a_against_pending_copyout() {
                tracing::warn!("[serve] guard_buffer_a failed: {}", e);
            }
        }
        let mut prefill_rounds = 0usize;
        while !pending_prefills.is_empty() {
            prefill_rounds += 1;
            for cmd in std::mem::take(&mut pending_prefills) {
                {
                    let mut ctx = WorkerCtx {
                        active: &mut active,
                        prefilling: &mut prefilling,
                        decode_engine: &mut decode_engine,
                        kv_allocator: &mut kv_allocator,
                        enable_prefix_caching,
                    };
                    if drain_control(control, &mut ctx) {
                        return Ok(());
                    }
                }
                if let Err(e) = handle_prefill(
                    &mut PrefillCtx {
                        runner: &mut runner,
                        active: &mut active,
                        prefilling: &mut prefilling,
                        kv_allocator: &mut kv_allocator,
                        control,
                        data,
                        eos_ids,
                        enable_prefix_caching,
                        cap_batch,
                    },
                    &cmd,
                ) {
                    if matches!(e, OpError::Shutdown) {
                        tracing::info!("[serve] prefill interrupted by shutdown.");
                        return Ok(());
                    }
                    tracing::info!("[serve] prefill error: {}", e);
                }
            }

            if prefill_rounds >= MAX_CONSECUTIVE_PREFILL_ROUNDS {
                break;
            }

            pending_prefills = drain_data(data);
            // Drop the 1ms wait — if nothing is queued, proceed to decode
            // immediately. This removes up to 16ms of stall per serve loop
            // iteration when the prefill queue drains mid-burst.
            // pending_prefills = wait_for_prefill_quiet(data, Duration::from_millis(1));
        }

        // ── Decode step ──
        //
        // Drives every active sequence one token forward (pipelined 1-deep).
        // Also runs when `active` is empty but a step is still in flight, so the
        // last issued step is finalized and its tokens are sent.
        if !active.is_empty() || decode_engine.has_pending() {
            if let Some(max_steps) = profile_cuda_steps {
                if !profile_started {
                    // SAFETY: extern profiler API; returns a cudaError code.
                    let rc = unsafe { cudaProfilerStart() };
                    if rc != 0 {
                        tracing::warn!(rc, "cudaProfilerStart returned non-zero");
                    }
                    profile_started = true;
                    tracing::info!(
                        "[profile] cudaProfilerStart (will stop after {} steps)",
                        max_steps
                    );
                }
            }

            if let Err(e) = decode_engine.run_step(
                &mut runner,
                &mut active,
                &mut prefilling,
                &mut kv_allocator,
                control,
                data,
                eos_ids,
                enable_prefix_caching,
            ) {
                if matches!(e, OpError::Shutdown) {
                    tracing::info!("[serve] decode interrupted by shutdown.");
                    return Ok(());
                }
                tracing::info!("[serve] decode error: {}", e);
            }

            if let Some(max_steps) = profile_cuda_steps {
                if profile_started {
                    profile_step_count += 1;
                    if profile_step_count >= max_steps {
                        // SAFETY: extern profiler API; returns a cudaError code.
                        let rc = unsafe { cudaProfilerStop() };
                        if rc != 0 {
                            tracing::warn!(rc, "cudaProfilerStop returned non-zero");
                        }
                        tracing::info!(
                            "[profile] cudaProfilerStop after {} steps. Exiting.",
                            profile_step_count
                        );
                        return Ok(());
                    }
                }
            }
        }

        maybe_heartbeat(
            control,
            active.len() + prefilling.len(),
            &mut last_heartbeat,
            heartbeat_interval,
            &kv_allocator,
        );
    }
}

fn drain_control(control: &ControlPump, ctx: &mut WorkerCtx<'_>) -> bool {
    loop {
        match control.try_recv(0) {
            Ok(Some((msg, req_id))) => match msg {
                SchedulerControlMessage::Shutdown => {
                    tracing::info!("[serve] Shutdown received, exiting.");
                    return true;
                }
                SchedulerControlMessage::Cancel(c) => {
                    apply_cancel(control, ctx, c.sequence_id, req_id);
                }
                SchedulerControlMessage::FreeKvIndices(free) => {
                    if !free.indices.is_empty() {
                        ctx.kv_allocator.free(&free.indices);
                    }
                }
                SchedulerControlMessage::Preempt(p) => {
                    apply_preempt(ctx, &p.sequence_ids, &p.free_indices);
                }
                SchedulerControlMessage::Ping => {
                    if let Err(e) = control.send(WorkerControlMessage::Pong, req_id) {
                        tracing::info!("[serve] failed to send Pong: {}", e);
                    }
                }
                SchedulerControlMessage::Drain(d) => {
                    apply_drain(control, ctx, d.mode, req_id);
                }
                SchedulerControlMessage::UnloadModel(u) => {
                    if req_id.is_correlated() {
                        if let Err(e) = control.send(
                            WorkerControlMessage::UnloadAck(UnloadAck {
                                model_instance_id: u.model_instance_id,
                            }),
                            req_id,
                        ) {
                            tracing::info!("[serve] failed to send UnloadAck: {}", e);
                        }
                    }
                    tracing::info!("[serve] UnloadModel received, exiting.");
                    return true;
                }
                _ => {}
            },
            _ => break,
        }
    }
    false
}

/// Cancel one sequence: evict it from `active`/`prefilling`, release its KV,
/// resync decode rows, and ack if the request was correlated.
fn apply_cancel(
    control: &ControlPump,
    ctx: &mut WorkerCtx<'_>,
    sequence_id: u64,
    req_id: RequestId,
) {
    let removed = ctx.active.remove(&sequence_id);
    let removed_prefill = ctx.prefilling.remove(&sequence_id);
    let removed_flag = removed.is_some() || removed_prefill.is_some();
    if let Some(removed) = removed {
        ctx.kv_allocator
            .release_owned(&removed.block_table, ctx.enable_prefix_caching);
        ctx.decode_engine.retain_active(ctx.active);
        tracing::info!("[serve] cancelled seq {}", sequence_id);
    }
    if let Some(removed) = removed_prefill {
        ctx.kv_allocator
            .release_owned(&removed.block_table, ctx.enable_prefix_caching);
        tracing::info!("[serve] cancelled prefilling seq {}", sequence_id);
    }
    if req_id.is_correlated() {
        if let Err(e) = control.send(
            WorkerControlMessage::CancelAck(CancelAck {
                sequence_id,
                removed: removed_flag,
            }),
            req_id,
        ) {
            tracing::info!("[serve] failed to send CancelAck: {}", e);
        }
    }
}

/// Preempt a set of victims: evict each from `active`/`prefilling`, release
/// their KV, resync decode rows, then return any scheduler-freed slots.
fn apply_preempt(ctx: &mut WorkerCtx<'_>, sequence_ids: &[u64], free_indices: &[u32]) {
    for sid in sequence_ids {
        if let Some(removed) = ctx.active.remove(sid) {
            ctx.kv_allocator
                .release_owned(&removed.block_table, ctx.enable_prefix_caching);
            tracing::info!("[serve] preempted seq {}", sid);
        }
        if let Some(removed) = ctx.prefilling.remove(sid) {
            ctx.kv_allocator
                .release_owned(&removed.block_table, ctx.enable_prefix_caching);
            tracing::info!("[serve] preempted prefilling seq {}", sid);
        }
    }
    ctx.decode_engine.retain_active(ctx.active);
    if !free_indices.is_empty() {
        ctx.kv_allocator.free(free_indices);
    }
}

/// Drain: on `Immediate`, evict every sequence and release all KV; always ack
/// with the post-drain remaining-request count if correlated.
fn apply_drain(control: &ControlPump, ctx: &mut WorkerCtx<'_>, mode: DrainMode, req_id: RequestId) {
    if matches!(mode, DrainMode::Immediate) {
        // P1: Iterate drain() directly instead of collecting into a Vec.
        for (_, seq) in ctx.active.drain() {
            ctx.kv_allocator
                .release_owned(&seq.block_table, ctx.enable_prefix_caching);
        }
        for (_, seq) in ctx.prefilling.drain() {
            ctx.kv_allocator
                .release_owned(&seq.block_table, ctx.enable_prefix_caching);
        }
        // Free the in-flight decode step's slots (not yet in any block table)
        // before clearing, else they leak from the pool for the process life.
        ctx.decode_engine.reclaim_pending(ctx.kv_allocator);
        ctx.decode_engine.clear();
    }
    if req_id.is_correlated() {
        if let Err(e) = control.send(
            WorkerControlMessage::DrainAck(DrainAck {
                remaining_requests: ctx.active.len() + ctx.prefilling.len(),
            }),
            req_id,
        ) {
            tracing::info!("[serve] failed to send DrainAck: {}", e);
        }
    }
}

fn drain_data(data: &DataPump) -> Vec<PrefillBatchCmd> {
    let mut pending_prefills = Vec::new();
    loop {
        match data.try_recv_batch(0) {
            Ok(Some(BatchCommand::Prefill(p))) => pending_prefills.push(p),
            Ok(Some(BatchCommand::DiffusionBatch(_))) => {
                if let Err(e) =
                    data.send_diffusion_output(&DiffusionBatchOutput { results: vec![] })
                {
                    tracing::info!("[serve] failed to send empty diffusion output: {}", e);
                }
            }
            _ => break,
        }
    }
    pending_prefills
}

fn wait_for_prefill_quiet(data: &DataPump, quiet: Duration) -> Vec<PrefillBatchCmd> {
    let timeout_ms = quiet.as_millis().max(1) as i64;
    let mut items = [data.recv_socket().as_poll_item(zmq::POLLIN)];
    match zmq::poll(&mut items, timeout_ms) {
        Ok(_) => {}
        Err(zmq::Error::EINTR) => return Vec::new(),
        Err(e) => {
            tracing::info!("[serve] prefill quiet poll error: {:?}", e);
            return Vec::new();
        }
    }
    let data_ready = items[0].is_readable();
    drop(items);
    if data_ready {
        drain_data(data)
    } else {
        Vec::new()
    }
}

fn maybe_heartbeat(
    control: &ControlPump,
    active_n: usize,
    last: &mut Instant,
    interval: Duration,
    kv_allocator: &GlobalKvAllocator,
) {
    if last.elapsed() >= interval {
        if let Err(e) = control.send_heartbeat(
            active_n,
            Some(kv_allocator.outstanding()),
            Some(kv_allocator.total_free()),
            Some(kv_allocator.released_len() as u32),
        ) {
            tracing::info!("[serve] failed to send heartbeat: {}", e);
        }
        *last = Instant::now();
    }
}
