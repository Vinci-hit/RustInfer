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
use crate::application::worker_scheduler::handle_fused_step;
use crate::application::worker_state::{ActiveSeqMap, PrefillSeqMap};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::domain::model::DecoderModel;
use crate::domain::ports::OpError;
use crate::infrastructure::cuda::{Cuda, CudaScope, device_utils};
use crate::infrastructure::transport::control_pump::ControlPump;
use crate::infrastructure::transport::data_pump::DataPump;
use crate::models::loader::LoadConfig;

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
        let probed = raw.saturating_sub(reserve).max(1);
        // Cap the auto-sized pool at the working set the worker can ever hold
        // in use: at most `max_batch_seqs` sequences, each at most
        // `max_seq_len` tokens. The scheduler never admits more than
        // `max_batch_seqs` running sequences, so any block beyond this is
        // unreachable — it only wastes VRAM AND inflates every O(num_blocks)
        // op in `GlobalKvAllocator`. `release_owned`→`free()` does a full-pool
        // compact+merge on EVERY sequence completion; at a VRAM-filling ~780k
        // pool that is ~2ms/completion (measured), which lands in serve-loop
        // ticks alongside prefills and directly inflates TTFT on slower CPUs.
        // With prefix caching ON, extra blocks cache evicted prefixes, so keep
        // the full probe; with it OFF (real-time recycling) the live working
        // set is the only thing that can be allocated, so clamp to it.
        let derived = if bs.load.enable_prefix_caching {
            probed
        } else {
            let working_set = bs
                .load
                .max_batch_seqs
                .saturating_mul(max_blocks_per_seq)
                .max(1);
            probed.min(working_set)
        };
        tracing::info!(
            "[bootstrap] KV mem probe: free={:.2}GiB total={:.2}GiB fraction={} bytes/block={} probed={} -> num_blocks={} (~{:.2}GiB KV pool)",
            free as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64,
            fraction,
            bytes_per_block,
            probed,
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
        tracing::info!("[bootstrap] graph priming skipped, eager decode: {:?}", e);
    } else {
        // Capture every decode graph now, before serving, so no live decode
        // step ever pays an inline capture stall (the dominant TTFT/TPOT tail
        // spike under load — see `Runtime::prewarm_decode_graphs`).
        let t_warm = Instant::now();
        match runner.prewarm_decode_graphs() {
            Ok(()) => tracing::info!(
                "[bootstrap] decode graphs prewarmed ({} sizes) in {:.2}s",
                bs.capture_sizes.len(),
                t_warm.elapsed().as_secs_f64(),
            ),
            Err(e) => tracing::info!("[bootstrap] decode graph prewarm skipped: {:?}", e),
        }
        // Warm the prefill path across a length grid so the first live prefill
        // of each shape pays no inline allocator/library cost (the residual
        // TTFT p99 tail after decode-graph prewarm). Grid is coarse — the CUDA
        // allocator bins sizes, so nearby lengths reuse warmed pools.
        let prefill_grid: Vec<usize> = [
            8usize, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 768,
            1024,
        ]
        .into_iter()
        .filter(|&l| l <= bs.max_seq_len)
        .collect();
        let t_pf = Instant::now();
        match runner.prewarm_prefill_shapes(&prefill_grid) {
            Ok(()) => tracing::info!(
                "[bootstrap] prefill shapes prewarmed ({} lengths) in {:.2}s",
                prefill_grid.len(),
                t_pf.elapsed().as_secs_f64(),
            ),
            Err(e) => tracing::info!("[bootstrap] prefill prewarm skipped: {:?}", e),
        }
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

    // Per-step stall tracer (off unless RUSTINFER_STEP_TRACE set). Logs any
    // serve-loop *work* iteration (prefill+decode, excluding idle poll wait)
    // whose wall time >= RUSTINFER_STEP_TRACE_MS (default 20ms), with the
    // batch shape + KV pool state that produced it. One run → fingerprint of
    // the wandering once-per-run stall.
    let trace_steps = std::env::var_os("RUSTINFER_STEP_TRACE").is_some();
    let step_trace_ms: f64 = std::env::var("RUSTINFER_STEP_TRACE_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20.0);
    if trace_steps {
        tracing::info!("[step-trace] enabled, threshold={:.1}ms", step_trace_ms);
    }

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
                decode_engine.transient_reserved_slots(),
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

        // Stall tracer: start timing the *work* section (idle poll wait above
        // is intentionally excluded — it is not a stall).
        let work_t0 = if trace_steps {
            Some(Instant::now())
        } else {
            None
        };
        let mut tr_pf_seqs = 0usize;
        let mut tr_pf_tokens = 0usize;
        let tr_pf_ms = 0f64;
        let mut tr_dec_ms = 0f64;
        let mut tr_dec_batch = 0usize;

        // ── Fused step: prefill chunks + decode in ONE ragged forward ──
        //
        // Every pending prefill chunk and every active decode row go through a
        // single eager ragged forward (`handle_fused_step`). This pays the fixed
        // per-forward host overhead once per step instead of once per prefill,
        // and in-flight decode rows advance a token in that same forward rather
        // than stalling behind the prefills. When nothing is prefilling, steady-
        // state decode keeps the fast graphed ABC pipeline (`run_step`).
        let prefill_rounds = if pending_prefills.is_empty() {
            0usize
        } else {
            1usize
        };
        if !pending_prefills.is_empty() {
            if trace_steps {
                for cmd in &pending_prefills {
                    tr_pf_seqs += cmd.segments.len();
                    tr_pf_tokens += cmd.input_ids.len();
                }
                tr_dec_batch = active.len();
            }
            let fused_t0 = if trace_steps {
                Some(Instant::now())
            } else {
                None
            };
            if let Err(e) = handle_fused_step(
                &mut runner,
                &mut decode_engine,
                &mut active,
                &mut prefilling,
                &mut kv_allocator,
                control,
                data,
                eos_ids,
                enable_prefix_caching,
                cap_batch,
                cap_num_tokens,
                std::mem::take(&mut pending_prefills),
            ) {
                if matches!(e, OpError::Shutdown) {
                    tracing::info!("[serve] fused step interrupted by shutdown.");
                    return Ok(());
                }
                if e.is_fatal() {
                    escalate_fatal_and_exit(control, &active, &prefilling, &e);
                }
                tracing::info!("[serve] fused step error: {}", e);
            }
            if let Some(t) = fused_t0 {
                tr_dec_ms = t.elapsed().as_secs_f64() * 1e3;
            }
        } else if !active.is_empty() || decode_engine.has_pending() {
            // ── Pure decode step (graphed ABC pipeline, 1-deep) ──
            //
            // Runs when nothing is prefilling. Also runs when `active` is empty
            // but a step is still in flight, so the last issued step is
            // finalized and its tokens are sent.
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

            let dec_t0 = if trace_steps {
                tr_dec_batch = active.len();
                Some(Instant::now())
            } else {
                None
            };
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
                if e.is_fatal() {
                    escalate_fatal_and_exit(control, &active, &prefilling, &e);
                }
                tracing::info!("[serve] decode error: {}", e);
            }
            if let Some(t) = dec_t0 {
                tr_dec_ms = t.elapsed().as_secs_f64() * 1e3;
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

        if let Some(t0) = work_t0 {
            let iter_ms = t0.elapsed().as_secs_f64() * 1e3;
            if iter_ms >= step_trace_ms {
                tracing::info!(
                    "[step-trace] work={:.1}ms pf={:.1}ms(seqs={} tok={} rounds={}) dec={:.1}ms(batch={}) active={} kv[out={} free={} rel={}]",
                    iter_ms,
                    tr_pf_ms,
                    tr_pf_seqs,
                    tr_pf_tokens,
                    prefill_rounds,
                    tr_dec_ms,
                    tr_dec_batch,
                    active.len(),
                    kv_allocator.outstanding(),
                    kv_allocator.total_free(),
                    kv_allocator.released_len(),
                );
            }
        }

        maybe_heartbeat(
            control,
            active.len() + prefilling.len(),
            &mut last_heartbeat,
            heartbeat_interval,
            &kv_allocator,
            decode_engine.transient_reserved_slots(),
        );
    }
}

/// A poisoned device/context surfaced a fatal error. Notify the scheduler so it
/// fails every in-flight sequence, then exit the process. We deliberately call
/// `std::process::exit` rather than returning `Err` and unwinding: CUDA `Drop`
/// impls (cudaFree / device-sync) can hang or double-fault on a poisoned
/// context, and the OS reclaims device memory on process exit anyway.
fn escalate_fatal_and_exit(
    control: &ControlPump,
    active: &ActiveSeqMap,
    prefilling: &PrefillSeqMap,
    err: &OpError,
) -> ! {
    let mut sids: Vec<u64> = active.keys().copied().collect();
    sids.extend(prefilling.keys().copied());
    sids.sort_unstable();
    sids.dedup();
    tracing::error!(
        error = %err,
        num_seqs = sids.len(),
        "[serve] FATAL device error — notifying scheduler, exiting worker"
    );
    crate::application::decode_common::send_fatal_step_error(
        control,
        sids,
        format!("worker fatal device error: {}", err),
    );
    std::process::exit(1);
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

fn maybe_heartbeat(
    control: &ControlPump,
    active_n: usize,
    last: &mut Instant,
    interval: Duration,
    kv_allocator: &GlobalKvAllocator,
    transient_reserved: usize,
) {
    if last.elapsed() >= interval {
        if let Err(e) = control.send_heartbeat(
            active_n,
            Some(kv_allocator.outstanding()),
            Some(transient_reserved.min(u32::MAX as usize) as u32),
            Some(kv_allocator.total_free()),
            Some(kv_allocator.released_len() as u32),
        ) {
            tracing::info!("[serve] failed to send heartbeat: {}", e);
        }
        *last = Instant::now();
    }
}
