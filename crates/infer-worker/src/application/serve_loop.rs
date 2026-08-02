use std::sync::Arc;
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
use crate::application::runtime::{
    Runtime, RuntimePeerGroup, RuntimePeerWatchdog, spawn_monitored_follower,
};
use crate::application::sampler_stack::GreedySampler;
use crate::application::worker_scheduler::handle_fused_step;
use crate::application::worker_state::{ActiveSeqMap, PrefillSeqMap};
use crate::domain::exec::{ExecScope, RankPair, TopologyShape};
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::domain::model::DecoderModel;
use crate::domain::ports::{CollectiveOps, CommAxis, OpError};
use crate::infrastructure::cuda::{Cuda, CudaScope, NcclCommunicator, device_utils};
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
    /// Upper bound for one mirrored TP startup/command round.
    pub peer_timeout: Duration,
    /// Upper bound for NCCL/follower model and Runtime startup.
    pub peer_startup_timeout: Duration,
    /// Rank-0 member of the single-process TP communicator group.
    pub tp_communicator: Option<Arc<NcclCommunicator>>,
    /// CUDA devices in TP-rank order. Used for group-wide capacity probing.
    pub tp_devices: &'a [Cuda],
}

/// Everything a non-zero TP rank needs to construct its thread-local Runtime.
///
/// The factory itself runs on the rank thread. This keeps the model's
/// `Rc<ForwardScratch>` and all CUDA allocations on the thread that owns and
/// executes them; only this small, owned bootstrap value crosses the boundary.
#[derive(Debug, Clone)]
pub struct RuntimeFollowerInit {
    pub rank: usize,
    pub size: usize,
    pub pp_rank: usize,
    pub pp_size: usize,
    pub bootstrap_pool_blocks: usize,
    pub block_size: usize,
    pub max_blocks_per_seq: usize,
    pub max_seq_len: usize,
    pub cap_num_tokens: usize,
    pub cap_batch: usize,
    pub capture_sizes: Vec<usize>,
}

impl RuntimeFollowerInit {
    /// Bind this rank's pre-created NCCL communicator to its execution scope.
    /// The controller creates the whole single-process group atomically before
    /// any rank thread starts loading weights.
    pub fn build_scope(
        &self,
        cuda: Cuda,
        communicator: Arc<NcclCommunicator>,
    ) -> crate::domain::ports::OpResult<CudaScope> {
        let topology = TopologyShape {
            tp: RankPair {
                rank: self.rank,
                size: self.size,
            },
            pp: RankPair {
                rank: self.pp_rank,
                size: self.pp_size,
            },
            dp: RankPair { rank: 0, size: 1 },
            node: RankPair { rank: 0, size: 1 },
        };
        CudaScope::new(cuda)
            .with_topology(topology)?
            .with_tp_communicator(communicator)
    }

    /// Finish Runtime construction after this rank's communicator and model
    /// shard are ready.
    pub fn build_runtime<M>(
        self,
        model: M,
        scope: CudaScope,
    ) -> crate::domain::ports::OpResult<Runtime<bf16, Cuda, M>>
    where
        M: DecoderModel<bf16, Cuda>,
    {
        Runtime::new(
            model,
            scope,
            Box::new(GreedySampler),
            self.bootstrap_pool_blocks,
            self.block_size,
            self.max_blocks_per_seq,
            self.max_seq_len,
            self.cap_num_tokens,
            self.cap_batch,
            self.capture_sizes,
        )
    }
}

/// Builds one non-zero rank entirely inside its long-lived Runtime thread.
pub type RuntimeFollowerFactory<M> = Box<
    dyn FnOnce(RuntimeFollowerInit) -> crate::domain::ports::OpResult<Runtime<bf16, Cuda, M>>
        + Send,
>;

#[derive(Debug)]
struct KvCapacityProbe {
    rank: usize,
    device_id: i32,
    total: usize,
    free_before: usize,
    free_after: usize,
    probed: usize,
    capacity: usize,
}

fn tp_memory_info(devices: &[Cuda], stage: &str) -> Result<Vec<(usize, usize)>, String> {
    devices
        .iter()
        .enumerate()
        .map(|(rank, cuda)| {
            let scope = CudaScope::new(cuda.clone());
            let _active_device = scope.enter();
            device_utils::mem_get_info().map_err(|error| {
                format!(
                    "cudaMemGetInfo for TP rank {rank} on cuda:{} during {stage}: {error:?}",
                    cuda.device_id
                )
            })
        })
        .collect()
}

pub fn run_with_model<M>(
    control: &ControlPump,
    data: &DataPump,
    model: M,
    bs: Bootstrap<'_>,
    follower_factories: Vec<RuntimeFollowerFactory<M>>,
    eos_ids: &[i32],
    profile_cuda_steps: Option<u32>,
) -> Result<(), String>
where
    M: DecoderModel<bf16, Cuda> + 'static,
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

    let bytes_per_block =
        model_dims.num_layers * 2 * bs.block_size * model_dims.kv_dim * std::mem::size_of::<bf16>();

    let cap_num_tokens = bs.load.max_batch_tokens;
    let cap_batch = bs.load.max_batch_seqs;

    if bs.load.tp_rank != 0 {
        return Err(format!(
            "the scheduler-facing worker must own TP rank 0, got rank {}/{}",
            bs.load.tp_rank, bs.load.tp_size
        ));
    }
    let expected_followers = bs.load.tp_size.saturating_sub(1);
    if follower_factories.len() != expected_followers {
        return Err(format!(
            "TP{} requires {} Runtime followers, got {}",
            bs.load.tp_size,
            expected_followers,
            follower_factories.len()
        ));
    }
    if bs.tp_devices.len() != bs.load.tp_size {
        return Err(format!(
            "TP{} requires {} CUDA devices for capacity probing, got {}",
            bs.load.tp_size,
            bs.load.tp_size,
            bs.tp_devices.len()
        ));
    }
    let rank0_device = bs.tp_devices.first().ok_or("TP device list is empty")?;
    if rank0_device.device_id != bs.cuda.device_id
        || !Arc::ptr_eq(&rank0_device.config, &bs.cuda.config)
    {
        return Err(format!(
            "TP rank-0 device cuda:{} does not match bootstrap cuda:{}",
            rank0_device.device_id, bs.cuda.device_id
        ));
    }

    // KV-pool sizing (auto path) is profile-driven: build the Runtime with a
    // small throwaway KV pool, run one worst-case eager forward so the fixed
    // activation workspace + lazy library allocations are all resident, THEN
    // probe free memory and size the real pool from what is actually left.
    //
    // The old design probed `cudaMemGetInfo` immediately after weight load —
    // before `Runtime::new` allocated the (GiB-scale, `max_batch_tokens × vocab`)
    // logits/activation workspace — so `mem_fraction_static` had to leave that
    // fixed cost as a *percentage* of free memory. A large batch/vocab then blew
    // the fractional headroom and OOM'd *after* the KV pool was already sized.
    // Profiling first makes the probe see the true footprint, so the fraction
    // means "fraction of usable memory for KV", as users expect.
    let bootstrap_pool_blocks = if bs.num_blocks_override != 0 {
        // Explicit override: skip probing entirely and build at the final size.
        tracing::info!(
            "[bootstrap] num_blocks override from CLI: {} (skipping GPU mem probe)",
            bs.num_blocks_override,
        );
        bs.num_blocks_override + 1
    } else {
        crate::application::tuning::PROFILE_KV_BLOCKS.min(
            // Never allocate a profiling pool larger than the final pool could
            // ever be (a tiny device / huge model): cap at the working set.
            bs.load
                .max_batch_seqs
                .saturating_mul(max_blocks_per_seq)
                .saturating_add(1)
                .max(2),
        )
    };

    let topology = TopologyShape {
        tp: RankPair {
            rank: bs.load.tp_rank,
            size: bs.load.tp_size,
        },
        pp: RankPair {
            rank: bs.load.pp_rank,
            size: bs.load.pp_size,
        },
        dp: RankPair { rank: 0, size: 1 },
        node: RankPair { rank: 0, size: 1 },
    };
    let mut peer_handles = Vec::with_capacity(expected_followers);
    let mut peer_watchdog = None;
    let scope = if topology.tp.size > 1 {
        let watchdog = RuntimePeerWatchdog::fail_stop()
            .map_err(|error| format!("start TP fail-stop watchdog: {error}"))?;
        let failure_notifier = watchdog.notifier();
        for (offset, factory) in follower_factories.into_iter().enumerate() {
            let rank = offset + 1;
            let init = RuntimeFollowerInit {
                rank,
                size: topology.tp.size,
                pp_rank: topology.pp.rank,
                pp_size: topology.pp.size,
                bootstrap_pool_blocks,
                block_size: bs.block_size,
                max_blocks_per_seq,
                max_seq_len: bs.max_seq_len,
                cap_num_tokens,
                cap_batch,
                capture_sizes: bs.capture_sizes.clone(),
            };
            let handle = spawn_monitored_follower::<bf16, Cuda, M, _, _>(
                rank,
                factory,
                init,
                failure_notifier.clone(),
            )
            .map_err(|error| format!("spawn TP Runtime rank {rank}: {error}"))?;
            peer_handles.push(handle);
        }
        peer_watchdog = Some(watchdog);
        let communicator = bs.tp_communicator.clone().ok_or_else(|| {
            format!(
                "TP{} requires a pre-created rank-0 NCCL communicator",
                topology.tp.size
            )
        })?;
        CudaScope::new(bs.cuda.clone())
            .with_topology(topology)
            .and_then(|scope| scope.with_tp_communicator(communicator))
            .map_err(|error| format!("initialize rank-0 CUDA execution scope: {error}"))?
    } else {
        if bs.tp_communicator.is_some() {
            return Err("TP1 must not receive an NCCL communicator".into());
        }
        CudaScope::new(bs.cuda.clone())
            .with_topology(topology)
            .map_err(|error| format!("invalid CUDA execution topology: {error}"))?
    };
    if topology.tp.size > 1 && <Cuda as CollectiveOps>::comm(&scope, CommAxis::Tp).is_none() {
        return Err(format!(
            "TP weights loaded for rank {}/{}, but no TP communicator is configured",
            topology.tp.rank, topology.tp.size
        ));
    }
    let mut runner: Runtime<bf16, Cuda, M> = Runtime::new(
        model,
        scope,
        Box::new(GreedySampler),
        bootstrap_pool_blocks,
        bs.block_size,
        max_blocks_per_seq,
        bs.max_seq_len,
        cap_num_tokens,
        cap_batch,
        bs.capture_sizes.clone(),
    )
    .map_err(|e| format!("Runtime::new: {:?}", e))?;
    if !peer_handles.is_empty() {
        let watchdog = peer_watchdog
            .take()
            .ok_or("TP Runtime followers require a fail-stop watchdog")?;
        let peers = RuntimePeerGroup::with_watchdog(
            peer_handles,
            bs.peer_startup_timeout,
            bs.peer_timeout,
            watchdog,
        )
        .map_err(|error| format!("create TP Runtime peer group: {error}"))?;
        runner
            .install_peer_group(peers)
            .map_err(|error| format!("start TP Runtime followers: {error}"))?;
    }

    let num_blocks = if bs.num_blocks_override != 0 {
        bs.num_blocks_override
    } else {
        let fraction = bs.load.kv_cache_memory_fraction.unwrap_or(0.9).clamp(
            crate::application::tuning::KV_MEM_FRACTION_MIN,
            crate::application::tuning::KV_MEM_FRACTION_MAX,
        );
        // Probe now — AFTER `Runtime::new` allocated the fixed activation
        // workspace (the GiB-scale logits buffer that used to OOM), but before
        // the dummy forward. The delta against the post-dummy probe below is the
        // lazy library footprint (diagnostic only).
        let before = tp_memory_info(bs.tp_devices, "before profile forward")?;
        // Worst-case eager forward: exercises the activation workspace and forces
        // the lazy cuBLASLt/cuDNN/recycling-pool allocations the first live
        // forward would otherwise make. `graph` is still None ⇒ eager ⇒ no graph
        // captured, no KV-base pointer baked (so the resize below is safe).
        runner
            .profile_forward()
            .map_err(|e| format!("profile_forward: {:?}", e))?;
        // This probe reflects weights + activation workspace + committed library
        // state — everything resident except the (throwaway) profiling KV pool.
        let after = tp_memory_info(bs.tp_devices, "after profile forward")?;
        // Cap the auto-sized pool at the working set the worker can ever hold in
        // use: at most `max_batch_seqs` sequences, each at most `max_seq_len`
        // tokens. The scheduler never admits more than `max_batch_seqs` running
        // sequences, so any block beyond this is unreachable — it only wastes
        // VRAM AND inflates every O(num_blocks) op in `GlobalKvAllocator`.
        // `release_owned`→`free()` does a full-pool compact+merge on EVERY
        // sequence completion; at a VRAM-filling ~780k pool that is ~2ms/
        // completion (measured), landing in serve-loop ticks alongside prefills
        // and directly inflating TTFT on slower CPUs. With prefix caching ON,
        // extra blocks cache evicted prefixes, so keep the full probe; with it
        // OFF (real-time recycling) the live working set is the only thing that
        // can be allocated, so clamp to it.
        let working_set = bs
            .load
            .max_batch_seqs
            .saturating_mul(max_blocks_per_seq)
            .max(1);
        let probes: Vec<KvCapacityProbe> = bs
            .tp_devices
            .iter()
            .enumerate()
            .zip(before.into_iter().zip(after))
            .map(|((rank, cuda), ((free_before, total), (free_after, _)))| {
                // Hold back a fixed reserve for incremental prewarm allocations,
                // then add one graph-scratch block plus 0.5% fragmentation slack.
                let usable =
                    free_after.saturating_sub(crate::application::tuning::PREWARM_HEADROOM_BYTES);
                let budget = (usable as f64 * fraction as f64) as usize;
                let raw = budget / bytes_per_block.max(1);
                let reserve = 1 + (raw / 200).max(1);
                let probed = raw.saturating_sub(reserve).max(1);
                let capacity = if bs.load.enable_prefix_caching {
                    probed
                } else {
                    probed.min(working_set)
                };
                KvCapacityProbe {
                    rank,
                    device_id: cuda.device_id,
                    total,
                    free_before,
                    free_after,
                    probed,
                    capacity,
                }
            })
            .collect();
        let limiting = probes
            .iter()
            .min_by_key(|probe| probe.capacity)
            .ok_or("TP KV capacity probe produced no results")?;
        let derived = limiting.capacity;
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        for probe in &probes {
            tracing::info!(
                rank = probe.rank,
                device = probe.device_id,
                total_gib = gib(probe.total),
                free_after_workspace_gib = gib(probe.free_before),
                free_after_dummy_gib = gib(probe.free_after),
                lazy_libs_gib = gib(probe.free_before.saturating_sub(probe.free_after)),
                probed_blocks = probe.probed,
                capacity_blocks = probe.capacity,
                "TP rank KV capacity probe"
            );
        }
        tracing::info!(
            limiting_rank = limiting.rank,
            limiting_device = limiting.device_id,
            fraction,
            bytes_per_block,
            num_blocks = derived,
            kv_pool_gib = gib(derived.saturating_mul(bytes_per_block)),
            prewarm_reserve_gib = gib(crate::application::tuning::PREWARM_HEADROOM_BYTES),
            "[bootstrap] group-wide KV capacity uses the minimum across TP ranks"
        );
        // Swap the throwaway profiling pool for the real one (frees the profile
        // pool first). Runs before `prime_graphs`, so no graph references the old
        // KV base and there is no live sequence state to migrate.
        runner
            .resize_kv_pool(derived + 1)
            .map_err(|e| format!("resize_kv_pool: {:?}", e))?;
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

    // Install the decode CUDA-graph runner. On CUDA (arena reserved) this
    // enables real capture-on-first-hit / replay for decode-only batches whose
    // size matches a capture slot; on backends without graph support it is a
    // no-op and decode stays eager.
    if let Err(e) = runner.prime_graphs() {
        if e.is_fatal() {
            return Err(format!("fatal graph priming failure: {e}"));
        }
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
            Err(e) if e.is_fatal() => {
                return Err(format!("fatal decode graph prewarm failure: {e}"));
            }
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
            Err(e) if e.is_fatal() => {
                return Err(format!("fatal prefill prewarm failure: {e}"));
            }
            Err(e) => tracing::info!("[bootstrap] prefill prewarm skipped: {:?}", e),
        }
        let t_mixed = Instant::now();
        if runner.mixed_eager_mode() {
            // Eager-mixed mode (unified FA3): mixed graphs are never replayed,
            // so skip the 104-bucket capture pass and warm the eager fused
            // path's GEMM token buckets instead.
            match runner.prewarm_mixed_eager_shapes(eos_ids) {
                Ok(n) => tracing::info!(
                    "[bootstrap] mixed eager shapes prewarmed ({} token buckets) in {:.2}s",
                    n,
                    t_mixed.elapsed().as_secs_f64(),
                ),
                Err(e) if e.is_fatal() => {
                    return Err(format!("fatal mixed eager prewarm failure: {e}"));
                }
                Err(e) => tracing::info!("[bootstrap] mixed eager prewarm skipped: {:?}", e),
            }
        } else {
            let attn = if runner.mixed_fa3_graph_mode() {
                "FA3"
            } else {
                "CuTe-split"
            };
            match runner.prewarm_mixed_graphs(eos_ids) {
                Ok(n) => tracing::info!(
                    "[bootstrap] mixed ABC graphs prewarmed ({} buckets, {} attention) in {:.2}s",
                    n,
                    attn,
                    t_mixed.elapsed().as_secs_f64(),
                ),
                Err(e) if e.is_fatal() => {
                    return Err(format!("fatal mixed graph prewarm failure: {e}"));
                }
                Err(e) => tracing::info!("[bootstrap] mixed graph prewarm skipped: {:?}", e),
            }
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

    // Prefill cmds deferred by the fused-step token budget (a burst larger
    // than one prewarmed mixed-graph bucket is spread across consecutive
    // steps). Carried across iterations, consumed ahead of fresh arrivals.
    let mut deferred_prefills: Vec<PrefillBatchCmd> = Vec::new();

    loop {
        let drain_result = {
            let mut ctx = WorkerCtx {
                active: &mut active,
                prefilling: &mut prefilling,
                decode_engine: &mut decode_engine,
                kv_allocator: &mut kv_allocator,
                enable_prefix_caching,
            };
            drain_control(control, &mut runner, &mut ctx)
        };
        match drain_result {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(error) if error.is_fatal() => {
                escalate_fatal_and_exit(control, &active, &prefilling, &error)
            }
            Err(error) => return Err(format!("handle worker control: {error}")),
        }

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
        let mut pending_prefills = std::mem::take(&mut deferred_prefills);
        pending_prefills.extend(drain_data(data));
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

            let (data_ready, control_ready) = {
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
                // Snapshot readiness before the PollItems release their socket
                // borrows at the end of this scope.
                (items[0].is_readable(), items[1].is_readable())
            };

            // Data plane ready → pull every queued prefill in one go.
            if data_ready {
                pending_prefills.extend(drain_data(data));
            }
            // Control plane ready → handle it now (may be a cancel/shutdown
            // that voids the prefill we are about to run).
            if control_ready {
                let drain_result = {
                    let mut ctx = WorkerCtx {
                        active: &mut active,
                        prefilling: &mut prefilling,
                        decode_engine: &mut decode_engine,
                        kv_allocator: &mut kv_allocator,
                        enable_prefix_caching,
                    };
                    drain_control(control, &mut runner, &mut ctx)
                };
                match drain_result {
                    Ok(true) => return Ok(()),
                    Ok(false) => {}
                    Err(error) if error.is_fatal() => {
                        escalate_fatal_and_exit(control, &active, &prefilling, &error)
                    }
                    Err(error) => return Err(format!("handle worker control: {error}")),
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
        // Pending prefill chunks and every active decode row go through a
        // single ragged forward (`handle_fused_step`; prewarmed mixed-graph
        // replay when the shape bucket is covered, eager otherwise). This pays
        // the fixed per-forward host overhead once per step instead of once
        // per prefill, and in-flight decode rows advance a token in that same
        // forward rather than stalling behind the prefills. Admission is
        // bounded to one mixed-graph token bucket per step; surplus cmds carry
        // over via `deferred_prefills`. When nothing is prefilling, steady-
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
                &mut deferred_prefills,
            ) {
                if matches!(e, OpError::Shutdown) {
                    tracing::info!("[serve] fused step interrupted by shutdown.");
                    finalize_for_clean_exit(
                        &mut runner,
                        &mut decode_engine,
                        &mut kv_allocator,
                        control,
                        &active,
                        &prefilling,
                    )?;
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
            if let Some(max_steps) = profile_cuda_steps
                && !profile_started
            {
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
                    finalize_for_clean_exit(
                        &mut runner,
                        &mut decode_engine,
                        &mut kv_allocator,
                        control,
                        &active,
                        &prefilling,
                    )?;
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

            if let Some(max_steps) = profile_cuda_steps
                && profile_started
            {
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
                    finalize_for_clean_exit(
                        &mut runner,
                        &mut decode_engine,
                        &mut kv_allocator,
                        control,
                        &active,
                        &prefilling,
                    )?;
                    return Ok(());
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

/// Close an issued async step before a deliberate process exit. This keeps the
/// TP peer group and its watchdog in the idle state required for deterministic
/// follower shutdown and join.
fn finalize_for_clean_exit<M>(
    runner: &mut Runtime<bf16, Cuda, M>,
    decode_engine: &mut DecodeEngine,
    kv_allocator: &mut GlobalKvAllocator,
    control: &ControlPump,
    active: &ActiveSeqMap,
    prefilling: &PrefillSeqMap,
) -> Result<(), String>
where
    M: DecoderModel<bf16, Cuda>,
{
    match decode_engine.finalize_and_reclaim_pending(runner, kv_allocator) {
        Ok(()) => Ok(()),
        Err(error) if error.is_fatal() => {
            escalate_fatal_and_exit(control, active, prefilling, &error)
        }
        Err(error) => Err(format!(
            "finalize pending TP work during clean worker exit: {error}"
        )),
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

fn drain_control<M>(
    control: &ControlPump,
    runner: &mut Runtime<bf16, Cuda, M>,
    ctx: &mut WorkerCtx<'_>,
) -> crate::domain::ports::OpResult<bool>
where
    M: DecoderModel<bf16, Cuda>,
{
    while let Ok(Some((msg, req_id))) = control.try_recv(0) {
        match msg {
            SchedulerControlMessage::Shutdown => {
                tracing::info!("[serve] Shutdown received, exiting.");
                ctx.decode_engine
                    .finalize_and_reclaim_pending(runner, ctx.kv_allocator)?;
                return Ok(true);
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
                apply_drain(control, runner, ctx, d.mode, req_id)?;
            }
            SchedulerControlMessage::UnloadModel(u) => {
                ctx.decode_engine
                    .finalize_and_reclaim_pending(runner, ctx.kv_allocator)?;
                if req_id.is_correlated()
                    && let Err(e) = control.send(
                        WorkerControlMessage::UnloadAck(UnloadAck {
                            model_instance_id: u.model_instance_id,
                        }),
                        req_id,
                    )
                {
                    tracing::info!("[serve] failed to send UnloadAck: {}", e);
                }
                tracing::info!("[serve] UnloadModel received, exiting.");
                return Ok(true);
            }
            _ => {}
        }
    }
    Ok(false)
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
    if req_id.is_correlated()
        && let Err(e) = control.send(
            WorkerControlMessage::CancelAck(CancelAck {
                sequence_id,
                removed: removed_flag,
            }),
            req_id,
        )
    {
        tracing::info!("[serve] failed to send CancelAck: {}", e);
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
fn apply_drain<M>(
    control: &ControlPump,
    runner: &mut Runtime<bf16, Cuda, M>,
    ctx: &mut WorkerCtx<'_>,
    mode: DrainMode,
    req_id: RequestId,
) -> crate::domain::ports::OpResult<()>
where
    M: DecoderModel<bf16, Cuda>,
{
    if matches!(mode, DrainMode::Immediate) {
        // Close rank-local and peer-side async decode spans before any block
        // can be returned to the allocator and reused.
        ctx.decode_engine
            .finalize_and_reclaim_pending(runner, ctx.kv_allocator)?;
        // P1: Iterate drain() directly instead of collecting into a Vec.
        for (_, seq) in ctx.active.drain() {
            ctx.kv_allocator
                .release_owned(&seq.block_table, ctx.enable_prefix_caching);
        }
        for (_, seq) in ctx.prefilling.drain() {
            ctx.kv_allocator
                .release_owned(&seq.block_table, ctx.enable_prefix_caching);
        }
        ctx.decode_engine.clear();
    }
    if req_id.is_correlated()
        && let Err(e) = control.send(
            WorkerControlMessage::DrainAck(DrainAck {
                remaining_requests: ctx.active.len() + ctx.prefilling.len(),
            }),
            req_id,
        )
    {
        tracing::info!("[serve] failed to send DrainAck: {}", e);
    }
    Ok(())
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
