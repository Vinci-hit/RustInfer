//! rustinfer-worker binary — entry point for the inference worker process.
//!
//! Loads a Llama-3 / Qwen-3 model on a CUDA device, performs the control
//! plane bootstrap (Hello → LoadModel → Ready), then runs an LLM serve loop:
//!
//!   * receive `PrefillBatchCmd` over the data plane PULL socket
//!   * run prefill via `ModelRunner::step_batch` (paged KV)
//!   * keep an internal `active_decodes` table; each iteration runs all
//!     active decodes in one batched step until they hit EOS / max_tokens
//!   * push `StepOutput` over the data plane PUSH socket
//!
//! The worker owns the decode self-loop — the scheduler never re-sends
//! per-step decode commands.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use clap::Parser;
use half::bf16;
use serde::Deserialize;

// cudaProfiler API for precise nsys capture
#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn cudaProfilerStart() -> u32;
    fn cudaProfilerStop() -> u32;
}

use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
use infer_protocol::scheduler_to_worker_data::{
    BatchCommand, PrefillBatchCmd, PrefillSegmentCompletion,
};
use infer_protocol::worker_to_scheduler_control::{
    WorkerCapacity, WorkerControlMessage, WorkerStepError,
};
use infer_protocol::worker_to_scheduler_data::{
    AssignedIndices, DiffusionBatchOutput, GeneratedToken, StepOutput,
};

use infer_worker::application::model_runner::{ModelRunner, SeqStep};
use infer_worker::domain::global_kv_alloc::GlobalKvAllocator;
use infer_worker::domain::ports::OpResult;
use infer_worker::infrastructure::cuda::Cuda;
use infer_worker::infrastructure::io::SafetensorsReader;
use infer_worker::models::loader::{LoadConfig, RopeScaling, WeightLoader};
use infer_worker::infrastructure::transport::control_pump::ControlPump;
use infer_worker::infrastructure::transport::data_pump::DataPump;

#[derive(Parser, Debug)]
#[command(name = "rustinfer-worker", version = "0.3.0")]
struct Args {
    /// Scheduler control plane endpoint (DEALER → ROUTER).
    #[arg(long, alias = "worker-control-endpoint", default_value = "ipc:///tmp/rustinfer-worker-control.ipc")]
    control_endpoint: String,

    /// Scheduler → Worker data plane endpoint (PUSH → PULL).
    /// Worker connects with PULL.
    #[arg(long, alias = "worker-pull-endpoint", default_value = "ipc:///tmp/rustinfer-worker-in.ipc")]
    data_recv_endpoint: String,

    /// Worker → Scheduler data plane endpoint (PUSH → PULL).
    /// Worker connects with PUSH.
    #[arg(long, alias = "worker-push-endpoint", default_value = "ipc:///tmp/rustinfer-worker-out.ipc")]
    data_send_endpoint: String,

    #[arg(long, default_value = "worker-0")]
    worker_id: String,

    /// Paged KV block size. MUST be 1 (every token = 1 slot in global KV pool).
    /// Kept as CLI knob for diagnostics only — do not change in production.
    #[arg(long, default_value_t = 1)]
    block_size: usize,

    /// Override number of physical paged blocks. If 0, derived from LoadModel.
    /// Use with --profile or diagnostics only.
    #[arg(long, default_value_t = 0)]
    num_blocks_override: usize,

    /// Number of decode steps to profile with cudaProfilerApi.
    /// When set, calls cudaProfilerStart() before the first decode step
    /// and cudaProfilerStop() after N steps, then exits.
    /// Use with: nsys profile --capture-range=cudaProfilerApi ...
    #[arg(long)]
    profile_cuda_steps: Option<u32>,

    /// Log level (forwarded to tracing-subscriber).
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[derive(Debug, Deserialize)]
struct HfConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    vocab_size: usize,
    #[serde(default = "default_max_position")]
    max_position_embeddings: usize,
    #[serde(default = "default_rms_eps")]
    rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default)]
    rope_scaling: Option<HfRopeScaling>,
    #[serde(default)]
    architectures: Vec<String>,
}

fn default_max_position() -> usize { 4096 }
fn default_rms_eps() -> f32 { 1e-5 }
fn default_rope_theta() -> f64 { 10000.0 }

#[derive(Debug, Deserialize)]
struct HfRopeScaling {
    #[serde(default)]
    rope_type: Option<String>,
    #[serde(default)]
    factor: Option<f32>,
    #[serde(default)]
    low_freq_factor: Option<f32>,
    #[serde(default)]
    high_freq_factor: Option<f32>,
    #[serde(default)]
    original_max_position_embeddings: Option<u32>,
}

fn build_load_config(cfg: &HfConfig, max_seq_len: usize) -> LoadConfig {
    let head_dim = cfg.head_dim.unwrap_or_else(|| cfg.hidden_size / cfg.num_attention_heads);
    let rope_scaling = cfg.rope_scaling.as_ref().and_then(|rs| {
        let is_llama3 = rs.rope_type.as_deref() == Some("llama3");
        if !is_llama3 { return None; }
        Some(RopeScaling {
            factor: rs.factor?,
            low_freq_factor: rs.low_freq_factor?,
            high_freq_factor: rs.high_freq_factor?,
            original_max_position_embeddings: rs.original_max_position_embeddings?,
        })
    });
    LoadConfig {
        dim: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        layer_num: cfg.num_hidden_layers,
        head_num: cfg.num_attention_heads,
        kv_head_num: cfg.num_key_value_heads,
        head_dim,
        vocab_size: cfg.vocab_size,
        seq_len: max_seq_len.max(cfg.max_position_embeddings.min(max_seq_len)),
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: cfg.rope_theta,
        rope_scaling,
    }
}

fn parse_device_id(spec: &str) -> Result<i32, String> {
    let suffix = spec.strip_prefix("cuda:").ok_or_else(|| format!("expected cuda:N, got '{}'", spec))?;
    suffix.parse().map_err(|e: std::num::ParseIntError| format!("invalid device id: {}", e))
}

/// Per-sequence decode state held by the worker between iterations.
struct ActiveSeq {
    last_token: i32,
    kv_len: usize,
    block_table: Vec<u32>,
    max_tokens: usize,
    generated_count: usize,
    /// When true, EOS tokens do not terminate the sequence; it decodes
    /// all the way to `max_tokens` (fixed-length benchmarking).
    ignore_eos: bool,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.clone().into()),
        )
        .init();
    eprintln!("rustinfer-worker v0.3.0");
    eprintln!("  worker_id = {}", args.worker_id);
    eprintln!("  control   = {}", args.control_endpoint);
    eprintln!("  data_recv = {}", args.data_recv_endpoint);
    eprintln!("  data_send = {}", args.data_send_endpoint);
    eprintln!("  (model/device/limits come from scheduler LoadModel)");

    // ── 1. ZMQ ──
    let zmq_ctx = zmq::Context::new();
    let control = ControlPump::new(&zmq_ctx, args.worker_id.clone(), &args.control_endpoint)?;
    let data = DataPump::new(&zmq_ctx, &args.data_recv_endpoint, &args.data_send_endpoint)?;

    // ── 2. Hello ──
    control.send_hello()?;
    eprintln!("[bootstrap] Hello sent, waiting for LoadModel...");

    // ── 3. Wait for LoadModel (skipping SchedulerHello / unrelated msgs) ──
    let mut server_heartbeat_ms: Option<u64> = None;
    let load = loop {
        let (msg, _req_id) = control.recv()?;
        match msg {
            SchedulerControlMessage::LoadModel(l) => break l,
            SchedulerControlMessage::Hello(h) => {
                eprintln!(
                    "[bootstrap] SchedulerHello: protocol={} heartbeat={}ms",
                    h.protocol_version, h.heartbeat_interval_ms,
                );
                server_heartbeat_ms = Some(h.heartbeat_interval_ms);
            }
            other => {
                eprintln!("[bootstrap] ignoring pre-LoadModel control msg: {:?}", other);
            }
        }
    };
    eprintln!(
        "[bootstrap] LoadModel: path={} max_seqs={} max_tokens={} max_model_len={}",
        load.model_path, load.max_batch_seqs, load.max_batch_tokens, load.max_model_len,
    );

    // ── 4. Load model ──
    let device_id = parse_device_id(&load.device)?;
    let cuda = Cuda::new(device_id).map_err(|e| format!("Cuda::new: {:?}", e))?;
    let cfg_path = Path::new(&load.model_path).join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).map_err(|e| format!("read {}: {}", cfg_path.display(), e))?;
    let hf_cfg: HfConfig = serde_json::from_slice(&cfg_bytes)
        .map_err(|e| format!("parse {}: {}", cfg_path.display(), e))?;
    let max_seq_len = load.max_model_len;
    let load_cfg = build_load_config(&hf_cfg, max_seq_len);
    eprintln!(
        "[bootstrap] arch={} layers={} dim={} heads={}/{} vocab={}",
        hf_cfg.architectures.first().cloned().unwrap_or_default(),
        hf_cfg.num_hidden_layers, hf_cfg.hidden_size,
        hf_cfg.num_attention_heads, hf_cfg.num_key_value_heads, hf_cfg.vocab_size,
    );

    let st_path = Path::new(&load.model_path);
    let reader = SafetensorsReader::open(st_path).map_err(|e| format!("open weights: {}", e))?;
    let loader = WeightLoader::new(&reader);
    let load_start = Instant::now();
    eprintln!("[bootstrap] loading weights for model_type='{}'", load.model_type);

    // ── 5/6/7. Build runner + send Ready + run serve loop, dispatched on model_type ──
    let bootstrap = Bootstrap {
        load: &load,
        cuda: &cuda,
        load_cfg: &load_cfg,
        max_seq_len,
        block_size: args.block_size,
        num_blocks_override: args.num_blocks_override,
        server_heartbeat_ms,
    };
    let eos_ids: Vec<i32> = match load.model_type.as_str() {
        "qwen3" => vec![151643, 151645], // <|endoftext|>, <|im_end|>
        _ => vec![128001, 128008, 128009], // Llama 3.x default
    };

    match load.model_type.as_str() {
        "qwen3" => {
            let model = loader.load_qwen3::<bf16, Cuda>(&load_cfg, &cuda)
                .map_err(|e| format!("load_qwen3: {:?}", e))?;
            eprintln!("[bootstrap] weights loaded in {:.2}s", load_start.elapsed().as_secs_f32());
            run_with_model(&control, &data, model, bootstrap, &eos_ids, args.profile_cuda_steps)?;
        }
        _ => {
            let model = loader.load_llama3::<bf16, Cuda>(&load_cfg, &cuda)
                .map_err(|e| format!("load_llama3: {:?}", e))?;
            eprintln!("[bootstrap] weights loaded in {:.2}s", load_start.elapsed().as_secs_f32());
            run_with_model(&control, &data, model, bootstrap, &eos_ids, args.profile_cuda_steps)?;
        }
    }

    Ok(())
}

/// Bundle of bootstrap parameters that don't depend on the model type.
struct Bootstrap<'a> {
    load: &'a infer_protocol::scheduler_to_worker_control::LoadModel,
    cuda: &'a Cuda,
    load_cfg: &'a LoadConfig,
    max_seq_len: usize,
    block_size: usize,
    num_blocks_override: usize,
    server_heartbeat_ms: Option<u64>,
}

fn run_with_model<M>(
    control: &ControlPump,
    data: &DataPump,
    model: M,
    bs: Bootstrap<'_>,
    eos_ids: &[i32],
    profile_cuda_steps: Option<u32>,
) -> Result<(), String>
where
    M: infer_worker::domain::model::LlmModel<bf16, Cuda>,
{
    let _ = bs.load_cfg;
    let max_blocks_per_seq = (bs.max_seq_len + bs.block_size - 1) / bs.block_size;
    // KV pool sizing. Weights are already resident at this point (load
    // happened before `run_with_model`), so `cudaMemGetInfo` reflects the
    // post-load free pool. Allocate `kv_cache_memory_fraction` of that free
    // memory to the paged KV pool; the rest is left for the forward /
    // batch workspaces, the CUDA-Graph scratch block, and graph capture
    // buffers. `--num-blocks-override` (CLI) still overrides for diagnostics.
    let num_blocks = if bs.num_blocks_override != 0 {
        eprintln!(
            "[bootstrap] num_blocks override from CLI: {} (skipping GPU mem probe)",
            bs.num_blocks_override,
        );
        bs.num_blocks_override
    } else {
        // Bytes per scheduler-visible block: K and V tensors, one per
        // layer, each shaped [1, block_size, kv_dim] of bf16 (2 bytes).
        let bytes_per_block =
            model.num_layers() * 2 * bs.block_size * model.kv_dim() * std::mem::size_of::<bf16>();
        let fraction = bs.load.kv_cache_memory_fraction.unwrap_or(0.9).clamp(0.05, 0.98);
        let (free, total) = infer_worker::infrastructure::cuda::device_utils::mem_get_info()
            .map_err(|e| format!("cudaMemGetInfo: {:?}", e))?;
        let budget = (free as f64 * fraction as f64) as usize;
        // Reserve one block for the CUDA-Graph scratch (`pool_blocks` adds
        // +1 below), so derive at least 1 visible block.
        let derived = (budget / bytes_per_block.max(1)).saturating_sub(1).max(1);
        eprintln!(
            "[bootstrap] KV mem probe: free={:.2}GiB total={:.2}GiB fraction={} bytes/block={} → num_blocks={} (≈{:.2}GiB KV pool)",
            free as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64,
            fraction,
            bytes_per_block,
            derived,
            (derived * bytes_per_block) as f64 / (1u64 << 30) as f64,
        );
        derived
    };
    // Worker-side pool size = scheduler-visible blocks + 1 scratch block
    // for CUDA Graph padding. The last physical block (id = pool_blocks-1)
    // is reserved on the worker; the scheduler only sees `num_blocks` and
    // never hands out the scratch id.
    let pool_blocks = num_blocks + 1;
    eprintln!(
        "[bootstrap] paged KV pool: block_size={} num_blocks={} pool_blocks={} (last block reserved as graph scratch) max_blocks_per_seq={}",
        bs.block_size, num_blocks, pool_blocks, max_blocks_per_seq,
    );
    // Forward workspace caps: cap_num_tokens = max_batch_tokens (worst-case
    // single-step ragged batch), cap_batch = max_batch_seqs.
    let cap_num_tokens = bs.load.max_batch_tokens;
    let cap_batch = bs.load.max_batch_seqs;
    // f32 flash-decode workspace size for the worst-case batch.
    let flash_decode_capacity_f32 =
        infer_worker::infrastructure::cuda::kernels::attention_paged::flash_decode_workspace_capacity_f32(
            cap_batch.max(1), 128, 256,
        );
    let mut runner: ModelRunner<bf16, Cuda, M> = ModelRunner::new(
        model, bs.cuda.clone(),
        pool_blocks, bs.block_size, max_blocks_per_seq, bs.max_seq_len,
        cap_num_tokens, cap_batch, flash_decode_capacity_f32,
        // Decode-only graph capture sizes — full contiguous coverage 1..=32
        // so every decode batch size up to max_batch_seqs replays its own
        // graph with zero padding (no wasted compute on padded sequences).
        (1..=32).collect(),
    ).map_err(|e| format!("ModelRunner::new: {:?}", e))?;

    // Prime CUDA Graphs for decode-only batches in 1..=32.
    // The runner reserves the LAST physical block (`pool_blocks-1`) as a
    // graph scratch block; the scheduler only sees `num_blocks` and never
    // allocates that id, so production sequences cannot collide with the
    // graph-scratch K/V writes.
    if let Err(e) = runner.prime_graphs_cuda() {
        eprintln!("[bootstrap] CUDA Graph priming FAILED, continuing in eager mode: {:?}", e);
    } else {
        eprintln!(
            "[bootstrap] CUDA Graphs primed for decode-only batches in {:?}",
            runner.capture_sizes,
        );
    }

    let max_total_kv_tokens = num_blocks * bs.block_size;
    control.send_ready(
        bs.load.model_instance_id.clone(),
        bs.load.model_path.clone(),
        bs.load.model_type.clone(),
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
    eprintln!(
        "[bootstrap] Ready sent. max_total_kv_tokens={}. Entering serve loop...",
        max_total_kv_tokens,
    );

    let hb_ms = bs.server_heartbeat_ms.unwrap_or(1000).max(200);
    let heartbeat_interval = Duration::from_millis(hb_ms);
    eprintln!("[serve] heartbeat interval = {:?}", heartbeat_interval);
    let mut last_heartbeat = Instant::now();
    let mut active: HashMap<u64, ActiveSeq> = HashMap::new();
    // Phase 7B: worker-owned global KV-slot allocator. With block_size=1,
    // `num_blocks` == total token-slot capacity. The legacy +1 scratch
    // block stays reserved on the runner side; the allocator tracks only
    // the visible `num_blocks` indices the scheduler sees.
    let mut kv_allocator = GlobalKvAllocator::new(num_blocks as u32);
    eprintln!(
        "[serve] worker-owned KV allocator: total={} (block_size={})",
        num_blocks, bs.block_size,
    );
    let enable_prefix_caching = bs.load.enable_prefix_caching;
    eprintln!(
        "[serve] prefix caching: {}",
        if enable_prefix_caching { "enabled (RadixTree)" } else { "disabled (real-time recycling)" },
    );
    let mut profile_step_count: u32 = 0;
    let mut profile_started = false;

    loop {
        // Drain all pending control messages (non-blocking).
        loop {
            match control.try_recv(0) {
                Ok(Some((msg, _req_id))) => match msg {
                    SchedulerControlMessage::Shutdown => {
                        eprintln!("[serve] Shutdown received, exiting.");
                        return Ok(());
                    }
                    SchedulerControlMessage::Cancel(c) => {
                        if let Some(removed) = active.remove(&c.sequence_id) {
                            if !removed.block_table.is_empty() {
                                if enable_prefix_caching {
                                    kv_allocator.free(&removed.block_table);
                                } else {
                                    kv_allocator.release(&removed.block_table);
                                }
                            }
                            eprintln!("[serve] cancelled seq {}", c.sequence_id);
                        }
                    }
                    // Phase 7B: scheduler tells us which global KV indices
                    // to release back to the allocator (driven by RadixTree
                    // LRU eviction). Worker is purely passive here.
                    SchedulerControlMessage::FreeKvIndices(free) => {
                        if !free.indices.is_empty() {
                            kv_allocator.free(&free.indices);
                        }
                    }
                    // Scheduler-driven Level-2 victim preemption.
                    SchedulerControlMessage::Preempt(p) => {
                        for sid in &p.sequence_ids {
                            if let Some(removed) = active.remove(sid) {
                                if !removed.block_table.is_empty() {
                                    if enable_prefix_caching {
                                        kv_allocator.free(&removed.block_table);
                                    } else {
                                        kv_allocator.release(&removed.block_table);
                                    }
                                }
                                eprintln!("[serve] preempted seq {}", sid);
                            }
                        }
                    }
                    SchedulerControlMessage::Ping => {
                        let _ = control.send(WorkerControlMessage::Pong, _req_id);
                    }
                    _ => {}
                },
                _ => break,
            }
        }

        let mut pending_prefills: Vec<PrefillBatchCmd> = Vec::new();
        loop {
            match data.try_recv_batch(0) {
                Ok(Some(BatchCommand::Prefill(p))) => pending_prefills.push(p),
                Ok(Some(BatchCommand::DiffusionBatch(_))) => {
                    let _ = data.send_diffusion_output(&DiffusionBatchOutput { results: vec![] });
                }
                _ => break,
            }
        }

        if pending_prefills.is_empty() && active.is_empty() {
            maybe_heartbeat(control, active.len(), &mut last_heartbeat, heartbeat_interval);
            let idle_wait_ms = (heartbeat_interval.as_millis() as i64 / 2).max(50);
            match data.try_recv_batch(idle_wait_ms) {
                Ok(Some(BatchCommand::Prefill(p))) => pending_prefills.push(p),
                Ok(Some(BatchCommand::DiffusionBatch(_))) => {
                    let _ = data.send_diffusion_output(&DiffusionBatchOutput { results: vec![] });
                }
                _ => {}
            }
            if pending_prefills.is_empty() && active.is_empty() {
                continue;
            }
        }

        for cmd in pending_prefills {
            if let Err(e) = handle_prefill(
                &mut runner,
                &cmd,
                &mut active,
                &mut kv_allocator,
                control,
                data,
                eos_ids,
                enable_prefix_caching,
                cap_batch,
            ) {
                eprintln!("[serve] prefill error: {}", e);
            }
        }

        if !active.is_empty() {
            // Profiler: start on first decode step
            if let Some(max_steps) = profile_cuda_steps {
                if !profile_started {
                    unsafe { cudaProfilerStart(); }
                    profile_started = true;
                    eprintln!("[profile] cudaProfilerStart (will stop after {} steps)", max_steps);
                }
            }

            if let Err(e) = run_decode_step(
                &mut runner,
                &mut active,
                &mut kv_allocator,
                control,
                data,
                eos_ids,
                enable_prefix_caching,
            ) {
                eprintln!("[serve] decode error: {}", e);
            }

            // Profiler: stop after N steps and exit
            if let Some(max_steps) = profile_cuda_steps {
                if profile_started {
                    profile_step_count += 1;
                    if profile_step_count >= max_steps {
                        unsafe { cudaProfilerStop(); }
                        eprintln!("[profile] cudaProfilerStop after {} steps. Exiting.", profile_step_count);
                        return Ok(());
                    }
                }
            }
        }

        maybe_heartbeat(control, active.len(), &mut last_heartbeat, heartbeat_interval);

        // Edge-triggered low-water signaling is gone: KV pressure now
        // travels through `AllocFailed` events emitted only when an
        // `alloc_indices` call actually fails. See `wait_for_relief`.
    }
}

fn maybe_heartbeat(
    control: &ControlPump,
    active_n: usize,
    last: &mut Instant,
    interval: Duration,
) {
    if last.elapsed() >= interval {
        let _ = control.send_heartbeat(active_n);
        *last = Instant::now();
    }
}

/// Block-poll the control plane until the scheduler ships KV relief
/// (either a `FreeKvIndices` or a `Preempt`) or `wait_ms` elapses.
///
/// Other control messages that arrive during the wait (Cancel,
/// Shutdown, Ping) are NOT drained here — they are left in the socket
/// for the next iteration's regular drain. The only exception is the
/// two relief variants this function explicitly applies and consumes.
/// Returns `true` if relief landed, `false` on timeout.
fn wait_for_relief(
    control: &ControlPump,
    kv_allocator: &mut GlobalKvAllocator,
    active: &mut HashMap<u64, ActiveSeq>,
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
                        if !entry.block_table.is_empty() {
                            if enable_prefix_caching {
                                kv_allocator.free(&entry.block_table);
                            } else {
                                kv_allocator.release(&entry.block_table);
                            }
                        }
                    }
                }
                return true;
            }
            // For Shutdown / Cancel / Ping we'd ideally hand them back
            // for the next drain; ZMQ has no peek API. Apply Shutdown
            // and Cancel inline so the worker doesn't deadlock here;
            // Ping/Pong is cheap enough to answer immediately.
            Ok(Some((SchedulerControlMessage::Shutdown, _))) => {
                eprintln!("[serve] Shutdown received during wait_for_relief");
                std::process::exit(0);
            }
            Ok(Some((SchedulerControlMessage::Cancel(c), _))) => {
                if let Some(removed) = active.remove(&c.sequence_id) {
                    if !removed.block_table.is_empty() {
                        if enable_prefix_caching {
                            kv_allocator.free(&removed.block_table);
                        } else {
                            kv_allocator.release(&removed.block_table);
                        }
                    }
                    eprintln!("[serve] cancelled seq {} (during relief wait)", c.sequence_id);
                }
            }
            Ok(Some((SchedulerControlMessage::Ping, req_id))) => {
                let _ = control.send(WorkerControlMessage::Pong, req_id);
            }
            // Any other message: drop and keep waiting.
            Ok(Some(_)) => continue,
            _ => continue,
        }
    }
    false
}

/// Try to allocate `n_initial` slots. On `AllocFull`, signal the
/// scheduler with `AllocFailed{round=0}`, block-poll for relief
/// (FreeKvIndices or Preempt) for up to 500ms, retry. If still
/// failing, escalate to `round=1` (victim preemption) and retry one
/// more time. After two rounds the worker gives up and returns
/// `None` — caller should fail the batch as a last-resort tier-3
/// fallback.
///
/// Importantly: if a Preempt arrives during relief we may have fewer
/// active seqs than we started with; the retry recomputes the
/// allocation size against the post-preempt `active` map (decoder
/// path) by passing `n_initial.min(active.len())`. The prefill path
/// passes a pre-computed `n_initial` derived from `cmd.segments` and
/// must use exactly that count regardless of preempt activity.
fn alloc_with_relief(
    kv_allocator: &mut GlobalKvAllocator,
    control: &ControlPump,
    active: &mut HashMap<u64, ActiveSeq>,
    n_initial: u32,
    enable_prefix_caching: bool,
) -> Option<Vec<u32>> {
    const RELIEF_TIMEOUT_MS: i64 = 500;
    let mut round: u8 = 0;
    let mut n = n_initial;
    loop {
        if n == 0 {
            // Edge case: caller asked for zero slots, or all active
            // seqs got preempted out from under us. Return an empty vec
            // so the caller can short-circuit cleanly.
            return Some(Vec::new());
        }
        match kv_allocator.alloc_indices(n) {
            Ok(v) => return Some(v),
            Err(e) => {
                eprintln!(
                    "[serve] alloc({}) failed at round={}: {} — requesting relief",
                    n, round, e
                );
                let _ = control.send_alloc_failed(n, round);
                if !wait_for_relief(control, kv_allocator, active, RELIEF_TIMEOUT_MS, enable_prefix_caching) {
                    eprintln!(
                        "[serve] relief timed out at round={} (still need {} slots)",
                        round, n
                    );
                    return None;
                }
                if round >= 1 {
                    eprintln!(
                        "[serve] relief returned but alloc still short after round 1 — giving up"
                    );
                    return None;
                }
                round += 1;
                // Decoder path: active may have shrunk. Recompute `n`
                // capped to active.len() so we never over-allocate.
                // Prefill path callers pass an explicit `n_initial`
                // that does NOT track `active` — but Preempt can only
                // touch already-active seqs and a prefill that hasn't
                // been installed in `active` yet is invisible to the
                // scheduler's victim policy. So recomputing against
                // active.len() is safe: it's a no-op for prefill (n
                // stays > active.len()) and a useful trim for decode.
                let active_cap = active.len() as u32;
                if n > active_cap && n_initial == active_cap {
                    n = active_cap;
                }
            }
        }
    }
}

/// Run one prefill batch (one PrefillBatchCmd = potentially many segments).
///
/// Phase 7B: each segment's KV slots come from the worker-owned
/// `GlobalKvAllocator`, not from `seg.block_table` (which is now ignored on
/// the new path). Optional `seg.prefix_hint` is prepended verbatim — those
/// slots already hold valid KV from a previous request and we trust the
/// scheduler not to evict them mid-step (RadixTree pinning, plan §3).
///
/// On success the resulting `StepOutput` carries `assigned_indices` so the
/// scheduler can extend its RadixTree chain for each seq.
fn handle_prefill<M>(
    runner: &mut ModelRunner<bf16, Cuda, M>,
    cmd: &PrefillBatchCmd,
    active: &mut HashMap<u64, ActiveSeq>,
    kv_allocator: &mut GlobalKvAllocator,
    control: &ControlPump,
    data: &DataPump,
    eos_ids: &[i32],
    enable_prefix_caching: bool,
    cap_batch: usize,
) -> OpResult<()>
where
    M: infer_worker::domain::model::LlmModel<bf16, Cuda>,
{
    // Admission guard (defense in depth): the decode step batches every
    // active seq in one `build_plan` call sized to `cap_batch`. If a prefill
    // batch would push the active set past that cap, the subsequent decode
    // step rejects the whole batch and fails every request in it. The
    // scheduler's seq budget should already prevent this, but a stale or
    // off-by-one count must not be able to crash a decode batch — so reject
    // the overflow segments here, non-fatally, before allocating any KV.
    let new_decode_segments = cmd
        .segments
        .iter()
        .filter(|s| matches!(
            s.completion,
            PrefillSegmentCompletion::FinishPrefillAndStartDecode
        ))
        .count();
    if active.len() + new_decode_segments > cap_batch {
        let overflow_ids: Vec<u64> = cmd.segments.iter().map(|s| s.sequence_id).collect();
        eprintln!(
            "[serve] prefill rejected: active ({}) + new ({}) > cap ({}) — failing {} seqs",
            active.len(),
            new_decode_segments,
            cap_batch,
            overflow_ids.len(),
        );
        let _ = control.send(
            WorkerControlMessage::StepError(WorkerStepError {
                sequence_ids: overflow_ids,
                message: format!(
                    "worker batch slot exhausted: active={} + new={} > cap={}",
                    active.len(), new_decode_segments, cap_batch,
                ),
                fatal: false,
            }),
            infer_protocol::control_envelope::RequestId(0),
        );
        return Ok(());
    }

    // 1. Compute total new KV slots and allocate one contiguous range.
    let mut per_seg_new_tokens: Vec<u32> = Vec::with_capacity(cmd.segments.len());
    let mut per_seg_prefix_hint: Vec<Vec<u32>> = Vec::with_capacity(cmd.segments.len());
    let mut total_new: u32 = 0;
    for (i, seg) in cmd.segments.iter().enumerate() {
        let seg_len = (seg.segment_end - seg.segment_start) as u32;
        let prefix_hit = seg
            .prefix_hint
            .as_ref()
            .map(|h| h.len() as u32)
            .unwrap_or(0);
        // We write KV for every token in the segment that is NOT covered by
        // a prefix hint. For segments past `segment_start > 0`, the prefix
        // is by construction earlier than `segment_start`, so all tokens in
        // this segment are fresh.
        let new_tokens = if seg.segment_start == 0 {
            seg_len.saturating_sub(prefix_hit)
        } else {
            seg_len
        };
        per_seg_new_tokens.push(new_tokens);
        per_seg_prefix_hint.push(seg.prefix_hint.clone().unwrap_or_default());
        total_new = total_new.saturating_add(new_tokens);
        let _ = i;
    }

    let base_indices = match alloc_with_relief(
        kv_allocator,
        control,
        active,
        total_new,
        enable_prefix_caching,
    ) {
        Some(v) => v,
        None => {
            // Two rounds of relief failed. Fail the entire batch
            // (legacy behavior preserved as the final tier-3 escape).
            eprintln!(
                "[serve] prefill alloc still failing after relief — failing batch (n={})",
                total_new
            );
            let failed_ids: Vec<u64> =
                cmd.segments.iter().map(|s| s.sequence_id).collect();
            let _ = control.send(
                WorkerControlMessage::StepError(WorkerStepError {
                    sequence_ids: failed_ids,
                    message: format!("worker KV pool exhausted (n={})", total_new),
                    fatal: false,
                }),
                infer_protocol::control_envelope::RequestId(0),
            );
            return Ok(());
        }
    };

    // 2. Build SeqStep per segment, attaching the freshly-allocated indices.
    let mut steps: Vec<SeqStep> = Vec::with_capacity(cmd.segments.len());
    let mut per_seg_indices: Vec<Vec<u32>> = Vec::with_capacity(cmd.segments.len());
    let mut idx_cursor = 0usize;
    for (i, seg) in cmd.segments.iter().enumerate() {
        let new_tokens = per_seg_new_tokens[i] as usize;
        let prefix = &per_seg_prefix_hint[i];
        let range = cmd.segment_token_range(i);
        // For the first segment of a fresh prompt we skip the prefix-hit
        // input tokens (their KV is already on the worker). Continuation
        // chunks (segment_start > 0) feed all their tokens.
        let (input_ids, positions): (Vec<i32>, Vec<i32>) = if seg.segment_start == 0 && !prefix.is_empty() {
            let prefix_len = prefix.len();
            let prompt = &cmd.input_ids[range.clone()];
            let trimmed: Vec<i32> = prompt.iter().skip(prefix_len).copied().collect();
            let positions: Vec<i32> =
                ((prefix_len as i32)..(prefix_len as i32 + trimmed.len() as i32)).collect();
            (trimmed, positions)
        } else {
            let prompt = cmd.input_ids[range.clone()].to_vec();
            let positions: Vec<i32> =
                (seg.segment_start as i32..seg.segment_end as i32).collect();
            (prompt, positions)
        };

        // block_table = prefix_hint ++ allocated indices
        let new_indices: Vec<u32> = base_indices[idx_cursor..idx_cursor + new_tokens].to_vec();
        idx_cursor += new_tokens;
        let mut block_table: Vec<u32> = Vec::with_capacity(prefix.len() + new_tokens);
        block_table.extend_from_slice(prefix);
        block_table.extend_from_slice(&new_indices);

        let kv_write_start = block_table.len() as i32 - new_tokens as i32; // == prefix_len
        let kv_len_after = block_table.len() as i32;
        steps.push(SeqStep {
            input_ids,
            positions,
            kv_write_start,
            kv_len_after,
            block_table,
        });
        per_seg_indices.push(new_indices);
    }
    debug_assert_eq!(idx_cursor, total_new as usize);

    // 3. Run forward.
    let first_tokens = runner.step_batch(&steps)?;

    // 4. Build StepOutput including assigned_indices.
    // Build assigned_indices: one entry per contiguous run within each seq's
    // allocated indices. `alloc_indices` may return non-contiguous indices
    // when the free pool is fragmented; `AssignedIndices{base,len}` only
    // describes contiguous runs, so we emit multiple entries per seq when
    // needed.
    let mut assigned: Vec<AssignedIndices> = Vec::new();
    for (i, seg) in cmd.segments.iter().enumerate() {
        let indices = &per_seg_indices[i];
        if indices.is_empty() {
            continue;
        }
        let mut run_start = 0usize;
        while run_start < indices.len() {
            let mut run_end = run_start + 1;
            while run_end < indices.len() && indices[run_end] == indices[run_end - 1] + 1 {
                run_end += 1;
            }
            let len = (run_end - run_start) as u32;
            assigned.push(AssignedIndices {
                sequence_id: seg.sequence_id,
                base: indices[run_start],
                len: len.min(u16::MAX as u32) as u16,
            });
            run_start = run_end;
        }
    }

    let mut output = StepOutput {
        prefill_done: Vec::new(),
        tokens: Vec::new(),
        assigned_indices: assigned,
    };
    for (i, seg) in cmd.segments.iter().enumerate() {
        let token = first_tokens[i];
        let new_indices = &per_seg_indices[i];
        match seg.completion {
            PrefillSegmentCompletion::ContinuePrefill => {
                output.prefill_done.push(seg.sequence_id);
            }
            PrefillSegmentCompletion::FinishPrefillAndStartDecode => {
                output.prefill_done.push(seg.sequence_id);
                let finished = (!seg.ignore_eos && eos_ids.contains(&token)) || seg.max_tokens <= 1;
                output.tokens.push(GeneratedToken {
                    sequence_id: seg.sequence_id,
                    token_id: token,
                    finished,
                });
                if !finished {
                    let prefix = &per_seg_prefix_hint[i];
                    let mut bt: Vec<u32> = Vec::with_capacity(prefix.len() + new_indices.len());
                    bt.extend_from_slice(prefix);
                    bt.extend_from_slice(new_indices);
                    let kv_len = bt.len();
                    active.insert(seg.sequence_id, ActiveSeq {
                        last_token: token,
                        kv_len,
                        block_table: bt,
                        max_tokens: seg.max_tokens,
                        generated_count: 1,
                        ignore_eos: seg.ignore_eos,
                    });
                } else if !enable_prefix_caching {
                    // Real-time recycling: prefill that finished immediately
                    // without starting decode — release its KV blocks now.
                    let mut bt: Vec<u32> = Vec::with_capacity(new_indices.len());
                    bt.extend_from_slice(new_indices);
                    if !bt.is_empty() {
                        kv_allocator.release(&bt);
                    }
                }
            }
        }
    }
    let _ = data.send_step_output(&output);
    Ok(())
}

/// Run one decode iteration over all active seqs in a single batched step.
///
/// Phase 7B: each step allocates one contiguous range from the worker-owned
/// allocator (`alloc_segment(N)` where N = #active seqs), distributes one
/// slot to each seq, and emits `assigned_indices` so the scheduler can
/// extend its RadixTree chains.
fn run_decode_step<M>(
    runner: &mut ModelRunner<bf16, Cuda, M>,
    active: &mut HashMap<u64, ActiveSeq>,
    kv_allocator: &mut GlobalKvAllocator,
    control: &ControlPump,
    data: &DataPump,
    eos_ids: &[i32],
    enable_prefix_caching: bool,
) -> OpResult<()>
where
    M: infer_worker::domain::model::LlmModel<bf16, Cuda>,
{
    if active.is_empty() {
        return Ok(());
    }
    // Allocate one slot per active seq. Using alloc_indices (vs alloc_segment)
    // means we degrade gracefully when the free pool is fragmented — decode
    // doesn't need contiguous slots either.
    //
    // The active set may change inside `alloc_with_relief` if the
    // scheduler picks Decoding victims at round 1 — we re-snapshot
    // `order` after relief completes.
    let initial_n = active.len() as u32;
    let new_indices = match alloc_with_relief(
        kv_allocator,
        control,
        active,
        initial_n,
        enable_prefix_caching,
    ) {
        Some(v) => v,
        None => {
            // Relief ran out. Fall back to legacy "fail every active
            // seq" so clients don't sit on a 120s timeout.
            let order: Vec<u64> = active.keys().copied().collect();
            eprintln!("[serve] decode alloc still failing after relief — failing {} seqs", order.len());
            let _ = control.send(
                WorkerControlMessage::StepError(WorkerStepError {
                    sequence_ids: order.clone(),
                    message: "worker KV pool exhausted at decode".to_string(),
                    fatal: false,
                }),
                infer_protocol::control_envelope::RequestId(0),
            );
            for sid in &order {
                let _ = active.remove(sid);
            }
            return Ok(());
        }
    };
    // Now build a stable iteration order for the surviving active set.
    // alloc_with_relief may have shrunk `active` or returned more slots
    // than we now need; trim if so.
    let mut order: Vec<u64> = active.keys().copied().collect();
    order.sort_unstable();
    if order.is_empty() {
        // All active seqs were preempted. Free the slots we just got.
        if !new_indices.is_empty() {
            kv_allocator.free(&new_indices);
        }
        return Ok(());
    }
    // alloc_with_relief returned exactly `active.len()` slots when it
    // re-tried with the post-preempt count. Defensive: if we
    // over-allocated (shouldn't happen), give back the tail.
    let new_indices: Vec<u32> = if new_indices.len() > order.len() {
        let (take, give_back) = new_indices.split_at(order.len());
        kv_allocator.free(give_back);
        take.to_vec()
    } else {
        new_indices
    };
    debug_assert_eq!(new_indices.len(), order.len());

    let mut steps: Vec<SeqStep> = Vec::with_capacity(order.len());
    let mut assigned: Vec<AssignedIndices> = Vec::with_capacity(order.len());
    for (i, &sid) in order.iter().enumerate() {
        let new_idx = new_indices[i];
        let seq = active.get_mut(&sid).unwrap();
        // Build block_table = current history ++ new slot.
        let mut bt = seq.block_table.clone();
        bt.push(new_idx);
        steps.push(SeqStep {
            input_ids: vec![seq.last_token],
            positions: vec![seq.kv_len as i32],
            kv_write_start: seq.kv_len as i32,
            kv_len_after: (seq.kv_len + 1) as i32,
            block_table: bt,
        });
        assigned.push(AssignedIndices {
            sequence_id: sid,
            base: new_idx,
            len: 1,
        });
    }
    let new_tokens = runner.step_batch_with_graph(&steps)?;

    let mut output = StepOutput {
        prefill_done: Vec::new(),
        tokens: Vec::new(),
        assigned_indices: assigned,
    };
    let mut to_remove: Vec<u64> = Vec::new();
    for (i, &sid) in order.iter().enumerate() {
        let token = new_tokens[i];
        let new_idx = new_indices[i];
        let seq = active.get_mut(&sid).unwrap();
        seq.last_token = token;
        seq.kv_len += 1;
        seq.generated_count += 1;
        seq.block_table.push(new_idx);
        let finished = (!seq.ignore_eos && eos_ids.contains(&token)) || seq.generated_count >= seq.max_tokens;
        output.tokens.push(GeneratedToken {
            sequence_id: sid,
            token_id: token,
            finished,
        });
        if finished {
            to_remove.push(sid);
        }
    }
    for sid in &to_remove {
        // When prefix caching is enabled the scheduler will issue a
        // `FreeKvIndices` after its RadixTree LRU evicts the chain.
        // When disabled, the worker immediately releases the blocks
        // to the released pool for real-time recycling.
        let removed = active.remove(sid);
        if let Some(removed) = removed {
            if !enable_prefix_caching && !removed.block_table.is_empty() {
                kv_allocator.release(&removed.block_table);
            }
        }
    }
    let _ = data.send_step_output(&output);
    Ok(())
}
