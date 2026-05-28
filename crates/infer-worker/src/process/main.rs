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

use infer_protocol::scheduler_to_worker_control::{SchedulerControlMessage, GrantBlocks};
use infer_protocol::scheduler_to_worker_data::{
    BatchCommand, PrefillBatchCmd, PrefillSegmentCompletion, PrefillSegmentMeta, SamplingParams,
};
use infer_protocol::worker_to_scheduler_control::{WorkerCapacity, WorkerControlMessage, NeedBlocks, NeedBlocksReason};
use infer_protocol::worker_to_scheduler_data::{DiffusionBatchOutput, GeneratedToken, StepOutput};

use infer_worker::app::model_runner::{ModelRunner, SeqStep};
use infer_worker::domain::ports::OpResult;
use infer_worker::infra::cuda::Cuda;
use infer_worker::infra::io::SafetensorsReader;
use infer_worker::models::loader::{LoadConfig, RopeScaling, WeightLoader};
use infer_worker::process::control_pump::ControlPump;
use infer_worker::process::data_pump::DataPump;

#[derive(Parser, Debug)]
#[command(name = "rustinfer-worker", version = "0.3.0")]
struct Args {
    /// Path to model weights (optional — the scheduler's `LoadModel`
    /// payload provides the canonical path).
    #[arg(long)]
    model_path: Option<String>,

    /// CUDA device (e.g. "cuda:0").
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Model architecture: "llama3" or "qwen3" (only llama3 implemented for now).
    #[arg(long, default_value = "llama3")]
    model_type: String,

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

    #[arg(long, default_value_t = 8192)]
    max_batch_tokens: usize,

    #[arg(long, default_value_t = 32)]
    max_batch_seqs: usize,

    /// Maximum seq length per request (== max_blocks_per_seq * block_size).
    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,

    /// Paged KV block size (must match scheduler).
    #[arg(long, default_value_t = 16)]
    block_size: usize,

    /// Number of physical paged blocks. If 0, derived from max_batch_seqs * max_seq_len.
    #[arg(long, default_value_t = 0)]
    num_blocks: usize,

    #[arg(long, default_value_t = 1000)]
    heartbeat_interval_ms: u64,

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
    sequence_id: u64,
    last_token: i32,
    kv_len: usize,
    block_table: Vec<u32>,
    block_size: usize,
    max_tokens: usize,
    generated_count: usize,
    sampling: SamplingParams,
    /// True if we've already sent NeedBlocks for this seq and are awaiting grant.
    block_requested: bool,
}

impl ActiveSeq {
    fn from_segment(seg: &PrefillSegmentMeta, first_token: i32) -> Self {
        Self {
            sequence_id: seg.sequence_id,
            last_token: first_token,
            kv_len: seg.prompt_len as usize,
            block_table: seg.block_table.clone(),
            block_size: seg.block_size as usize,
            max_tokens: seg.max_tokens,
            generated_count: 1, // first token already produced by prefill argmax
            sampling: seg.sampling_params.clone(),
            block_requested: false,
        }
    }

    /// Block capacity in tokens = block_table.len() * block_size.
    fn block_capacity(&self) -> usize {
        self.block_table.len() * self.block_size
    }

    /// True if the next decode step can proceed without new blocks.
    fn has_block_space(&self) -> bool {
        self.kv_len + 1 <= self.block_capacity()
    }

    /// True if we should request new blocks (approaching boundary, within 1 block).
    /// Skip if we haven't decoded at least 2 steps yet (avoid racing with
    /// the StepOutput that transitions session to Decoding in scheduler).
    fn needs_block_soon(&self) -> bool {
        !self.block_requested
            && self.generated_count >= 2
            && self.kv_len + self.block_size >= self.block_capacity()
    }
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
    eprintln!("  model     = {} ({})", args.model_path.as_deref().unwrap_or("<from LoadModel>"), args.model_type);
    eprintln!("  device    = {}", args.device);
    eprintln!("  control   = {}", args.control_endpoint);
    eprintln!("  data_recv = {}", args.data_recv_endpoint);
    eprintln!("  data_send = {}", args.data_send_endpoint);

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
    let device_id = parse_device_id(&args.device)?;
    let cuda = Cuda::new(device_id).map_err(|e| format!("Cuda::new: {:?}", e))?;
    let cfg_path = Path::new(&load.model_path).join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path).map_err(|e| format!("read {}: {}", cfg_path.display(), e))?;
    let hf_cfg: HfConfig = serde_json::from_slice(&cfg_bytes)
        .map_err(|e| format!("parse {}: {}", cfg_path.display(), e))?;
    let max_seq_len = load.max_model_len.max(args.max_seq_len);
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
        num_blocks_arg: args.num_blocks,
        server_heartbeat_ms,
        heartbeat_interval_ms_arg: args.heartbeat_interval_ms,
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
            run_with_model(&control, &data, model, bootstrap, &eos_ids)?;
        }
        _ => {
            let model = loader.load_llama3::<bf16, Cuda>(&load_cfg, &cuda)
                .map_err(|e| format!("load_llama3: {:?}", e))?;
            eprintln!("[bootstrap] weights loaded in {:.2}s", load_start.elapsed().as_secs_f32());
            run_with_model(&control, &data, model, bootstrap, &eos_ids)?;
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
    num_blocks_arg: usize,
    server_heartbeat_ms: Option<u64>,
    heartbeat_interval_ms_arg: u64,
}

fn run_with_model<M>(
    control: &ControlPump,
    data: &DataPump,
    model: M,
    bs: Bootstrap<'_>,
    eos_ids: &[i32],
) -> Result<(), String>
where
    M: infer_worker::domain::model::LlmModel<bf16, Cuda>,
{
    let _ = bs.load_cfg;
    let max_blocks_per_seq = (bs.max_seq_len + bs.block_size - 1) / bs.block_size;
    let num_blocks = if bs.num_blocks_arg == 0 {
        max_blocks_per_seq * bs.load.max_batch_seqs
    } else {
        bs.num_blocks_arg
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
        infer_worker::infra::cuda::kernels::attention_paged::flash_decode_workspace_capacity_f32(
            cap_batch.max(1), 128, 256,
        );
    let mut runner: ModelRunner<bf16, Cuda, M> = ModelRunner::new(
        model, bs.cuda.clone(),
        pool_blocks, bs.block_size, max_blocks_per_seq, bs.max_seq_len,
        cap_num_tokens, cap_batch, flash_decode_capacity_f32,
        // Decode-only graph capture sizes — pad up to nearest power of 2.
        vec![1, 2, 4, 8, 16, 32],
    ).map_err(|e| format!("ModelRunner::new: {:?}", e))?;

    // Prime CUDA Graphs for decode-only batches in {1,2,4,8,16}.
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

    let hb_ms = bs.server_heartbeat_ms
        .unwrap_or(bs.heartbeat_interval_ms_arg)
        .min(bs.heartbeat_interval_ms_arg);
    let heartbeat_interval = Duration::from_millis(hb_ms.max(200));
    eprintln!("[serve] heartbeat interval = {:?}", heartbeat_interval);
    let mut last_heartbeat = Instant::now();
    let mut active: HashMap<u64, ActiveSeq> = HashMap::new();

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
                        if active.remove(&c.sequence_id).is_some() {
                            eprintln!("[serve] cancelled seq {}", c.sequence_id);
                        }
                    }
                    SchedulerControlMessage::GrantBlocks(grant) => {
                        if let Some(seq) = active.get_mut(&grant.sequence_id) {
                            seq.block_table.extend(grant.block_ids.iter().copied());
                            seq.block_requested = false;
                        }
                    }
                    SchedulerControlMessage::GrantBlocksDenied(denied) => {
                        active.remove(&denied.sequence_id);
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
            if let Err(e) = handle_prefill(&mut runner, &cmd, &mut active, data, eos_ids) {
                eprintln!("[serve] prefill error: {}", e);
            }
        }

        if !active.is_empty() {
            // Async: request blocks for seqs approaching capacity (non-blocking).
            request_blocks_if_needed(&mut active, control);

            // If any seq can decode, do it. Otherwise wait for grants.
            let any_runnable = active.values().any(|s| s.has_block_space());
            if any_runnable {
                if let Err(e) = run_decode_step(&mut runner, &mut active, data, eos_ids) {
                    eprintln!("[serve] decode error: {}", e);
                }
            } else {
                // All seqs waiting for blocks — drain all pending grants from control.
                loop {
                    match control.try_recv(20) {
                        Ok(Some((SchedulerControlMessage::GrantBlocks(grant), _))) => {
                            if let Some(seq) = active.get_mut(&grant.sequence_id) {
                                seq.block_table.extend(grant.block_ids.iter().copied());
                                seq.block_requested = false;
                            }
                        }
                        Ok(Some((SchedulerControlMessage::GrantBlocksDenied(denied), _))) => {
                            active.remove(&denied.sequence_id);
                        }
                        Ok(Some((SchedulerControlMessage::Shutdown, _))) => return Ok(()),
                        Ok(Some((SchedulerControlMessage::Cancel(c), _))) => {
                            active.remove(&c.sequence_id);
                        }
                        _ => break, // no more messages or timeout
                    }
                }
            }
        }

        maybe_heartbeat(control, active.len(), &mut last_heartbeat, heartbeat_interval);
    }
    Ok(())
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

/// Run one prefill batch (one PrefillBatchCmd = potentially many segments).
fn handle_prefill<M>(
    runner: &mut ModelRunner<bf16, Cuda, M>,
    cmd: &PrefillBatchCmd,
    active: &mut HashMap<u64, ActiveSeq>,
    data: &DataPump,
    eos_ids: &[i32],
) -> OpResult<()>
where
    M: infer_worker::domain::model::LlmModel<bf16, Cuda>,
{
    let mut steps: Vec<SeqStep> = Vec::with_capacity(cmd.segments.len());
    for (i, seg) in cmd.segments.iter().enumerate() {
        let range = cmd.segment_token_range(i);
        let prompt = cmd.input_ids[range].to_vec();
        let positions: Vec<i32> =
            (seg.segment_start as i32..seg.segment_end as i32).collect();
        steps.push(SeqStep {
            input_ids: prompt,
            positions,
            kv_write_start: seg.segment_start as i32,
            kv_len_after: seg.segment_end as i32,
            block_table: seg.block_table.clone(),
        });
    }
    let first_tokens = runner.step_batch(&steps)?;

    let mut output = StepOutput { prefill_done: Vec::new(), tokens: Vec::new() };
    for (i, seg) in cmd.segments.iter().enumerate() {
        let token = first_tokens[i];
        match seg.completion {
            PrefillSegmentCompletion::ContinuePrefill => {
                // KV-only chunked prefill segment; not finishing the seq.
                output.prefill_done.push(seg.sequence_id);
            }
            PrefillSegmentCompletion::FinishPrefillAndStartDecode => {
                output.prefill_done.push(seg.sequence_id);
                let finished = eos_ids.contains(&token) || seg.max_tokens <= 1;
                output.tokens.push(GeneratedToken {
                    sequence_id: seg.sequence_id,
                    token_id: token,
                    finished,
                });
                if !finished {
                    active.insert(seg.sequence_id, ActiveSeq::from_segment(seg, token));
                }
            }
        }
    }
    let _ = data.send_step_output(&output);
    Ok(())
}

/// Non-blocking: request new blocks for sequences approaching capacity.
/// Requests 4 blocks at a time to amortize round-trip latency.
fn request_blocks_if_needed(
    active: &mut HashMap<u64, ActiveSeq>,
    control: &ControlPump,
) {
    for seq in active.values_mut() {
        if seq.needs_block_soon() {
            let request_blocks: u32 = 4; // 4 blocks = 64 tokens of headroom
            let req = NeedBlocks {
                worker_id: String::new(),
                model_instance_id: String::new(),
                sequence_id: seq.sequence_id,
                current_blocks: seq.block_table.len() as u32,
                required_blocks: seq.block_table.len() as u32 + request_blocks,
                request_blocks,
                reason: NeedBlocksReason::DecodeExtend,
            };
            let _ = control.send(
                WorkerControlMessage::NeedBlocks(req),
                infer_protocol::control_envelope::RequestId(0),
            );
            seq.block_requested = true;
        }
    }
}

/// Run one decode iteration over all active seqs in a single batched step.
fn run_decode_step<M>(
    runner: &mut ModelRunner<bf16, Cuda, M>,
    active: &mut HashMap<u64, ActiveSeq>,
    data: &DataPump,
    eos_ids: &[i32],
) -> OpResult<()>
where
    M: infer_worker::domain::model::LlmModel<bf16, Cuda>,
{
    if active.is_empty() {
        return Ok(());
    }
    // Build steps in a stable order, skipping sequences that have no block space.
    let mut order: Vec<u64> = active.keys().copied()
        .filter(|sid| active[sid].has_block_space())
        .collect();
    order.sort_unstable();
    if order.is_empty() {
        return Ok(()); // all sequences waiting for blocks
    }
    let mut steps: Vec<SeqStep> = Vec::with_capacity(order.len());
    for &sid in &order {
        let seq = &active[&sid];
        steps.push(SeqStep {
            input_ids: vec![seq.last_token],
            positions: vec![seq.kv_len as i32],
            kv_write_start: seq.kv_len as i32,
            kv_len_after: (seq.kv_len + 1) as i32,
            block_table: seq.block_table.clone(),
        });
    }
    let new_tokens = runner.step_batch_with_graph(&steps)?;

    let mut output = StepOutput { prefill_done: Vec::new(), tokens: Vec::new() };
    let mut to_remove: Vec<u64> = Vec::new();
    for (i, &sid) in order.iter().enumerate() {
        let token = new_tokens[i];
        let seq = active.get_mut(&sid).unwrap();
        seq.last_token = token;
        seq.kv_len += 1;
        seq.generated_count += 1;
        let finished = eos_ids.contains(&token) || seq.generated_count >= seq.max_tokens;
        output.tokens.push(GeneratedToken {
            sequence_id: sid,
            token_id: token,
            finished,
        });
        if finished {
            to_remove.push(sid);
        }
    }
    for sid in to_remove {
        active.remove(&sid);
    }
    let _ = data.send_step_output(&output);
    Ok(())
}
