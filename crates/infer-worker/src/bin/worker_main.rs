//! rustinfer-worker binary — entry point for the inference worker process.
//!
//! Loads a Llama-3 / Qwen-3 model on a CUDA device, performs the control
//! plane bootstrap (Hello → LoadModel → Ready), then runs an LLM serve loop:
//!
//!   * receive `PrefillBatchCmd` over the data plane PULL socket
//!   * run prefill via `Runtime::step` (paged KV)
//!   * keep an internal `active_decodes` table; each iteration runs all
//!     active decodes in one batched step until they hit EOS / max_tokens
//!   * push `StepOutput` over the data plane PUSH socket
//!
//! The worker owns the decode self-loop — the scheduler never re-sends
//! per-step decode commands.

use std::path::Path;
use std::time::Instant;

use clap::Parser;
use half::bf16;
use serde::Deserialize;

use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;

use infer_worker::application::serve_loop::{Bootstrap, run_with_model};
use infer_worker::infrastructure::cuda::Cuda;
use infer_worker::infrastructure::io::SafetensorsReader;
use infer_worker::infrastructure::transport::control_pump::ControlPump;
use infer_worker::infrastructure::transport::data_pump::DataPump;
use infer_worker::models::loader::{LoadConfig, RopeScaling, WeightLoader};

#[derive(Parser, Debug)]
#[command(name = "rustinfer-worker", version = "0.3.0")]
struct Args {
    /// Path to the shared TOML launch config.
    #[arg(long, default_value = "rustinfer.toml")]
    config: String,

    /// Number of decode steps to profile with cudaProfilerApi.
    /// Diagnostic-only override (not part of the shared config). When set,
    /// calls cudaProfilerStart() before the first decode step and
    /// cudaProfilerStop() after N steps, then exits.
    /// Use with: nsys profile --capture-range=cudaProfilerApi ...
    #[arg(long)]
    profile_cuda_steps: Option<u32>,
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

fn default_max_position() -> usize {
    4096
}
fn default_rms_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10000.0
}

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
    let head_dim = cfg
        .head_dim
        .unwrap_or_else(|| cfg.hidden_size / cfg.num_attention_heads);
    let rope_scaling = cfg.rope_scaling.as_ref().and_then(|rs| {
        let is_llama3 = rs.rope_type.as_deref() == Some("llama3");
        if !is_llama3 {
            return None;
        }
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
    let suffix = spec
        .strip_prefix("cuda:")
        .ok_or_else(|| format!("expected cuda:N, got '{}'", spec))?;
    suffix
        .parse()
        .map_err(|e: std::num::ParseIntError| format!("invalid device id: {}", e))
}

macro_rules! dispatch_worker_model {
    (
        model_type = $model_type:expr,
        loader = $loader:expr,
        load_cfg = $load_cfg:expr,
        cuda = $cuda:expr,
        control = $control:expr,
        data = $data:expr,
        bootstrap = $bootstrap:expr,
        eos_ids = $eos_ids:expr,
        profile_cuda_steps = $profile_cuda_steps:expr,
        load_start = $load_start:expr,
        shipped { $( $arch:literal => $load_method:ident ),+ $(,)? },
        default => $default_method:ident
    ) => {{
        match $model_type.as_str() {
            $(
                $arch => {
                    let model = $loader
                        .$load_method::<bf16, Cuda>($load_cfg, $cuda)
                        .map_err(|e| format!("{}: {:?}", stringify!($load_method), e))?;
                    eprintln!(
                        "[bootstrap] weights loaded in {:.2}s",
                        $load_start.elapsed().as_secs_f32()
                    );
                    run_with_model(
                        $control,
                        $data,
                        model,
                        $bootstrap,
                        $eos_ids,
                        $profile_cuda_steps,
                    )?;
                }
            )+
            _ => {
                let model = $loader
                    .$default_method::<bf16, Cuda>($load_cfg, $cuda)
                    .map_err(|e| format!("{}: {:?}", stringify!($default_method), e))?;
                eprintln!(
                    "[bootstrap] weights loaded in {:.2}s",
                    $load_start.elapsed().as_secs_f32()
                );
                run_with_model(
                    $control,
                    $data,
                    model,
                    $bootstrap,
                    $eos_ids,
                    $profile_cuda_steps,
                )?;
            }
        }
        Ok::<(), String>(())
    }};
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let cfg = infer_protocol::RustInferConfig::load(&args.config)?;
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| cfg.log_level.clone().into()),
        )
        .init();

    let control_endpoint = cfg.worker_control_endpoint();
    let data_recv_endpoint = cfg.worker_in_endpoint();
    let data_send_endpoint = cfg.worker_out_endpoint();
    let block_size = cfg.paged_block_size;
    let num_blocks_override = cfg.num_blocks;

    eprintln!("rustinfer-worker v0.3.0");
    eprintln!("  worker_id = {}", cfg.worker_id);
    eprintln!("  control   = {}", control_endpoint);
    eprintln!("  data_recv = {}", data_recv_endpoint);
    eprintln!("  data_send = {}", data_send_endpoint);
    eprintln!("  (model/device/limits come from scheduler LoadModel)");

    // ── 1. ZMQ ──
    let zmq_ctx = zmq::Context::new();
    let control = ControlPump::new(
        &zmq_ctx,
        cfg.worker_id.clone(),
        cfg.device.clone(),
        &control_endpoint,
    )?;
    let data = DataPump::new(&zmq_ctx, &data_recv_endpoint, &data_send_endpoint)?;

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
                eprintln!(
                    "[bootstrap] ignoring pre-LoadModel control msg: {:?}",
                    other
                );
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
    let cfg_bytes =
        std::fs::read(&cfg_path).map_err(|e| format!("read {}: {}", cfg_path.display(), e))?;
    let hf_cfg: HfConfig = serde_json::from_slice(&cfg_bytes)
        .map_err(|e| format!("parse {}: {}", cfg_path.display(), e))?;
    let max_seq_len = load.max_model_len;
    let load_cfg = build_load_config(&hf_cfg, max_seq_len);
    eprintln!(
        "[bootstrap] arch={} layers={} dim={} heads={}/{} vocab={}",
        hf_cfg.architectures.first().cloned().unwrap_or_default(),
        hf_cfg.num_hidden_layers,
        hf_cfg.hidden_size,
        hf_cfg.num_attention_heads,
        hf_cfg.num_key_value_heads,
        hf_cfg.vocab_size,
    );

    let st_path = Path::new(&load.model_path);
    let reader = SafetensorsReader::open(st_path).map_err(|e| format!("open weights: {}", e))?;
    let loader = WeightLoader::new(&reader);
    let load_start = Instant::now();

    // Model type is derived from the model's config.json, NOT from
    // `load.model_type` (which the scheduler fills for its own logging).
    // This guarantees the worker dispatch always matches the loaded weights.
    let model_type = infer_protocol::resolve_model_type(&load.model_path)?;
    eprintln!(
        "[bootstrap] loading weights for model_type='{}' (derived from config.json)",
        model_type
    );

    // ── 5/6/7. Build runner + send Ready + run serve loop, dispatched on model_type ──
    let bootstrap = Bootstrap {
        load: &load,
        cuda: &cuda,
        load_cfg: &load_cfg,
        max_seq_len,
        block_size,
        num_blocks_override,
        server_heartbeat_ms,
        model_type: model_type.clone(),
        capture_sizes: cfg.capture_sizes.clone(),
    };
    let eos_ids: Vec<i32> = match model_type.as_str() {
        "qwen3" => vec![151643, 151645],   // <|endoftext|>, <|im_end|>
        _ => vec![128001, 128008, 128009], // Llama 3.x default
    };

    dispatch_worker_model!(
        model_type = model_type,
        loader = loader,
        load_cfg = &load_cfg,
        cuda = &cuda,
        control = &control,
        data = &data,
        bootstrap = bootstrap,
        eos_ids = &eos_ids,
        profile_cuda_steps = args.profile_cuda_steps,
        load_start = load_start,
        shipped {
            "qwen3" => load_qwen3,
        },
        default => load_llama3
    )?;

    Ok(())
}
