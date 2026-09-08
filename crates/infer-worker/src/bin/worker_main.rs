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
use std::time::{Duration, Instant};

use clap::Parser;
use half::bf16;
use serde::Deserialize;

use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
use infer_protocol::worker_to_scheduler_control::WORKER_CONTROL_PROTOCOL_VERSION;

use infer_worker::application::serve_loop::{
    Bootstrap, RuntimeFollowerFactory, RuntimeFollowerInit, run_with_model,
};
use infer_worker::application::tensor_parallel::{LocalTpBootstrap, TpRankResource};
use infer_worker::domain::dtype::quant::QuantScheme;
use infer_worker::domain::model::DecoderModel;
use infer_worker::domain::ports::{OpError, OpResult};
use infer_worker::infrastructure::cuda::{Cuda, CudaMemoryPlan, device_utils};
use infer_worker::infrastructure::io::SafetensorsReader;
use infer_worker::infrastructure::transport::control_pump::ControlPump;
use infer_worker::infrastructure::transport::data_pump::DataPump;
use infer_worker::models::loader::{LinearAttnConfig, LoadConfig, RopeScaling, WeightLoader};
use infer_worker::models::{llama3, qwen3, qwen3_5, qwen3_moe};

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
    /// compressed-tensors / llm-compressor quantization block. Present only for
    /// quantized checkpoints; we support int4 `pack-quantized` on the MLP.
    #[serde(default)]
    quantization_config: Option<HfQuantConfig>,

    // ── Sparse MoE fields (Qwen3 MoE; absent / zero for dense models) ──
    #[serde(default)]
    num_experts: usize,
    #[serde(default)]
    num_experts_per_tok: usize,
    #[serde(default)]
    moe_intermediate_size: usize,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default)]
    decoder_sparse_step: usize,

    // ── Qwen3.5 hybrid-stack fields (absent / defaulted for Llama3 & Qwen3) ──
    /// Per-layer attention selector, e.g. `["linear_attention", ..., "full_attention"]`.
    /// Non-empty only for the hybrid Gated-DeltaNet stack.
    #[serde(default)]
    layer_types: Vec<String>,
    /// Fallback when `layer_types` is absent: every `full_attention_interval`-th
    /// layer is full attention, the rest are linear (Gated DeltaNet).
    #[serde(default)]
    full_attention_interval: Option<usize>,
    /// Full-attn `[gate | query]` output gate (`q_proj` emits 2× q_dim).
    #[serde(default)]
    attn_output_gate: bool,
    /// Partial RoPE fraction (Qwen3.5 full-attn: 0.25 → 64 of 256). May also
    /// live under `rope_parameters`; [`build_load_config`] prefers that.
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
    /// Gated-DeltaNet key/query head count.
    #[serde(default)]
    linear_num_key_heads: Option<usize>,
    /// Gated-DeltaNet value head count (one recurrent state each).
    #[serde(default)]
    linear_num_value_heads: Option<usize>,
    /// Gated-DeltaNet per-head key/query dim.
    #[serde(default)]
    linear_key_head_dim: Option<usize>,
    /// Gated-DeltaNet per-head value dim.
    #[serde(default)]
    linear_value_head_dim: Option<usize>,
    /// Gated-DeltaNet causal-conv kernel width.
    #[serde(default)]
    linear_conv_kernel_dim: Option<usize>,
    /// Newer checkpoints nest `rope_theta` / `partial_rotary_factor` here.
    #[serde(default)]
    rope_parameters: Option<HfRopeParameters>,
}

/// Qwen3.5's nested `rope_parameters` block. Older configs keep `rope_theta`
/// flat at the top level; this captures the newer nested form.
#[derive(Debug, Deserialize)]
struct HfRopeParameters {
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
}

/// Subset of the HuggingFace `quantization_config` block we act on. This covers
/// both compressed-tensors INT4 metadata and HuggingFace's blockwise FP8
/// metadata. The per-layer INT4 `ignore` list (attention, lm_head) is honored
/// implicitly because the loader only reads packed tensors for the MLP
/// projections.
#[derive(Debug, Deserialize)]
struct HfQuantConfig {
    #[serde(default)]
    quant_method: Option<String>,
    #[serde(default)]
    weight_block_size: Option<[usize; 2]>,
    #[serde(default)]
    fmt: Option<String>,
    #[serde(default)]
    activation_scheme: Option<String>,
    #[serde(default)]
    config_groups: std::collections::HashMap<String, HfQuantGroup>,
}

#[derive(Debug, Deserialize)]
struct HfQuantGroup {
    #[serde(default)]
    weights: Option<HfQuantWeights>,
    #[serde(default)]
    targets: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HfQuantWeights {
    #[serde(default)]
    num_bits: Option<u32>,
    #[serde(default)]
    group_size: Option<usize>,
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

/// Parse a HuggingFace `config.json` into our flat [`HfConfig`].
///
/// Qwen3.5 (`Qwen3_5ForConditionalGeneration`) nests every text-model field
/// under `text_config` (with `vision_config` / MTP fields as siblings we skip
/// in v1). Older Llama3 / Qwen3 configs are already flat. We normalize by
/// lifting `text_config`'s keys to the root — root keys win on conflict so an
/// explicit top-level override is respected — then deserialize the flat shape.
/// This keeps `HfConfig` a single flat struct rather than forking the parser on
/// model family.
fn parse_hf_config(bytes: &[u8]) -> Result<HfConfig, String> {
    let mut root: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| format!("parse config json: {}", e))?;

    if let Some(text_cfg) = root.get("text_config").cloned()
        && let (Some(root_map), Some(text_map)) = (root.as_object_mut(), text_cfg.as_object())
    {
        for (k, v) in text_map {
            root_map.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }

    serde_json::from_value(root).map_err(|e| format!("deserialize HfConfig: {}", e))
}

/// Build the per-layer full-vs-linear attention selector for the hybrid stack.
///
/// Prefers the explicit `layer_types` list; falls back to
/// `full_attention_interval` (every k-th layer is full, matching HF's
/// `(i + 1) % interval == 0` convention seen in Qwen3.5: layers 3,7,…,31).
fn build_layer_is_full(cfg: &HfConfig) -> Vec<bool> {
    if !cfg.layer_types.is_empty() {
        return cfg
            .layer_types
            .iter()
            .map(|t| t == "full_attention")
            .collect();
    }
    let interval = cfg.full_attention_interval.unwrap_or(0);
    (0..cfg.num_hidden_layers)
        .map(|i| interval != 0 && (i + 1) % interval == 0)
        .collect()
}

/// Assemble the Gated-DeltaNet config when the checkpoint declares a hybrid
/// stack (any `linear_attention` layer). Returns `None` for homogeneous
/// full-attention decoders (Llama3 / Qwen3).
fn build_linear_attn(cfg: &HfConfig) -> Option<LinearAttnConfig> {
    let layer_is_full = build_layer_is_full(cfg);
    // A hybrid stack has at least one non-full (linear) layer AND the GDN dims.
    let has_linear = layer_is_full.iter().any(|&f| !f);
    if !has_linear {
        return None;
    }
    Some(LinearAttnConfig {
        num_key_heads: cfg.linear_num_key_heads?,
        num_value_heads: cfg.linear_num_value_heads?,
        key_head_dim: cfg.linear_key_head_dim?,
        value_head_dim: cfg.linear_value_head_dim?,
        conv_kernel_dim: cfg.linear_conv_kernel_dim?,
        layer_is_full,
    })
}

/// Derive the MLP int4 quant scheme from `quantization_config`, or `None` for a
/// dense model. We enable int4 only when a config group is 4-bit and targets
/// the MLP `gate/up/down` projections (the shape this build's kernel supports).
fn derive_mlp_quant(cfg: &HfConfig) -> Option<QuantScheme> {
    let qc = cfg.quantization_config.as_ref()?;
    // compressed-tensors is the format llm-compressor emits for W4A16.
    if qc.quant_method.as_deref() != Some("compressed-tensors") {
        return None;
    }
    for group in qc.config_groups.values() {
        let w = match &group.weights {
            Some(w) => w,
            None => continue,
        };
        if w.num_bits != Some(4) {
            continue;
        }
        let targets_mlp = group
            .targets
            .iter()
            .any(|t| t.contains("gate_proj") || t.contains("up_proj") || t.contains("down_proj"));
        if !targets_mlp {
            continue;
        }
        let mut scheme = QuantScheme::AWQ_INT4_G128;
        if let Some(g) = w.group_size {
            scheme.group = g;
        }
        return Some(scheme);
    }
    None
}

/// Validate HuggingFace blockwise-FP8 metadata and return the weight scale
/// block shape. Other quantization methods are handled independently (for
/// example compressed-tensors INT4 above), so they do not enable FP8.
fn derive_fp8_block(cfg: &HfConfig) -> Result<Option<[usize; 2]>, String> {
    let Some(qc) = cfg.quantization_config.as_ref() else {
        return Ok(None);
    };
    if qc.quant_method.as_deref() != Some("fp8") {
        return Ok(None);
    }

    if qc.fmt.as_deref() != Some("e4m3") {
        return Err(format!(
            "unsupported FP8 format {:?}; expected 'e4m3'",
            qc.fmt.as_deref()
        ));
    }
    if qc.activation_scheme.as_deref() != Some("dynamic") {
        return Err(format!(
            "unsupported FP8 activation_scheme {:?}; expected 'dynamic'",
            qc.activation_scheme.as_deref()
        ));
    }

    let block = qc
        .weight_block_size
        .ok_or_else(|| "FP8 quantization requires weight_block_size".to_string())?;
    if block.contains(&0) {
        return Err(format!(
            "FP8 weight_block_size dimensions must be non-zero, got {:?}",
            block
        ));
    }
    if block != [128, 128] {
        return Err(format!(
            "unsupported FP8 weight_block_size {:?}; native CUDA kernels require [128, 128]",
            block
        ));
    }
    Ok(Some(block))
}

/// Resolve the stop-token ids for a checkpoint. EOS is an attribute of the
/// model's config, not of its architecture family, so we read it rather than
/// hardcoding per `model_type`:
///   1. `generation_config.json` `eos_token_id` (scalar or array) — authoritative
///   2. `config.json` `eos_token_id` (scalar or array)
///   3. an architecture-keyed default, with a warning (config-less checkpoints)
fn read_eos_ids(model_path: &str, model_type: &str) -> Vec<i32> {
    // Pull `eos_token_id` from one JSON file: accepts an int or an int array.
    fn eos_from(path: &Path) -> Option<Vec<i32>> {
        let bytes = std::fs::read(path).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        let e = v.get("eos_token_id")?;
        if let Some(n) = e.as_i64() {
            Some(vec![n as i32])
        } else if let Some(arr) = e.as_array() {
            let ids: Vec<i32> = arr
                .iter()
                .filter_map(|x| x.as_i64().map(|n| n as i32))
                .collect();
            (!ids.is_empty()).then_some(ids)
        } else {
            None
        }
    }

    let dir = Path::new(model_path);
    if let Some(ids) = eos_from(&dir.join("generation_config.json")) {
        return ids;
    }
    // NOTE: for qwen3_5 the only top-level `eos_token_id` is nested under
    // `text_config` and holds just `<|endoftext|>` (248044) — chat generation
    // must ALSO stop on `<|im_end|>` (248046). We deliberately do not pick up
    // that incomplete nested value here; the `qwen3_5` default below is the
    // complete stop set. Top-level (flat) configs are still honored.
    if let Some(ids) = eos_from(&dir.join("config.json")) {
        return ids;
    }

    // No config eos — fall back on an architecture default, but say so, since a
    // wrong stop-token set silently produces run-on or truncated generations.
    let default = match model_type {
        // <|endoftext|>=248044, <|im_end|>=248046
        "qwen3_5" => vec![248044, 248046],
        "qwen3" | "qwen3_moe" => vec![151643, 151645], // <|endoftext|>, <|im_end|>
        _ => vec![128001, 128008, 128009],             // Llama 3.x default
    };
    eprintln!(
        "[bootstrap] no eos_token_id in generation_config.json/config.json; \
         falling back to {} default {:?}",
        model_type, default
    );
    default
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

fn build_load_config(cfg: &HfConfig, max_seq_len: usize) -> Result<LoadConfig, String> {
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

    // rope_theta may be flat (Llama3/Qwen3) or nested under rope_parameters
    // (Qwen3.5). Nested wins when present.
    let rope_theta = cfg
        .rope_parameters
        .as_ref()
        .and_then(|rp| rp.rope_theta)
        .unwrap_or(cfg.rope_theta);

    // partial_rotary_factor: nested rope_parameters wins, then flat, else 1.0
    // (full rotary). rotary_dim rounds to an even count of head dims.
    let partial = cfg
        .rope_parameters
        .as_ref()
        .and_then(|rp| rp.partial_rotary_factor)
        .or(cfg.partial_rotary_factor)
        .unwrap_or(1.0);
    let rotary_dim = (((head_dim as f32) * partial) as usize) & !1;

    let linear_attn = build_linear_attn(cfg);

    Ok(LoadConfig {
        dim: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        layer_num: cfg.num_hidden_layers,
        head_num: cfg.num_attention_heads,
        kv_head_num: cfg.num_key_value_heads,
        head_dim,
        vocab_size: cfg.vocab_size,
        seq_len: max_seq_len.max(cfg.max_position_embeddings.min(max_seq_len)),
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta,
        rope_scaling,
        mlp_quant: derive_mlp_quant(cfg),
        fp8_block: derive_fp8_block(cfg)?,
        rotary_dim,
        attn_output_gate: cfg.attn_output_gate,
        linear_attn,
        num_experts: cfg.num_experts,
        experts_per_tok: cfg.num_experts_per_tok,
        moe_intermediate_size: cfg.moe_intermediate_size,
        norm_topk_prob: cfg.norm_topk_prob,
        decoder_sparse_step: cfg.decoder_sparse_step,
    })
}

fn parse_device_id(spec: &str) -> Result<i32, String> {
    let suffix = spec
        .strip_prefix("cuda:")
        .ok_or_else(|| format!("expected cuda:N, got '{}'", spec))?;
    suffix
        .parse()
        .map_err(|e: std::num::ParseIntError| format!("invalid device id: {}", e))
}

/// Build one factory per process-local TP follower. The factory is invoked
/// inside that rank's long-lived Runtime thread, so neither the model nor its
/// `Rc`-backed forward scratch ever crosses a thread boundary.
fn make_follower_factories<M, F>(
    ranks: Vec<TpRankResource>,
    model_path: String,
    load_cfg: LoadConfig,
    build: F,
) -> Vec<RuntimeFollowerFactory<M>>
where
    M: DecoderModel<bf16, Cuda> + 'static,
    F: for<'a> Fn(&WeightLoader<'a>, &LoadConfig, &Cuda) -> OpResult<M> + Copy + Send + 'static,
{
    ranks
        .into_iter()
        .map(|resource| {
            let TpRankResource {
                global_rank,
                global_size,
                local_rank,
                cuda,
                communicator,
            } = resource;
            let model_path = model_path.clone();
            let load_cfg = load_cfg.clone();
            Box::new(move |init: RuntimeFollowerInit| {
                if init.rank != global_rank || init.size != global_size {
                    return Err(OpError::Shape(format!(
                        "TP follower factory rank {global_rank}/{global_size} received init {}/{}",
                        init.rank, init.size
                    )));
                }

                // CUDA's current device is thread-local. The `Cuda` handles
                // were created by the controller thread, so establish this
                // rank's device before any allocator/model call on the new
                // Runtime thread and keep it current for that thread's life.
                device_utils::set_current_device(cuda.device_id).map_err(|error| {
                    OpError::Kernel(format!(
                        "set TP rank {global_rank} CUDA device {}: {error}",
                        cuda.device_id
                    ))
                })?;

                let communicator = communicator.ok_or_else(|| {
                    OpError::Shape(format!(
                        "TP follower rank {global_rank}/{} has no NCCL communicator",
                        global_size
                    ))
                })?;
                let scope = init.build_scope(cuda.clone(), communicator)?;
                let reader = SafetensorsReader::open(Path::new(&model_path)).map_err(|error| {
                    OpError::Kernel(format!("open TP rank {global_rank} weights: {error}"))
                })?;
                let loader = WeightLoader::with_tensor_parallel(&reader, global_rank, global_size)?;
                let started = Instant::now();
                let model = build(&loader, &load_cfg, &cuda)?;
                tracing::info!(
                    rank = global_rank,
                    local_rank,
                    size = global_size,
                    elapsed_seconds = started.elapsed().as_secs_f32(),
                    "TP follower weights loaded"
                );
                init.build_runtime(model, scope)
            }) as RuntimeFollowerFactory<M>
        })
        .collect()
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
                // Enforce the control-protocol version instead of just logging
                // it: a mismatched scheduler build would otherwise fail later
                // with opaque msgpack decode errors mid-batch.
                if h.protocol_version != WORKER_CONTROL_PROTOCOL_VERSION {
                    return Err(format!(
                        "scheduler speaks control protocol v{} but this worker requires v{}; \
                         rebuild/redeploy the mismatched side",
                        h.protocol_version, WORKER_CONTROL_PROTOCOL_VERSION,
                    ));
                }
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
        "[bootstrap] LoadModel: path={} max_seqs={} max_tokens={} max_model_len={} tp={}/{} pp={}/{}",
        load.model_path,
        load.max_batch_seqs,
        load.max_batch_tokens,
        load.max_model_len,
        load.tp_rank,
        load.tp_size,
        load.pp_rank,
        load.pp_size,
    );
    if load.pp_rank != 0 || load.pp_size != 1 {
        return Err(format!(
            "pipeline parallelism is not implemented; expected pp_rank/pp_size=0/1, got {}/{}",
            load.pp_rank, load.pp_size
        ));
    }
    if load.tp_rank != 0 || load.tp_size == 0 {
        return Err(format!(
            "single-process TP requires tp_rank=0 and tp_size>0, got tp_rank/tp_size={}/{}",
            load.tp_rank, load.tp_size
        ));
    }

    // ── 4. Load model ──
    let device_id = parse_device_id(&load.device)?;
    const MIB: usize = 1024 * 1024;
    let cuda_memory = cfg.cuda_memory;
    let memory_plan = CudaMemoryPlan {
        kernel_workspace_bytes: cuda_memory.kernel_workspace_mib * MIB,
        graph_arena_bytes: cuda_memory.graph_arena_mib * MIB,
        pool_retain_bytes: cuda_memory.pool_retain_mib * MIB,
    };
    eprintln!(
        "[bootstrap] cuda_memory: kernel={}MiB graph={}MiB pool={}MiB",
        cuda_memory.kernel_workspace_mib, cuda_memory.graph_arena_mib, cuda_memory.pool_retain_mib,
    );
    // Deployment placement and CUDA/NCCL creation live behind one bootstrap
    // boundary. Today the worker owns the whole global group; a future
    // one-process-per-node path can provide a distributed constructor without
    // changing model loading or the process-local Runtime peer loop.
    let tp_bootstrap = LocalTpBootstrap::single_process(
        load.tp_size,
        device_id,
        memory_plan,
        Duration::from_secs(cfg.tp_startup_timeout_secs),
    )
    .map_err(|error| format!("initialize local TP ranks: {error}"))?;
    let tp_placement = tp_bootstrap.placement();
    if load.tp_rank != tp_placement.local_rank_start() {
        return Err(format!(
            "LoadModel leader rank {}/{} does not match process-local TP rank start {}",
            load.tp_rank,
            load.tp_size,
            tp_placement.local_rank_start()
        ));
    }
    let all_devices = tp_bootstrap.devices();
    let mut rank_resources = tp_bootstrap.into_ranks();
    for resource in &rank_resources {
        eprintln!(
            "[bootstrap] TP global rank {}/{} (local rank {}) -> cuda:{}",
            resource.global_rank,
            tp_placement.global_size(),
            resource.local_rank,
            resource.cuda.device_id
        );
    }
    let leader = rank_resources.remove(0);
    let cuda = leader.cuda;
    let leader_communicator = leader.communicator;
    let follower_ranks = rank_resources;
    if follower_ranks.len() != tp_placement.local_rank_count().saturating_sub(1) {
        return Err(format!(
            "TP{} process placement requires {} followers, got {}",
            tp_placement.global_size(),
            tp_placement.local_rank_count().saturating_sub(1),
            follower_ranks.len()
        ));
    }
    device_utils::set_current_device(cuda.device_id)
        .map_err(|error| format!("set TP rank 0 CUDA device {}: {error}", cuda.device_id))?;
    let cfg_path = Path::new(&load.model_path).join("config.json");
    let cfg_bytes =
        std::fs::read(&cfg_path).map_err(|e| format!("read {}: {}", cfg_path.display(), e))?;
    let hf_cfg: HfConfig =
        parse_hf_config(&cfg_bytes).map_err(|e| format!("parse {}: {}", cfg_path.display(), e))?;
    let max_seq_len = load.max_model_len;
    let mut load_cfg = build_load_config(&hf_cfg, max_seq_len)
        .map_err(|e| format!("invalid quantization config: {}", e))?;
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
    let loader = WeightLoader::with_tensor_parallel(&reader, load.tp_rank, load.tp_size)
        .map_err(|e| format!("invalid tensor-parallel topology: {}", e))?;
    let load_start = Instant::now();

    // Reconcile the config's quant claim with the actual weights: only enable
    // the int4 MLP path when packed tensors are really present. A mismatch
    // (quantized config but dense weights, or vice-versa) falls back to dense
    // rather than failing the load.
    if let Some(scheme) = load_cfg.mlp_quant {
        let has_packed = loader.has_tensor("model.layers.0.mlp.gate_proj.weight_packed");
        if has_packed {
            eprintln!(
                "[bootstrap] MLP int4 quant enabled (pack-quantized, group_size={})",
                scheme.group
            );
        } else {
            eprintln!(
                "[bootstrap] config declares int4 MLP quant but no weight_packed tensors found; \
                 loading as dense"
            );
            load_cfg.mlp_quant = None;
        }
    }

    if let Some([block_n, block_k]) = load_cfg.fp8_block {
        let probe_name = "model.layers.0.self_attn.q_proj.weight";
        let probe = loader
            .read_view(probe_name)
            .map_err(|e| format!("FP8 checkpoint is missing '{}': {}", probe_name, e))?;
        if probe.dtype() != safetensors::Dtype::F8_E4M3 {
            return Err(format!(
                "config declares block FP8 but '{}' has dtype {:?}, expected F8_E4M3",
                probe_name,
                probe.dtype()
            ));
        }
        let scale_name = format!("{}_scale_inv", probe_name);
        if !loader.has_tensor(&scale_name) {
            return Err(format!(
                "config declares block FP8 but scale tensor '{}' is missing",
                scale_name
            ));
        }
        eprintln!(
            "[bootstrap] block FP8 checkpoint detected ({}x{}); keeping E4M3 linear weights quantized on device",
            block_n, block_k
        );
    }

    // Model type is derived from the model's config.json, NOT from
    // `load.model_type` (which the scheduler fills for its own logging).
    // This guarantees the worker dispatch always matches the loaded weights.
    let model_type = infer_protocol::resolve_model_type(&load.model_path)?;
    eprintln!(
        "[bootstrap] loading weights for model_type='{}' (derived from config.json)",
        model_type
    );

    // ── 5/6/7. Build runner + send Ready + run serve loop, dispatched on model_type ──
    let eos_ids: Vec<i32> = read_eos_ids(&load.model_path, &model_type);

    let make_bootstrap = || Bootstrap {
        load: &load,
        tp_placement,
        cuda: &cuda,
        load_cfg: &load_cfg,
        max_seq_len,
        block_size,
        num_blocks_override,
        server_heartbeat_ms,
        model_type: model_type.clone(),
        capture_sizes: cfg.capture_sizes.clone(),
        peer_timeout: Duration::from_secs(cfg.tp_operation_timeout_secs),
        peer_startup_timeout: Duration::from_secs(cfg.tp_startup_timeout_secs),
        tp_communicator: leader_communicator.clone(),
        tp_devices: &all_devices,
    };

    match model_type.as_str() {
        "llama3" => {
            let model = llama3::build::<bf16, Cuda>(&loader, &load_cfg, &cuda)
                .map_err(|e| format!("llama3::build: {:?}", e))?;
            let followers = make_follower_factories(
                follower_ranks,
                load.model_path.clone(),
                load_cfg.clone(),
                llama3::build::<bf16, Cuda>,
            );
            eprintln!(
                "[bootstrap] weights loaded in {:.2}s",
                load_start.elapsed().as_secs_f32()
            );
            run_with_model(
                &control,
                &data,
                model,
                make_bootstrap(),
                followers,
                &eos_ids,
                args.profile_cuda_steps,
            )?;
        }
        "qwen3" => {
            let model = qwen3::build::<bf16, Cuda>(&loader, &load_cfg, &cuda)
                .map_err(|e| format!("qwen3::build: {:?}", e))?;
            let followers = make_follower_factories(
                follower_ranks,
                load.model_path.clone(),
                load_cfg.clone(),
                qwen3::build::<bf16, Cuda>,
            );
            eprintln!(
                "[bootstrap] weights loaded in {:.2}s",
                load_start.elapsed().as_secs_f32()
            );
            run_with_model(
                &control,
                &data,
                model,
                make_bootstrap(),
                followers,
                &eos_ids,
                args.profile_cuda_steps,
            )?;
        }
        "qwen3_5" => {
            let mut model = qwen3_5::build::<bf16, Cuda>(&loader, &load_cfg, &cuda)
                .map_err(|e| format!("qwen3_5::build: {:?}", e))?;
            let full_config: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(
                    std::path::Path::new(&load.model_path).join("config.json"),
                )
                .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            if full_config.get("vision_config").is_some() {
                model
                    .load_vision(&loader, &full_config, &cuda)
                    .map_err(|e| format!("vision load: {e:?}"))?;
            }
            let followers = make_follower_factories(
                follower_ranks,
                load.model_path.clone(),
                load_cfg.clone(),
                qwen3_5::build::<bf16, Cuda>,
            );
            eprintln!(
                "[bootstrap] weights loaded in {:.2}s",
                load_start.elapsed().as_secs_f32()
            );
            run_with_model(
                &control,
                &data,
                model,
                make_bootstrap(),
                followers,
                &eos_ids,
                args.profile_cuda_steps,
            )?;
        }
        "qwen3_moe" => {
            let model = qwen3_moe::build::<bf16, Cuda>(&loader, &load_cfg, &cuda)
                .map_err(|e| format!("qwen3_moe::build: {:?}", e))?;
            let followers = make_follower_factories(
                follower_ranks,
                load.model_path.clone(),
                load_cfg.clone(),
                qwen3_moe::build::<bf16, Cuda>,
            );
            eprintln!(
                "[bootstrap] weights loaded in {:.2}s",
                load_start.elapsed().as_secs_f32()
            );
            run_with_model(
                &control,
                &data,
                model,
                make_bootstrap(),
                followers,
                &eos_ids,
                args.profile_cuda_steps,
            )?;
        }
        other => {
            return Err(format!(
                "unsupported model_type '{}'; supported models: {}",
                other,
                infer_protocol::supported_model_types_csv()
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod config_tests {
    use super::*;

    /// Minimal qwen3_5 config with everything nested under `text_config`, plus a
    /// sibling `vision_config` parsed separately by the vision loader. Trimmed from the real
    /// Qwen3.5-4B config.json (layer_types cut to one [L,L,L,F] period).
    const QWEN3_5_JSON: &str = r#"{
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": true,
        "vision_config": { "model_type": "qwen3_5", "hidden_size": 1024, "depth": 24 },
        "text_config": {
            "attn_output_gate": true,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 262144,
            "num_attention_heads": 16,
            "num_hidden_layers": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "vocab_size": 248320,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25
            }
        }
    }"#;

    /// A flat Qwen3-style config (no text_config nesting, no hybrid stack).
    const QWEN3_FLAT_JSON: &str = r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 40960
    }"#;

    const QWEN3_MOE_JSON: &str = r#"{
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 40960,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "decoder_sparse_step": 1,
        "norm_topk_prob": true
    }"#;

    /// Quantization-relevant fields and model dimensions copied from the
    /// target Qwen3-4B-FP8 checkpoint's `config.json`.
    const QWEN3_4B_FP8_JSON: &str = r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 40960,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128]
        }
    }"#;

    #[test]
    fn parses_nested_text_config() {
        let cfg = parse_hf_config(QWEN3_5_JSON.as_bytes()).expect("parse qwen3_5");
        // Fields lifted out of text_config.
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.intermediate_size, 9216);
        assert_eq!(cfg.num_hidden_layers, 4);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, Some(256));
        assert_eq!(cfg.vocab_size, 248320);
        assert!(cfg.attn_output_gate);
        assert_eq!(cfg.linear_num_key_heads, Some(16));
        assert_eq!(cfg.linear_num_value_heads, Some(32));
        assert_eq!(cfg.linear_key_head_dim, Some(128));
        assert_eq!(cfg.linear_conv_kernel_dim, Some(4));
        assert_eq!(cfg.layer_types.len(), 4);
        // rope_theta lives under the nested rope_parameters block.
        let rp = cfg.rope_parameters.as_ref().expect("rope_parameters");
        assert_eq!(rp.rope_theta, Some(10_000_000.0));
        assert_eq!(rp.partial_rotary_factor, Some(0.25));
    }

    #[test]
    fn hybrid_dims_and_partial_rope() {
        let cfg = parse_hf_config(QWEN3_5_JSON.as_bytes()).unwrap();
        let lc = build_load_config(&cfg, 4096).expect("build qwen3_5 load config");

        // Nested rope_theta wins over the flat default.
        assert_eq!(lc.rope_theta, 10_000_000.0);
        // partial_rotary_factor 0.25 * head_dim 256 = 64 (even).
        assert_eq!(lc.rotary_dim, 64);
        assert!(lc.attn_output_gate);

        let la = lc.linear_attn.expect("hybrid stack detected");
        assert_eq!(la.num_key_heads, 16);
        assert_eq!(la.num_value_heads, 32);
        assert_eq!(la.key_dim(), 2048);
        assert_eq!(la.value_dim(), 4096);
        assert_eq!(la.conv_dim(), 8192); // 2048 + 2048 + 4096
        // layer_types = [L,L,L,F] → last layer is full.
        assert_eq!(la.layer_is_full, vec![false, false, false, true]);
        assert_eq!(la.num_full_layers(), 1);
        assert_eq!(la.num_linear_layers(), 3);
    }

    #[test]
    fn flat_config_has_no_linear_attn() {
        let cfg = parse_hf_config(QWEN3_FLAT_JSON.as_bytes()).expect("parse flat qwen3");
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 28);
        let lc = build_load_config(&cfg, 4096).expect("build flat qwen3 load config");
        // Homogeneous full-attention decoder: no hybrid stack, full rotary.
        assert!(lc.linear_attn.is_none());
        assert!(!lc.attn_output_gate);
        assert_eq!(lc.rotary_dim, lc.head_dim); // partial factor defaults to 1.0
        assert_eq!(lc.head_dim, 128);
        assert_eq!(lc.rope_theta, 1_000_000.0);
        assert!(lc.fp8_block.is_none());
    }

    #[test]
    fn qwen3_moe_config_maps_sparse_fields() {
        let cfg = parse_hf_config(QWEN3_MOE_JSON.as_bytes()).expect("parse qwen3_moe");
        assert_eq!(cfg.num_experts, 128);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.moe_intermediate_size, 768);
        assert_eq!(cfg.decoder_sparse_step, 1);
        assert!(cfg.norm_topk_prob);

        let lc = build_load_config(&cfg, 4096).expect("build qwen3_moe load config");
        assert_eq!(lc.num_experts, 128);
        assert_eq!(lc.experts_per_tok, 8);
        assert_eq!(lc.moe_intermediate_size, 768);
        assert_eq!(lc.decoder_sparse_step, 1);
        assert!(lc.norm_topk_prob);
        assert!(lc.linear_attn.is_none());
    }

    #[test]
    fn full_attention_interval_fallback() {
        // Same model but layer_types omitted — selector derived from interval.
        let json = QWEN3_5_JSON.replace(
            r#""layer_types": ["linear_attention","linear_attention","linear_attention","full_attention"],"#,
            "",
        );
        let cfg = parse_hf_config(json.as_bytes()).expect("parse without layer_types");
        assert!(cfg.layer_types.is_empty());
        let la = build_linear_attn(&cfg).expect("interval-derived hybrid");
        // interval=4, num_layers=4 → (i+1)%4==0 → only layer index 3 is full.
        assert_eq!(la.layer_is_full, vec![false, false, false, true]);
    }

    #[test]
    fn qwen3_4b_fp8_config_maps_block_shape() {
        let cfg = parse_hf_config(QWEN3_4B_FP8_JSON.as_bytes()).expect("parse qwen3-4b-fp8");
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.intermediate_size, 9728);
        assert_eq!(cfg.num_hidden_layers, 36);

        let lc = build_load_config(&cfg, 4096).expect("supported FP8 config");
        assert_eq!(lc.fp8_block, Some([128, 128]));
        assert!(lc.mlp_quant.is_none());
    }

    #[test]
    fn rejects_unsupported_fp8_variants() {
        let cases = [
            (
                "format",
                QWEN3_4B_FP8_JSON.replace(r#""fmt": "e4m3""#, r#""fmt": "e5m2""#),
                "expected 'e4m3'",
            ),
            (
                "activation scheme",
                QWEN3_4B_FP8_JSON.replace(
                    r#""activation_scheme": "dynamic""#,
                    r#""activation_scheme": "static""#,
                ),
                "expected 'dynamic'",
            ),
            (
                "zero block dimension",
                QWEN3_4B_FP8_JSON.replace(
                    r#""weight_block_size": [128, 128]"#,
                    r#""weight_block_size": [0, 128]"#,
                ),
                "must be non-zero",
            ),
            (
                "unsupported block shape",
                QWEN3_4B_FP8_JSON.replace(
                    r#""weight_block_size": [128, 128]"#,
                    r#""weight_block_size": [64, 128]"#,
                ),
                "require [128, 128]",
            ),
            (
                "missing block shape",
                QWEN3_4B_FP8_JSON.replace(
                    r#""weight_block_size": [128, 128]"#,
                    r#""weight_block_size": null"#,
                ),
                "requires weight_block_size",
            ),
        ];

        for (variant, json, expected) in cases {
            let cfg = parse_hf_config(json.as_bytes())
                .unwrap_or_else(|e| panic!("parse unsupported {variant} fixture: {e}"));
            let err = match build_load_config(&cfg, 4096) {
                Ok(_) => panic!("unsupported FP8 {variant} was accepted"),
                Err(err) => err,
            };
            assert!(
                err.contains(expected),
                "unexpected error for {variant}: {err}"
            );
        }
    }
}

#[cfg(test)]
mod qwen35_checkpoint_tests {
    use super::*;
    use infer_worker::domain::cache::{LinearBatch, LinearLayerState, ModelCacheView};
    use infer_worker::domain::component::{Hidden, LayerRange};
    use infer_worker::domain::exec::StepCtx;
    use infer_worker::domain::forward_scratch::ForwardScratch;
    use infer_worker::domain::gdn_scratch::GdnScratch;
    use infer_worker::domain::kv::{KvIndexTensors, KvQuantTier, PagedKvLayer, PagedKvPool};
    use infer_worker::domain::model::SampleRows;
    use infer_worker::domain::plan::{BatchKind, BatchPlan};
    use infer_worker::domain::tensor::Tensor;
    use std::collections::HashMap;

    /// Opt-in checkpoint smoke test and fixed-token layer/logit diagnostic.
    #[test]
    #[ignore = "requires QWEN35_MODEL_PATH and a CUDA device with enough free memory"]
    fn qwen35_checkpoint_load_and_forward() {
        let path = std::env::var("QWEN35_MODEL_PATH").expect("QWEN35_MODEL_PATH");
        let ordinal = std::env::var("QWEN35_DEVICE")
            .unwrap_or_else(|_| "0".into())
            .parse()
            .unwrap();
        let cfg =
            parse_hf_config(&std::fs::read(Path::new(&path).join("config.json")).unwrap()).unwrap();
        let dump_dir = std::env::var("QWEN35_DUMP_DIR").ok();
        let steps: Vec<Vec<i32>> = if let Ok(path) = std::env::var("QWEN35_INPUT_JSON") {
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
        } else {
            vec![vec![1, 2], vec![3]]
        };
        let capacity = steps.iter().map(Vec::len).max().unwrap();
        let total: usize = steps.iter().map(Vec::len).sum();
        let blocks = total.max(4);
        let cfg = build_load_config(&cfg, blocks.max(16)).unwrap();
        let dump = |name: String, values: Vec<bf16>| {
            if let Some(dir) = &dump_dir {
                std::fs::create_dir_all(dir).unwrap();
                let bytes: Vec<u8> = values
                    .iter()
                    .flat_map(|v| v.to_f32().to_le_bytes())
                    .collect();
                std::fs::write(Path::new(dir).join(name), bytes).unwrap();
            }
        };
        let reader = SafetensorsReader::open(&path).unwrap();
        let cuda = Cuda::new(ordinal).unwrap();
        let loader = WeightLoader::new(&reader);
        let mut model = qwen3_5::build::<bf16, Cuda>(&loader, &cfg, &cuda).unwrap();
        let dims = model.dims();
        assert_eq!(model.cache_layout().num_full_layers(), 8);
        assert_eq!(model.cache_layout().linear_dims().len(), 24);
        assert_eq!(
            model.decoder.embed.table.data_ptr(),
            model
                .decoder
                .lm_head
                .proj
                .weight
                .as_dense()
                .unwrap()
                .data_ptr()
        );
        model.install_scratch(ForwardScratch::new(&cuda, dims, capacity, 1).unwrap());
        model
            .install_gdn_scratch(
                GdnScratch::new(
                    &cuda,
                    dims.dim,
                    model.cache_layout().linear_dims()[0],
                    capacity,
                )
                .unwrap(),
            )
            .unwrap();
        let mut kv = PagedKvPool {
            layers: (0..8)
                .map(|_| PagedKvLayer {
                    k: Tensor::zeros([blocks, 1, dims.kv_dim], &cuda).unwrap(),
                    v: Tensor::zeros([blocks, 1, dims.kv_dim], &cuda).unwrap(),
                })
                .collect(),
            num_blocks: blocks,
            block_size: 1,
            kv_dim: dims.kv_dim,
            quant: KvQuantTier::None,
            seq_kv_len: HashMap::new(),
        };
        let mut states: Vec<_> = model
            .cache_layout()
            .linear_dims()
            .iter()
            .map(|&d| LinearLayerState::new(d, 1, &cuda).unwrap())
            .collect();
        let ints = |v: &[i32]| Tensor::from_host_slice(v, [v.len()], &cuda).unwrap();
        let scope = cuda.scope();
        let mut direct_tokens = Vec::new();
        let mut start = 0i32;
        for (step, ids) in steps.iter().enumerate() {
            let n = ids.len();
            let positions: Vec<i32> = (start..start + n as i32).collect();
            let (cu, req, tile) = BatchPlan::plan_ragged_tiles(&[n as i32]);
            let plan = BatchPlan {
                kind: if n == 1 {
                    BatchKind::DecodeOnly
                } else {
                    BatchKind::Ragged
                },
                num_tokens: n,
                batch: 1,
                q_lens: vec![n as i32],
                kv_lens: vec![start + n as i32],
                seq_positions: vec![start],
                rope_positions: positions.clone(),
                max_blocks_per_seq: blocks,
                block_size: 1,
                total_q_tiles: req.len() as i32,
            };
            let index = KvIndexTensors {
                block_tables: Tensor::from_host_slice(
                    &(0..blocks as i32).collect::<Vec<_>>(),
                    [1, blocks],
                    &cuda,
                )
                .unwrap(),
                cu_q_lens: ints(&cu),
                kv_lens: ints(&plan.kv_lens),
                seq_positions: ints(&[start]),
                seq_lens_step: ints(&[n as i32]),
                rope_positions: ints(&positions),
                block2req: ints(&req),
                block2tile: ints(&tile),
                valid_q_tiles: ints(&[req.len() as i32]),
                valid_suffix_q_tiles: ints(&[req.len() as i32]),
            };
            let linear = LinearBatch::new(&[0], &[n as i32], 1, &cuda).unwrap();
            let mut cache = ModelCacheView::hybrid(&mut kv, &index, &mut states, &linear);
            let ctx = StepCtx::new(&scope, &plan);
            let mut hidden = Hidden {
                stream: Tensor::zeros([n, dims.dim], &cuda).unwrap(),
                pending: None,
            };
            model.embed(&ints(&ids), &mut hidden, &ctx).unwrap();
            dump(
                format!("step{step}_embed.f32"),
                hidden.stream.to_host_vec().unwrap(),
            );
            if dump_dir.is_some() {
                for layer in 0..dims.num_layers {
                    model
                        .decode_layers(
                            LayerRange {
                                start: layer,
                                end: layer + 1,
                            },
                            &mut hidden,
                            &mut cache,
                            &ctx,
                        )
                        .unwrap();
                    dump(
                        format!("step{step}_layer{layer}.f32"),
                        hidden.stream.to_host_vec().unwrap(),
                    );
                }
            } else {
                model
                    .decode_layers(
                        LayerRange {
                            start: 0,
                            end: dims.num_layers,
                        },
                        &mut hidden,
                        &mut cache,
                        &ctx,
                    )
                    .unwrap();
            }
            let logits_tensor = model
                .finalize(&hidden, SampleRows::LastPerSeq, &ctx)
                .unwrap()
                .0;
            let sampled =
                <Cuda as infer_worker::domain::ports::FusedOps>::argmax(&ctx, &logits_tensor)
                    .unwrap();
            if let Some(dir) = &dump_dir {
                std::fs::write(
                    Path::new(dir).join(format!("step{step}_sampled.json")),
                    serde_json::to_vec(&sampled).unwrap(),
                )
                .unwrap();
            }
            let logits = logits_tensor.to_host_vec().unwrap();
            dump(format!("step{step}_logits.f32"), logits.clone());
            direct_tokens.push(
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| {
                        a.1.to_f32()
                            .total_cmp(&b.1.to_f32())
                            .then_with(|| b.0.cmp(&a.0))
                    })
                    .unwrap()
                    .0 as i32,
            );
            assert_eq!(
                sampled[0],
                *direct_tokens.last().unwrap(),
                "device/host argmax mismatch at step {step}"
            );
            assert_eq!(logits.len(), dims.vocab_size);
            assert!(logits.iter().all(|v| v.to_f32().is_finite()));
            assert!(logits.windows(2).any(|v| v[0] != v[1]));
            eprintln!(
                "Qwen3.5 checkpoint: {n} tokens at position {start}, finite logits={}",
                logits.len()
            );
            start += n as i32;
        }

        if dump_dir.is_some() {
            return;
        }

        use infer_worker::application::runtime::Runtime;
        use infer_worker::application::sampler_stack::GreedySampler;
        use infer_worker::domain::plan::{SeqStep, StepRequest, StopCriteria};
        let mut runtime = Runtime::new(
            model,
            cuda.scope(),
            Box::new(GreedySampler),
            4,
            1,
            4,
            16,
            3,
            2,
            vec![1, 2],
        )
        .unwrap();
        runtime.profile_forward().unwrap();
        runtime.prime_graphs().unwrap();
        assert!(runtime.graph.is_some());
        for (i, (start, ids)) in [(0, vec![1, 2]), (2, vec![3])].into_iter().enumerate() {
            let n = ids.len();
            let req = StepRequest {
                seqs: vec![SeqStep {
                    sequence_id: 42,
                    input_ids: ids,
                    positions: (start..start + n as i32).collect(),
                    kv_write_start: start,
                    kv_len_after: start + n as i32,
                    block_table: vec![0, 1, 2, 3],
                }],
                sampling: vec![Default::default()],
                stop: StopCriteria {
                    eos_ids: vec![],
                    generated_counts: vec![0],
                    max_tokens: vec![16],
                    ignore_eos: vec![true],
                },
                draft_tokens: vec![],
            };
            let output = runtime.step(&req).unwrap();
            assert_eq!(output.tokens[0][0].token_id, direct_tokens[i]);
        }
        runtime.release_sequence(42);
        eprintln!("Qwen3.5 Runtime prefill/decode match direct-model greedy tokens");
    }

    /// Compare the real checkpoint through eager, cold capture and prewarmed
    /// replay, including ABC serving, padded batches and recurrent slot reuse.
    #[test]
    #[ignore = "requires QWEN35_MODEL_PATH and a CUDA device with enough free memory"]
    fn qwen35_decode_graph_matches_eager() {
        use infer_worker::application::runtime::{GraphDecision, Runtime};
        use infer_worker::application::sampler_stack::GreedySampler;
        use infer_worker::domain::exec::ExecScope;
        use infer_worker::domain::plan::{SeqStep, StepRequest, StopCriteria};

        let path = std::env::var("QWEN35_MODEL_PATH").expect("QWEN35_MODEL_PATH");
        let ordinal = std::env::var("QWEN35_DEVICE")
            .unwrap_or_else(|_| "0".into())
            .parse()
            .unwrap();
        let cfg =
            parse_hf_config(&std::fs::read(Path::new(&path).join("config.json")).unwrap()).unwrap();
        let cfg = build_load_config(&cfg, 64).unwrap();
        let reader = SafetensorsReader::open(&path).unwrap();
        let cuda = Cuda::new(ordinal).unwrap();
        let model = qwen3_5::build::<bf16, Cuda>(&WeightLoader::new(&reader), &cfg, &cuda).unwrap();
        let mut runtime = Runtime::new(
            model,
            cuda.scope(),
            Box::new(GreedySampler),
            257,
            1,
            64,
            64,
            32,
            4,
            vec![1, 2, 4],
        )
        .unwrap();
        let request = |rows: &[(u64, usize, i32, Vec<i32>)]| StepRequest {
            seqs: rows
                .iter()
                .map(|(id, slot, start, ids)| SeqStep {
                    sequence_id: *id,
                    input_ids: ids.clone(),
                    positions: (*start..*start + ids.len() as i32).collect(),
                    kv_write_start: *start,
                    kv_len_after: *start + ids.len() as i32,
                    block_table: (*slot * 64..(*slot + 1) * 64).map(|b| b as u32).collect(),
                })
                .collect(),
            sampling: vec![Default::default(); rows.len()],
            stop: StopCriteria {
                eos_ids: vec![],
                generated_counts: vec![0; rows.len()],
                max_tokens: vec![100; rows.len()],
                ignore_eos: vec![true; rows.len()],
            },
            draft_tokens: vec![],
        };
        let trace = vec![
            (
                vec![],
                request(&[
                    (10, 0, 0, vec![1, 2, 3]),
                    (20, 1, 0, vec![4, 5, 6]),
                    (30, 2, 0, vec![7, 8, 9]),
                    (40, 3, 0, vec![10, 11, 12]),
                ]),
            ),
            (
                vec![],
                request(&[
                    (10, 0, 3, vec![13]),
                    (20, 1, 3, vec![14]),
                    (30, 2, 3, vec![15]),
                    (40, 3, 3, vec![16]),
                ]),
            ),
            (
                vec![],
                request(&[
                    (30, 2, 4, vec![17]),
                    (10, 0, 4, vec![18]),
                    (40, 3, 4, vec![19]),
                ]),
            ),
            (vec![], request(&[(20, 1, 4, vec![20])])),
            (
                vec![],
                request(&[(40, 3, 5, vec![21]), (20, 1, 5, vec![22])]),
            ),
            (
                vec![10, 30],
                request(&[(50, 0, 0, vec![23, 24, 25]), (40, 3, 6, vec![26])]),
            ),
            (
                vec![],
                request(&[
                    (20, 1, 6, vec![27]),
                    (50, 0, 3, vec![28]),
                    (40, 3, 7, vec![29]),
                ]),
            ),
            (
                vec![20, 40, 50],
                request(&[
                    (0, 0, 0, vec![1, 2, 3]),
                    (1, 1, 0, vec![4, 5, 6]),
                    (2, 2, 0, vec![7, 8, 9]),
                    (3, 3, 0, vec![10, 11, 12]),
                ]),
            ),
            (
                vec![],
                request(&[
                    (3, 3, 3, vec![16]),
                    (2, 2, 3, vec![15]),
                    (1, 1, 3, vec![14]),
                    (0, 0, 3, vec![13]),
                ]),
            ),
            (
                vec![],
                request(&[
                    (0, 0, 4, vec![18]),
                    (3, 3, 4, vec![19]),
                    (2, 2, 4, vec![17]),
                ]),
            ),
            (vec![], request(&[(1, 1, 4, vec![20])])),
            (vec![], request(&[(1, 1, 5, vec![22]), (3, 3, 5, vec![21])])),
        ];
        let mut reference: Vec<(Vec<i32>, Vec<bf16>)> = Vec::new();
        for mode in ["eager", "cold", "prewarmed", "abc"] {
            runtime.retain_sequences([]);
            runtime.kv_pool.seq_kv_len.clear();
            if mode != "eager" {
                cuda.config.invalidate_all_graphs();
                runtime.prime_graphs().unwrap();
                if mode != "cold" {
                    runtime.prewarm_decode_graphs().unwrap();
                    for size in [1, 2, 4] {
                        assert!(runtime.scope.graph_ready(size));
                    }
                }
            }
            let mut max_relative_l2 = 0.0f64;
            for (step, (release, req)) in trace.iter().enumerate() {
                for &id in release {
                    runtime.release_sequence(id);
                }
                let q_lens: Vec<i32> = req.seqs.iter().map(|s| s.input_ids.len() as i32).collect();
                let (_, tiles, _) = BatchPlan::plan_ragged_tiles(&q_lens);
                let plan = BatchPlan {
                    kind: if q_lens.iter().all(|&n| n == 1) {
                        BatchKind::DecodeOnly
                    } else {
                        BatchKind::Ragged
                    },
                    num_tokens: q_lens.iter().map(|&n| n as usize).sum(),
                    batch: req.seqs.len(),
                    q_lens,
                    kv_lens: req.seqs.iter().map(|s| s.kv_len_after).collect(),
                    seq_positions: req.seqs.iter().map(|s| s.kv_write_start).collect(),
                    rope_positions: req
                        .seqs
                        .iter()
                        .flat_map(|s| s.positions.iter().copied())
                        .collect(),
                    max_blocks_per_seq: 64,
                    block_size: 1,
                    total_q_tiles: tiles.len() as i32,
                };
                let decode = matches!(plan.kind, BatchKind::DecodeOnly);
                if !decode {
                    assert!(matches!(runtime.decide(&plan), GraphDecision::Eager));
                }
                let tokens: Vec<i32> = if mode == "abc" && decode {
                    let n = req.seqs.len();
                    runtime
                        .issue_decode_abc(
                            req,
                            0,
                            &vec![0; n],
                            &vec![100; n],
                            &vec![true; n],
                            &[],
                            None,
                            false,
                        )
                        .unwrap();
                    runtime
                        .finalize_decode_abc(n)
                        .unwrap()
                        .active
                        .iter()
                        .map(|t| t.token_id)
                        .collect()
                } else {
                    runtime
                        .step(req)
                        .unwrap()
                        .tokens
                        .iter()
                        .map(|row| row[0].token_id)
                        .collect()
                };
                let hidden = Hidden {
                    stream: runtime.hidden.stream.narrow(0, 0, plan.num_tokens).unwrap(),
                    pending: None,
                };
                let ctx = StepCtx::new(&runtime.scope, &plan);
                let logits = runtime
                    .model
                    .finalize(&hidden, SampleRows::LastPerSeq, &ctx)
                    .unwrap()
                    .0
                    .to_host_vec()
                    .unwrap();
                if mode == "eager" {
                    reference.push((tokens, logits));
                } else {
                    let (expected_tokens, expected_logits) = &reference[step];
                    assert_eq!(&tokens, expected_tokens, "{mode} step {step}");
                    let error: f64 = logits
                        .iter()
                        .zip(expected_logits)
                        .map(|(a, b)| (a.to_f64() - b.to_f64()).powi(2))
                        .sum();
                    let norm: f64 = expected_logits.iter().map(|v| v.to_f64().powi(2)).sum();
                    let relative_l2 = (error / norm).sqrt();
                    assert!(
                        relative_l2.is_finite() && relative_l2 < 0.005,
                        "{mode} step {step}: logits relative L2 {relative_l2}"
                    );
                    max_relative_l2 = max_relative_l2.max(relative_l2);
                }
            }
            if mode != "eager" {
                for size in [1, 2, 4] {
                    assert!(runtime.scope.graph_ready(size));
                }
            }
            eprintln!(
                "Qwen3.5 {mode}: {} steps matched, max logits relative L2={max_relative_l2:.6}",
                trace.len()
            );
        }
        // Measure steady batch=1 decode without diagnostic downloads. Requests
        // use fixed tokens so both modes execute the same length progression.
        for graphed in [false, true] {
            runtime.retain_sequences([]);
            let runner = if graphed { None } else { runtime.graph.take() };
            runtime
                .step(&request(&[(99, 0, 0, vec![1, 2, 3])]))
                .unwrap();
            let mut times = Vec::new();
            for start in 3..35 {
                let req = request(&[(99, 0, start, vec![13])]);
                let begin = std::time::Instant::now();
                runtime.step(&req).unwrap();
                if start >= 7 {
                    times.push(begin.elapsed().as_secs_f64() * 1000.0);
                }
            }
            times.sort_by(f64::total_cmp);
            eprintln!(
                "Qwen3.5 decode batch=1 graph={graphed}: median {:.3} ms ({} steps)",
                times[times.len() / 2],
                times.len()
            );
            if let Some(runner) = runner {
                runtime.graph = Some(runner);
            }
        }
    }
    #[test]
    #[ignore = "requires QWEN35_MODEL_PATH, QWEN35_VISION_REFERENCE and CUDA"]
    fn qwen35_multimodal_precision_and_recompute() {
        use infer_protocol::multimodal::{IMAGE_TOKEN_ID, ImageInput, ImageSpan, MultimodalInput};
        use infer_worker::application::runtime::Runtime;
        use infer_worker::application::sampler_stack::GreedySampler;
        use infer_worker::domain::exec::ExecScope;
        use infer_worker::domain::plan::{SeqStep, StepRequest, StopCriteria};
        let path = std::env::var("QWEN35_MODEL_PATH").unwrap();
        let reference = std::env::var("QWEN35_VISION_REFERENCE").unwrap();
        let reference = Path::new(&reference);
        let ordinal = std::env::var("QWEN35_DEVICE")
            .unwrap_or_else(|_| "7".into())
            .parse()
            .unwrap();
        let config_bytes = std::fs::read(Path::new(&path).join("config.json")).unwrap();
        let config_json: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let cfg = build_load_config(&parse_hf_config(&config_bytes).unwrap(), 512).unwrap();
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(reference.join("metadata.json")).unwrap())
                .unwrap();
        let ids: Vec<i32> = serde_json::from_value(meta["input_ids"].clone()).unwrap();
        let expected: Vec<i32> = serde_json::from_value(meta["generated_ids"].clone()).unwrap();
        let image = ImageInput {
            grid_thw: serde_json::from_value(meta["grid_thw"].clone()).unwrap(),
            patches: std::fs::read(reference.join("patches.bf16")).unwrap(),
        };
        let input = std::sync::Arc::new(MultimodalInput {
            spans: vec![ImageSpan {
                image_index: 0,
                token_start: ids.iter().position(|&id| id == IMAGE_TOKEN_ID).unwrap() as u32,
                token_len: image.num_tokens() as u32,
            }],
            images: vec![image],
            original_prompt_len: ids.len() as u32,
        });
        input.validate_tokens(&ids).unwrap();
        let reader = SafetensorsReader::open(&path).unwrap();
        let cuda = Cuda::new(ordinal).unwrap();
        let loader = WeightLoader::new(&reader);
        let mut model = qwen3_5::build::<bf16, Cuda>(&loader, &cfg, &cuda).unwrap();
        model.load_vision(&loader, &config_json, &cuda).unwrap();
        let dump_compare = |name: &str, tensor: &Tensor<bf16, Cuda>| {
            let actual: Vec<f32> = tensor
                .to_host_vec()
                .unwrap()
                .iter()
                .map(|v| v.to_f32())
                .collect();
            std::fs::write(
                reference.join(format!("rust-{name}.f32")),
                actual
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let bytes = std::fs::read(reference.join(format!("{name}.f32"))).unwrap();
            let reference: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(actual.len(), reference.len());
            assert!(actual.iter().all(|v| v.is_finite()));
            let relative = (actual
                .iter()
                .zip(&reference)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>()
                / reference
                    .iter()
                    .map(|b| (*b as f64).powi(2))
                    .sum::<f64>()
                    .max(1e-20))
            .sqrt();
            eprintln!("{name}: relative L2={relative:.6}");
            if name == "vision_merger" {
                assert!(relative < 0.08);
            }
            relative
        };
        model
            .vision
            .as_ref()
            .unwrap()
            .forward_with_trace(&input.images[0], &cuda.scope(), |name, t| {
                dump_compare(&format!("vision_{name}"), t);
                Ok(())
            })
            .unwrap();
        let mut runtime = Runtime::new(
            model,
            cuda.scope(),
            Box::new(GreedySampler),
            2049,
            1,
            512,
            512,
            128,
            4,
            vec![1, 2, 4],
        )
        .unwrap();
        let request = |tokens: Vec<i32>, start: usize| StepRequest {
            seqs: vec![SeqStep {
                sequence_id: 42,
                positions: (start as i32..(start + tokens.len()) as i32).collect(),
                kv_write_start: start as i32,
                kv_len_after: (start + tokens.len()) as i32,
                block_table: (0..512).collect(),
                input_ids: tokens,
            }],
            sampling: vec![Default::default()],
            stop: StopCriteria {
                eos_ids: vec![],
                generated_counts: vec![0],
                max_tokens: vec![100],
                ignore_eos: vec![true],
            },
            draft_tokens: vec![],
        };
        let measure = |runtime: &mut Runtime<bf16, Cuda, qwen3_5::Qwen3_5Model<bf16, Cuda>>,
                       req: &StepRequest,
                       name: &str| {
            let seq = &req.seqs[0];
            let n = seq.input_ids.len();
            let plan = BatchPlan {
                kind: if n == 1 {
                    BatchKind::DecodeOnly
                } else {
                    BatchKind::Ragged
                },
                num_tokens: n,
                batch: 1,
                q_lens: vec![n as i32],
                kv_lens: vec![seq.kv_len_after],
                seq_positions: vec![seq.kv_write_start],
                rope_positions: seq.positions.clone(),
                max_blocks_per_seq: 512,
                block_size: 1,
                total_q_tiles: 1,
            };
            let hidden = Hidden {
                stream: runtime.hidden.stream.narrow(0, 0, n).unwrap(),
                pending: None,
            };
            let logits = runtime
                .model
                .finalize(
                    &hidden,
                    SampleRows::LastPerSeq,
                    &StepCtx::new(&runtime.scope, &plan),
                )
                .unwrap();
            // HF BF16 eager vs HF BF16 SDPA on this fixture differs by 3.59%
            // relative L2. Keep a numerical bound plus exact greedy-token checks.
            assert!(dump_compare(name, &logits.0) < 0.06);
        };
        let mut all_outputs = Vec::new();
        for chunk in [128, 31, 26] {
            runtime.release_sequence(42);
            runtime.register_multimodal(42, input.clone()).unwrap();
            let mut last = 0;
            for (i, part) in ids.chunks(chunk).enumerate() {
                let req = request(part.to_vec(), i * chunk);
                last = runtime.step(&req).unwrap().tokens[0][0].token_id;
                if (i + 1) * chunk >= ids.len() {
                    measure(&mut runtime, &req, "step0_logits");
                }
            }
            let mut generated = vec![last];
            for i in 1..expected.len() {
                let req = request(vec![last], ids.len() + i - 1);
                last = runtime.step(&req).unwrap().tokens[0][0].token_id;
                measure(&mut runtime, &req, &format!("step{i}_logits"));
                generated.push(last);
            }
            eprintln!("multimodal chunk={chunk}: {generated:?}, expected={expected:?}");
            all_outputs.push(generated);
        }
        // Recompute an evicted request including four already generated tokens.
        runtime.release_sequence(42);
        runtime.register_multimodal(42, input.clone()).unwrap();
        let mut replay = ids.clone();
        replay.extend_from_slice(&expected[..4]);
        let mut token = 0;
        for (i, part) in replay.chunks(31).enumerate() {
            token = runtime
                .step(&request(part.to_vec(), i * 31))
                .unwrap()
                .tokens[0][0]
                .token_id;
        }
        assert_eq!(token, expected[4]);
        let (runs, hits, bytes) = runtime.visual_cache_stats();
        assert_eq!(runs, 1);
        assert!(hits > 0 && bytes > 0);
        eprintln!("vision cache: runs={runs}, hits={hits}, bytes={bytes}; recompute matched");
        for output in all_outputs {
            assert_eq!(output, expected);
        }
        runtime.release_sequence(42);
        // Transition from image eager execution back to text ABC without stale overrides.
        let req = request(vec![1], 0);
        runtime
            .issue_decode_abc(&req, 0, &[0], &[100], &[true], &[], None, false)
            .unwrap();
        runtime.finalize_decode_abc(1).unwrap();

        // Reuse text-captured graphs for image decode. Distinct image counts
        // give distinct rope_delta values, even when physical positions match.
        let mut twice = (*input).clone();
        let mut second_span = twice.spans[0].clone();
        second_span.image_index = 1;
        second_span.token_start += ids.len() as u32;
        twice.images.push(twice.images[0].clone());
        twice.spans.push(second_span);
        twice.original_prompt_len *= 2;
        let twice = std::sync::Arc::new(twice);
        let mut double_ids = ids.clone();
        double_ids.extend_from_slice(&ids);
        twice.validate_tokens(&double_ids).unwrap();
        assert_ne!(
            input.positions().unwrap().rope_delta,
            twice.positions().unwrap().rope_delta
        );
        let rows = |rows: &[(u64, usize, usize, Vec<i32>)]| StepRequest {
            seqs: rows
                .iter()
                .map(|(id, slot, start, tokens)| SeqStep {
                    sequence_id: *id,
                    input_ids: tokens.clone(),
                    positions: (*start as i32..(*start + tokens.len()) as i32).collect(),
                    kv_write_start: *start as i32,
                    kv_len_after: (*start + tokens.len()) as i32,
                    block_table: (*slot * 512..(*slot + 1) * 512).map(|b| b as u32).collect(),
                })
                .collect(),
            sampling: vec![Default::default(); rows.len()],
            stop: StopCriteria {
                eos_ids: vec![],
                generated_counts: vec![0; rows.len()],
                max_tokens: vec![100; rows.len()],
                ignore_eos: vec![true; rows.len()],
            },
            draft_tokens: vec![],
        };
        // release ids, request; start=0 re-registers durable image inputs.
        let mut trace = Vec::new();
        for (id, slot, prompt) in [(42, 0, &ids), (43, 1, &double_ids)] {
            // Include q_len=1 inside the image span; its graph must stay eager.
            let first_image = input.spans[0].token_start as usize;
            let mut start = 0;
            for end in [first_image, first_image + 1, prompt.len()] {
                for part in prompt[start..end].chunks(31) {
                    trace.push((vec![], rows(&[(id, slot, start, part.to_vec())])));
                    start += part.len();
                }
            }
        }
        trace.push((vec![], rows(&[(44, 2, 0, vec![1, 2, 3])])));
        let n = ids.len();
        trace.push((
            vec![],
            rows(&[
                (42, 0, n, vec![760]),
                (43, 1, n * 2, vec![760]),
                (44, 2, 3, vec![13]),
            ]),
        ));
        trace.push((
            vec![],
            rows(&[
                (44, 2, 4, vec![13]),
                (43, 1, n * 2 + 1, vec![760]),
                (42, 0, n + 1, vec![760]),
            ]),
        ));
        let mut finish_middle = rows(&[
            (44, 2, 5, vec![13]),
            (43, 1, n * 2 + 2, vec![760]),
            (42, 0, n + 2, vec![760]),
        ]);
        finish_middle.stop.max_tokens[1] = 1;
        trace.push((vec![], finish_middle));
        trace.push((
            vec![43],
            rows(&[(44, 2, 6, vec![13]), (42, 0, n + 3, vec![760])]),
        ));
        trace.push((
            vec![],
            rows(&[(42, 0, n + 4, vec![760]), (44, 2, 7, vec![13])]),
        ));
        // Cancel the text row, reuse its slots for a new image request.
        for (i, part) in ids.chunks(31).enumerate() {
            trace.push((
                if i == 0 { vec![44] } else { vec![] },
                rows(&[(45, 2, i * 31, part.to_vec())]),
            ));
        }
        trace.push((
            vec![],
            rows(&[(45, 2, n, vec![760]), (42, 0, n + 5, vec![760])]),
        ));
        // Preempt 42 and reconstruct both KV and GDN history before replay.
        let mut recompute = ids.clone();
        recompute.extend_from_slice(&[760; 6]);
        for (i, part) in recompute.chunks(31).enumerate() {
            trace.push((
                if i == 0 { vec![42] } else { vec![] },
                rows(&[(42, 0, i * 31, part.to_vec())]),
            ));
        }
        trace.push((
            vec![],
            rows(&[(42, 0, n + 6, vec![760]), (45, 2, n + 1, vec![760])]),
        ));
        trace.push((vec![45], rows(&[(42, 0, n + 7, vec![760])])));
        trace.push((vec![42], rows(&[(42, 0, 0, vec![1, 2, 3])])));
        trace.push((vec![], rows(&[(42, 0, 3, vec![13])])));

        let mut baseline = Vec::new();
        for mode in ["eager", "cold", "prewarmed", "abc"] {
            runtime.retain_sequences([]);
            runtime.graph = None;
            cuda.config.invalidate_all_graphs();
            if mode != "eager" {
                runtime.prime_graphs().unwrap();
                if mode != "cold" {
                    runtime.prewarm_decode_graphs().unwrap();
                }
            }
            let mut device_rows = Vec::new();
            let mut device_tokens = Vec::new();
            let mut max_l2 = 0.0f64;
            let mut reused = 0;
            for (step, (release, req)) in trace.iter().enumerate() {
                for &id in release {
                    runtime.release_sequence(id);
                }
                for seq in &req.seqs {
                    if seq.kv_write_start == 0
                        && (seq.sequence_id == 43
                            || seq.sequence_id == 45
                            || (seq.sequence_id == 42 && seq.input_ids != [1, 2, 3]))
                    {
                        runtime
                            .register_multimodal(
                                seq.sequence_id,
                                if seq.sequence_id == 43 {
                                    twice.clone()
                                } else {
                                    input.clone()
                                },
                            )
                            .unwrap();
                    }
                }
                let decode = req.seqs.iter().all(|s| {
                    s.input_ids.len() == 1
                        && (!runtime.has_multimodal_sequence(s.sequence_id)
                            || s.kv_write_start as usize
                                >= if s.sequence_id == 43 {
                                    double_ids.len()
                                } else {
                                    ids.len()
                                })
                });
                let q_lens: Vec<i32> = req.seqs.iter().map(|s| s.input_ids.len() as i32).collect();
                let plan = BatchPlan {
                    kind: if q_lens.iter().all(|&n| n == 1) {
                        BatchKind::DecodeOnly
                    } else {
                        BatchKind::Ragged
                    },
                    num_tokens: q_lens.iter().map(|&n| n as usize).sum(),
                    batch: req.seqs.len(),
                    total_q_tiles: BatchPlan::plan_ragged_tiles(&q_lens).1.len() as i32,
                    q_lens,
                    kv_lens: req.seqs.iter().map(|s| s.kv_len_after).collect(),
                    seq_positions: req.seqs.iter().map(|s| s.kv_write_start).collect(),
                    rope_positions: req
                        .seqs
                        .iter()
                        .flat_map(|s| s.positions.iter().copied())
                        .collect(),
                    max_blocks_per_seq: 512,
                    block_size: 1,
                };
                let tokens = if mode == "abc" && decode {
                    let order: Vec<_> = req.seqs.iter().map(|s| s.sequence_id).collect();
                    let reuse = device_rows == order;
                    let prefix = if reuse {
                        // Reuse A only where it already contains this trace's
                        // fixed input token; upload the divergent suffix.
                        req.seqs
                            .iter()
                            .zip(&device_tokens)
                            .take_while(|(s, t)| s.input_ids[0] == **t)
                            .count()
                    } else {
                        0
                    };
                    let survivors: Vec<_> = req
                        .seqs
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| req.stop.max_tokens[*i] > 1)
                        .collect();
                    let next_slots: Vec<u32> = survivors
                        .iter()
                        .map(|(_, s)| s.block_table[s.kv_len_after as usize])
                        .collect();
                    runtime
                        .issue_decode_abc(
                            req,
                            prefix,
                            &req.stop.generated_counts,
                            &req.stop.max_tokens,
                            &req.stop.ignore_eos,
                            &[],
                            Some(&next_slots),
                            reuse,
                        )
                        .unwrap();
                    let out = runtime.finalize_decode_abc(req.seqs.len()).unwrap();
                    device_rows = out
                        .active
                        .iter()
                        .map(|r| req.seqs[r.src_row].sequence_id)
                        .collect();
                    device_tokens = out.active.iter().map(|r| r.token_id).collect();
                    reused += usize::from(reuse);
                    let mut tokens = vec![0; req.seqs.len()];
                    for row in out.active.iter().chain(&out.finished) {
                        tokens[row.src_row] = row.token_id;
                    }
                    tokens
                } else {
                    device_rows.clear();
                    runtime
                        .step(req)
                        .unwrap()
                        .tokens
                        .iter()
                        .map(|row| row[0].token_id)
                        .collect::<Vec<_>>()
                };
                let hidden = Hidden {
                    stream: runtime.hidden.stream.narrow(0, 0, plan.num_tokens).unwrap(),
                    pending: None,
                };
                let logits = runtime
                    .model
                    .finalize(
                        &hidden,
                        SampleRows::LastPerSeq,
                        &StepCtx::new(&runtime.scope, &plan),
                    )
                    .unwrap()
                    .0
                    .to_host_vec()
                    .unwrap();
                if mode == "eager" {
                    baseline.push((tokens, logits));
                } else {
                    assert_eq!(tokens, baseline[step].0, "multimodal {mode} step={step}");
                    let expected = &baseline[step].1;
                    let l2 = (logits
                        .iter()
                        .zip(expected)
                        .map(|(a, b)| (a.to_f64() - b.to_f64()).powi(2))
                        .sum::<f64>()
                        / expected.iter().map(|v| v.to_f64().powi(2)).sum::<f64>())
                    .sqrt();
                    assert!(
                        l2.is_finite() && l2 < 0.005,
                        "multimodal {mode} step={step}: L2={l2}"
                    );
                    max_l2 = max_l2.max(l2);
                }
            }
            if mode == "abc" {
                assert!(reused >= 2, "device control reuse was not exercised");
            }
            if mode == "prewarmed" || mode == "abc" {
                for size in [1, 2, 4] {
                    assert!(runtime.scope.graph_ready(size));
                }
            }
            eprintln!(
                "multimodal {mode}: {} steps matched; max logits L2={max_l2:.6}, device control reuse={reused}",
                trace.len()
            );
        }
        // Diagnostic latency comparison after both paths are warm. No timing
        // assertion: this opt-in test can share the GPU with other workloads.
        let mut timings = [Vec::new(), Vec::new()];
        for round in 0..3 {
            for graphed in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                runtime.retain_sequences([]);
                let graph = if graphed { None } else { runtime.graph.take() };
                runtime.register_multimodal(42, input.clone()).unwrap();
                runtime.step(&request(ids.clone(), 0)).unwrap();
                for i in 0..20 {
                    let req = request(vec![760], ids.len() + i);
                    let start = std::time::Instant::now();
                    runtime.step(&req).unwrap();
                    if i >= 4 {
                        timings[usize::from(graphed)].push(start.elapsed().as_secs_f64() * 1000.0);
                    }
                }
                if let Some(graph) = graph {
                    runtime.graph = Some(graph);
                }
            }
        }
        for (graphed, times) in timings.iter_mut().enumerate() {
            times.sort_by(f64::total_cmp);
            eprintln!(
                "multimodal decode batch=1 graph={}: median {:.3} ms ({} steps)",
                graphed != 0,
                times[times.len() / 2],
                times.len()
            );
        }
    }
}

#[cfg(test)]
#[path = "checkpoint_tests/qwen3_moe.rs"]
mod qwen3_moe_checkpoint_tests;
