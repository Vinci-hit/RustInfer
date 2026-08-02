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
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use half::bf16;
use serde::Deserialize;

use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
use infer_protocol::worker_to_scheduler_control::WORKER_CONTROL_PROTOCOL_VERSION;

use infer_worker::application::runtime::RuntimePeerWatchdog;
use infer_worker::application::serve_loop::{
    Bootstrap, RuntimeFollowerFactory, RuntimeFollowerInit, run_with_model,
};
use infer_worker::domain::dtype::quant::QuantScheme;
use infer_worker::domain::model::DecoderModel;
use infer_worker::domain::ports::{OpError, OpResult};
use infer_worker::infrastructure::cuda::{Cuda, CudaMemoryPlan, NcclCommunicator, device_utils};
use infer_worker::infrastructure::io::SafetensorsReader;
use infer_worker::infrastructure::transport::control_pump::ControlPump;
use infer_worker::infrastructure::transport::data_pump::DataPump;
use infer_worker::models::loader::{LinearAttnConfig, LoadConfig, RopeScaling, WeightLoader};
use infer_worker::models::{llama3, qwen3, qwen3_moe};

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
    /// Per-layer mixer selector, e.g. `["linear_attention", ..., "full_attention"]`.
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

/// Build the per-layer full-vs-linear mixer selector for the hybrid stack.
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

struct TpFollowerResource {
    rank: usize,
    cuda: Cuda,
    communicator: Arc<NcclCommunicator>,
}

/// Build one factory per non-zero TP rank. The factory is invoked inside that
/// rank's long-lived Runtime thread, so neither the model nor its `Rc`-backed
/// forward scratch ever crosses a thread boundary.
fn make_follower_factories<M, F>(
    devices: Vec<TpFollowerResource>,
    model_path: String,
    load_cfg: LoadConfig,
    tp_size: usize,
    build: F,
) -> Vec<RuntimeFollowerFactory<M>>
where
    M: DecoderModel<bf16, Cuda> + 'static,
    F: for<'a> Fn(&WeightLoader<'a>, &LoadConfig, &Cuda) -> OpResult<M> + Copy + Send + 'static,
{
    devices
        .into_iter()
        .map(|resource| {
            let TpFollowerResource {
                rank,
                cuda,
                communicator,
            } = resource;
            let model_path = model_path.clone();
            let load_cfg = load_cfg.clone();
            Box::new(move |init: RuntimeFollowerInit| {
                if init.rank != rank || init.size != tp_size {
                    return Err(OpError::Shape(format!(
                        "TP follower factory rank {rank}/{tp_size} received init {}/{}",
                        init.rank, init.size
                    )));
                }

                // CUDA's current device is thread-local. The `Cuda` handles
                // were created by the controller thread, so establish this
                // rank's device before any allocator/model call on the new
                // Runtime thread and keep it current for that thread's life.
                device_utils::set_current_device(cuda.device_id).map_err(|error| {
                    OpError::Kernel(format!(
                        "set TP rank {rank} CUDA device {}: {error}",
                        cuda.device_id
                    ))
                })?;

                let scope = init.build_scope(cuda.clone(), communicator)?;
                let reader = SafetensorsReader::open(Path::new(&model_path)).map_err(|error| {
                    OpError::Kernel(format!("open TP rank {rank} weights: {error}"))
                })?;
                let loader = WeightLoader::with_tensor_parallel(&reader, rank, tp_size)?;
                let started = Instant::now();
                let model = build(&loader, &load_cfg, &cuda)?;
                tracing::info!(
                    rank,
                    size = tp_size,
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
            "the scheduler-facing worker requires tp_rank=0 and tp_size>0, got {}/{}",
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
    // A single worker process owns one CUDA Runtime rank per device. `device`
    // names rank 0; higher TP ranks use consecutive device ids. Construct all
    // devices up front so an invalid/missing GPU fails before the whole
    // single-process NCCL group is created.
    let mut rank_devices = Vec::with_capacity(load.tp_size);
    for rank in 0..load.tp_size {
        let offset = i32::try_from(rank)
            .map_err(|_| format!("TP rank {rank} does not fit a CUDA device id"))?;
        let rank_device_id = device_id
            .checked_add(offset)
            .ok_or_else(|| format!("CUDA device id overflow: base={device_id} TP rank={rank}"))?;
        let cuda = Cuda::with_memory_plan(rank_device_id, memory_plan).map_err(|e| {
            format!(
                "Cuda::with_memory_plan for TP rank {rank}/{} on cuda:{rank_device_id}: {:?}",
                load.tp_size, e
            )
        })?;
        eprintln!(
            "[bootstrap] TP rank {}/{} -> cuda:{}",
            rank, load.tp_size, rank_device_id
        );
        rank_devices.push((rank, cuda));
    }
    // All TP ranks live in this process, so create the communicator group in
    // one NCCL call. This cannot strand rank 0 waiting for a follower thread
    // that failed before entering a per-rank rendezvous.
    let all_devices: Vec<Cuda> = rank_devices.iter().map(|(_, cuda)| cuda.clone()).collect();
    let mut communicators = if load.tp_size > 1 {
        let watchdog = RuntimePeerWatchdog::fail_stop()
            .map_err(|error| format!("start NCCL initialization watchdog: {error}"))?;
        let timeout = Duration::from_secs(cfg.tp_startup_timeout_secs);
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or("NCCL initialization deadline overflowed Instant")?;
        watchdog
            .arm(0, "NCCL group initialization", deadline)
            .map_err(|error| format!("arm NCCL initialization watchdog: {error}"))?;
        let initialized = NcclCommunicator::init_all(&all_devices);
        watchdog
            .disarm(0)
            .map_err(|error| format!("disarm NCCL initialization watchdog: {error}"))?;
        drop(watchdog);
        initialized.map_err(|error| format!("initialize TP{} NCCL group: {error}", load.tp_size))?
    } else {
        Vec::new()
    };
    let (_, cuda) = rank_devices.remove(0);
    let leader_communicator = if communicators.is_empty() {
        None
    } else {
        Some(communicators.remove(0))
    };
    let follower_devices: Vec<_> = rank_devices
        .into_iter()
        .zip(communicators)
        .map(|((rank, cuda), communicator)| TpFollowerResource {
            rank,
            cuda,
            communicator,
        })
        .collect();
    if follower_devices.len() != load.tp_size.saturating_sub(1) {
        return Err(format!(
            "TP{} communicator/device mismatch: got {} followers",
            load.tp_size,
            follower_devices.len()
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
                follower_devices,
                load.model_path.clone(),
                load_cfg.clone(),
                load.tp_size,
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
                follower_devices,
                load.model_path.clone(),
                load_cfg.clone(),
                load.tp_size,
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
        "qwen3_moe" => {
            let model = qwen3_moe::build::<bf16, Cuda>(&loader, &load_cfg, &cuda)
                .map_err(|e| format!("qwen3_moe::build: {:?}", e))?;
            let followers = make_follower_factories(
                follower_devices,
                load.model_path.clone(),
                load_cfg.clone(),
                load.tp_size,
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
    /// sibling `vision_config` we must ignore in v1. Trimmed from the real
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
