//! RustInfer Scheduler binary entry point.

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use infer_scheduler::application::SchedulerEngine;
use infer_scheduler::config::{SchedulerConfig, SchedulerMode};
use infer_scheduler::domain::BlockSize;
use infer_scheduler::domain::policy::{ContinuousBatchingPolicy, DiffusionPolicy, SchedulingPolicy};
use infer_protocol::scheduler_to_worker_control::LoadModel;
use infer_scheduler::infrastructure::transport::control_plane::{ControlPlane, ControlPlaneConfig};
use infer_scheduler::infrastructure::transport::zmq_transport::{ZmqFrontendTransport, ZmqWorkerTransport};

#[derive(Parser, Debug)]
#[command(name = "rustinfer-scheduler")]
#[command(about = "RustInfer Scheduler — production-grade continuous batching scheduler")]
struct Args {
    /// ZMQ frontend endpoint (ROUTER socket, connects to HTTP server).
    #[arg(long, default_value = "ipc:///tmp/rustinfer.ipc")]
    frontend_endpoint: String,

    /// ZMQ Worker PUSH endpoint (sends batch commands).
    #[arg(long, default_value = "ipc:///tmp/rustinfer-worker-in.ipc")]
    worker_push_endpoint: String,

    /// ZMQ Worker PULL endpoint (receives step outputs).
    #[arg(long, default_value = "ipc:///tmp/rustinfer-worker-out.ipc")]
    worker_pull_endpoint: String,

    /// ZMQ Worker control endpoint (lifecycle handshake and readiness).
    #[arg(long, default_value = "ipc:///tmp/rustinfer-worker-control.ipc")]
    worker_control_endpoint: String,

    /// Optional model path to assign to the Worker via LoadModel.
    #[arg(long)]
    model: Option<String>,

    /// Model type assigned via LoadModel when --model is present.
    #[arg(long, default_value = "llama3")]
    model_type: String,

    /// Device assigned via LoadModel when --model is present.
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Static memory fraction reserved for model runtime planning.
    #[arg(long, default_value = "1.0")]
    mem_fraction_static: f32,

    /// Maximum batch tokens per iteration.
    #[arg(long, default_value = "1024")]
    max_batch_tokens: usize,

    /// Maximum concurrent sequences in flight.
    #[arg(long, default_value = "32")]
    max_batch_seqs: usize,

    /// Maximum model sequence length (prompt + generation).
    #[arg(long, default_value = "4096")]
    max_model_len: usize,

    /// Scheduler mode: "llm" (default) or "diffusion".
    #[arg(long, default_value = "llm")]
    mode: String,

    /// Paged KV block size (tokens per block). Default 16.
    #[arg(long, default_value_t = 16)]
    paged_block_size: usize,

    /// Chunked prefill: max tokens per prefill chunk.
    /// None (default) = no chunking (full prompt in one shot).
    /// Set to e.g. 512 or 2048 to split long prompts across iterations.
    #[arg(long)]
    chunked_prefill_size: Option<usize>,

    /// Enable RadixTree prefix caching. Only meaningful in paged KV mode.
    #[arg(long, default_value_t = false)]
    enable_prefix_caching: bool,

    /// Log level.
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn parse_block_size(s: usize) -> BlockSize {
    let raw = u32::try_from(s).unwrap_or(16);
    BlockSize::new(if raw == 0 { 16 } else { raw })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.clone().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let paged_block_size = parse_block_size(args.paged_block_size);
    let scheduler_mode = match args.mode.as_str() {
        "diffusion" => SchedulerMode::Diffusion,
        _ => SchedulerMode::Llm,
    };

    tracing::info!("RustInfer Scheduler v0.2.0 starting...");
    tracing::info!("  Mode: {:?}", scheduler_mode);
    tracing::info!("  Frontend: {}", args.frontend_endpoint);
    tracing::info!("  Worker PUSH: {}", args.worker_push_endpoint);
    tracing::info!("  Worker PULL: {}", args.worker_pull_endpoint);
    tracing::info!("  Worker Control: {}", args.worker_control_endpoint);
    tracing::info!("  Assigned model: {}", args.model.as_deref().unwrap_or("<worker-cli>"));
    tracing::info!("  Assigned model type: {}", args.model_type);
    tracing::info!("  max_batch_seqs: {}", args.max_batch_seqs);
    tracing::info!("  max_batch_tokens: {}", args.max_batch_tokens);
    tracing::info!("  max_model_len: {}", args.max_model_len);
    tracing::info!("  paged_block_size: {}", paged_block_size);
    tracing::info!("  chunked_prefill_size: {:?}", args.chunked_prefill_size);
    tracing::info!("  enable_prefix_caching: {}", args.enable_prefix_caching);

    // Build config.
    let mut config = SchedulerConfig {
        mode: scheduler_mode,
        max_num_seqs: args.max_batch_seqs,
        max_batch_tokens: args.max_batch_tokens,
        max_model_len: args.max_model_len,
        paged_block_size,
        chunked_prefill_size: args.chunked_prefill_size,
        enable_prefix_caching: args.enable_prefix_caching,
        frontend_endpoint: args.frontend_endpoint.clone(),
        worker_push_endpoint: args.worker_push_endpoint.clone(),
        worker_pull_endpoint: args.worker_pull_endpoint.clone(),
        ..Default::default()
    };

    // Create transports.
    let frontend = ZmqFrontendTransport::new(&args.frontend_endpoint)?;
    let worker = ZmqWorkerTransport::new(&args.worker_push_endpoint, &args.worker_pull_endpoint)?;

    let load_model = args.model.clone().map(|model_path| LoadModel {
        model_instance_id: "default".to_string(),
        model_path,
        model_type: args.model_type.clone(),
        device: args.device.clone(),
        max_batch_tokens: args.max_batch_tokens,
        max_batch_seqs: args.max_batch_seqs,
        max_model_len: args.max_model_len,
        mem_fraction_static: args.mem_fraction_static,
        tp_rank: 0,
        tp_size: 1,
        pp_rank: 0,
        pp_size: 1,
        kv_cache_mode: Some(format!("paged:{}", paged_block_size.raw())),
        kv_cache_memory_fraction: Some(args.mem_fraction_static),
    });

    // Bind the control-plane ROUTER, optionally assign a model, then
    // wait for `WorkerReady`. The same socket continues to serve
    // runtime control once the handshake completes.
    let cp_cfg = ControlPlaneConfig::default();
    let (mut control_plane, worker_group) =
        ControlPlane::bootstrap(&args.worker_control_endpoint, load_model, cp_cfg).await?;
    tracing::info!(
        "WorkerGroup ready: group_id={} model_instance_id={} ranks={} effective_max_batch_tokens={} effective_max_batch_seqs={}",
        worker_group.group_id,
        worker_group.model_instance_id,
        worker_group.rank_count(),
        worker_group.effective_capacity.max_batch_tokens,
        worker_group.effective_capacity.max_batch_seqs,
    );
    let workers = control_plane.workers();
    let default_worker = workers
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("control plane reports no workers after bootstrap"))?;
    let control_cmd = control_plane.cmd_tx();
    let control_events = control_plane.take_event_rx();
    // Hand the ControlPlane to the engine via a leak — the engine's lifetime
    // matches the process lifetime, and ControlPlane's Drop performs graceful
    // shutdown. Boxing + leaking keeps it alive for the duration of run().
    let _control_plane_handle: &'static ControlPlane = Box::leak(Box::new(control_plane));

    if let Some(max_total_kv_tokens) = worker_group.effective_capacity.max_total_kv_tokens {
        let block_size = paged_block_size.as_usize();
        config.num_gpu_blocks = max_total_kv_tokens / block_size;
        tracing::info!(
            "Paged KV capacity from worker profile: num_gpu_blocks={} block_size={} max_total_kv_tokens={}",
            config.num_gpu_blocks,
            block_size,
            max_total_kv_tokens,
        );
    }

    // The worker is the sole owner of physical block allocation; the
    // scheduler only tracks slot accounting via `KvBudget` + `RadixTree`,
    // wired up inside `SchedulerEngine::new`.

    let policy: Box<dyn SchedulingPolicy> = match scheduler_mode {
        SchedulerMode::Diffusion => Box::new(DiffusionPolicy::new(args.max_batch_seqs)),
        SchedulerMode::Llm => Box::new(ContinuousBatchingPolicy::new(config.chunked_prefill_size)),
    };

    // Build and run engine.
    let engine = SchedulerEngine::new(
        config,
        policy,
        worker_group,
        frontend,
        worker,
        control_cmd,
        control_events,
        default_worker,
    );

    tracing::info!("Scheduler engine running...");
    engine.run().await?;

    Ok(())
}
