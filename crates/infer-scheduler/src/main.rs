//! RustInfer Scheduler binary entry point.

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use infer_scheduler::cache::kv_manager::KvManager;
use infer_scheduler::cache::slot_kv_manager::SlotKvManager;
use infer_scheduler::cache::paged_kv_manager::PagedKvManager;
use infer_scheduler::cache::noop_kv_manager::NoopKvManager;
use infer_scheduler::config::{KvCacheMode, SchedulerConfig, SchedulerMode};
use infer_scheduler::core::SchedulerEngine;
use infer_scheduler::policy::{ContinuousBatchingPolicy, DiffusionPolicy, SchedulingPolicy};
use infer_scheduler::transport::zmq_transport::{ZmqFrontendTransport, ZmqWorkerTransport};
use infer_scheduler::WorkerGroup;
use infer_worker::worker::control_protocol::LoadModel;

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

    /// Maximum concurrent sequences (= slot count in slot mode).
    #[arg(long, default_value = "32")]
    max_batch_seqs: usize,

    /// Maximum model sequence length (prompt + generation).
    #[arg(long, default_value = "4096")]
    max_model_len: usize,

    /// Scheduler mode: "llm" (default) or "diffusion".
    #[arg(long, default_value = "llm")]
    mode: String,

    /// KV cache mode: "slot" (default) or "paged:BLOCK_SIZE". Only for LLM mode.
    #[arg(long, default_value = "slot")]
    kv_cache_mode: String,

    /// Chunked prefill: max tokens per prefill chunk.
    /// None (default) = no chunking (full prompt in one shot).
    /// Set to e.g. 512 or 2048 to split long prompts across iterations.
    #[arg(long)]
    chunked_prefill_size: Option<usize>,

    /// Log level.
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn parse_kv_cache_mode(s: &str) -> KvCacheMode {
    if s == "slot" {
        KvCacheMode::Slot
    } else if let Some(rest) = s.strip_prefix("paged:") {
        let block_size: usize = rest.parse().unwrap_or(16);
        KvCacheMode::Paged { block_size }
    } else {
        tracing::warn!("Unknown kv-cache-mode '{}', defaulting to slot", s);
        KvCacheMode::Slot
    }
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

    let kv_mode = parse_kv_cache_mode(&args.kv_cache_mode);
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
    tracing::info!("  kv_cache_mode: {:?}", kv_mode);
    tracing::info!("  chunked_prefill_size: {:?}", args.chunked_prefill_size);

    // Build config.
    let config = SchedulerConfig {
        mode: scheduler_mode,
        max_num_seqs: args.max_batch_seqs,
        max_batch_tokens: args.max_batch_tokens,
        max_model_len: args.max_model_len,
        kv_cache_mode: kv_mode,
        chunked_prefill_size: args.chunked_prefill_size,
        frontend_endpoint: args.frontend_endpoint.clone(),
        worker_push_endpoint: args.worker_push_endpoint.clone(),
        worker_pull_endpoint: args.worker_pull_endpoint.clone(),
        ..Default::default()
    };

    // Create KV manager and policy based on mode.
    let kv_manager: Box<dyn KvManager> = match scheduler_mode {
        SchedulerMode::Diffusion => Box::new(NoopKvManager::new()),
        SchedulerMode::Llm => match kv_mode {
            KvCacheMode::Slot => {
                Box::new(SlotKvManager::new(args.max_batch_seqs, args.max_model_len))
            }
            KvCacheMode::Paged { block_size } => {
                Box::new(PagedKvManager::new(config.num_gpu_blocks, block_size))
            }
        },
    };

    let policy: Box<dyn SchedulingPolicy> = match scheduler_mode {
        SchedulerMode::Diffusion => Box::new(DiffusionPolicy::new(args.max_batch_seqs)),
        SchedulerMode::Llm => Box::new(ContinuousBatchingPolicy::new(config.chunked_prefill_size)),
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
    });

    // Phase 2 control-plane gate: optionally assign model, then wait for WorkerReady.
    let ready = infer_scheduler::transport::worker_control::wait_for_worker_ready(
        &args.worker_control_endpoint,
        load_model,
    )?;
    tracing::info!(
        "Worker ready gate opened: id={} model_instance_id={} model_type={} device={}",
        ready.worker_id,
        ready.model_instance_id,
        ready.model_type,
        ready.device,
    );
    let worker_group = WorkerGroup::from_single_ready(ready);
    tracing::info!(
        "WorkerGroup ready: group_id={} model_instance_id={} ranks={} effective_max_batch_tokens={} effective_max_batch_seqs={}",
        worker_group.group_id,
        worker_group.model_instance_id,
        worker_group.rank_count(),
        worker_group.effective_capacity.max_batch_tokens,
        worker_group.effective_capacity.max_batch_seqs,
    );

    // Build and run engine.
    let engine = SchedulerEngine::new(config, policy, kv_manager, worker_group, frontend, worker);

    tracing::info!("Scheduler engine running...");
    engine.run().await?;

    Ok(())
}
