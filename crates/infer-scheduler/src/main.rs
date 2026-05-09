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

    // Build and run engine.
    let engine = SchedulerEngine::new(config, policy, kv_manager, frontend, worker);

    tracing::info!("Scheduler engine running...");
    engine.run().await?;

    Ok(())
}
