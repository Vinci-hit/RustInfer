//! RustInfer Scheduler binary entry point.

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use infer_scheduler::application::SchedulerEngine;
use infer_scheduler::config::{SchedulerConfig, SchedulerMode};
use infer_scheduler::domain::BlockSize;
use infer_scheduler::domain::policy::{ContinuousBatchingPolicy, DiffusionPolicy, SchedulingPolicy};
use infer_protocol::scheduler_to_worker_control::LoadModel;
use infer_protocol::{resolve_model_type, RustInferConfig};
use infer_scheduler::infrastructure::transport::control_plane::{ControlPlane, ControlPlaneConfig};
use infer_scheduler::infrastructure::transport::zmq_transport::{ZmqFrontendTransport, ZmqWorkerTransport};

#[derive(Parser, Debug)]
#[command(name = "rustinfer-scheduler")]
#[command(about = "RustInfer Scheduler — production-grade continuous batching scheduler")]
struct Args {
    /// Path to the shared TOML launch config.
    #[arg(long, default_value = "rustinfer.toml")]
    config: String,
}

fn parse_block_size(s: usize) -> BlockSize {
    let raw = u32::try_from(s).unwrap_or(1);
    BlockSize::new(if raw == 0 { 1 } else { raw })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = RustInferConfig::load(&args.config).map_err(|e| anyhow::anyhow!(e))?;

    // Initialize logging.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| cfg.log_level.clone().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let frontend_endpoint = cfg.frontend_endpoint();
    let worker_push_endpoint = cfg.worker_in_endpoint();
    let worker_pull_endpoint = cfg.worker_out_endpoint();
    let worker_control_endpoint = cfg.worker_control_endpoint();

    let paged_block_size = parse_block_size(cfg.paged_block_size);
    let scheduler_mode = match cfg.mode.as_str() {
        "diffusion" => SchedulerMode::Diffusion,
        _ => SchedulerMode::Llm,
    };

    // model_type is derived from the model's config.json (never a CLI flag),
    // so the worker dispatch always matches the loaded weights.
    let model_type = resolve_model_type(&cfg.model).map_err(|e| anyhow::anyhow!(e))?;

    tracing::info!("RustInfer Scheduler v0.2.0 starting...");
    tracing::info!("  Mode: {:?}", scheduler_mode);
    tracing::info!("  Frontend: {}", frontend_endpoint);
    tracing::info!("  Worker PUSH: {}", worker_push_endpoint);
    tracing::info!("  Worker PULL: {}", worker_pull_endpoint);
    tracing::info!("  Worker Control: {}", worker_control_endpoint);
    tracing::info!("  Assigned model: {}", cfg.model);
    tracing::info!("  Assigned model type: {}", model_type);
    tracing::info!("  max_batch_seqs: {}", cfg.max_batch_seqs);
    tracing::info!("  max_batch_tokens: {}", cfg.max_batch_tokens);
    tracing::info!("  max_model_len: {}", cfg.max_model_len);
    tracing::info!("  paged_block_size: {}", paged_block_size);
    tracing::info!("  chunked_prefill_size: {:?}", cfg.chunked_prefill());
    tracing::info!("  enable_prefix_caching: {}", cfg.enable_prefix_caching);

    // Build config.
    let mut config = SchedulerConfig {
        mode: scheduler_mode,
        max_num_seqs: cfg.max_batch_seqs,
        max_batch_tokens: cfg.max_batch_tokens,
        max_model_len: cfg.max_model_len,
        paged_block_size,
        chunked_prefill_size: cfg.chunked_prefill(),
        enable_prefix_caching: cfg.enable_prefix_caching,
        frontend_endpoint: frontend_endpoint.clone(),
        worker_push_endpoint: worker_push_endpoint.clone(),
        worker_pull_endpoint: worker_pull_endpoint.clone(),
        ..Default::default()
    };

    // Create transports.
    let frontend = ZmqFrontendTransport::new(&frontend_endpoint)?;
    let worker = ZmqWorkerTransport::new(&worker_push_endpoint, &worker_pull_endpoint)?;

    let load_model = Some(LoadModel {
        model_instance_id: "default".to_string(),
        model_path: cfg.model.clone(),
        model_type: model_type.clone(),
        device: cfg.device.clone(),
        max_batch_tokens: cfg.max_batch_tokens,
        max_batch_seqs: cfg.max_batch_seqs,
        max_model_len: cfg.max_model_len,
        mem_fraction_static: cfg.mem_fraction_static,
        tp_rank: 0,
        tp_size: 1,
        pp_rank: 0,
        pp_size: 1,
        kv_cache_mode: Some(format!("paged:{}", paged_block_size.raw())),
        kv_cache_memory_fraction: Some(cfg.mem_fraction_static),
        enable_prefix_caching: cfg.enable_prefix_caching,
    });

    // Bind the control-plane ROUTER, optionally assign a model, then
    // wait for `WorkerReady`. The same socket continues to serve
    // runtime control once the handshake completes.
    let cp_cfg = ControlPlaneConfig::default();
    let (mut control_plane, worker_group) =
        ControlPlane::bootstrap(&worker_control_endpoint, load_model, cp_cfg).await?;
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
        SchedulerMode::Diffusion => Box::new(DiffusionPolicy::new(cfg.max_batch_seqs)),
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
