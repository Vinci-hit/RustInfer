//! RustInfer Scheduler binary entry point.

use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use infer_protocol::scheduler_to_worker_control::LoadModel;
use infer_protocol::{RustInferConfig, resolve_model_type};
use infer_scheduler::application::SchedulerEngine;
use infer_scheduler::config::{SchedulerConfig, SchedulerMode};
use infer_scheduler::domain::BlockSize;
use infer_scheduler::domain::policy::{
    ContinuousBatchingPolicy, DiffusionPolicy, SchedulingPolicy,
};
use infer_scheduler::infrastructure::transport::control_plane::{ControlPlane, ControlPlaneConfig};
use infer_scheduler::infrastructure::transport::zmq_transport::{
    ZmqFrontendTransport, ZmqWorkerTransport,
};

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
    match cfg.batch_wait() {
        Some(d) => tracing::info!("  batch_wait: {:?} (throughput mode)", d),
        None => tracing::info!("  batch_wait: off (low-latency mode)"),
    }

    // Build config from the shared launch config (single mapping + validation).
    let mut config = SchedulerConfig::from_launch(&cfg, scheduler_mode, paged_block_size);

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
    let mut cp_cfg = ControlPlaneConfig::default();
    cp_cfg.heartbeat_interval = Duration::from_secs(1);
    cp_cfg.heartbeat_timeout = Duration::from_secs(cfg.worker_heartbeat_timeout_secs.max(1));
    tracing::info!(
        "  worker heartbeat_interval: {:?}",
        cp_cfg.heartbeat_interval
    );
    tracing::info!("  worker heartbeat_timeout: {:?}", cp_cfg.heartbeat_timeout);
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
    // Keep the ControlPlane owned for the whole run. `cmd_tx()` / `take_event_rx()`
    // hand the engine independent owned handles, so the plane itself does not need
    // to be borrowed or leaked. Holding it in this binding keeps the router thread
    // and liveness watchdog alive for the duration of `engine.run()`, and lets its
    // `Drop` run on exit — performing graceful shutdown (Shutdown to the worker,
    // router-thread join, pending-RPC drain) that the previous `Box::leak` skipped.

    config.apply_worker_capacity(worker_group.effective_capacity.max_total_kv_tokens);

    // The worker is the sole owner of physical block allocation; the
    // scheduler only tracks slot accounting via `KvBudget` + `RadixTree`,
    // wired up inside `SchedulerEngine::new`.

    let policy: Box<dyn SchedulingPolicy> = match scheduler_mode {
        SchedulerMode::Diffusion => Box::new(DiffusionPolicy::new(cfg.max_batch_seqs)),
        SchedulerMode::Llm => Box::new(
            ContinuousBatchingPolicy::new(config.chunked_prefill_size)
                .with_admission(config.max_prefill_seqs_per_iter, config.prefill_sjf),
        ),
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
    // Run the engine until it exits on its own OR a shutdown signal arrives.
    // Catching SIGTERM/SIGINT here is what prevents an orphaned worker holding
    // the GPU: on a signal we fall through to `drop(control_plane)`, which sends
    // `Shutdown` to the worker and joins the router thread. Without this, a
    // `systemctl stop` / `docker stop` (SIGTERM) would kill the scheduler
    // without ever telling the worker to exit.
    tokio::select! {
        res = engine.run() => {
            if res.is_err() {
                // The engine failed in-flight/waiting requests on its way out,
                // but those responses are only queued to the detached frontend
                // ZMQ thread. Give it a beat to flush to the sockets before
                // the process exit tears it down.
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            res?;
        }
        sig = wait_for_shutdown_signal() => {
            tracing::info!("Received {}; shutting down scheduler and notifying worker.", sig);
        }
    }

    // Engine loop has exited (or a signal arrived); tear the control plane down
    // gracefully (Shutdown to the worker + router-thread join) before returning.
    drop(control_plane);
    Ok(())
}

/// Resolve when the process receives SIGINT (Ctrl-C) or SIGTERM (the signal
/// `systemctl stop` / `docker stop` send by default). Returns the signal name
/// for logging. On non-unix targets only Ctrl-C is observed.
async fn wait_for_shutdown_signal() -> &'static str {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => "SIGINT",
                    _ = term.recv() => "SIGTERM",
                }
            }
            Err(e) => {
                tracing::warn!("failed to install SIGTERM handler ({}); Ctrl-C only", e);
                let _ = tokio::signal::ctrl_c().await;
                "SIGINT"
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        "SIGINT"
    }
}
