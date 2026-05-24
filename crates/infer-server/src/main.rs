use anyhow::{Context, Result};
use clap::Parser;
use std::net::SocketAddr;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use infer_server::{
    AppState, ServerConfig, ZmqClient,
    router::build_router,
    state::ModelInfo,
};

/// A managed child process with a label.
struct ManagedChild {
    label: String,
    child: Child,
}

impl ManagedChild {
    fn new(label: impl Into<String>, child: Child) -> Self {
        Self {
            label: label.into(),
            child,
        }
    }

    /// Check if the process has exited.
    fn try_wait(&mut self) -> Option<std::process::ExitStatus> {
        self.child.try_wait().ok().flatten()
    }

    /// Send SIGTERM (Unix) or kill (Windows).
    fn terminate(&mut self) {
        #[cfg(unix)]
        {
            let pid = nix::unistd::Pid::from_raw(self.child.id() as i32);
            let _ = nix::sys::signal::kill(pid, nix::sys::signal::Signal::SIGTERM);
        }
        #[cfg(not(unix))]
        {
            let _ = self.child.kill();
        }
    }

    /// Force kill.
    fn kill(&mut self) {
        let _ = self.child.kill();
    }
}

/// Gracefully shut down all child processes.
async fn shutdown_all(children: &mut [ManagedChild]) {
    // Send SIGTERM to all.
    for managed in children.iter_mut() {
        tracing::info!("[{}] Sending SIGTERM...", managed.label);
        managed.terminate();
    }

    // Wait up to 5 seconds for graceful exit.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let all_exited = children.iter_mut().all(|m| m.try_wait().is_some());
        if all_exited {
            tracing::info!("All components exited gracefully.");
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Force kill any remaining.
    for managed in children.iter_mut() {
        if managed.try_wait().is_none() {
            tracing::warn!("[{}] Force killing (timeout)...", managed.label);
            managed.kill();
        }
    }
}

/// Clean up IPC socket files.
fn cleanup_ipc(pid: u32) {
    let patterns = [
        format!("/tmp/rustinfer-{}-frontend.ipc", pid),
        format!("/tmp/rustinfer-{}-worker-in.ipc", pid),
        format!("/tmp/rustinfer-{}-worker-out.ipc", pid),
        format!("/tmp/rustinfer-{}-worker-control.ipc", pid),
    ];
    for path in &patterns {
        if std::path::Path::new(path).exists() {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = ServerConfig::parse();

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| config.log_level.clone().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let pid = std::process::id();
    let devices = config.devices();
    let assigned_device = devices.first().copied().unwrap_or("cuda:0").to_string();
    if devices.len() > 1 {
        tracing::warn!(
            "Currently starting one WorkerGroup rank only; ignoring extra devices: {:?}",
            &devices[1..],
        );
    }
    let model_name = config.effective_model_name();

    // Auto-generate IPC endpoints
    let frontend_ep = format!("ipc:///tmp/rustinfer-{}-frontend.ipc", pid);
    let worker_in_ep = format!("ipc:///tmp/rustinfer-{}-worker-in.ipc", pid);
    let worker_out_ep = format!("ipc:///tmp/rustinfer-{}-worker-out.ipc", pid);
    let worker_control_ep = format!("ipc:///tmp/rustinfer-{}-worker-control.ipc", pid);

    tracing::info!("╔══════════════════════════════════════════════════╗");
    tracing::info!("║          RustInfer Server v0.1.0                 ║");
    tracing::info!("╚══════════════════════════════════════════════════╝");
    tracing::info!("  Model: {}", config.model);
    tracing::info!("  Model type: {}", config.model_type);
    tracing::info!("  Devices: {:?}", devices);
    tracing::info!("  API Server Port: {}", config.port);
    tracing::info!("  max_batch_tokens: {}", config.max_batch_tokens);
    tracing::info!("  max_batch_seqs: {}", config.max_batch_seqs);
    tracing::info!("  max_model_len: {}", config.max_model_len);
    tracing::info!("  chunked_prefill_size: {:?}", config.chunked_prefill_size);

    let mut children: Vec<ManagedChild> = Vec::new();

    // ═══════════════════════════════════════════════════════════════════════════
    //  1. Start Scheduler (binds IPC sockets)
    // ═══════════════════════════════════════════════════════════════════════════
    tracing::info!("[scheduler] Starting...");
    let mut scheduler_cmd = Command::new("rustinfer-scheduler");
    scheduler_cmd
        .arg("--frontend-endpoint").arg(&frontend_ep)
        .arg("--worker-push-endpoint").arg(&worker_in_ep)
        .arg("--worker-pull-endpoint").arg(&worker_out_ep)
        .arg("--worker-control-endpoint").arg(&worker_control_ep)
        .arg("--model").arg(&config.model)
        .arg("--model-type").arg(&config.model_type)
        .arg("--device").arg(&assigned_device)
        .arg("--max-batch-tokens").arg(config.max_batch_tokens.to_string())
        .arg("--max-batch-seqs").arg(config.max_batch_seqs.to_string())
        .arg("--max-model-len").arg(config.max_model_len.to_string())
        .arg("--log-level").arg(&config.log_level);

    if let Some(chunk_size) = config.chunked_prefill_size {
        scheduler_cmd.arg("--chunked-prefill-size").arg(chunk_size.to_string());
    }

    // Forward KV-cache and runtime planning flags so paged mode can be exercised
    // end-to-end via the all-in-one launcher.
    scheduler_cmd
        .arg("--kv-cache-mode").arg(&config.kv_cache_mode)
        .arg("--mem-fraction-static").arg(config.mem_fraction_static.to_string());
    if config.enable_prefix_caching {
        scheduler_cmd.arg("--enable-prefix-caching");
    }

    let scheduler_child = scheduler_cmd
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("Failed to spawn rustinfer-scheduler. Is it in PATH?")?;

    children.push(ManagedChild::new("scheduler", scheduler_child));

    // ═══════════════════════════════════════════════════════════════════════════
    //  2. Start WorkerGroup rank 0
    // ═══════════════════════════════════════════════════════════════════════════
    tracing::info!("[worker:0] Starting on {}...", assigned_device);

    let worker_child = Command::new("rustinfer-worker")
        .arg("--device").arg(&assigned_device)
        .arg("--worker-pull-endpoint").arg(&worker_in_ep)
        .arg("--worker-push-endpoint").arg(&worker_out_ep)
        .arg("--worker-control-endpoint").arg(&worker_control_ep)
        .arg("--max-batch-tokens").arg(config.max_batch_tokens.to_string())
        .arg("--max-batch-seqs").arg(config.max_batch_seqs.to_string())
        .arg("--log-level").arg(&config.log_level)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context(format!("Failed to spawn rustinfer-worker for {}", assigned_device))?;

    tracing::info!("[scheduler] PID={}", children.last().unwrap().child.id());
    tracing::info!("[worker:0] PID={}", worker_child.id());
    children.push(ManagedChild::new("worker:0", worker_child));

    tokio::time::sleep(Duration::from_millis(500)).await;

    // ═══════════════════════════════════════════════════════════════════════════
    //  3. Initialize API Server
    // ═══════════════════════════════════════════════════════════════════════════
    tracing::info!("[server] Initializing Tokenizer and ZMQ Client...");
    let tokenizer_path = std::path::Path::new(&config.model).join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    tracing::info!("Tokenizer loaded (vocab_size={})", tokenizer.get_vocab_size(true));

    let client = ZmqClient::new(&frontend_ep, config.request_timeout_secs).await?;
    tracing::info!("Connected to scheduler via {}", frontend_ep);

    let model_info = ModelInfo {
        model_id: model_name,
        owned_by: "rustinfer".to_string(),
    };

    let state = Arc::new(AppState {
        client,
        tokenizer,
        config: config.clone(),
        model_info,
    });

    let app = build_router(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    tracing::info!("API Server listening on http://{}:{}", config.host, config.port);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Create a broadcast channel for shutdown
    let (shutdown_tx, _) = tokio::sync::broadcast::channel::<()>(1);
    
    // Spawn axum server
    let mut axum_rx = shutdown_tx.subscribe();
    let server_task = tokio::spawn(async move {
        let result = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = axum_rx.recv().await;
                tracing::info!("Axum server shutting down gracefully.");
            })
            .await;
        if let Err(e) = result {
            tracing::error!("Axum server error: {}", e);
        }
    });

    // ═══════════════════════════════════════════════════════════════════════════
    //  4. Monitor: wait for CTRL+C or any child exit
    // ═══════════════════════════════════════════════════════════════════════════
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Received Ctrl+C, initiating shutdown...");
                break;
            }
            _ = tokio::time::sleep(Duration::from_millis(500)) => {
                let mut crashed: Option<(String, std::process::ExitStatus)> = None;
                for managed in children.iter_mut() {
                    if let Some(status) = managed.try_wait() {
                        tracing::error!(
                            "[{}] exited unexpectedly with status: {}",
                            managed.label, status
                        );
                        crashed = Some((managed.label.clone(), status));
                        break;
                    }
                }
                if let Some((label, status)) = crashed {
                    tracing::error!("Component '{}' crashed (exit: {}). Initiating shutdown.", label, status);
                    break;
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  5. Graceful shutdown
    // ═══════════════════════════════════════════════════════════════════════════
    let _ = shutdown_tx.send(()); // Signal Axum to shut down
    
    tracing::info!("Shutting down child processes...");
    shutdown_all(&mut children).await;
    cleanup_ipc(pid);

    tracing::info!("Waiting for Axum server to exit...");
    let _ = server_task.await;

    tracing::info!("All components stopped. Goodbye!");
    Ok(())
}
