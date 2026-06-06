use anyhow::{Context, Result};
use clap::Parser;
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use infer_server::ServerConfig;

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
    let mut config = ServerConfig::parse();

    // Model type is derived from the model's config.json, not a CLI flag, so
    // that the chat template / worker dispatch always match the loaded weights.
    config.model_type = ServerConfig::resolve_model_type(&config.model)
        .with_context(|| format!("resolve model type from {}", config.model))?;

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
    let _model_name = config.effective_model_name();

    // Auto-generate IPC endpoints
    let frontend_ep = format!("ipc:///tmp/rustinfer-{}-frontend.ipc", pid);
    let worker_in_ep = format!("ipc:///tmp/rustinfer-{}-worker-in.ipc", pid);
    let worker_out_ep = format!("ipc:///tmp/rustinfer-{}-worker-out.ipc", pid);
    let worker_control_ep = format!("ipc:///tmp/rustinfer-{}-worker-control.ipc", pid);

    tracing::info!("╔══════════════════════════════════════════════════╗");
    tracing::info!("║     RustInfer Scheduler & Worker v0.1.0           ║");
    tracing::info!("╚══════════════════════════════════════════════════╝");
    tracing::info!("  Model: {}", config.model);
    tracing::info!("  Model type: {}", config.model_type);
    tracing::info!("  Devices: {:?}", devices);
    tracing::info!("  max_batch_tokens: {}", config.max_batch_tokens);
    tracing::info!("  max_batch_seqs: {}", config.max_batch_seqs);
    tracing::info!("  max_model_len: {}", config.max_model_len);
    tracing::info!("  chunked_prefill_size: {:?}", config.chunked_prefill_size);
    tracing::info!("  ignore_eos: {}", config.ignore_eos);

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

    // Forward paged KV block size + runtime planning fraction to the scheduler.
    // `kv_cache_mode` is kept on the server CLI as `paged:<N>` for backwards
    // compatibility; the scheduler now takes the block size directly.
    let paged_block_size: usize = config
        .kv_cache_mode
        .strip_prefix("paged:")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    scheduler_cmd
        .arg("--paged-block-size").arg(paged_block_size.to_string())
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

    let mut worker_cmd = Command::new("rustinfer-worker");
    worker_cmd
        .arg("--device").arg(&assigned_device)
        .arg("--worker-pull-endpoint").arg(&worker_in_ep)
        .arg("--worker-push-endpoint").arg(&worker_out_ep)
        .arg("--worker-control-endpoint").arg(&worker_control_ep)
        .arg("--max-batch-tokens").arg(config.max_batch_tokens.to_string())
        .arg("--max-batch-seqs").arg(config.max_batch_seqs.to_string())
        .arg("--log-level").arg(&config.log_level);
    if let Some(n) = config.num_blocks {
        worker_cmd.arg("--num-blocks").arg(n.to_string());
    }
    let worker_child = worker_cmd
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context(format!("Failed to spawn rustinfer-worker for {}", assigned_device))?;

    tracing::info!("[scheduler] PID={}", children.last().unwrap().child.id());
    tracing::info!("[worker:0] PID={}", worker_child.id());
    children.push(ManagedChild::new("worker:0", worker_child));

    tokio::time::sleep(Duration::from_millis(500)).await;

    // ═══════════════════════════════════════════════════════════════════════════
    //  3. Print connection info for API server
    // ═══════════════════════════════════════════════════════════════════════════
    tracing::info!("╔══════════════════════════════════════════════════╗");
    tracing::info!("║         Scheduler & Worker Running                ║");
    tracing::info!("╚══════════════════════════════════════════════════╝");
    tracing::info!("  Frontend Endpoint: {}", frontend_ep);
    tracing::info!("  Worker In Endpoint: {}", worker_in_ep);
    tracing::info!("  Worker Out Endpoint: {}", worker_out_ep);
    tracing::info!("  Worker Control Endpoint: {}", worker_control_ep);
    tracing::info!("");
    tracing::info!("To start the API server in another terminal, run:");
    tracing::info!("  rustinfer-api --model {} --frontend-endpoint {} --port 8000",
        config.model, frontend_ep);

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
    tracing::info!("Shutting down child processes...");
    shutdown_all(&mut children).await;
    cleanup_ipc(pid);

    tracing::info!("All components stopped. Goodbye!");
    Ok(())
}
