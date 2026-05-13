//! RustInfer Launcher — unified entry point for LLM inference serving.
//!
//! Spawns and manages scheduler, worker, and HTTP server as child processes.
//! One command to start the entire inference stack.

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(name = "rustinfer-launch")]
#[command(about = "RustInfer Launcher — one command to start the full inference stack")]
struct Args {
    /// Model path (directory containing model weights + tokenizer.json).
    #[arg(short, long)]
    model: String,

    /// Model type: llama3, qwen3.
    #[arg(long, default_value = "llama3")]
    model_type: String,

    /// Device(s): comma-separated. e.g. "cuda:0" or "cuda:0,cuda:1" for multi-GPU.
    #[arg(short, long, default_value = "cuda:0")]
    device: String,

    /// HTTP server port.
    #[arg(short, long, default_value = "8000")]
    port: u16,

    /// Maximum batch tokens per iteration.
    #[arg(long, default_value = "4096")]
    max_batch_tokens: usize,

    /// Maximum concurrent sequences.
    #[arg(long, default_value = "32")]
    max_batch_seqs: usize,

    /// Maximum model sequence length.
    #[arg(long, default_value = "8192")]
    max_model_len: usize,

    /// Chunked prefill size (None = disabled).
    #[arg(long)]
    chunked_prefill_size: Option<usize>,

    /// Model name for /v1/models endpoint.
    #[arg(long)]
    model_name: Option<String>,

    /// Log level.
    #[arg(long, default_value = "info")]
    log_level: String,
}

impl Args {
    /// Parse device list from comma-separated string.
    fn devices(&self) -> Vec<&str> {
        self.device.split(',').map(|s| s.trim()).collect()
    }

    /// Derive model name from path if not explicitly set.
    fn effective_model_name(&self) -> String {
        self.model_name.clone().unwrap_or_else(|| {
            std::path::Path::new(&self.model)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("default")
                .to_string()
        })
    }
}

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

    let pid = std::process::id();
    let devices = args.devices();
    let assigned_device = devices.first().copied().unwrap_or("cuda:0").to_string();
    if devices.len() > 1 {
        tracing::warn!(
            "Phase 4 currently starts one WorkerGroup rank only; ignoring extra devices: {:?}",
            &devices[1..],
        );
    }
    let model_name = args.effective_model_name();

    // Auto-generate IPC endpoints (unique per process to avoid conflicts).
    let frontend_ep = format!("ipc:///tmp/rustinfer-{}-frontend.ipc", pid);
    let worker_in_ep = format!("ipc:///tmp/rustinfer-{}-worker-in.ipc", pid);
    let worker_out_ep = format!("ipc:///tmp/rustinfer-{}-worker-out.ipc", pid);
    let worker_control_ep = format!("ipc:///tmp/rustinfer-{}-worker-control.ipc", pid);

    tracing::info!("╔══════════════════════════════════════════════════╗");
    tracing::info!("║          RustInfer Launcher v0.1.0               ║");
    tracing::info!("╚══════════════════════════════════════════════════╝");
    tracing::info!("  Model: {}", args.model);
    tracing::info!("  Model type: {}", args.model_type);
    tracing::info!("  Devices: {:?}", devices);
    tracing::info!("  Port: {}", args.port);
    tracing::info!("  max_batch_tokens: {}", args.max_batch_tokens);
    tracing::info!("  max_batch_seqs: {}", args.max_batch_seqs);
    tracing::info!("  max_model_len: {}", args.max_model_len);
    tracing::info!("  chunked_prefill_size: {:?}", args.chunked_prefill_size);

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
        .arg("--model").arg(&args.model)
        .arg("--model-type").arg(&args.model_type)
        .arg("--device").arg(&assigned_device)
        .arg("--max-batch-tokens").arg(args.max_batch_tokens.to_string())
        .arg("--max-batch-seqs").arg(args.max_batch_seqs.to_string())
        .arg("--max-model-len").arg(args.max_model_len.to_string())
        .arg("--log-level").arg(&args.log_level);

    if let Some(chunk_size) = args.chunked_prefill_size {
        scheduler_cmd.arg("--chunked-prefill-size").arg(chunk_size.to_string());
    }

    let scheduler_child = scheduler_cmd
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("Failed to spawn rustinfer-scheduler. Is it in PATH?")?;

    children.push(ManagedChild::new("scheduler", scheduler_child));
    tracing::info!("[scheduler] PID={}", children.last().unwrap().child.id());

    // Brief wait for scheduler to bind IPC sockets.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // ═══════════════════════════════════════════════════════════════════════════
    //  2. Start WorkerGroup rank 0 (single-rank group for now)
    // ═══════════════════════════════════════════════════════════════════════════

    tracing::info!("[worker:0] Starting on {}...", assigned_device);

    let worker_child = Command::new("rustinfer-worker")
        .arg("--device").arg(&assigned_device)
        .arg("--worker-pull-endpoint").arg(&worker_in_ep)
        .arg("--worker-push-endpoint").arg(&worker_out_ep)
        .arg("--worker-control-endpoint").arg(&worker_control_ep)
        .arg("--max-batch-tokens").arg(args.max_batch_tokens.to_string())
        .arg("--max-batch-seqs").arg(args.max_batch_seqs.to_string())
        .arg("--log-level").arg(&args.log_level)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context(format!("Failed to spawn rustinfer-worker for {}", assigned_device))?;

    tracing::info!("[worker:0] PID={}", worker_child.id());
    children.push(ManagedChild::new("worker:0", worker_child));

    // Wait for workers to connect and load model.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // ═══════════════════════════════════════════════════════════════════════════
    //  3. Start HTTP Server (connect to scheduler)
    // ═══════════════════════════════════════════════════════════════════════════

    tracing::info!("[server] Starting on port {}...", args.port);

    let server_child = Command::new("rustinfer-server")
        .arg("--port").arg(args.port.to_string())
        .arg("--engine-endpoint").arg(&frontend_ep)
        .arg("--tokenizer").arg(&args.model)
        .arg("--model-name").arg(&model_name)
        .arg("--log-level").arg(&args.log_level)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("Failed to spawn rustinfer-server. Is it in PATH?")?;

    tracing::info!("[server] PID={}", server_child.id());
    children.push(ManagedChild::new("server", server_child));

    // ═══════════════════════════════════════════════════════════════════════════
    //  4. Monitor: wait for CTRL+C or any child exit
    // ═══════════════════════════════════════════════════════════════════════════

    tracing::info!("════════════════════════════════════════════════════");
    tracing::info!("  All components started successfully.");
    tracing::info!("  API: http://0.0.0.0:{}/v1", args.port);
    tracing::info!("  Press Ctrl+C to stop.");
    tracing::info!("════════════════════════════════════════════════════");

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Received Ctrl+C, shutting down...");
                break;
            }
            _ = tokio::time::sleep(Duration::from_millis(500)) => {
                // Check if any child has exited unexpectedly.
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
                    shutdown_all(&mut children).await;
                    cleanup_ipc(pid);
                    anyhow::bail!("Component '{}' crashed (exit: {})", label, status);
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  5. Graceful shutdown
    // ═══════════════════════════════════════════════════════════════════════════

    shutdown_all(&mut children).await;
    cleanup_ipc(pid);
    tracing::info!("All components stopped. Goodbye!");
    Ok(())
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
    ];
    for path in &patterns {
        if std::path::Path::new(path).exists() {
            let _ = std::fs::remove_file(path);
        }
    }
}
