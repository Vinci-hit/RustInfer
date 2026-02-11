//! Worker Binary Entry Point
//!
//! This is the standalone worker process that connects to a Scheduler
//! and handles inference requests.
//!
//! # Usage
//!
//! ```bash
//! # Start a single worker (GPU 0)
//! cargo run --bin infer-worker --features server -- \
//!     --rank 0 \
//!     --world-size 1 \
//!     --scheduler-url "ipc:///tmp/rustinfer-scheduler.ipc"
//!
//! # Start worker for tensor parallelism (GPU 1 of 2)
//! cargo run --bin infer-worker --features server -- \
//!     --rank 1 \
//!     --world-size 2 \
//!     --scheduler-url "tcp://localhost:5555"
//! ```
//!
//! # Architecture
//!
//! 1. Worker starts with rank/device info
//! 2. Worker connects to Scheduler
//! 3. Worker sends Register message
//! 4. Scheduler sends LoadModel command (with model path)
//! 5. Worker loads model from specified path
//! 6. Scheduler sends InitKVCache command
//! 7. Worker is ready for inference

use clap::Parser;
use infer_worker::worker::WorkerServer;

/// RustInfer Worker Process
///
/// Connects to a Scheduler and handles model inference requests.
#[derive(Parser, Debug)]
#[command(name = "infer-worker")]
#[command(author = "RustInfer Team")]
#[command(version = "0.1.0")]
#[command(about = "RustInfer Worker - GPU inference worker process")]
struct Args {
    /// Worker rank (0-indexed, maps to GPU device)
    #[arg(long, default_value_t = 0)]
    rank: usize,

    /// Total number of workers (world size for tensor parallelism)
    #[arg(long, default_value_t = 1)]
    world_size: usize,

    /// Scheduler ZeroMQ endpoint URL
    ///
    /// Examples:
    /// - IPC: "ipc:///tmp/rustinfer-scheduler.ipc"
    /// - TCP: "tcp://localhost:5555"
    #[arg(long, default_value = "ipc:///tmp/rustinfer-scheduler.ipc")]
    scheduler_url: String,

    /// Enable verbose logging
    #[arg(long, short, default_value_t = false)]
    verbose: bool,

    /// Run in mock mode (for testing spawner, no real scheduler connection)
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// Fail after N seconds (mock mode only)
    #[arg(long)]
    fail_after: Option<u64>,

    /// Crash at startup with probability 0.0-1.0 (mock mode only)
    #[arg(long)]
    crash_chance: Option<f32>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Print startup banner
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║              RustInfer Worker v0.1.0                       ║");
    if args.mock {
        println!("║              [MOCK MODE - FOR TESTING]                      ║");
    }
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Rank:        {:>4} / {:<4}                                 ║", args.rank, args.world_size);
    println!("║  Scheduler:   {:<43} ║", truncate_str(&args.scheduler_url, 43));
    println!("║  Model:       <Waiting for Scheduler>                        ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    // Validate arguments
    if args.rank >= args.world_size {
        anyhow::bail!(
            "Invalid rank {} for world size {}",
            args.rank,
            args.world_size
        );
    }

    // Run mock mode or real worker
    if args.mock {
        run_mock_worker(&args).await
    } else {
        run_real_worker(&args).await
    }
}

/// Run worker in mock mode (for testing spawner)
async fn run_mock_worker(args: &Args) -> anyhow::Result<()> {
    let worker_id = format!("worker-{}", args.rank);

    println!("\n[{}] Starting mock worker (rank={}/{}, PID={})",
        worker_id, args.rank, args.world_size, std::process::id());
    println!("[{}] Mock mode: simulating worker initialization...", worker_id);

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    println!("[{}] Initialized CUDA context", worker_id);

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    println!("[{}] Connected to scheduler (mock)", worker_id);

    println!("[{}] Ready", worker_id);

    // Crash at startup if chance is set
    if let Some(chance) = args.crash_chance {
        if chance > 0.0 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            // Deterministic "random" based on PID for repeatability
            let mut hasher = DefaultHasher::new();
            std::process::id().hash(&mut hasher);
            let hash = hasher.finish() as f32 / u64::MAX as f32;

            if hash < chance {
                eprintln!("[{}] Simulated crash at startup!", worker_id);
                return Err(anyhow::anyhow!("Mock startup crash"));
            }
        }
    }

    // Setup shutdown signal
    let shutdown = tokio::signal::ctrl_c();
    tokio::pin!(shutdown);

    // Main loop
    let _start_time = std::time::Instant::now();
    let mut tick_count = 0u64;

    loop {
        tokio::select! {
            _ = &mut shutdown => {
                println!("[{}] Received shutdown signal", worker_id);
                break;
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {
                tick_count += 1;

                // Simulated failure after N seconds
                if let Some(fail_secs) = args.fail_after {
                    if tick_count >= fail_secs {
                        eprintln!("[{}] Simulated failure after {} seconds", worker_id, fail_secs);
                        return Err(anyhow::anyhow!("Mock worker failure"));
                    }
                }

                // Periodic status output
                if tick_count % 5 == 0 {
                    println!("[{}] Heartbeat: {}s elapsed, processed 0 requests",
                        worker_id, tick_count);
                }
            }
        }
    }

    println!("[{}] Shutdown complete", worker_id);
    Ok(())
}

/// Run worker in real mode (normal operation)
async fn run_real_worker(args: &Args) -> anyhow::Result<()> {
    // Create and start WorkerServer
    println!("\n[Worker-{}] Initializing...", args.rank);

    let mut server = WorkerServer::new(
        args.rank,
        args.world_size,
        &args.scheduler_url,
    )
    .await?;

    println!("[Worker-{}] Ready. Waiting for commands from Scheduler...\n", args.rank);

    // Setup signal handler for graceful shutdown
    let shutdown_signal = setup_shutdown_signal();

    // Run the main loop with shutdown handling
    tokio::select! {
        result = server.run_loop() => {
            if let Err(e) = result {
                eprintln!("[Worker-{}] Error in main loop: {}", args.rank, e);
                return Err(e);
            }
        }
        _ = shutdown_signal => {
            println!("\n[Worker-{}] Received shutdown signal, cleaning up...", args.rank);
            server.shutdown();
        }
    }

    println!("[Worker-{}] Shutdown complete.", args.rank);
    Ok(())
}

/// Setup signal handler for graceful shutdown (Ctrl+C)
async fn setup_shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
}

/// Truncate string to max length with ellipsis
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
