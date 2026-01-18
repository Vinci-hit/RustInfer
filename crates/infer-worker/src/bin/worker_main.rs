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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Print startup banner
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║              RustInfer Worker v0.1.0                       ║");
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
