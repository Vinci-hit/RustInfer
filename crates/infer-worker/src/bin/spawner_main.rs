//! RustInfer Worker Spawner - Main Entry Point
//!
//! This binary automatically discovers available devices and launches
//! multiple Worker processes, one per device.
//!
//! # Usage
//!
//! ```bash
//! # Auto-discover all GPUs
//! cargo run --bin infer-spawner --features server -- \
//!     --scheduler-url "tcp://localhost:5555" \
//!     --auto
//!
//! # Use specific GPUs
//! cargo run --bin infer-spawner --features server -- \
//!     --scheduler-url "tcp://localhost:5555" \
//!     --gpu-ids 0,1,2,3
//!
//! # With restart policy
//! cargo run --bin infer-spawner --features server -- \
//!     --scheduler-url "tcp://localhost:5555" \
//!     --auto \
//!     --restart-policy on-failure \
//!     --max-retries 3
//! ```

use std::process;
use anyhow::Result;

#[cfg(feature = "server")]
use clap::Parser;

use infer_worker::spawner::{SpawnerArgs, WorkerSpawner, SpawnerConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    let args = SpawnerArgs::parse();

    // Print startup banner
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║            RustInfer Worker Spawner v0.1.0                ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Scheduler:   {:<44} ║", truncate_str(&args.scheduler_url, 44));
    println!("║  Auto-discover:{:<11}                                        ║", args.auto);
    if let Some(ref gpu_ids) = args.gpu_ids {
        println!("║  GPU IDs:     {:<45} ║", format!("{:?}", gpu_ids));
    }
    println!("╚════════════════════════════════════════════════════════════╝");

    // Convert args to config
    let config: SpawnerConfig = args.try_into()
        .map_err(|e| anyhow::anyhow!("Invalid configuration: {}", e))?;

    // Create spawner
    let mut spawner = WorkerSpawner::new(config);

    // Spawn all workers
    if let Err(e) = spawner.spawn_all().await {
        eprintln!("[Spawner] ✗ Failed to start workers: {}", e);
        process::exit(1);
    }

    // Start monitoring loop
    if let Err(e) = spawner.monitor_workers().await {
        eprintln!("[Spawner] ✗ Monitoring failed: {}", e);
        process::exit(1);
    }

    println!("[Spawner] Shutdown complete.");
    Ok(())
}

/// Truncate string to max length with ellipsis
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
