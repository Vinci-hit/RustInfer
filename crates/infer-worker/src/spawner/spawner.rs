//! Worker Spawner Module
//!
//! Handles launching and managing multiple Worker processes.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use anyhow::{Context, Result};

use super::config::{RestartPolicy, SpawnerConfig, WorkerPlan};
use super::device::{DeviceDiscovery, AvailableDevice};

/// Worker process status
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStatus {
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed(String),
}

/// A single Worker process managed by the Spawner
pub struct WorkerProcess {
    pub worker_id: String,
    pub rank: usize,
    pub device_id: i32,
    pub device_info: AvailableDevice,
    pub status: WorkerStatus,
    pub started_at: Instant,
    pub restart_count: u32,
    child: Option<Child>,
}

impl WorkerProcess {
    /// Create a new WorkerProcess (not yet started)
    pub fn new(plan: &WorkerPlan) -> Self {
        Self {
            worker_id: plan.worker_id.clone(),
            rank: plan.rank,
            device_id: plan.device_id,
            device_info: plan.device.clone(),
            status: WorkerStatus::Starting,
            started_at: Instant::now(),
            restart_count: 0,
            child: None,
        }
    }

    /// Check if the worker process is still running
    pub fn is_running(&self) -> bool {
        matches!(self.status, WorkerStatus::Running) && self.child.is_some()
    }

    /// Try to wait for the process to complete
    pub fn try_wait(&mut self) -> Option<std::process::ExitStatus> {
        if let Some(ref mut child) = self.child {
            child.try_wait().ok().flatten()
        } else {
            None
        }
    }

    /// Kill the worker process
    pub fn kill(&mut self) -> Result<()> {
        if let Some(ref mut child) = self.child {
            child.kill().context("Failed to kill worker process")?;
            self.status = WorkerStatus::Stopped;
            Ok(())
        } else {
            Ok(())
        }
    }
}

/// Worker Spawner - manages multiple Worker processes
pub struct WorkerSpawner {
    config: SpawnerConfig,
    workers: HashMap<String, WorkerProcess>,
}

impl WorkerSpawner {
    /// Create a new WorkerSpawner
    pub fn new(config: SpawnerConfig) -> Self {
        Self {
            config,
            workers: HashMap::new(),
        }
    }

    /// Discover available devices and generate worker plans
    pub fn discover_and_plan(&mut self) -> Result<Vec<WorkerPlan>> {
        println!("[Spawner] Starting device discovery...");

        // Discover devices
        let devices = DeviceDiscovery::discover_all()
            .context("Failed to discover devices")?;

        // Filter devices based on constraints
        let filtered_devices = DeviceDiscovery::filter_devices(
            devices,
            &self.config.device_constraints,
        );

        DeviceDiscovery::print_summary(&filtered_devices);

        if filtered_devices.is_empty() {
            anyhow::bail!("No available devices found matching the constraints");
        }

        // Generate worker plans
        let plans = self.generate_plans(filtered_devices);

        println!("[Spawner] Generated {} worker plan(s)", plans.len());

        Ok(plans)
    }

    /// Generate worker plans from available devices
    pub fn generate_plans(&self, devices: Vec<AvailableDevice>) -> Vec<WorkerPlan> {
        let world_size = devices.len();

        devices
            .into_iter()
            .enumerate()
            .map(|(rank, device)| {
                let mut plan = WorkerPlan::new(
                    rank,
                    world_size,
                    device.clone(),
                    self.config.scheduler_url.clone(),
                );

                // Add extra env vars from config if needed
                if self.config.verbose {
                    plan = plan.with_env("RUST_LOG", &self.config.log_level);
                } else {
                    plan = plan.with_env("RUST_LOG", "warn");
                }

                plan
            })
            .collect()
    }

    /// Start all workers
    pub async fn spawn_all(&mut self) -> Result<()> {
        let plans = self.discover_and_plan()?;

        println!("\n[Spawner] Starting worker processes...");
        println!("{}", "=".repeat(70));

        for plan in &plans {
            self.spawn_worker(plan)?;
        }

        println!("{}", "=".repeat(70));
        println!("[Spawner] All {} workers started", plans.len());
        println!();

        // Wait for all workers to be ready if requested
        if self.config.wait_for_all {
            println!("[Spawner] Waiting for all workers to initialize...");
            tokio::time::sleep(Duration::from_secs(2)).await;
            println!("[Spawner] All workers should be ready now");
        }

        Ok(())
    }

    /// Start a single worker process
    pub fn spawn_worker(&mut self, plan: &WorkerPlan) -> Result<()> {
        println!(
            "[Spawner] Starting {} on device {}...",
            plan.worker_id,
            plan.device_id
        );

        // Build command
        let args = plan.build_command_args();
        let env = plan.build_env();

        // Get infer-worker binary path (same directory as current binary)
        let current_exe = std::env::current_exe()
            .context("Failed to get current executable path")?;
        let exe_dir = current_exe
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Failed to get executable directory"))?;
        let exe_path = exe_dir.join("infer-worker");

        // Spawn the process
        let mut child = Command::new(&exe_path)
            .args(&args)
            .envs(&env)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "Failed to spawn worker {}: {:?} with env {:?}",
                    plan.worker_id, args, env
                )
            })?;

        // Capture stdout/stderr for logging
        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            let worker_id = plan.worker_id.clone();
            std::thread::spawn(move || {
                for line in reader.lines() {
                    if let Ok(line) = line {
                        println!("[{}] {}", worker_id, line);
                    }
                }
            });
        }

        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            let worker_id = plan.worker_id.clone();
            std::thread::spawn(move || {
                for line in reader.lines() {
                    if let Ok(line) = line {
                        eprintln!("[{}] {}", worker_id, line);
                    }
                }
            });
        }

        // Create worker process record
        let mut worker = WorkerProcess::new(plan);
        worker.child = Some(child);
        worker.status = WorkerStatus::Running;
        worker.started_at = Instant::now();

        let pid = worker.child.as_ref().map(|c| c.id());

        self.workers.insert(plan.worker_id.clone(), worker);

        println!("[Spawner] ✓ {} started (PID: {:?})", plan.worker_id, pid);

        Ok(())
    }

    /// Monitor all workers and handle failures
    pub async fn monitor_workers(&mut self) -> Result<()> {
        println!("[Spawner] Monitoring workers (press Ctrl+C to stop)...\n");

        // Set up Ctrl+C handler
        let shutdown = tokio::signal::ctrl_c();

        tokio::pin!(shutdown);

        loop {
            tokio::select! {
                _ = &mut shutdown => {
                    println!("\n[Spawner] Received shutdown signal");
                    self.shutdown().await?;
                    return Ok(());
                }
                _ = tokio::time::sleep(self.config.health_check_interval) => {
                    self.check_workers()?;
                }
            }
        }
    }

    /// Check status of all workers
    fn check_workers(&mut self) -> Result<()> {
        let mut failed_workers = Vec::new();
        let mut restart_workers = Vec::new();

        for (worker_id, worker) in self.workers.iter_mut() {
            if let Some(exit_status) = worker.try_wait() {
                println!("[Spawner] ⚠ Worker {} exited with status: {:?}", worker_id, exit_status);

                match &self.config.restart_policy {
                    RestartPolicy::Never => {
                        worker.status = WorkerStatus::Failed("exited".to_string());
                        failed_workers.push(worker_id.clone());
                    }
                    RestartPolicy::Always => {
                        restart_workers.push(worker_id.clone());
                    }
                    RestartPolicy::OnFailure { max_retries } => {
                        if worker.restart_count < *max_retries {
                            restart_workers.push(worker_id.clone());
                        } else {
                            worker.status = WorkerStatus::Failed(
                                format!("exceeded max retries ({}),", max_retries)
                            );
                            failed_workers.push(worker_id.clone());
                        }
                    }
                }
            }
        }

        // Restart workers
        for worker_id in &restart_workers {
            self.restart_worker(worker_id)?;
        }

        // If all workers failed, exit
        if !failed_workers.is_empty() {
            let running_count = self.workers.values().filter(|w| w.is_running()).count();
            if running_count == 0 {
                anyhow::bail!("All workers have failed");
            }
        }

        Ok(())
    }

    /// Restart a failed worker
    fn restart_worker(&mut self, worker_id: &str) -> Result<()> {
        let worker = self.workers.get_mut(worker_id)
            .ok_or_else(|| anyhow::anyhow!("Worker {} not found", worker_id))?;

        worker.restart_count += 1;
        println!(
            "[Spawner] Restarting {} (attempt {})...",
            worker_id, worker.restart_count
        );

        // TODO: Re-generate plan and spawn new worker
        // For now, just mark as failed
        worker.status = WorkerStatus::Failed("restart not implemented".to_string());

        Ok(())
    }

    /// Shutdown all workers
    pub async fn shutdown(&mut self) -> Result<()> {
        println!("[Spawner] Shutting down all workers...");

        for (worker_id, worker) in self.workers.iter_mut() {
            println!("[Spawner] Stopping {}...", worker_id);
            if let Err(e) = worker.kill() {
                eprintln!("[Spawner] Failed to stop {}: {}", worker_id, e);
            }
        }

        self.workers.clear();
        println!("[Spawner] All workers stopped");

        Ok(())
    }

    /// Get number of running workers
    pub fn running_count(&self) -> usize {
        self.workers.values().filter(|w| w.is_running()).count()
    }

    /// Get total number of workers
    pub fn total_count(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for WorkerSpawner {
    fn drop(&mut self) {
        // Best effort cleanup
        for worker in self.workers.values_mut() {
            let _ = worker.kill();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_status() {
        let status = WorkerStatus::Running;
        assert_eq!(status, WorkerStatus::Running);
    }
}
