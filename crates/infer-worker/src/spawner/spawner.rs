//! Worker Spawner Module
//!
//! Handles launching and managing multiple Worker processes.

use std::collections::HashMap;
use std::process::ExitStatus;
use std::time::{Duration, Instant};
use anyhow::{Context, Result};

use super::config::{RestartPolicy, SpawnerConfig, WorkerPlan};
use super::device::{DeviceDiscovery, AvailableDevice};
use super::process_manager::ProcessManager;
use super::logger::AggregateLogger;
use super::tui::{WorkerDisplayInfo, Tui, TuiApp};

/// Worker process status
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStatus {
    Starting,
    Running,
    Stopping,
    Stopped,
    WaitingForRestart { delay_ms: u64 },
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
    pub last_restart_time: Instant,
    process_manager: Option<ProcessManager>,
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
            last_restart_time: Instant::now(),
            process_manager: None,
        }
    }

    /// Check if the worker process is still running
    pub fn is_running(&self) -> bool {
        matches!(self.status, WorkerStatus::Running) && self.process_manager.is_some()
    }

    /// Try to wait for the process to complete
    pub fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        if let Some(ref mut pm) = self.process_manager {
            pm.try_wait()
        } else {
            Ok(None)
        }
    }

    /// Kill the worker process
    pub fn kill(&mut self) -> Result<()> {
        if let Some(ref mut pm) = self.process_manager {
            pm.kill().context("Failed to kill worker process")?;
            self.status = WorkerStatus::Stopped;
        }
        Ok(())
    }

    /// Set the process manager
    pub fn set_process_manager(&mut self, pm: ProcessManager) {
        self.process_manager = Some(pm);
    }

    /// Get process ID
    pub fn pid(&self) -> Option<u32> {
        self.process_manager.as_ref().and_then(|pm| pm.id())
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }
}

/// Worker Spawner - manages multiple Worker processes
pub struct WorkerSpawner {
    config: SpawnerConfig,
    workers: HashMap<String, WorkerProcess>,
    plans: HashMap<String, WorkerPlan>,  // Store plans for restarts
    logger: AggregateLogger,
    tui: Tui,
}

impl WorkerSpawner {
    /// Create a new WorkerSpawner
    pub fn new(config: SpawnerConfig) -> Result<Self> {
        let logger = AggregateLogger::new(config.log_dir.as_deref())?;
        let tui = Tui::new();

        Ok(Self {
            config,
            workers: HashMap::new(),
            plans: HashMap::new(),
            logger,
            tui,
        })
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

        // Store plans for future restarts
        for plan in &plans {
            self.plans.insert(plan.worker_id.clone(), plan.clone());
        }

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
            plan.worker_id, plan.device_id
        );

        // Register worker in logger
        self.logger
            .register_worker(&plan.worker_id, plan.device_id as u32)?;

        // Build command
        let args = plan.build_command_args();
        let env = plan.build_env();

        // Get infer-worker binary path
        let current_exe = std::env::current_exe()
            .context("Failed to get current executable path")?;
        let exe_dir = current_exe
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Failed to get executable directory"))?;
        let exe_path = exe_dir.join("infer-worker");

        // Create process manager with logging
        let worker_id = plan.worker_id.clone();
        let logger = self.logger.clone();
        let worker_id_for_logging = worker_id.clone();

        let process_manager = ProcessManager::spawn(
            &worker_id,
            &exe_path,
            &args,
            &env,
            move |line: &str| {
                let _ = logger.info(&worker_id_for_logging, line);
            },
        )?;

        // Create worker process record
        let mut worker = WorkerProcess::new(plan);
        worker.set_process_manager(process_manager);
        worker.status = WorkerStatus::Running;
        worker.started_at = Instant::now();

        let pid = worker.pid();

        // Update TUI
        let app = self.tui.app();
        let mut app = app.lock().unwrap();

        // Check if worker already exists in TUI (e.g., from restart)
        if app.workers.iter().any(|w| w.worker_id == plan.worker_id) {
            app.update_worker(&plan.worker_id, "Running".to_string());
        } else {
            app.add_worker(WorkerDisplayInfo {
                worker_id: worker.worker_id.clone(),
                device_id: worker.device_id as u32,
                status: "Running".to_string(),
                restart_count: worker.restart_count,
                uptime: Duration::from_secs(0),
                memory_used: None,
                memory_total: None,
            });
        }

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
        let mut ready_to_restart = Vec::new();

        // Check for workers waiting to restart (delay expired)
        for (worker_id, worker) in self.workers.iter_mut() {
            if let WorkerStatus::WaitingForRestart { delay_ms } = &worker.status {
                let elapsed_ms = worker.last_restart_time.elapsed().as_millis() as u64;
                if elapsed_ms >= *delay_ms {
                    ready_to_restart.push(worker_id.clone());
                }
            }
        }

        // Perform actual restart for workers with expired delay
        for worker_id in ready_to_restart {
            self.do_restart_worker(&worker_id)?;
        }

        // Check for exited workers
        for (worker_id, worker) in self.workers.iter_mut() {
            if let Ok(Some(exit_status)) = worker.try_wait() {
                // Only handle if not already waiting for restart
                if !matches!(worker.status, WorkerStatus::WaitingForRestart { .. }) {
                    println!("[Spawner] ⚠ Worker {} exited with status: {:?}", worker_id, exit_status);
                    let _ = self.logger.warn(worker_id, &format!("Exited with status: {:?}", exit_status));

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
                                    format!("exceeded max retries ({})", max_retries)
                                );
                                failed_workers.push(worker_id.clone());
                            }
                        }
                    }
                }
            }
        }

        // Schedule restart for workers with delay
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

    /// Restart a failed worker with exponential backoff
    fn restart_worker(&mut self, worker_id: &str) -> Result<()> {
        let worker = self.workers.get_mut(worker_id)
            .ok_or_else(|| anyhow::anyhow!("Worker {} not found", worker_id))?;

        // Calculate exponential backoff delay
        let delay_ms = {
            let base_ms = 1000u64;
            let max_ms = 60000u64;
            let delay = base_ms * 2_u64.pow(worker.restart_count);
            delay.min(max_ms)
        };

        println!(
            "[Spawner] ↻ Restarting {} (attempt {}, waiting {}ms)...",
            worker_id, worker.restart_count + 1,
            delay_ms
        );

        let _ = self.logger.info(
            worker_id,
            &format!("Scheduled restart (attempt {}, delay {}ms)",
                worker.restart_count + 1, delay_ms)
        );

        worker.restart_count += 1;
        worker.last_restart_time = Instant::now();
        worker.status = WorkerStatus::WaitingForRestart { delay_ms };

        // Update TUI
        let app = self.tui.app();
        let mut app = app.lock().unwrap();
        app.update_worker(worker_id, format!("Waiting ({}/{}s)", delay_ms / 1000, 60));

        Ok(())
    }

    /// Perform actual restart after delay has expired
    fn do_restart_worker(&mut self, worker_id: &str) -> Result<()> {
        println!("[Spawner] → Starting {} now...", worker_id);

        let _ = self.logger.info(worker_id, "Restarting now");

        // Get the plan for this worker
        let plan = self.plans.get(worker_id)
            .ok_or_else(|| anyhow::anyhow!("No plan found for worker {}", worker_id))?
            .clone();

        // Kill old process if still alive
        if let Some(worker) = self.workers.get_mut(worker_id) {
            let _ = worker.kill();
        }

        // Re-spawn the worker
        self.spawn_worker(&plan)?;

        println!("[Spawner] ✓ {} restarted", worker_id);

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
