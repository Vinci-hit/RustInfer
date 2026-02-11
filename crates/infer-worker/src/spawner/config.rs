//! Spawner Configuration Module
//!
//! Configuration structures for the Worker Spawner.

use std::collections::HashMap;
use std::time::Duration;

use clap::Parser;
use serde::{Deserialize, Serialize};

use super::device::{AvailableDevice, DeviceConstraints};

/// Spawner configuration
///
/// Contains all settings needed to launch and manage multiple Worker processes.
#[derive(Debug, Clone)]
pub struct SpawnerConfig {
    /// Scheduler ZeroMQ endpoint URL
    pub scheduler_url: String,

    /// Host ID (for multi-machine deployment)
    pub host_id: Option<String>,

    /// Node list (for multi-machine deployment)
    pub node_list: Option<HashMap<String, Vec<AvailableDevice>>>,

    /// Device constraints for filtering available devices
    pub device_constraints: DeviceConstraints,

    /// Restart policy for failed workers
    pub restart_policy: RestartPolicy,

    /// Health check interval
    pub health_check_interval: Duration,

    /// Worker startup timeout
    pub worker_startup_timeout: Duration,

    /// Whether to wait for all workers to start before returning
    pub wait_for_all: bool,

    /// Log level for workers
    pub log_level: String,

    /// Whether to enable verbose output
    pub verbose: bool,

    /// Log directory for worker logs
    pub log_dir: Option<std::path::PathBuf>,

    /// Enable TUI monitoring dashboard
    pub enable_tui: bool,
}

impl Default for SpawnerConfig {
    fn default() -> Self {
        Self {
            scheduler_url: "ipc:///tmp/rustinfer-scheduler.ipc".to_string(),
            host_id: None,
            node_list: None,
            device_constraints: DeviceConstraints::default(),
            restart_policy: RestartPolicy::Never,
            health_check_interval: Duration::from_secs(5),
            worker_startup_timeout: Duration::from_secs(30),
            wait_for_all: false,
            log_level: "info".to_string(),
            verbose: false,
            log_dir: Some(std::path::PathBuf::from("./logs")),
            enable_tui: true,
        }
    }
}

impl SpawnerConfig {
    /// Create a new SpawnerConfig with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set scheduler URL
    pub fn with_scheduler_url(mut self, url: impl Into<String>) -> Self {
        self.scheduler_url = url.into();
        self
    }

    /// Set host ID
    pub fn with_host_id(mut self, host_id: impl Into<String>) -> Self {
        self.host_id = Some(host_id.into());
        self
    }

    /// Set device constraints
    pub fn with_device_constraints(mut self, constraints: DeviceConstraints) -> Self {
        self.device_constraints = constraints;
        self
    }

    /// Set restart policy
    pub fn with_restart_policy(mut self, policy: RestartPolicy) -> Self {
        self.restart_policy = policy;
        self
    }

    /// Set wait for all flag
    pub fn with_wait_for_all(mut self, wait: bool) -> Self {
        self.wait_for_all = wait;
        self
    }

    /// Set log level
    pub fn with_log_level(mut self, level: impl Into<String>) -> Self {
        self.log_level = level.into();
        self
    }

    /// Set verbose flag
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set log directory
    pub fn with_log_dir(mut self, log_dir: impl Into<std::path::PathBuf>) -> Self {
        self.log_dir = Some(log_dir.into());
        self
    }

    /// Enable/disable TUI monitoring
    pub fn with_tui(mut self, enable_tui: bool) -> Self {
        self.enable_tui = enable_tui;
        self
    }
}

/// Restart policy for failed workers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestartPolicy {
    /// Never restart workers
    Never,
    /// Always restart workers on failure
    Always,
    /// Restart workers on failure, but with a maximum retry count
    OnFailure { max_retries: u32 },
}

impl std::fmt::Display for RestartPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Never => write!(f, "never"),
            Self::Always => write!(f, "always"),
            Self::OnFailure { max_retries } => write!(f, "on-failure:{}", max_retries),
        }
    }
}

impl std::str::FromStr for RestartPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();
        match parts[0] {
            "never" => Ok(Self::Never),
            "always" => Ok(Self::Always),
            "on-failure" => {
                let max_retries = if parts.len() > 1 {
                    parts[1]
                        .parse()
                        .map_err(|_| "Invalid max_retries value".to_string())?
                } else {
                    3
                };
                Ok(Self::OnFailure { max_retries })
            }
            _ => Err(format!("Unknown restart policy: {}", s)),
        }
    }
}

/// Worker startup plan
///
/// Contains all information needed to start a single Worker process.
#[derive(Debug, Clone)]
pub struct WorkerPlan {
    /// Unique worker identifier
    pub worker_id: String,

    /// Worker rank (0-indexed)
    pub rank: usize,

    /// Total world size
    pub world_size: usize,

    /// Device to bind to
    pub device: AvailableDevice,

    /// Device ID
    pub device_id: i32,

    /// Tensor parallelism rank
    pub tp_rank: u32,

    /// Tensor parallelism world size
    pub tp_world_size: u32,

    /// Scheduler URL
    pub scheduler_url: String,

    /// Environment variables to pass to worker
    pub env_vars: HashMap<String, String>,
}

impl WorkerPlan {
    /// Create a new WorkerPlan
    pub fn new(
        rank: usize,
        world_size: usize,
        device: AvailableDevice,
        scheduler_url: String,
    ) -> Self {
        let device_id = match &device {
            AvailableDevice::Cpu => 0,
            AvailableDevice::Cuda(info) => info.device_id,
        };

        Self {
            worker_id: format!("worker-{}", rank),
            rank,
            world_size,
            device,
            device_id,
            tp_rank: rank as u32,
            tp_world_size: world_size as u32,
            scheduler_url,
            env_vars: HashMap::new(),
        }
    }

    /// Add an environment variable
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Build command line arguments for worker_main
    pub fn build_command_args(&self) -> Vec<String> {
        vec![
            "--rank".to_string(),
            self.rank.to_string(),
            "--world-size".to_string(),
            self.world_size.to_string(),
            "--scheduler-url".to_string(),
            self.scheduler_url.clone(),
        ]
    }

    /// Build environment variables for the worker process
    pub fn build_env(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();

        // Add custom env vars
        for (key, value) in &self.env_vars {
            env.insert(key.clone(), value.clone());
        }

        // Add standard env vars
        env.insert(
            "RUSTINFER_WORKER_ID".to_string(),
            self.worker_id.clone(),
        );
        env.insert(
            "RUSTINFER_WORKER_RANK".to_string(),
            self.rank.to_string(),
        );
        env.insert(
            "RUSTINFER_WORLD_SIZE".to_string(),
            self.world_size.to_string(),
        );
        env.insert(
            "RUSTINFER_DEVICE_ID".to_string(),
            self.device_id.to_string(),
        );
        env.insert(
            "RUSTINFER_SCHEDULER_URL".to_string(),
            self.scheduler_url.clone(),
        );

        // For CUDA devices, set CUDA_VISIBLE_DEVICES
        if let AvailableDevice::Cuda(info) = &self.device {
            env.insert(
                "CUDA_VISIBLE_DEVICES".to_string(),
                info.device_id.to_string(),
            );
        }

        env
    }
}

/// Command-line arguments for infer-spawner
#[derive(Parser, Debug)]
#[command(name = "infer-spawner")]
#[command(author = "RustInfer Team")]
#[command(version = "0.1.0")]
#[command(about = "RustInfer Worker Spawner - Auto-launch multiple worker processes")]
pub struct SpawnerArgs {
    /// Scheduler ZeroMQ endpoint URL
    ///
    /// Examples:
    /// - IPC: "ipc:///tmp/rustinfer-scheduler.ipc"
    /// - TCP: "tcp://localhost:5555"
    #[arg(long, default_value = "ipc:///tmp/rustinfer-scheduler.ipc")]
    pub scheduler_url: String,

    /// Automatically discover and use all available devices
    #[arg(long)]
    pub auto: bool,

    /// Specify GPU IDs to use (comma-separated, e.g., 0,1,2,3)
    #[arg(long, value_delimiter = ',')]
    pub gpu_ids: Option<Vec<i32>>,

    /// Exclude GPU IDs (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub exclude_gpu_ids: Option<Vec<i32>>,

    /// Minimum GPU memory (e.g., 8GB, 16GB)
    #[arg(long)]
    pub min_gpu_memory: Option<String>,

    /// Maximum number of workers to start
    #[arg(long)]
    pub max_workers: Option<usize>,

    /// Include CPU as a compute device
    #[arg(long)]
    pub include_cpu: bool,

    /// Restart policy: never, always, on-failure[:N]
    #[arg(long, default_value = "never")]
    pub restart_policy: String,

    /// Maximum restart attempts (only for on-failure policy)
    #[arg(long, default_value_t = 3)]
    pub max_retries: u32,

    /// Wait for all workers to start before returning
    #[arg(long)]
    pub wait_for_all: bool,

    /// Host ID (for multi-machine deployment)
    #[arg(long)]
    pub host_id: Option<String>,

    /// Node list (format: "host1=0,1,2;host2=0,1,2")
    #[arg(long)]
    pub node_list: Option<String>,

    /// Log level: error, warn, info, debug, trace
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Enable verbose output
    #[arg(long, short)]
    pub verbose: bool,
}

impl TryFrom<SpawnerArgs> for SpawnerConfig {
    type Error = String;

    fn try_from(args: SpawnerArgs) -> Result<Self, Self::Error> {
        // Parse restart policy
        let restart_policy = args
            .restart_policy
            .parse()
            .map_err(|e| format!("Invalid restart policy: {}", e))?;

        // Parse GPU memory constraint
        let min_memory_mb = args.min_gpu_memory.and_then(|s| {
            let s = s.to_lowercase();
            if s.ends_with("gb") {
                s.trim_end_matches("gb").parse::<u64>().ok().map(|v| v * 1024)
            } else if s.ends_with("mb") {
                s.trim_end_matches("mb").parse().ok()
            } else if s.ends_with("g") {
                s.trim_end_matches("g").parse::<u64>().ok().map(|v| v * 1024)
            } else if s.ends_with("m") {
                s.trim_end_matches("m").parse().ok()
            } else {
                s.parse().ok()
            }
        });

        // Build device constraints
        let device_constraints = DeviceConstraints {
            min_memory_mb,
            exclude_device_ids: args.exclude_gpu_ids.unwrap_or_default(),
            include_device_ids: args.gpu_ids,
            max_devices: args.max_workers,
            include_cpu: args.include_cpu,
        };

        // Parse node list if provided
        let node_list = if let Some(node_list_str) = args.node_list {
            Some(parse_node_list(&node_list_str)?)
        } else {
            None
        };

        Ok(Self {
            scheduler_url: args.scheduler_url,
            host_id: args.host_id,
            node_list,
            device_constraints,
            restart_policy,
            health_check_interval: Duration::from_secs(5),
            worker_startup_timeout: Duration::from_secs(30),
            wait_for_all: args.wait_for_all,
            log_level: args.log_level,
            verbose: args.verbose,
            log_dir: Some(std::path::PathBuf::from("./logs")),
            enable_tui: true,
        })
    }
}

/// Parse node list string into HashMap
///
/// Format: "host1=0,1,2;host2=0,1,2"
fn parse_node_list(s: &str) -> Result<HashMap<String, Vec<AvailableDevice>>, String> {
    let mut result = HashMap::new();

    for entry in s.split(';') {
        let parts: Vec<&str> = entry.split('=').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid node list entry: {}", entry));
        }

        let host = parts[0].trim().to_string();
        let device_ids: Vec<i32> = parts[1]
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        // For now, we just store the device IDs
        // In the future, this would query actual device info
        for id in device_ids {
            result
                .entry(host.clone())
                .or_insert_with(Vec::new)
                .push(AvailableDevice::Cuda(crate::spawner::device::CudaDeviceInfo {
                    device_id: id,
                    name: format!("Device {}", id),
                    total_memory: 0,
                    free_memory: 0,
                    compute_capability: (0, 0),
                    multiprocessor_count: 0,
                }));
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_plan_build_args() {
        let plan = WorkerPlan::new(
            0,
            4,
            AvailableDevice::Cuda(crate::spawner::device::CudaDeviceInfo {
                device_id: 0,
                name: "Test GPU".to_string(),
                total_memory: 16 * 1024 * 1024 * 1024,
                free_memory: 8 * 1024 * 1024 * 1024,
                compute_capability: (8, 0),
                multiprocessor_count: 108,
            }),
            "tcp://localhost:5555".to_string(),
        );

        let args = plan.build_command_args();
        assert_eq!(args, vec![
            "--rank", "0",
            "--world-size", "4",
            "--scheduler-url", "tcp://localhost:5555",
        ]);
    }

    #[test]
    fn test_worker_plan_build_env() {
        let plan = WorkerPlan::new(
            0,
            4,
            AvailableDevice::Cpu,
            "tcp://localhost:5555".to_string(),
        );

        let env = plan.build_env();
        assert_eq!(env.get("RUSTINFER_WORKER_ID"), Some(&"worker-0".to_string()));
        assert_eq!(env.get("RUSTINFER_WORKER_RANK"), Some(&"0".to_string()));
        assert_eq!(env.get("CUDA_VISIBLE_DEVICES"), None);
    }

    #[test]
    fn test_restart_policy_parse() {
        assert_eq!(
            "never".parse::<RestartPolicy>().unwrap(),
            RestartPolicy::Never
        );
        assert_eq!(
            "always".parse::<RestartPolicy>().unwrap(),
            RestartPolicy::Always
        );
        assert_eq!(
            "on-failure:5".parse::<RestartPolicy>().unwrap(),
            RestartPolicy::OnFailure { max_retries: 5 }
        );
    }
}
