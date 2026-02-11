//! Worker Auto-Spawner Module
//!
//! Provides automatic launching and monitoring of multiple Worker processes.

pub mod config;
pub mod device;
pub mod spawner;
pub mod process_manager;
pub mod logger;
pub mod tui;

pub use config::{RestartPolicy, SpawnerConfig, SpawnerArgs, WorkerPlan};
pub use device::{AvailableDevice, CudaDeviceInfo, DeviceConstraints, DeviceDiscovery};
pub use spawner::{WorkerProcess, WorkerSpawner, WorkerStatus};
pub use process_manager::ProcessManager;
pub use logger::AggregateLogger;
pub use tui::{Tui, TuiApp, WorkerDisplayInfo};
