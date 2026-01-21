//! Worker Auto-Spawner Module
//!
//! Provides automatic launching and monitoring of multiple Worker processes.

pub mod config;
pub mod device;
pub mod spawner;

pub use config::{RestartPolicy, SpawnerConfig, SpawnerArgs, WorkerPlan};
pub use device::{AvailableDevice, CudaDeviceInfo, DeviceConstraints, DeviceDiscovery};
pub use spawner::{WorkerProcess, WorkerSpawner, WorkerStatus};
