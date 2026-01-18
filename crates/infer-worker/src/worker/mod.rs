//! Worker Module - Inference Worker for LLM Serving
//!
//! This module provides the core inference worker that manages:
//! - Device resources (GPU/CPU)
//! - Model instance
//! - KV Cache memory pool
//! - Sampler for token generation
//! - CUDA configuration (streams, handles, etc.)
//!
//! # Architecture
//!
//! ```text
//! +------------------+
//! |     Worker       |  <- Main entry point for inference
//! +------------------+
//!          |
//!    +-----+-----+-----+-----+
//!    |     |     |     |     |
//!    v     v     v     v     v
//! Device Model KVCache Sampler CudaConfig
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let worker_config = WorkerConfig::new(0, model_path)?;
//! let worker = Worker::new(worker_config)?;
//! let output = worker.generate("Hello", 100)?;
//! ```

mod worker;
mod device_info;
mod config;

#[cfg(feature = "server")]
pub mod server;

pub use worker::{Worker, WorkerState, MemoryStats, PerformanceStats};
pub use device_info::DeviceInfo;
pub use config::WorkerConfig;

#[cfg(feature = "server")]
pub use server::WorkerServer;
