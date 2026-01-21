//! Worker Module - Inference Worker for LLM Serving
//!
//! This module provides the core inference worker that manages:
//! - Device resources (GPU/CPU)
//! - Model instance (with internal Sampler)
//! - KV Cache memory pool
//! - CUDA configuration (streams, handles, etc.)
//!
//! Note: Sampler is managed internally by the Model, not by the Worker.
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
//! Device Model KVCache CudaConfig
//!        (sampler inside)
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
