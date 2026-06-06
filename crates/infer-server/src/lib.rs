//! RustInfer HTTP API Server
//!
//! 提供 OpenAI 兼容的 HTTP API，通过 ZMQ 与 Scheduler 通信。

pub mod api;
pub mod chat;
pub mod client;
pub mod config;
pub mod error;
pub mod middleware;
pub mod router;
pub mod state;

// Re-exports
pub use client::ZmqClient;
pub use state::{AppState, SharedState};
