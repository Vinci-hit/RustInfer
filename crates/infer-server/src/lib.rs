//! RustInfer Server Library
//!
//! 提供 OpenAI-compatible HTTP API for RustInfer.

pub mod config;
pub mod state;
pub mod processor;
pub mod backend;
pub mod http;

// Re-exports
pub use config::ServerConfig;
pub use state::AppState;
pub use processor::{TokenizerWrapper, StreamDecoder};
