//! Transport Module - 传输层
//!
//! 负责与 Worker 和 Frontend 的通信

pub mod worker_proxy;
pub mod api_frontend;

pub use worker_proxy::WorkerProxy;
pub use api_frontend::{FrontendReceiver, create_frontend_channel};
