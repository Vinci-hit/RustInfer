//! # `infer-worker` — GPU Inference Runtime
//!
//! Internal architecture follows DDD (Domain-Driven Design):
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │ application/      应用层 (Runtime, DecodeEngine,          │
//! │                   ServeLoop)                             │
//! ├─────────────────────────────────────────────────────────┤
//! │ models/           具体模型 (Qwen3, Llama3)               │
//! ├─────────────────────────────────────────────────────────┤
//! │ domain/           域层 — 纯的，零 FFI，零 I/O             │
//! │   model/cache contracts, step requests, KV allocation   │
//! ├─────────────────────────────────────────────────────────┤
//! │ infrastructure/   基础设施 — I/O 与通信适配                │
//! │   io/, transport/, backend re-exports                   │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! Shared tensors, storage, execution types, and backend ports are imported
//! directly from `infer_core`. The domain and model layers depend on those
//! contracts; `infer-backend-cpu` and `infer-backend-cuda` implement them.
//! The application layer assembles models, backends, and I/O into the worker.

pub mod application;
pub mod components;
pub mod domain;
pub use infer_core::env_flags;
pub mod infrastructure;
pub mod models;
