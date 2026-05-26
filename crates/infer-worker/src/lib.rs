//! # `infer-worker` — GPU Inference Runtime
//!
//! Internal architecture follows DDD (Domain-Driven Design):
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │ process/          进程入口 (main, ZMQ)                   │
//! ├─────────────────────────────────────────────────────────┤
//! │ app/              应用层 (Runner, Scheduler, Graph)       │
//! ├─────────────────────────────────────────────────────────┤
//! │ models/           具体模型 (Qwen3, Llama3)               │
//! ├─────────────────────────────────────────────────────────┤
//! │ domain/           域层 — 纯的，零 FFI，零 I/O             │
//! │   types, tensor, ports, ops, model trait, runtime        │
//! ├─────────────────────────────────────────────────────────┤
//! │ infra/            基础设施 — 实现 domain 的 trait          │
//! │   cuda/, cpu/, buffer, io                                │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! **依赖方向**: domain ← infra ← app ← models ← process
//! domain 不 `use` infra 的任何东西（通过 trait 反转依赖）。

pub mod domain;
pub mod infra;
pub mod app;
pub mod models;
pub mod process;
