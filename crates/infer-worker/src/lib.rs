//! # `infer-worker` — GPU Inference Runtime
//!
//! Internal architecture follows DDD (Domain-Driven Design):
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │ application/      应用层 (ModelRunner, GraphRunner,       │
//! │                   ServeLoop, SubScheduler)               │
//! ├─────────────────────────────────────────────────────────┤
//! │ models/           具体模型 (Qwen3, Llama3, Diffusion)    │
//! ├─────────────────────────────────────────────────────────┤
//! │ domain/           域层 — 纯的，零 FFI，零 I/O             │
//! │   types, tensor, ports, ops, model trait, runtime        │
//! ├─────────────────────────────────────────────────────────┤
//! │ infrastructure/   基础设施 — 实现 domain 的 trait          │
//! │   cuda/, cpu/, io/, transport/                           │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! **依赖方向**: domain ← infrastructure ← models ← application
//! domain 不 `use` infrastructure 的任何东西（通过 trait 反转依赖）。

pub mod domain;
pub mod infrastructure;
pub mod application;
pub mod models;
