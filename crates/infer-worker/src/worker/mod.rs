//! Worker crate 的 runner 子系统。
//!
//! 本轮只保留：
//! - `batch_workspace`：GPU 资源池（所有 device tensor 地址稳定）
//! - `runner`：单进程 Runner，常驻线程 + SyncFlags 握手
//! - `control_client`：Worker 控制面客户端
//! - `sub_scheduler`：Worker 内部二级调度器，直接使用 ModelRunner API
//!
//! 旧的 `shared_buffers / server / runner_dummy` 老 Runner API 已删除。
pub mod batch_workspace;
pub mod control_client;
pub mod diffusion_server;
pub mod runner;
pub mod sub_scheduler;

pub use batch_workspace::BatchWorkspace;
pub use runner::ModelRunner;
pub use diffusion_server::DiffusionWorkerServer;
pub use runner::{StepMeta, SyncFlags, WorkerBatchMeta};
pub use sub_scheduler::SubScheduler;
