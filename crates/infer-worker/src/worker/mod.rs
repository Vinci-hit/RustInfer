//! Worker crate 的 runner 子系统。
//!
//! 本轮只保留：
//! - `batch_workspace`：GPU 资源池（所有 device tensor 地址稳定）
//! - `runner`：单进程 Runner，常驻线程 + SyncFlags 握手
//! - `protocol`：ZMQ 消息定义（PrefillBatchCmd / StepOutput）
//! - `server_new`：新版 Server，直接使用 ModelRunner API
//!
//! 旧的 `shared_buffers / server / runner_dummy` 一套（基于 ZMQ +
//! 老 Runner API）暂时从 crate 对外接口下架。待确认新 server 稳定后删除。
pub mod batch_workspace;
pub mod control_client;
pub mod control_protocol;
pub mod protocol;
pub mod runner;
pub mod server_new;

pub use batch_workspace::BatchWorkspace;
pub use runner::ModelRunner;
pub use runner::{StepMeta, SyncFlags, WorkerBatchMeta};
pub use server_new::WorkerServer;
