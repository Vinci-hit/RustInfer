//! Worker crate 的 runner 子系统。
//!
//! 本轮只保留：
//! - `batch_workspace`：GPU 资源池（所有 device tensor 地址稳定）
//! - `runner`：单进程 Runner，常驻线程 + SyncFlags 握手
//!
//! 旧的 `shared_buffers / server / protocol / runner_dummy` 一套（基于 ZMQ +
//! 老 Runner API）暂时从 crate 对外接口下架。待 server 按新 Runner API 重写时再
//! 恢复。代码本身保留在文件里，但不 pub mod，不参与编译。
pub mod batch_workspace;
pub mod runner;

pub use batch_workspace::BatchWorkspace;
pub use runner::ModelRunner;
pub use runner::{StepMeta, SyncFlags, WorkerBatchMeta};
