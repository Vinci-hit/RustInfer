//! 推理客户端抽象层
//!
//! 定义 `InferClient` trait，使 HTTP handler 与底层通信解耦。
//! 当前实现: ZmqClient（通过 ZMQ 与 Scheduler 通信）

pub mod zmq_client;

pub use zmq_client::{StreamHandle, ZmqClient};

use anyhow::Result;
use infer_protocol::scheduler_to_server::InferenceResponse;
use infer_protocol::server_to_scheduler::InferenceRequest;

/// 推理客户端 trait
///
/// - `infer`: 非流式，等待完整响应
/// - `infer_stream`: 流式，返回 mpsc Receiver 逐 chunk 接收
pub trait InferClient: Send + Sync + 'static {
    /// 非流式推理：发送请求，等待完整响应
    fn infer(
        &self,
        req: InferenceRequest,
    ) -> impl std::future::Future<Output = Result<InferenceResponse>> + Send;

    /// 流式推理：发送请求，返回带 Drop 取消语义的 stream handle。
    fn infer_stream(
        &self,
        req: InferenceRequest,
    ) -> impl std::future::Future<Output = Result<StreamHandle>> + Send;
}
