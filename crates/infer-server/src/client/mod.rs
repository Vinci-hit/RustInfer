//! 推理客户端抽象层
//!
//! 定义 `InferClient` trait，使 HTTP handler 与底层通信解耦。
//! 当前实现: ZmqClient（通过 ZMQ 与 Scheduler 通信）

pub mod zmq_client;

pub use zmq_client::ZmqClient;

use anyhow::Result;
use infer_protocol::{InferenceRequest, InferenceResponse, StreamChunk};
use tokio::sync::mpsc;

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

    /// 流式推理：发送请求，返回 chunk 接收通道
    ///
    /// Scheduler 每生成一个 token 就通过 ZMQ 回传一个 StreamChunk，
    /// 收到 ChunkType::Done 后通道关闭。
    fn infer_stream(
        &self,
        req: InferenceRequest,
    ) -> impl std::future::Future<Output = Result<mpsc::Receiver<StreamChunk>>> + Send;
}
