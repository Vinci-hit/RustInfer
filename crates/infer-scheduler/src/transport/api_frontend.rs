//! Transport Module - API Frontend
//!
//! 负责接收来自 HTTP Server 的推理请求

use infer_protocol::{InferRequest, SchedulerCommand};
use tokio::sync::mpsc;

/// 前端接收器
///
/// 从 HTTP Server 接收新的推理请求
pub struct FrontendReceiver {
    /// 接收 channel
    rx: mpsc::UnboundedReceiver<SchedulerCommand>,
}

impl FrontendReceiver {
    /// 创建新的 FrontendReceiver
    pub fn new(rx: mpsc::UnboundedReceiver<SchedulerCommand>) -> Self {
        Self { rx }
    }

    /// 非阻塞接收一个请求
    ///
    /// 返回 None 如果通道为空
    pub fn try_recv(&mut self) -> Option<InferRequest> {
        match self.rx.try_recv() {
            Ok(SchedulerCommand::AddRequest(req)) => Some(req),
            Ok(SchedulerCommand::AbortRequest(request_id)) => {
                // TODO: 处理取消请求
                tracing::warn!("Abort request not implemented: {}", request_id);
                None
            }
            Err(mpsc::error::TryRecvError::Empty) => None,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                tracing::error!("Frontend channel disconnected");
                None
            }
        }
    }

    /// 批量接收所有待处理的请求（非阻塞）
    pub fn try_recv_all(&mut self) -> Vec<InferRequest> {
        let mut requests = Vec::new();
        while let Some(req) = self.try_recv() {
            requests.push(req);
        }
        requests
    }

    /// 阻塞等待下一个请求
    pub async fn recv(&mut self) -> Option<InferRequest> {
        match self.rx.recv().await {
            Some(SchedulerCommand::AddRequest(req)) => Some(req),
            Some(SchedulerCommand::AbortRequest(request_id)) => {
                tracing::warn!("Abort request not implemented: {}", request_id);
                None
            }
            None => None,
        }
    }
}

/// 创建前端通信 channel
///
/// 返回 (sender, receiver)
pub fn create_frontend_channel() -> (
    mpsc::UnboundedSender<SchedulerCommand>,
    FrontendReceiver,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    (tx, FrontendReceiver::new(rx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_protocol::{SamplingParams, StoppingCriteria};

    fn create_test_request(id: &str) -> InferRequest {
        InferRequest {
            request_id: id.to_string(),
            input_token_ids: vec![1, 2, 3],
            sampling_params: SamplingParams {
                temperature: 1.0,
                top_p: 0.9,
                top_k: 50,
                repetition_penalty: 1.0,
                frequency_penalty: 0.0,
                seed: None,
            },
            stopping_criteria: StoppingCriteria {
                stop_token_ids: vec![2],
                max_new_tokens: 100,
                ignore_eos: false,
            },
            adapter_id: None,
        }
    }

    #[test]
    fn test_frontend_receiver_try_recv() {
        let (tx, mut rx) = create_frontend_channel();

        // 发送请求
        let req = create_test_request("req1");
        tx.send(SchedulerCommand::AddRequest(req)).unwrap();

        // 接收请求
        let received = rx.try_recv();
        assert!(received.is_some());
        assert_eq!(received.unwrap().request_id, "req1");

        // 通道为空
        let empty = rx.try_recv();
        assert!(empty.is_none());
    }

    #[test]
    fn test_frontend_receiver_try_recv_all() {
        let (tx, mut rx) = create_frontend_channel();

        // 发送多个请求
        for i in 0..5 {
            let req = create_test_request(&format!("req{}", i));
            tx.send(SchedulerCommand::AddRequest(req)).unwrap();
        }

        // 批量接收
        let requests = rx.try_recv_all();
        assert_eq!(requests.len(), 5);
    }

    #[tokio::test]
    async fn test_frontend_receiver_async() {
        let (tx, mut rx) = create_frontend_channel();

        // 异步发送
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            let req = create_test_request("req1");
            tx.send(SchedulerCommand::AddRequest(req)).unwrap();
        });

        // 异步接收（会阻塞等待）
        let received = rx.recv().await;
        assert!(received.is_some());
        assert_eq!(received.unwrap().request_id, "req1");
    }
}
