use anyhow::Result;
use infer_protocol::{InferenceRequest, InferenceResponse, EngineRequest, EngineResponse, EngineMetrics};
use std::collections::HashMap;
use std::thread;
use tokio::sync::oneshot;

/// ZMQ客户端 - API Server用于与Engine通信
pub struct ZmqClient {
    request_tx: std::sync::mpsc::Sender<(EngineRequest, oneshot::Sender<EngineResponse>)>,
}

impl ZmqClient {
    /// 创建新的ZMQ客户端
    pub async fn new(endpoint: &str) -> Result<Self> {
        let endpoint = endpoint.to_string();
        let (request_tx, request_rx) = std::sync::mpsc::channel::<(EngineRequest, oneshot::Sender<EngineResponse>)>();

        // 在专用线程中运行ZMQ操作
        thread::spawn(move || {
            if let Err(e) = Self::zmq_thread(endpoint, request_rx) {
                tracing::error!("ZMQ thread error: {:?}", e);
            }
        });

        Ok(Self { request_tx })
    }

    /// ZMQ专用线程 - 处理所有ZMQ socket操作
    fn zmq_thread(
        endpoint: String,
        request_rx: std::sync::mpsc::Receiver<(EngineRequest, oneshot::Sender<EngineResponse>)>,
    ) -> Result<()> {
        let context = zmq::Context::new();
        let socket = context.socket(zmq::DEALER)?;
        socket.connect(&endpoint)?;

        tracing::info!("✅ ZMQ client connected to {}", endpoint);

        let mut pending_requests: HashMap<String, oneshot::Sender<EngineResponse>> = HashMap::new();

        socket.set_rcvtimeo(10)?;

        loop {
            // 处理新请求
            while let Ok((request, response_tx)) = request_rx.try_recv() {
                // For MetricsQuery, use a special ID
                let request_id = match &request {
                    EngineRequest::Inference(ireq) => ireq.request_id.clone(),
                    EngineRequest::MetricsQuery => "__metrics__".to_string(),
                };

                pending_requests.insert(request_id.clone(), response_tx);

                // 发送请求
                let data = match rmp_serde::to_vec(&request) {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::error!("Failed to serialize request: {:?}", e);
                        continue;
                    }
                };

                if let Err(e) = socket.send(&b""[..], zmq::SNDMORE) {
                    tracing::error!("Failed to send empty frame: {:?}", e);
                    continue;
                }
                if let Err(e) = socket.send(&data, 0) {
                    tracing::error!("Failed to send request: {:?}", e);
                    continue;
                }

                tracing::debug!("Sent request: {}", request_id);
            }

            // 接收响应
            match socket.recv_bytes(0) {
                Ok(_empty) => {
                    // 读取实际数据
                    match socket.recv_bytes(0) {
                        Ok(data) => {
                            // 解析响应
                            match rmp_serde::from_slice::<EngineResponse>(&data) {
                                Ok(response) => {
                                    let response_id = match &response {
                                        EngineResponse::Inference(iresp) => iresp.request_id.clone(),
                                        EngineResponse::Metrics(_) => "__metrics__".to_string(),
                                    };

                                    tracing::debug!("Received response: {}", response_id);

                                    if let Some(tx) = pending_requests.remove(&response_id) {
                                        if tx.send(response).is_err() {
                                            tracing::warn!("Failed to send response to handler (channel closed)");
                                        }
                                    } else {
                                        tracing::warn!("Received response for unknown request: {}", response_id);
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Failed to deserialize response: {:?}", e);
                                }
                            }
                        }
                        Err(zmq::Error::EAGAIN) => {
                            // Timeout, continue loop
                        }
                        Err(e) => {
                            tracing::error!("Failed to recv data: {:?}", e);
                        }
                    }
                }
                Err(zmq::Error::EAGAIN) => {
                    // Timeout, continue loop
                }
                Err(e) => {
                    tracing::error!("Failed to recv empty frame: {:?}", e);
                }
            }
        }
    }

    /// 发送推理请求并等待响应
    pub async fn send_request(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let (tx, rx) = oneshot::channel();

        self.request_tx.send((EngineRequest::Inference(request), tx))?;

        // 等待响应 (带超时)
        match tokio::time::timeout(tokio::time::Duration::from_secs(30), rx).await {
            Ok(Ok(EngineResponse::Inference(response))) => Ok(response),
            Ok(Ok(EngineResponse::Metrics(_))) => Err(anyhow::anyhow!("Unexpected metrics response")),
            Ok(Err(_)) => Err(anyhow::anyhow!("Response channel closed")),
            Err(_) => Err(anyhow::anyhow!("Request timeout after 30s")),
        }
    }

    /// 获取引擎和缓存指标
    pub async fn get_metrics(&self) -> Result<EngineMetrics> {
        let (tx, rx) = oneshot::channel();

        self.request_tx.send((EngineRequest::MetricsQuery, tx))?;

        // 等待响应 (带超时)
        match tokio::time::timeout(tokio::time::Duration::from_secs(5), rx).await {
            Ok(Ok(EngineResponse::Metrics(metrics))) => Ok(metrics),
            Ok(Ok(EngineResponse::Inference(_))) => Err(anyhow::anyhow!("Unexpected inference response")),
            Ok(Err(_)) => Err(anyhow::anyhow!("Response channel closed")),
            Err(_) => Err(anyhow::anyhow!("Metrics request timeout after 5s")),
        }
    }
}
