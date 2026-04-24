use anyhow::Result;
use infer_protocol::{InferenceRequest, InferenceResponse};
use crate::engine::InferenceEngine;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use tokio::sync::Mutex;

/// ZMQ服务器 - 接收API Server的请求并返回响应
pub async fn run(engine: InferenceEngine, endpoint: &str) -> Result<()> {
    tracing::info!("Starting ZMQ server...");

    // 共享状态
    let engine_arc = Arc::new(Mutex::new(engine));
    let identity_map: Arc<Mutex<HashMap<String, Vec<u8>>>> = Arc::new(Mutex::new(HashMap::new()));

    // 创建channel用于ZMQ线程与主线程通信
    let (incoming_tx, mut incoming_rx) = tokio::sync::mpsc::unbounded_channel::<(Vec<u8>, InferenceRequest)>();
    let (outgoing_tx, outgoing_rx) = std::sync::mpsc::channel::<(Vec<u8>, InferenceResponse)>();

    // 启动ZMQ专用线程
    let endpoint = endpoint.to_string();
    thread::spawn(move || {
        if let Err(e) = zmq_thread(endpoint, incoming_tx, outgoing_rx) {
            tracing::error!("ZMQ thread error: {:?}", e);
        }
    });

    // 启动后台调度线程
    let scheduler_engine = engine_arc.clone();
    let scheduler_identity_map = identity_map.clone();
    let scheduler_outgoing_tx = outgoing_tx.clone();

    tokio::spawn(async move {
        scheduler_loop(scheduler_engine, scheduler_identity_map, scheduler_outgoing_tx).await;
    });

    // 主接收循环 - 处理来自ZMQ线程的请求
    while let Some((identity, request)) = incoming_rx.recv().await {
        tracing::info!("Received request: {}", request.request_id);

        // 保存identity映射
        {
            let mut map = identity_map.lock().await;
            map.insert(request.request_id.clone(), identity.clone());
        }

        // 加入队列
        {
            let engine = engine_arc.lock().await;
            if let Err(e) = engine.enqueue_request(request.clone()).await {
                tracing::error!("Failed to enqueue request: {:?}", e);

                // 发送错误响应
                let error_resp = InferenceResponse {
                    request_id: request.request_id.clone(),
                    status: infer_protocol::ResponseStatus::Error,
                    text: None,
                    tokens: None,
                    num_tokens: 0,
                    error: Some(e.to_string()),
                    metrics: Default::default(),
                };

                if let Err(e) = outgoing_tx.send((identity.clone(), error_resp)) {
                    tracing::error!("Failed to send error response to ZMQ thread: {:?}", e);
                }

                // 清理identity映射
                let mut map = identity_map.lock().await;
                map.remove(&request.request_id);
            }
        }
    }

    Ok(())
}

/// ZMQ专用线程 - 处理所有socket操作
fn zmq_thread(
    endpoint: String,
    incoming_tx: tokio::sync::mpsc::UnboundedSender<(Vec<u8>, InferenceRequest)>,
    outgoing_rx: std::sync::mpsc::Receiver<(Vec<u8>, InferenceResponse)>,
) -> Result<()> {
    let context = zmq::Context::new();
    let socket = context.socket(zmq::ROUTER)?;
    socket.bind(&endpoint)?;

    tracing::info!("✅ ZMQ server listening on {}", endpoint);

    // 设置非阻塞模式 (降低超时以减少延迟: 100ms -> 10ms)
    socket.set_rcvtimeo(10)?;

    loop {
        // 接收请求
        match socket.recv_bytes(0) {
            Ok(identity) => {
                let _empty = socket.recv_bytes(0)?;
                let data = socket.recv_bytes(0)?;

                tracing::debug!("Received request from client ({}bytes)", data.len());

                // 解析请求
                match rmp_serde::from_slice::<InferenceRequest>(&data) {
                    Ok(request) => {
                        if incoming_tx.send((identity, request)).is_err() {
                            tracing::error!("Failed to send request to main loop");
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to deserialize request: {:?}", e);
                    }
                }
            }
            Err(zmq::Error::EAGAIN) => {
                // Timeout, continue to check for outgoing messages
            }
            Err(e) => {
                tracing::error!("Failed to recv request: {:?}", e);
            }
        }

        // 发送响应
        while let Ok((identity, response)) = outgoing_rx.try_recv() {
            let data = match rmp_serde::to_vec(&response) {
                Ok(d) => d,
                Err(e) => {
                    tracing::error!("Failed to serialize response: {:?}", e);
                    continue;
                }
            };

            if let Err(e) = socket.send(&identity, zmq::SNDMORE) {
                tracing::error!("Failed to send identity: {:?}", e);
                continue;
            }
            if let Err(e) = socket.send(&b""[..], zmq::SNDMORE) {
                tracing::error!("Failed to send empty frame: {:?}", e);
                continue;
            }
            if let Err(e) = socket.send(&data, 0) {
                tracing::error!("Failed to send response data: {:?}", e);
                continue;
            }

            tracing::debug!("Sent response for request: {}", response.request_id);
        }
    }
}

/// 调度循环 - 定期批量处理请求
async fn scheduler_loop(
    engine: Arc<Mutex<InferenceEngine>>,
    identity_map: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    outgoing_tx: std::sync::mpsc::Sender<(Vec<u8>, InferenceResponse)>,
) {
    // 降低调度间隔以减少延迟 (10ms -> 1ms)
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(1));

    loop {
        interval.tick().await;

        let responses = {
            let mut engine = engine.lock().await;
            let (queue_len, _) = engine.queue_status().await;

            if queue_len == 0 {
                continue;
            }

            tracing::debug!("Scheduler tick: {} requests in queue", queue_len);
            engine.process_batch().await
        };

        if responses.is_empty() {
            continue;
        }

        tracing::debug!("Processed {} responses, sending back to clients", responses.len());

        // 发送响应到ZMQ线程
        let mut map = identity_map.lock().await;

        for response in responses {
            // 查找对应的客户端identity
            if let Some(identity) = map.remove(&response.request_id) {
                if let Err(e) = outgoing_tx.send((identity, response.clone())) {
                    tracing::error!("Failed to send response to ZMQ thread: {:?}", e);
                } else {
                    tracing::info!(
                        "✅ Response queued for {}: {} tokens in {:.1}ms",
                        response.request_id,
                        response.num_tokens,
                        response.metrics.prefill_ms + response.metrics.decode_ms
                    );
                }
            } else {
                tracing::warn!("No identity found for request: {}", response.request_id);
            }
        }
    }
}
