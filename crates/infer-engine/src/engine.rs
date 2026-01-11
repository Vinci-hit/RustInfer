use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;
use infer_protocol::{InferenceRequest, InferenceResponse, InferenceMetrics, ResponseStatus};
use anyhow::Result;

/// 推理引擎 - 负责批量调度和推理
pub struct InferenceEngine {
    /// Llama3模型实例
    model: infer_core::model::llama3::Llama3,

    /// 请求队列
    request_queue: Arc<Mutex<VecDeque<QueuedRequest>>>,

    /// 配置
    config: EngineConfig,
}

/// 队列中的请求
pub struct QueuedRequest {
    pub request: InferenceRequest,
    pub enqueue_time: std::time::Instant,
}

/// 引擎配置
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub max_batch_size: usize,
    pub max_queue_size: usize,
    pub schedule_interval_ms: u64,
}

impl InferenceEngine {
    pub fn new(
        model: infer_core::model::llama3::Llama3,
        max_batch_size: usize,
        max_queue_size: usize,
        schedule_interval_ms: u64,
    ) -> Self {
        Self {
            model,
            request_queue: Arc::new(Mutex::new(VecDeque::new())),
            config: EngineConfig {
                max_batch_size,
                max_queue_size,
                schedule_interval_ms,
            },
        }
    }

    /// 添加请求到队列
    pub async fn enqueue_request(&self, request: InferenceRequest) -> Result<()> {
        let mut queue = self.request_queue.lock().await;

        if queue.len() >= self.config.max_queue_size {
            anyhow::bail!("Request queue is full");
        }

        queue.push_back(QueuedRequest {
            request,
            enqueue_time: std::time::Instant::now(),
        });

        tracing::debug!("Request enqueued, queue size: {}", queue.len());
        Ok(())
    }

    /// 批量处理请求 (核心调度逻辑)
    pub async fn process_batch(&mut self) -> Vec<InferenceResponse> {
        let mut queue = self.request_queue.lock().await;

        if queue.is_empty() {
            return Vec::new();
        }

        // 1. 从队列取出batch_size个请求
        let batch_size = std::cmp::min(self.config.max_batch_size, queue.len());
        let mut batch_requests: Vec<QueuedRequest> = Vec::new();

        for _ in 0..batch_size {
            if let Some(req) = queue.pop_front() {
                batch_requests.push(req);
            }
        }

        drop(queue); // 释放锁，避免阻塞新请求入队

        tracing::info!("Processing batch of {} requests", batch_requests.len());

        // 2. 处理batch (目前是逐个处理，未来可以真正batch)
        let mut responses = Vec::new();

        for queued_req in batch_requests {
            let req = queued_req.request;
            let queue_ms = queued_req.enqueue_time.elapsed().as_millis() as u64;

            let start = std::time::Instant::now();

            // 调用模型推理
            let result = self.model.generate(
                &req.prompt,
                req.max_tokens,
                false, // 非verbose模式
            );

            let response = match result {
                Ok((text, num_tokens, prefill_ms, decode_ms, decode_iterations)) => {
                    let total_time_ms = start.elapsed().as_millis() as u64;
                    let tokens_per_second = if total_time_ms > 0 {
                        (num_tokens as f64 / total_time_ms as f64) * 1000.0
                    } else {
                        0.0
                    };

                    tracing::info!(
                        "Request {} completed: {} tokens in {}ms ({:.1} tok/s)",
                        req.request_id, num_tokens, total_time_ms, tokens_per_second
                    );

                    InferenceResponse {
                        request_id: req.request_id,
                        status: ResponseStatus::Success,
                        text: Some(text),
                        tokens: None,
                        num_tokens: num_tokens,
                        error: None,
                        metrics: InferenceMetrics {
                            prefill_ms,
                            decode_ms,
                            queue_ms,
                            batch_size: 1, // TODO: 真正batch后这里会>1
                            tokens_per_second,
                            decode_iterations,
                        },
                    }
                }
                Err(e) => {
                    tracing::error!("Request {} failed: {:?}", req.request_id, e);
                    InferenceResponse {
                        request_id: req.request_id,
                        status: ResponseStatus::Error,
                        text: None,
                        tokens: None,
                        num_tokens: 0,
                        error: Some(e.to_string()),
                        metrics: InferenceMetrics {
                            queue_ms,
                            ..Default::default()
                        },
                    }
                }
            };

            responses.push(response);
        }

        responses
    }

    /// 获取队列状态
    pub async fn queue_status(&self) -> (usize, usize) {
        let queue = self.request_queue.lock().await;
        (queue.len(), self.config.max_queue_size)
    }
}
