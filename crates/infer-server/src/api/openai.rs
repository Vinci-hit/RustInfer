use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::zmq_client::ZmqClient;
use crate::chat::get_template;

// Request/Response Types
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,

    #[serde(default)]
    pub stream: bool,

    // Sampling parameters
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,

    // Performance metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub performance: Option<Performance>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Performance {
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub decode_iterations: usize,
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: u64,
    pub queue_ms: u64,
    pub batch_size: usize,
}

fn default_max_tokens() -> Option<usize> {
    Some(512)
}

// ============================================================================
// Handlers (ZMQ模式)
// ============================================================================

/// Chat completions handler (通过ZMQ连接到Engine)
#[axum::debug_handler]
pub async fn chat_completions(
    State(zmq_client): State<Arc<ZmqClient>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    let resp = chat_completions_impl(zmq_client, req).await?;
    Ok(Json(resp).into_response())
}

async fn chat_completions_impl(
    zmq_client: Arc<ZmqClient>,
    req: ChatCompletionRequest,
) -> anyhow::Result<ChatCompletionResponse> {
    tracing::info!("Received chat completion request: stream={}", req.stream);

    if req.stream {
        anyhow::bail!("Streaming not yet implemented");
    }

    // 应用chat template
    let template = get_template(&req.model);
    let prompt = template.apply(&req.messages)?;

    // 构造Engine请求
    let engine_req = infer_protocol::InferenceRequest {
        request_id: uuid::Uuid::new_v4().to_string(),
        prompt,
        max_tokens: req.max_tokens.unwrap_or(512),
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.9),
        top_k: req.top_k.unwrap_or(50),
        stop_sequences: req.stop.unwrap_or_default(),
        stream: false,
        priority: 0,
    };

    tracing::debug!("Sending request to engine: {}", engine_req.request_id);

    // 发送到Engine (通过ZMQ)
    let engine_resp = zmq_client.send_request(engine_req).await?;

    // 检查错误
    if let infer_protocol::ResponseStatus::Error = engine_resp.status {
        anyhow::bail!(
            "Engine error: {}",
            engine_resp.error.unwrap_or_else(|| "Unknown error".to_string())
        );
    }

    // 转换为OpenAI格式
    let openai_resp = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: engine_resp.text.unwrap_or_default(),
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 0, // TODO: 从tokenizer计算
            completion_tokens: engine_resp.num_tokens as u32,
            total_tokens: engine_resp.num_tokens as u32,
            performance: Some(Performance {
                prefill_ms: engine_resp.metrics.prefill_ms,
                decode_ms: engine_resp.metrics.decode_ms,
                decode_iterations: engine_resp.metrics.decode_iterations,
                tokens_per_second: engine_resp.metrics.tokens_per_second,
                time_to_first_token_ms: engine_resp.metrics.prefill_ms,
                queue_ms: engine_resp.metrics.queue_ms,
                batch_size: engine_resp.metrics.batch_size,
            }),
        },
    };

    tracing::info!(
        "Request completed: {} tokens in {:.1}ms ({:.1} tok/s)",
        engine_resp.num_tokens,
        engine_resp.metrics.prefill_ms + engine_resp.metrics.decode_ms,
        engine_resp.metrics.tokens_per_second
    );

    Ok(openai_resp)
}

pub async fn list_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "llama3",
                "object": "model",
                "owned_by": "rustinfer"
            }
        ]
    }))
}

// Error handling
#[derive(Debug)]
pub struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        tracing::error!("Request failed: {:?}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": self.0.to_string(),
                    "type": "internal_error"
                }
            })),
        )
            .into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
