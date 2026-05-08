use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::AppState;
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

    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub performance: Option<Performance>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Performance {
    pub total_ms: u64,
    pub tokens_per_second: f64,
}

fn default_max_tokens() -> Option<usize> {
    Some(512)
}

/// Chat completions handler
#[axum::debug_handler]
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    let resp = chat_completions_impl(state, req).await?;
    Ok(Json(resp).into_response())
}

async fn chat_completions_impl(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> anyhow::Result<ChatCompletionResponse> {
    if req.stream {
        anyhow::bail!("Streaming not yet implemented");
    }

    // 1. 应用 chat template
    let template = get_template(&req.model);
    let prompt = template.apply(&req.messages)?;

    // 2. Tokenize (在 HTTP Server 端做)
    let encoding = state.tokenizer.encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenize error: {}", e))?;
    let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let prompt_tokens = input_ids.len() as u32;

    // 3. 发给 Scheduler
    let engine_req = infer_protocol::InferenceRequest {
        request_id: uuid::Uuid::new_v4().to_string(),
        input_ids,
        max_tokens: req.max_tokens.unwrap_or(512),
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(-1),
        stream: false,
        priority: 0,
    };

    let engine_resp = state.zmq_client.send_request(engine_req).await?;

    // 4. 检查错误
    if let infer_protocol::ResponseStatus::Error = engine_resp.status {
        anyhow::bail!(
            "Scheduler error: {}",
            engine_resp.error.unwrap_or_else(|| "Unknown error".to_string())
        );
    }

    // 5. Decode output tokens → 文本 (在 HTTP Server 端做)
    let output_ids_u32: Vec<u32> = engine_resp.output_token_ids.iter()
        .map(|&id| id as u32).collect();
    let generated_text = state.tokenizer.decode(&output_ids_u32, true)
        .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
    let completion_tokens = engine_resp.output_token_ids.len() as u32;

    // 6. 转换为 OpenAI 格式
    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: generated_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            performance: Some(Performance {
                total_ms: engine_resp.metrics.total_ms,
                tokens_per_second: engine_resp.metrics.tokens_per_second,
            }),
        },
    })
}

pub async fn list_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{"id": "llama3", "object": "model", "owned_by": "rustinfer"}]
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
            Json(serde_json::json!({"error": {"message": self.0.to_string(), "type": "internal_error"}})),
        ).into_response()
    }
}

impl<E> From<E> for AppError where E: Into<anyhow::Error> {
    fn from(err: E) -> Self { Self(err.into()) }
}
