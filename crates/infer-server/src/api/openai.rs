use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, Sse},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::inference::InferenceEngine;

// Request/Response Types
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,

    #[serde(default)]
    pub stream: bool,

    // Future parameters (not implemented yet)
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
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
    pub time_to_first_token_ms: u64,  // Same as prefill_ms
}

#[derive(Debug, Serialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,

    // Usage with performance metrics (included in final chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

fn default_max_tokens() -> Option<usize> {
    Some(512)
}

// Handlers
pub async fn chat_completions(
    State(engine): State<Arc<Mutex<InferenceEngine>>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    tracing::info!("Received chat completion request: stream={}", req.stream);

    if req.stream {
        // Streaming response with SSE
        let stream = engine.lock().await.generate_stream(req).await?;
        Ok(Sse::new(stream).into_response())
    } else {
        // Non-streaming response
        let response = engine.lock().await.generate(req).await?;
        Ok(Json(response).into_response())
    }
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
    fn into_response(self) -> Response {
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
