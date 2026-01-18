use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::state::metrics::SystemMetrics;

#[derive(Clone)]
pub struct ApiClient {
    base_url: String,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: ChatMessage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub completion_tokens: u32,
    pub performance: Option<Performance>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Performance {
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub tokens_per_second: f64,
}

impl ApiClient {
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:8000".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .json()
            .await?;
        Ok(response)
    }

    pub async fn get_system_metrics(&self) -> Result<SystemMetrics> {
        let url = format!("{}/v1/metrics", self.base_url);
        let response = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;
        Ok(response)
    }
}
