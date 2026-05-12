use anyhow::Result;
use crate::state::metrics::SystemMetrics;
use super::types::*;

#[derive(Clone)]
pub struct ApiClient {
    base_url: String,
    client: reqwest::Client,
}

impl ApiClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn default() -> Self {
        Self::new("http://localhost:8000")
    }

    /// 非流式 chat completion
    pub async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(response)
    }

    /// 流式 chat completion — 返回 Response 供逐行读取 SSE
    pub async fn chat_completion_stream(&self, request: ChatRequest) -> Result<reqwest::Response> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?;
        Ok(response)
    }

    /// 解析一行 SSE data
    pub fn parse_sse_line(line: &str) -> Option<StreamChunk> {
        let line = line.trim();
        if line.is_empty() || line == "data: [DONE]" {
            return None;
        }
        if let Some(data) = line.strip_prefix("data: ") {
            serde_json::from_str(data).ok()
        } else {
            None
        }
    }

    /// 获取可用模型列表
    pub async fn list_models(&self) -> Result<Vec<ModelObject>> {
        let url = format!("{}/v1/models", self.base_url);
        let resp: ModelListResponse = self.client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(resp.data)
    }

    /// 获取系统 metrics
    pub async fn get_metrics(&self) -> Result<SystemMetrics> {
        let url = format!("{}/metrics", self.base_url);
        let response = self.client
            .get(&url)
            .send()
            .await?
            .json()
            .await?;
        Ok(response)
    }

    /// 健康检查
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        self.client.get(&url).send().await.is_ok()
    }
}
