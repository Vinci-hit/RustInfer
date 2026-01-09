use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: i64,
    pub metrics: Option<MessageMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageMetrics {
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub tokens_per_second: f64,
    pub total_tokens: u32,
}

impl Message {
    pub fn user(content: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role: "user".to_string(),
            content,
            timestamp: chrono::Utc::now().timestamp(),
            metrics: None,
        }
    }

    pub fn assistant(content: String, metrics: Option<MessageMetrics>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role: "assistant".to_string(),
            content,
            timestamp: chrono::Utc::now().timestamp(),
            metrics,
        }
    }
}
