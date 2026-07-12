use serde::{Deserialize, Serialize};

/// 单条消息
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: i64,
    pub metrics: Option<MessageMetrics>,
    pub is_streaming: bool,
}

/// 每条消息的推理性能指标
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageMetrics {
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub tokens_per_second: f64,
    pub total_tokens: u32,
}

/// 一个对话
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub messages: Vec<Message>,
    pub created_at: i64,
    pub updated_at: i64,
    pub model: String,
}

impl Message {
    pub fn user(content: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role: "user".to_string(),
            content,
            timestamp: chrono::Utc::now().timestamp(),
            metrics: None,
            is_streaming: false,
        }
    }

    pub fn assistant_streaming() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role: "assistant".to_string(),
            content: String::new(),
            timestamp: chrono::Utc::now().timestamp(),
            metrics: None,
            is_streaming: true,
        }
    }

    pub fn assistant(content: String, metrics: Option<MessageMetrics>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role: "assistant".to_string(),
            content,
            timestamp: chrono::Utc::now().timestamp(),
            metrics,
            is_streaming: false,
        }
    }
}

impl Conversation {
    pub fn new(model: String) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            title: "New Chat".to_string(),
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
            model,
        }
    }

    /// 根据首条用户消息自动生成标题
    pub fn auto_title(&mut self) {
        if let Some(first_user_msg) = self.messages.iter().find(|m| m.role == "user") {
            let title: String = first_user_msg.content.chars().take(30).collect();
            self.title = if first_user_msg.content.len() > 30 {
                format!("{}...", title)
            } else {
                title
            };
        }
    }
}
