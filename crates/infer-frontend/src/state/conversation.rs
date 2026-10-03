use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Attachment {
    pub id: String,
    pub name: String,
    pub mime_type: String,
    pub size: u64,
    pub data_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub id: String,
    pub role: String,
    pub content: String,
    pub timestamp: i64,
    pub metrics: Option<MessageMetrics>,
    pub is_streaming: bool,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub attachments: Vec<Attachment>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub interrupted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageMetrics {
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub tokens_per_second: f64,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub messages: Vec<Message>,
    pub created_at: i64,
    pub updated_at: i64,
    pub model: String,
    #[serde(default)]
    pub draft: String,
}

impl Message {
    pub fn user(content: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            role: "user".into(),
            content,
            timestamp: chrono::Utc::now().timestamp(),
            metrics: None,
            is_streaming: false,
            model: None,
            attachments: Vec::new(),
            error: None,
            interrupted: false,
        }
    }
    pub fn assistant_streaming() -> Self {
        Self {
            role: "assistant".into(),
            is_streaming: true,
            ..Self::user(String::new())
        }
    }
}

impl Conversation {
    pub fn new(model: String) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            title: "新对话".into(),
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
            model,
            draft: String::new(),
        }
    }
    pub fn auto_title(&mut self) {
        if let Some(message) = self.messages.iter().find(|m| m.role == "user") {
            let text = message.content.trim();
            self.title = if text.is_empty() {
                "图片对话".into()
            } else {
                let title: String = text.chars().take(26).collect();
                if text.chars().count() > 26 {
                    format!("{title}…")
                } else {
                    title
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn titles_count_characters_and_support_image_only_messages() {
        let mut conversation = Conversation::new("test".into());
        conversation
            .messages
            .push(Message::user("你好，这是中文对话".into()));
        conversation.auto_title();
        assert_eq!(conversation.title, "你好，这是中文对话");
        conversation.messages[0].content.clear();
        conversation.auto_title();
        assert_eq!(conversation.title, "图片对话");
    }
}
