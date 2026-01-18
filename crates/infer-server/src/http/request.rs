//! HTTP Request Types
//!
//! OpenAI-compatible request structures.

use serde::{Deserialize, Serialize};

/// Chat Completion 请求
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// 模型名称
    pub model: String,
    
    /// 对话消息列表
    pub messages: Vec<ChatMessage>,
    
    /// 最大生成 Token 数
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,
    
    /// 是否流式返回
    #[serde(default)]
    pub stream: bool,
    
    // ===== Sampling Parameters =====
    
    /// 温度 (0.0 ~ 2.0)
    #[serde(default = "default_temperature")]
    pub temperature: Option<f32>,
    
    /// Top-p sampling
    #[serde(default = "default_top_p")]
    pub top_p: Option<f32>,
    
    /// Top-k sampling
    #[serde(default)]
    pub top_k: Option<i32>,
    
    /// 重复惩罚
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    
    /// 频率惩罚
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    
    /// 停止词
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    
    /// 随机种子
    #[serde(default)]
    pub seed: Option<u64>,
}

/// 对话消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// 角色: system, user, assistant
    pub role: String,
    
    /// 消息内容
    pub content: String,
}

// ===== Default Functions =====

fn default_max_tokens() -> Option<usize> {
    Some(512)
}

fn default_temperature() -> Option<f32> {
    Some(1.0)
}

fn default_top_p() -> Option<f32> {
    Some(1.0)
}

impl ChatCompletionRequest {
    /// 将对话消息转换为单个文本 Prompt
    ///
    /// 使用简单的格式：
    /// ```
    /// <|system|>
    /// {system_message}
    /// <|user|>
    /// {user_message}
    /// <|assistant|>
    /// ```
    pub fn to_prompt(&self) -> String {
        let mut prompt = String::new();
        
        for msg in &self.messages {
            prompt.push_str(&format!("<|{}|>\n", msg.role));
            prompt.push_str(&msg.content);
            prompt.push('\n');
        }
        
        // 添加 assistant 角色提示
        prompt.push_str("<|assistant|>\n");
        
        prompt
    }
}
