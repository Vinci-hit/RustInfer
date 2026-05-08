//! Chat Template 系统
//!
//! 将 OpenAI 格式的 messages 转换为模型特定的 prompt 格式。

use anyhow::Result;
use crate::api::openai::types::ChatMessage;

/// Chat Template trait
pub trait ChatTemplate: Send + Sync {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String>;
}

/// Llama 3 格式
pub struct Llama3Template;

impl ChatTemplate for Llama3Template {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String> {
        let mut prompt = String::from("<|begin_of_text|>");

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|eot_id|>");
                }
                "user" => {
                    prompt.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|eot_id|>");
                }
                "assistant" => {
                    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|eot_id|>");
                }
                _ => {}
            }
        }

        // Add assistant header for generation
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

        Ok(prompt)
    }
}

/// Qwen 3 格式
pub struct Qwen3Template;

impl ChatTemplate for Qwen3Template {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String> {
        let mut prompt = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<|im_start|>system\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
                "user" => {
                    prompt.push_str("<|im_start|>user\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
                "assistant" => {
                    prompt.push_str("<|im_start|>assistant\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
                _ => {}
            }
        }

        // Add assistant header for generation
        prompt.push_str("<|im_start|>assistant\n");

        Ok(prompt)
    }
}

/// 根据模型名称获取对应的 chat template
pub fn get_template(model_name: &str) -> Box<dyn ChatTemplate + Send + Sync> {
    match model_name.to_lowercase().as_str() {
        name if name.contains("qwen") => Box::new(Qwen3Template),
        _ => Box::new(Llama3Template),
    }
}
