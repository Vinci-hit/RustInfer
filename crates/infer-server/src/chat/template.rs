//! Chat Template 系统
//!
//! 将 OpenAI 格式的 messages 转换为模型特定的 prompt 格式。

use crate::api::openai::types::ChatMessage;
use anyhow::Result;

/// Chat Template trait
pub trait ChatTemplate: Send + Sync {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String>;
}

/// Llama 3 格式
pub struct Llama3Template;

impl ChatTemplate for Llama3Template {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String> {
        // BOS (`<|begin_of_text|>`) is added automatically by the tokenizer
        // via `add_special_tokens=true` in `GenericHfTokenizer::encode`.
        // Embedding it in the template here would produce a duplicated BOS
        // (e.g. `[128000, 128000, ...]`), pushing the prompt off the
        // training distribution.
        let mut prompt = String::new();

        // Llama 3.2's bundled jinja chat_template ALWAYS emits a system
        // header — even when the caller provided no system message — and
        // injects "Cutting Knowledge Date" + "Today Date". Mirror that here
        // so RustInfer's prompt is byte-equivalent to HF's
        // `apply_chat_template(messages, add_generation_prompt=True)`.
        //
        // Date is fixed to the HF template's hard-coded fallback ("26 Jul 2024")
        // — matches HF's behavior when `strftime_now` is unavailable, and
        // keeps token sequences stable across days for reproducibility.
        let user_system = messages
            .iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.trim().to_string())
            .unwrap_or_default();

        prompt.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
        prompt.push_str("Cutting Knowledge Date: December 2023\n");
        prompt.push_str("Today Date: 26 Jul 2024\n\n");
        prompt.push_str(&user_system);
        prompt.push_str("<|eot_id|>");

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    // Already consumed above into the leading system block.
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

/// Qwen3.5 text chat uses an explicit thinking prefix by default.
pub struct Qwen35Template;

impl ChatTemplate for Qwen35Template {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String> {
        let mut prompt = Qwen3Template.apply(messages)?;
        prompt.push_str("<think>\n");
        Ok(prompt)
    }
}

/// 根据服务器实际加载的 `model_type` 选择对应的 chat template。
///
/// 必须基于服务端真实加载的模型类型（`ServerConfig::model_type`，取值
/// `"llama3"` / `"qwen3"`），而不是客户端请求里任意填写的 `req.model` 字段。
/// 否则当客户端用错误的模型名（如对 Qwen3 模型发送 `"llama3.2-1b"`）请求时，
/// 会套用错误的 prompt 模板，模型输出错误的控制 token 且永远不会发出正确的
/// 终止 token（如 Qwen3 的 `<|im_end|>`），导致生成停不下来、跑满 max_tokens。
pub fn get_template(model_type: &str) -> Box<dyn ChatTemplate + Send + Sync> {
    match model_type.to_lowercase().as_str() {
        "qwen3_5" | "qwen3_5_text" => Box::new(Qwen35Template),
        t if t.contains("qwen") => Box::new(Qwen3Template),
        _ => Box::new(Llama3Template),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35_generation_prefix_matches_checkpoint_default() {
        let messages = [ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        let base = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
        for model_type in ["qwen3_5", "qwen3_5_text"] {
            assert_eq!(
                get_template(model_type).apply(&messages).unwrap(),
                format!("{base}<think>\n")
            );
        }
        assert_eq!(get_template("qwen3").apply(&messages).unwrap(), base);
    }
}
