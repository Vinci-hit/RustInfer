//! Tokenizer with Stream Decoding Support
//!
//! 封装 Hugging Face tokenizers 库，提供流式解码能力。
//! 核心挑战：逐个 Token 解码时可能遇到 UTF-8 字节边界截断问题。

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer as HfTokenizer;

/// Tokenizer 封装
///
/// 线程安全，可以在多个请求间共享。
pub struct TokenizerWrapper {
    inner: Arc<HfTokenizer>,
    eos_token_ids: Vec<u32>,
}

impl TokenizerWrapper {
    /// 从文件加载 Tokenizer
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        // 查找 EOS Token IDs
        let eos_token_ids = Self::find_eos_tokens(&inner);
        
        tracing::info!("Tokenizer loaded, vocab_size={}, eos_tokens={:?}", 
            inner.get_vocab_size(true), eos_token_ids);
        
        Ok(Self {
            inner: Arc::new(inner),
            eos_token_ids,
        })
    }
    
    /// 查找 EOS Token IDs
    fn find_eos_tokens(tokenizer: &HfTokenizer) -> Vec<u32> {
        let mut eos_ids = Vec::new();
        
        // 常见的 EOS Token 字符串
        for token_str in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<|endoftext|>"] {
            if let Some(id) = tokenizer.token_to_id(token_str) {
                eos_ids.push(id);
                tracing::debug!("Found EOS token: {} -> {}", token_str, id);
            }
        }
        
        eos_ids
    }
    
    /// 编码文本为 Token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }
    
    /// 解码 Token IDs 为文本 (批量)
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.inner.decode(token_ids, false)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))
    }
    
    /// 检查是否是 EOS Token
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }
    
    /// 创建流式解码器
    pub fn create_decoder(&self) -> StreamDecoder {
        StreamDecoder::new(self.inner.clone())
    }
    
    /// 获取词汇表大小
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

/// 流式解码器
///
/// 逐个 Token 解码，处理 UTF-8 字节边界问题。
///
/// ## 问题
/// 当我们逐个解码 Token 时，可能出现：
/// - Token A 的最后几个字节 + Token B 的前几个字节 = 一个完整的 UTF-8 字符
/// - 如果只解码 Token A，会得到不完整的 UTF-8 序列
///
/// ## 解决方案
/// 维护一个 buffer，累积 Token IDs，只输出完整的 UTF-8 字符串。
pub struct StreamDecoder {
    tokenizer: Arc<HfTokenizer>,
    /// 累积的 Token IDs
    token_buffer: Vec<u32>,
    /// 已经输出的字符数 (用于增量输出)
    output_offset: usize,
}

impl StreamDecoder {
    /// 创建新的流式解码器
    pub fn new(tokenizer: Arc<HfTokenizer>) -> Self {
        Self {
            tokenizer,
            token_buffer: Vec::new(),
            output_offset: 0,
        }
    }
    
    /// 解码一个新 Token，返回新增的文本
    ///
    /// 返回 `None` 表示这个 Token 导致了 UTF-8 边界截断，
    /// 需要等待下一个 Token 才能输出完整的字符。
    pub fn decode(&mut self, token_id: u32) -> Option<String> {
        // 添加到 buffer
        self.token_buffer.push(token_id);
        
        // 尝试解码整个 buffer
        match self.tokenizer.decode(&self.token_buffer, false) {
            Ok(full_text) => {
                // 解码成功，提取新增的部分
                if full_text.len() > self.output_offset {
                    let new_text = full_text[self.output_offset..].to_string();
                    self.output_offset = full_text.len();
                    Some(new_text)
                } else {
                    // 没有新增文本（可能是特殊 Token）
                    None
                }
            }
            Err(_) => {
                // 解码失败（UTF-8 边界问题），等待下一个 Token
                None
            }
        }
    }
    
    /// 刷新剩余的 buffer
    ///
    /// 在流结束时调用，确保所有文本都被输出。
    pub fn flush(&mut self) -> Option<String> {
        if self.token_buffer.is_empty() {
            return None;
        }
        
        match self.tokenizer.decode(&self.token_buffer, false) {
            Ok(full_text) => {
                if full_text.len() > self.output_offset {
                    let remaining = full_text[self.output_offset..].to_string();
                    self.output_offset = full_text.len();
                    self.token_buffer.clear();
                    Some(remaining)
                } else {
                    self.token_buffer.clear();
                    None
                }
            }
            Err(e) => {
                tracing::warn!("Failed to flush decoder buffer: {}", e);
                self.token_buffer.clear();
                None
            }
        }
    }
    
    /// 重置解码器状态
    pub fn reset(&mut self) {
        self.token_buffer.clear();
        self.output_offset = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stream_decoder_basic() {
        // 这个测试需要真实的 tokenizer 文件，仅作示例
        // 实际测试中，你可以使用 mock tokenizer
    }
}
