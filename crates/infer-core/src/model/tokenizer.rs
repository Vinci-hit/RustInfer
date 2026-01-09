// src/tokenizer/mod.rs

use crate::base::error::{Error, Result};
use std::path::Path;
use tokenizers::tokenizer::Tokenizer as HfTokenizer;

pub trait Tokenizer: Send + Sync {
    /// 获取内部的 Hugging Face Tokenizer 实例的引用。
    /// 这样可以在需要时访问更底层的 API。
    fn as_hf_tokenizer(&self) -> &HfTokenizer;

    /// 获取词汇表的大小。
    fn vocab_size(&self) -> usize {
        // 直接从 hf_tokenizer 实例获取
        self.as_hf_tokenizer().get_vocab_size(true) // `true` 表示包含所有特殊 token
    }
    
    /// 将文本编码为 Token ID 序列。
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        self.as_hf_tokenizer().encode(text, true)
            .map_err(|e| Error::InternalError(format!("Tokenizer encode error: {}", e)))?
            .get_ids()
            .iter()
            // 在这里将 u32 转换为 i32
            .map(|&id| Ok(id as i32))
            .collect::<Result<Vec<i32>>>() // collect 可以在这里处理 TryFrom 错误
            .map_err(|_| Error::InternalError("Token ID out of i32 range".to_string()).into())
    }

    /// 将 Token ID 序列解码为文本。
    fn decode(&self, ids: &[i32]) -> Result<String> {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.as_hf_tokenizer().decode(&u32_ids, false)
            .map_err(|e| Error::InternalError(format!("Tokenizer decode error: {}", e)).into())
    }

    fn is_eos(&self, token_id: i32) -> bool;
}


/// 一个具体的 Tokenizer 实现，它直接持有一个 `hf_tokenizers::Tokenizer` 实例。
/// 我们可以用这一个结构体来代表所有从 `tokenizer.json` 加载的分词器。
pub struct GenericHfTokenizer {
    // 核心：内部持有一个来自 Hugging Face 库的 Tokenizer 实例
    hf_tokenizer: HfTokenizer,
    eos_token_id: Vec<u32>,
}

impl GenericHfTokenizer {
    /// 从 `tokenizer.json` 文件加载分词器。
    /// 这是创建我们所有分词器实例的统一入口。
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {

        let hf_tokenizer = HfTokenizer::from_file(path)
            .map_err(|e| Error::InternalError(format!("Failed to load tokenizer: {}", e)))?;
        
        // 调用我们新的、健壮的查找函数
        let eos_token_id = Self::find_eos_token_id(&hf_tokenizer)?; 
        Ok(Self { hf_tokenizer , eos_token_id})
    }
    fn find_eos_token_id(tokenizer: &HfTokenizer) -> Result<Vec<u32>> {
        // --- 方法 C: 尝试通过已知的 token 字符串直接查找 ID ---
        // 这是最后的、针对特定模型的后备方案
        let mut eos_token_id = Vec::new();
        for token_str in ["<|end_of_text|>", "</s>", "<|eot_id|>"] {
            if let Some(id) = tokenizer.token_to_id(token_str) {
                println!("[DEBUG] Found EOS token ID {} using token string '{}'", id, token_str);
                eos_token_id.push(id); // 找到了！
            }
        }
        if !eos_token_id.is_empty() {
            return Ok(eos_token_id);
        }
        // 如果所有方法都失败了，返回一个清晰的错误
        Err(Error::InternalError("EOS token ID could not be found in the tokenizer file using any known method.".to_string()).into())
    }
}

impl Tokenizer for GenericHfTokenizer {
    fn as_hf_tokenizer(&self) -> &HfTokenizer {
        &self.hf_tokenizer
    }

    fn is_eos(&self, token_id: i32) -> bool {
        // 我们将输入的 i32 token ID 转换为 u32，
        // 然后与我们存储的 u32 类型的 eos_token_id 进行比较。
        let token_id_u32 = token_id as u32;
        self.eos_token_id.contains(&token_id_u32)
    }
}
