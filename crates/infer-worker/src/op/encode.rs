
use crate::base::error::Result;

/// Encode 层抽象，封装 encode / decode / vocab 查询等操作。
pub trait EncodeLayer: Send + Sync {
    fn encode(&self, sentence: &str) -> Result<Vec<i32>>;
    fn decode_ids(&self, ids: &[i32]) -> Result<String>;
    fn decode(&self, token_id: i32) -> Result<String> {
        self.decode_ids(&[token_id])
    }
    fn is_sentence_ending(&self, token_id: i32) -> bool;
    fn vocab_size(&self) -> i32;
}