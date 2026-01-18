//! HTTP Layer
//!
//! 提供 OpenAI 兼容的 HTTP API。

pub mod handler;
pub mod request;
pub mod response;

pub use handler::chat_completions;
pub use request::{ChatCompletionRequest, ChatMessage};
pub use response::{ChatCompletionResponse, ChatCompletionChunk, ChatChoice, Usage};
