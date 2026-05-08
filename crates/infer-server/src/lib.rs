pub mod api;
pub mod chat;
pub mod zmq_client;

pub use chat::get_template;
pub use zmq_client::ZmqClient;

/// 共享应用状态
pub struct AppState {
    pub zmq_client: ZmqClient,
    pub tokenizer: tokenizers::Tokenizer,
}
