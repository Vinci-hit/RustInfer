// Public API for reusable components

pub mod api;
pub mod chat;
pub mod zmq_client;  // ZMQ客户端（分离式架构）

// Re-export commonly used types
pub use chat::get_template;
pub use zmq_client::ZmqClient;


