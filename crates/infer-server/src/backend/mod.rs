//! Backend Communication Module
//!
//! 负责与 Scheduler 的通信。

pub mod zmq_client;

pub use zmq_client::start_zmq_loop;
