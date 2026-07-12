//! Infrastructure layer — implements domain ports.
pub use infer_backend_cpu as cpu;
#[cfg(feature = "cuda")]
pub use infer_backend_cuda as cuda;
pub mod io;
pub mod transport;
