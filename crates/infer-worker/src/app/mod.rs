//! Application layer — orchestration of inference.

#[cfg(feature = "cuda")]
pub mod cuda_graph_runner;
pub mod model_runner;
