//! Domain ports — trait definitions that Infrastructure must implement.
//!
//! This is the hexagonal "port" layer. Domain code programs against these
//! traits; infrastructure provides the concrete adapters.

pub use infer_core::device;
pub use infer_core::error;
mod op_ports;

pub mod backend;
pub mod collective;
pub mod fused_ops;
pub mod math_ops;
pub mod sampler;

// Two op surfaces coexist intentionally (both load-bearing, not duplication):
//   * `CoreOps`/`DiffusionOps`/`OpBackend` (op_ports) over `types::Dtype` — the
//     scope-less surface used by the diffusion + model-layer path.
//   * `MathOps`/`FusedOps`/`LlmBackend` (math_ops/fused_ops/backend) over the
//     richer `dtype::Dtype` — the scope/stream-threaded surface used by the LLM
//     decode path, which needs the scope for CUDA-graph/stream capture.
// The two `Dtype`/`Float` trait pairs are the same seam: the old surface needs
// only storage metadata, the new one needs read/write_f64 + a type id.
pub use backend::LlmBackend;
pub use collective::{CollectiveOps, CommAxis, ReduceOp, ShardSpec, ShardedLoad};
pub use device::{AllocError, Allocator, Device, HostDevice, MemoryPort};
pub use error::{OpError, OpResult};
pub use fused_ops::FusedOps;
pub use math_ops::MathOps;
pub use op_ports::{CoreOps, DiffusionOps, OpBackend};
pub use sampler::{AcceptReject, SampleBatch, SampledToken, Sampler, SamplingParams};
