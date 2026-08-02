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
pub mod pipeline_ops;
pub mod sampler;
pub mod vocab_ops;

// Two op surfaces coexist intentionally:
//   * `CoreOps`/`DiffusionOps`/`OpBackend` is the scope-less diffusion surface.
//   * `MathOps`/`FusedOps`/`LlmBackend` threads an execution scope through the
//     LLM decode path for CUDA graph/stream capture.
// Both surfaces share the single canonical `types::Dtype`; `dtype::Dtype`
// re-exports that same trait for compatibility with the LLM module layout.
pub use backend::LlmBackend;
pub use collective::{CollectiveOps, CommAxis, ReduceOp, ShardSpec, ShardedLoad};
pub use device::{AllocError, Allocator, Device, HostDevice, MemoryPort};
pub use error::{OpError, OpResult};
pub use fused_ops::FusedOps;
pub use math_ops::MathOps;
pub use op_ports::{CoreOps, DiffusionOps, OpBackend};
pub use pipeline_ops::{
    CompactExtendControlArgs, DecodePipelineOps, MergeCompactDecodeArgs, MergeCompactMixedArgs,
};
pub use sampler::{AcceptReject, SampleBatch, SampledToken, Sampler, SamplingParams};
pub use vocab_ops::VocabOps;
