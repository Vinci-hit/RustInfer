//! Domain ports — trait definitions that Infrastructure must implement.
//!
//! This is the hexagonal "port" layer. Domain code programs against these
//! traits; infrastructure provides the concrete adapters.

pub use infer_core::device;
pub use infer_core::error;
mod op_ports;

pub mod backend;
pub mod collective;
pub mod diffusion_ops;
pub mod fused_ops;
pub mod math_ops;
pub mod sampler;

pub use backend::{Backend as V2Backend, DiffusionBackend, LlmBackend};
pub use collective::{CollectiveOps, CommAxis, ReduceOp, ShardSpec, ShardedLoad};
pub use device::{AllocError, Allocator, Device, HostDevice, MemoryPort};
pub use diffusion_ops::DiffusionOps as V2DiffusionOps;
pub use error::{OpError, OpResult};
pub use fused_ops::FusedOps;
pub use math_ops::MathOps;
pub use op_ports::{CoreOps, DiffusionOps, OpBackend};
pub use sampler::{AcceptReject, SampleBatch, SampledToken, Sampler, SamplingParams};
