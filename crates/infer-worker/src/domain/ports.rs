//! Domain ports — trait definitions that Infrastructure must implement.
//!
//! This is the hexagonal "port" layer. Domain code programs against these
//! traits; infrastructure provides the concrete adapters.

mod device;
mod error;
mod op_ports;

pub use device::{AllocError, Allocator, Device, HostDevice, MemoryPort};
pub use error::{OpError, OpResult};
pub use op_ports::{CoreOps, DiffusionOps, LlmOps, OpBackend};
