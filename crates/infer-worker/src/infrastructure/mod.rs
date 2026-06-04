//! Infrastructure layer — implements domain ports.
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod io;
pub mod transport;
