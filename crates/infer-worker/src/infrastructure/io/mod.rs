//! Infrastructure I/O — file-system adapters for loading weights.
//!
//! This sub-module isolates filesystem access from the rest of the worker.
//! Model loading currently uses `SafetensorsReader`; `GgufReader` exposes
//! validated file metadata and encoded tensor bytes for future model adapters.

pub mod gguf;
pub mod safetensors;

pub use gguf::GgufReader;
pub use safetensors::SafetensorsReader;
