//! Infrastructure I/O — file-system adapters for loading weights.
//!
//! This sub-module isolates filesystem access from the rest of the worker.
//! Only `models/loader.rs` (and the standalone llama3 demo bin) depend on
//! it, via the `SafetensorsReader` port.

pub mod safetensors;

pub use safetensors::SafetensorsReader;
