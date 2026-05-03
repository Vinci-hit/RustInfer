//! Tensor abstractions for the inference worker.
//!
//! The module is split by concern rather than by type:
//!
//! | File               | Role                                                      |
//! |--------------------|-----------------------------------------------------------|
//! | `dtype.rs`         | `Dtype` trait binding Rust scalars to `DataType`          |
//! | `dims.rs`          | Inline `[usize; MAX_RANK]` container for shape / strides  |
//! | `typed.rs`         | `TypedTensor<T>`: actual storage + stride-aware metadata  |
//! | `tensor.rs`        | Dtype-erased `Tensor` enum and core accessors             |
//! | `views.rs`         | Zero-copy view ops (narrow/select/permute/expand/…)       |
//! | `materialize.rs`   | `contiguous()` / `to_owned()` / `permute_into()`          |
//! | `io.rs`            | Allocation, device/dtype migration, safetensors, copies   |
//! | `ops.rs`           | Operator overloads and in-place ergonomics                |
//! | `tests.rs`         | Metadata / view / copy regression tests                   |
//!
//! The public surface — `Tensor`, `TypedTensor`, `Dtype`, `Dims`,
//! `MAX_RANK` — is re-exported from this file.

pub mod dims;
pub mod dtype;
pub mod io;
pub mod materialize;
pub mod ops;
pub mod tensor;
pub mod typed;
pub mod views;

#[cfg(test)]
mod tests;

pub use dims::{Dims, MAX_RANK};
pub use dtype::Dtype;
pub use tensor::Tensor;
pub use typed::TypedTensor;
