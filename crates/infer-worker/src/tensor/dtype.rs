//! Compile-time `Dtype` trait bridging Rust scalar types and the runtime
//! [`DataType`] enum.
//!
//! All element types that can back a `TypedTensor<T>` must implement this
//! trait. Keeping it `Send + Sync + Copy + 'static` means tensor storage can
//! cross thread boundaries and be memcpy'd freely.

use crate::base::DataType;
use half::{bf16, f16};

/// Marker trait for tensor element types.
///
/// `Copy` gives us byte-level copy semantics, which matches how we move
/// memory through `Buffer`. The associated `DTYPE` constant lets generic
/// code round-trip to the runtime [`DataType`].
pub trait Dtype: Send + Sync + Copy + 'static {
    const DTYPE: DataType;
}

impl Dtype for f32  { const DTYPE: DataType = DataType::F32;  }
impl Dtype for i32  { const DTYPE: DataType = DataType::I32;  }
impl Dtype for i8   { const DTYPE: DataType = DataType::I8;   }
impl Dtype for f16  { const DTYPE: DataType = DataType::F16;  }
impl Dtype for bf16 { const DTYPE: DataType = DataType::BF16; }
