//! Zero-cost dtype dispatch for CUDA kernels.
//!
//! Every numeric CUDA kernel exists as a `_f32` / `_bf16` / `_fp16` triple of
//! `extern "C"` entry points with an otherwise-identical ABI. Historically each
//! wrapper hand-wrote a `match T::DATA_TYPE { … }` to pick the right one, so the
//! "which dtypes does this op support" fact was an accident of which arms
//! someone typed, and an unsupported dtype failed at *runtime*.
//!
//! This module makes the dtype→entry-point mapping an **attribute of the type**:
//! each kernel declares a small binding trait (e.g. `AddKernel`) implemented once
//! per supported dtype, and its public wrapper is generic over that trait. After
//! monomorphization the call inlines straight to the concrete `extern` — exactly
//! the code the old `match` folded to — but now:
//!   * the dispatch logic is written once per op, not copy-pasted per dtype;
//!   * the supported-dtype set is a compile-time fact (a missing `impl` means
//!     `add::<i32>` fails to *compile*, not at runtime).
//!
//! The one place a runtime dtype must still be narrowed to a static type is the
//! `MathOps for Cuda` boundary, whose contract accepts any `T: Dtype`. That
//! single narrowing is centralized in [`narrow_float!`] rather than duplicated
//! across every kernel.

use half::{bf16, f16};

/// Element types the CUDA float kernels are built for. Implemented only by the
/// three IEEE-ish floats the `.cu` side ships (`f32`, `bf16`, `f16`); this is
/// the supertrait every per-kernel binding trait shares, so a kernel can only
/// ever be asked to run on a type that actually has a device kernel.
///
/// Sealed by construction: the trait is only implemented in this module.
pub trait CudaFloat: infer_core::types::Dtype {}

impl CudaFloat for f32 {}
impl CudaFloat for bf16 {}
impl CudaFloat for f16 {}

/// Narrow a generic `T: Dtype` (the `MathOps for Cuda` contract type) to the
/// concrete `CudaFloat` it names, then run `$body` with `$F` bound to that
/// static type. This is the *single* runtime dtype match in the CUDA backend;
/// all per-kernel dispatch downstream is static.
///
/// ```ignore
/// narrow_float!(T, "add", |F| {
///     // here `F: CudaFloat`, and the tensors can be viewed as `Tensor<F, _>`
///     add::<F>(stream, a.reinterpret::<F>(), b.reinterpret::<F>(), dst.reinterpret_mut::<F>())
/// })
/// ```
///
/// The arms are the only supported floats; anything else returns a descriptive
/// `OpError::Kernel`. Because `T::DATA_TYPE` is a `const`, each monomorphization
/// keeps only its matching arm after optimization — zero runtime cost.
#[macro_export]
macro_rules! narrow_float {
    ($t:ty, $op:expr, |$f:ident| $body:expr) => {{
        match <$t as infer_core::types::Dtype>::DATA_TYPE {
            infer_core::types::DataType::F32 => {
                type $f = f32;
                $body
            }
            infer_core::types::DataType::BF16 => {
                type $f = half::bf16;
                $body
            }
            infer_core::types::DataType::F16 => {
                type $f = half::f16;
                $body
            }
            other => Err(infer_core::ports::OpError::Kernel(format!(
                "{}: unsupported dtype {:?}",
                $op, other
            ))),
        }
    }};
}

/// Like [`narrow_float!`] but for kernels whose device side ships only `f32`
/// and `bf16` variants (e.g. `swiglu_packed`, `rope_interleaved`). `f16` is a
/// declared-unsupported dtype for these ops and returns an `OpError`, matching
/// the pre-refactor `match` arms exactly. Keeping it a *separate* macro means
/// the supported set of each op is still explicit at the call site — the two
/// coverage tiers never silently merge.
#[macro_export]
macro_rules! narrow_float_no_f16 {
    ($t:ty, $op:expr, |$f:ident| $body:expr) => {{
        match <$t as infer_core::types::Dtype>::DATA_TYPE {
            infer_core::types::DataType::F32 => {
                type $f = f32;
                $body
            }
            infer_core::types::DataType::BF16 => {
                type $f = half::bf16;
                $body
            }
            other => Err(infer_core::ports::OpError::Kernel(format!(
                "{}: unsupported dtype {:?}",
                $op, other
            ))),
        }
    }};
}
