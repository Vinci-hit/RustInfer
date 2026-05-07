//! Operator overloads and in-place ergonomic wrappers.
//!
//! Everything here is a thin shim over kernels in [`crate::op`]. The
//! conventions are:
//!
//! - `+`, `+=`, `*`, `*=`, `/`, `-` via the standard `std::ops` traits.
//!   Shape / dtype / device mismatches **panic** because operator traits
//!   cannot return `Result`. Callers that need graceful error handling
//!   must invoke the kernels directly.
//!
//! - Methods named like `silu()`, `tanh()`, `mul_row()` return `Result`
//!   and are the preferred surface in regular code.

use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg};

use crate::base::error::Result;
use crate::base::DataType;

use super::tensor::Tensor;

// ─────────────────────── CPU-indexing on f32 tensors ────────────────────

/// Element indexing for CPU-resident, contiguous, F32 tensors.
///
/// Provides `tensor[i]` syntax for direct scalar access. This is intended
/// for debugging and testing — for bulk data access, prefer
/// [`TypedTensor::as_slice`](super::typed::TypedTensor::as_slice).
///
/// # Panics
///
/// - If the tensor is not on CPU.
/// - If the tensor's dtype is not `F32`.
/// - If the tensor is not contiguous.
/// - If `index >= self.numel()`.
impl Index<usize> for Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        if self.device().is_cuda() || self.dtype() != DataType::F32 {
            panic!("Tensor::index: requires CPU F32 tensor");
        }
        if !self.is_contiguous() {
            panic!("Tensor::index: requires contiguous tensor");
        }
        let total = self.numel();
        assert!(index < total, "Tensor index OOB: {} >= {}", index, total);
        let slice = self.as_f32().unwrap().as_slice().unwrap();
        &slice[index]
    }
}

/// Mutable element indexing for CPU-resident, contiguous, F32 tensors.
///
/// Provides `tensor[i] = val` syntax for direct scalar mutation.
///
/// # Panics
///
/// Same conditions as the immutable [`Index`] implementation.
impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if self.device().is_cuda() || self.dtype() != DataType::F32 {
            panic!("Tensor::index_mut: requires CPU F32 tensor");
        }
        if !self.is_contiguous() {
            panic!("Tensor::index_mut: requires contiguous tensor");
        }
        let total = self.numel();
        assert!(index < total, "Tensor index OOB: {} >= {}", index, total);
        let slice = self.as_f32_mut().unwrap().as_slice_mut().unwrap();
        &mut slice[index]
    }
}

// ────────────────────────── arithmetic operators ────────────────────────

/// `&Tensor + &Tensor` — delegates to the `op::add` forward kernel.
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        let mut out = Tensor::empty(self.shape(), self.dtype(), self.device())
            .expect("Tensor + Tensor: allocation failed");
        crate::op::add::add(self, rhs, &mut out, None)
            .expect("Tensor + Tensor: forward failed");
        out
    }
}

/// `Tensor += &Tensor` — delegates to the in-place add kernel.
impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, rhs: &Tensor) {
        crate::op::add::add_inplace(rhs, self, None)
            .expect("Tensor += Tensor: forward failed");
    }
}

/// `-&Tensor` → element-wise negate (implemented as `* -1.0`).
impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor { self * (-1.0_f32) }
}

/// `&Tensor * f32` — broadcast scalar multiply.
impl Mul<f32> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f32) -> Tensor {
        let mut out = Tensor::empty(self.shape(), self.dtype(), self.device())
            .expect("Tensor * f32: allocation failed");
        crate::op::scalar::scalar_mul(self, &mut out, rhs)
            .expect("Tensor * f32: kernel failed");
        out
    }
}

/// `f32 * &Tensor` — commutativity sugar.
impl Mul<&Tensor> for f32 {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor { rhs * self }
}

/// `&Tensor / f32` — delegates to `* (1/rhs)`.
impl Div<f32> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: f32) -> Tensor { self * (1.0 / rhs) }
}

/// `&Tensor + f32` — broadcast scalar add.
impl Add<f32> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: f32) -> Tensor {
        let mut out = Tensor::empty(self.shape(), self.dtype(), self.device())
            .expect("Tensor + f32: allocation failed");
        crate::op::scalar::scalar_add(self, &mut out, rhs)
            .expect("Tensor + f32: kernel failed");
        out
    }
}

/// `Tensor += f32`.
impl AddAssign<f32> for Tensor {
    fn add_assign(&mut self, rhs: f32) {
        crate::op::scalar::scalar_add_inplace(self, rhs)
            .expect("Tensor += f32: kernel failed");
    }
}

/// `Tensor *= &Tensor` — element-wise multiply into self.
impl MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, rhs: &Tensor) {
        crate::op::ewise_mul::ewise_mul_inplace(self, rhs)
            .expect("Tensor *= &Tensor: kernel failed");
    }
}

/// `Tensor *= f32`. Canonical idiom for `*= -1.0` negation.
impl MulAssign<f32> for Tensor {
    fn mul_assign(&mut self, rhs: f32) {
        crate::op::scalar::scalar_mul_inplace(self, rhs)
            .expect("Tensor *= f32: kernel failed");
    }
}

// ───────────────────── in-place element-wise methods ────────────────────

impl Tensor {
    /// Applies the SiLU (Sigmoid Linear Unit) activation in place.
    ///
    /// `SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))`
    ///
    /// Also known as the "swish" activation. Widely used in modern transformer
    /// architectures (LLaMA, Mistral, etc.) for feed-forward network gating.
    ///
    /// # Errors
    ///
    /// Returns an error if the dtype/device combination is not supported
    /// by the SiLU kernel.
    pub fn silu(&mut self) -> Result<()> {
        crate::op::activation::silu_inplace(self)
    }

    /// Applies the tanh activation in place.
    ///
    /// `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
    ///
    /// Maps all values to the range `(-1, 1)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the dtype/device combination is not supported
    /// by the tanh kernel.
    pub fn tanh(&mut self) -> Result<()> {
        crate::op::activation::tanh_inplace(self)
    }

    /// Multiplies each row of `self` by the corresponding element of `row`.
    ///
    /// Semantics: `self[.., j] *= row[j]` where `row` is a 1-D tensor with
    /// `row.shape[0] == self.shape[ndim-1]` (the last dimension).
    ///
    /// This is commonly used for applying per-channel scaling (e.g. RMSNorm
    /// weight multiplication).
    ///
    /// # Errors
    ///
    /// Returns an error if shapes are incompatible or dtype/device mismatch.
    pub fn mul_row(&mut self, row: &Tensor) -> Result<()> {
        crate::op::broadcast_mul::broadcast_mul_inplace(self, row)
    }

    /// Applies interleaved Rotary Position Embedding (RoPE) in place.
    ///
    /// RoPE encodes positional information by rotating pairs of elements
    /// using precomputed cosine and sine tables.
    ///
    /// # Arguments
    ///
    /// - `self`: The input tensor with shape `[seq, n_heads, head_dim]` (F32 or BF16).
    /// - `cos`: Cosine table with shape `[seq, head_dim / 2]` in F32.
    /// - `sin`: Sine table with shape `[seq, head_dim / 2]` in F32.
    /// - `head_dim`: The dimension of each attention head.
    ///
    /// # Errors
    ///
    /// Returns an error if shapes/dtypes are incompatible or the kernel fails.
    pub fn rope_interleaved(&mut self, cos: &Tensor, sin: &Tensor, head_dim: usize) -> Result<()> {
        crate::op::rope_interleaved::apply_rope_interleaved(self, cos, sin, head_dim)
    }

    /// Broadcast-multiplies `self` by a 1-D scale tensor, returning a new tensor.
    ///
    /// Semantics: `out[..., j] = self[..., j] * scale[j]` where `scale`
    /// has the same length as the last dimension of `self`.
    ///
    /// Unlike [`mul_row`](Self::mul_row), this allocates a fresh contiguous
    /// output rather than modifying `self` in place.
    ///
    /// # Errors
    ///
    /// Returns an error if shapes are incompatible, dtype/device mismatch,
    /// or allocation fails.
    pub fn broadcast_mul(&self, scale: &Tensor) -> Result<Tensor> {
        let mut out = Tensor::empty(self.shape(), self.dtype(), self.device())?;
        crate::op::broadcast_mul::broadcast_mul(self, scale, &mut out)?;
        Ok(out)
    }
}
