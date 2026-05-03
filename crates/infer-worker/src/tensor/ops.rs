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
    /// `self[i] = silu(self[i])`. Panics on kernel failure.
    pub fn silu_(&mut self) {
        crate::op::activation::silu_inplace(self)
            .expect("Tensor::silu_: kernel failed");
    }

    /// `self[i] = tanh(self[i])`. Panics on kernel failure.
    pub fn tanh_(&mut self) {
        crate::op::activation::tanh_inplace(self)
            .expect("Tensor::tanh_: kernel failed");
    }

    /// Recoverable SiLU (result-returning variant).
    pub fn silu(&mut self) -> Result<()> {
        crate::op::activation::silu_inplace(self)
    }

    /// Recoverable tanh.
    pub fn tanh(&mut self) -> Result<()> {
        crate::op::activation::tanh_inplace(self)
    }

    /// `self[.., j] *= row[j]`, where `row.shape == [self.shape.last()]`.
    pub fn mul_row(&mut self, row: &Tensor) -> Result<()> {
        crate::op::broadcast_mul::broadcast_mul_inplace(self, row)
    }

    /// Interleaved RoPE applied in place.
    ///
    /// - `self`: `[seq, n_heads, head_dim]` (F32 or BF16 on device).
    /// - `cos`, `sin`: `[seq, head_dim / 2]` in F32.
    pub fn rope_interleaved(&mut self, cos: &Tensor, sin: &Tensor, head_dim: usize) -> Result<()> {
        crate::op::rope_interleaved::apply_rope_interleaved(self, cos, sin, head_dim)
    }

    /// Broadcast multiply: `out[..., j] = self[..., j] * scale[j]`.
    ///
    /// Allocates a fresh contiguous output.
    pub fn broadcast_mul(&self, scale: &Tensor) -> Result<Tensor> {
        let mut out = Tensor::empty(self.shape(), self.dtype(), self.device())?;
        crate::op::broadcast_mul::broadcast_mul(self, scale, &mut out)?;
        Ok(out)
    }
}
