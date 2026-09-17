//! Shared host-side contracts for the fixed-shape V4 kernels.
use crate::{Cuda, ffi};
use infer_core::dtype::Dtype;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

pub(super) struct Validation {
    op: &'static str,
    device: i32,
}

impl Validation {
    pub fn new(op: &'static str, device: i32) -> Self {
        Self { op, device }
    }

    pub fn tensor<T: Dtype>(&self, t: &Tensor<T, Cuda>) -> OpResult<(usize, usize)> {
        if !t.is_contiguous()
            || t.device().device_id != self.device
            || !(t.data_ptr() as usize).is_multiple_of(4)
        {
            return Err(OpError::Shape(format!(
                "{}: tensors must be contiguous, four-byte aligned, and on the scope device",
                self.op
            )));
        }
        let start = t.data_ptr() as usize;
        let end = t
            .numel()
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|bytes| start.checked_add(bytes))
            .ok_or_else(|| OpError::Shape(format!("{}: tensor range overflow", self.op)))?;
        Ok((start, end))
    }

    pub fn disjoint(&self, reads: &[(usize, usize)], writes: &[(usize, usize)]) -> OpResult<()> {
        for (i, &w) in writes.iter().enumerate() {
            if reads
                .iter()
                .chain(&writes[..i])
                .any(|r| r.0 < w.1 && w.0 < r.1)
            {
                return Err(OpError::Shape(format!(
                    "{}: writable tensors must not overlap another argument",
                    self.op
                )));
            }
        }
        Ok(())
    }

    pub fn launched(&self, status: i32) -> OpResult<()> {
        if status == ffi::cudaError_cudaSuccess as i32 {
            Ok(())
        } else {
            Err(OpError::Kernel(format!(
                "{}: CUDA launch error {status}",
                self.op
            )))
        }
    }
}
