//! Tensor<T, D> — the core domain object.

use std::marker::PhantomData;
use super::types::{Dtype, Shape, Strides};
use super::ports::Device;

/// Strongly-typed, device-aware tensor.
/// T = element dtype, D = device (enforced at compile time).
#[derive(Debug)]
pub struct Tensor<T: Dtype, D: Device> {
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) offset_elems: usize,
    pub(crate) numel: usize,
    pub(crate) is_contiguous: bool,
    /// Raw storage pointer (beginning of buffer).
    pub(crate) storage_ptr: *mut u8,
    /// Total storage size in bytes.
    pub(crate) storage_len: usize,
    /// Device this tensor lives on.
    pub(crate) device: D,
    pub(crate) _marker: PhantomData<T>,
}

// Tensor is Send+Sync if Device is (GPU memory is thread-safe).
unsafe impl<T: Dtype, D: Device> Send for Tensor<T, D> {}
unsafe impl<T: Dtype, D: Device> Sync for Tensor<T, D> {}

impl<T: Dtype, D: Device> Tensor<T, D> {
    #[inline] pub fn shape(&self) -> &Shape { &self.shape }
    #[inline] pub fn strides(&self) -> &Strides { &self.strides }
    #[inline] pub fn ndim(&self) -> usize { self.shape.ndim() }
    #[inline] pub fn numel(&self) -> usize { self.numel }
    #[inline] pub fn is_contiguous(&self) -> bool { self.is_contiguous }
    #[inline] pub fn device(&self) -> &D { &self.device }
    #[inline] pub fn data_ptr(&self) -> *const T {
        unsafe { (self.storage_ptr as *const T).add(self.offset_elems) }
    }
    #[inline] pub fn data_ptr_mut(&mut self) -> *mut T {
        unsafe { (self.storage_ptr as *mut T).add(self.offset_elems) }
    }
}

impl<T: Dtype, D: Device> Clone for Tensor<T, D> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape, strides: self.strides,
            offset_elems: self.offset_elems, numel: self.numel,
            is_contiguous: self.is_contiguous,
            storage_ptr: self.storage_ptr, storage_len: self.storage_len,
            device: self.device.clone(), _marker: PhantomData,
        }
    }
}
