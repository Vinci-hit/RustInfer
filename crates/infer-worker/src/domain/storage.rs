//! `Storage<D>` — RAII owner of a single device allocation.
//!
//! Domain entity. The only thing in the codebase allowed to call
//! `MemoryPort::alloc_bytes` / `MemoryPort::free_bytes`. Tensors hold an
//! `Arc<Storage<D>>`; cheap `Tensor::clone` or `Tensor::view` simply bumps
//! the refcount. The underlying allocation is freed when the last `Arc`
//! drops.
//!
//! This is the keystone of memory safety in this crate: as long as every
//! tensor goes through `Storage`, we cannot leak, double-free, or ship a
//! host pointer to a CUDA kernel.

use std::ptr::NonNull;
use std::sync::Arc;

use super::ports::{MemoryPort, OpResult};

/// RAII owner of a single contiguous allocation on device `D`.
///
/// Constructed via `Storage::alloc(&device, size)`. Freed automatically
/// when the last `Arc<Storage<D>>` drops.
pub struct Storage<D: MemoryPort> {
    ptr: NonNull<u8>,
    size: usize,
    device: D,
}

// SAFETY: `ptr` is owned exclusively by this Storage and is freed only in Drop;
// the underlying device memory may be touched concurrently by kernels but the
// Rust-level pointer ownership is single-owner.
unsafe impl<D: MemoryPort> Send for Storage<D> {}
unsafe impl<D: MemoryPort> Sync for Storage<D> {}

impl<D: MemoryPort> Storage<D> {
    /// Allocate `size` bytes on `device` (zero-initialized) and wrap in `Arc`
    /// for shared ownership.
    pub fn alloc(device: &D, size: usize) -> OpResult<Arc<Self>> {
        let ptr = device.alloc_bytes(size.max(1))?;
        Ok(Arc::new(Self {
            ptr,
            size,
            device: device.clone(),
        }))
    }

    /// Raw pointer to the start of the allocation.
    #[inline]
    pub fn ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Byte size of the allocation.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// The device this storage lives on.
    #[inline]
    pub fn device(&self) -> &D {
        &self.device
    }
}

impl<D: MemoryPort> Drop for Storage<D> {
    fn drop(&mut self) {
        // Synchronize before free so any async kernel still touching this
        // memory completes. Cheap on CPU (no-op); on CUDA waits on the
        // device's stream.
        let _ = self.device.synchronize();
        // SAFETY: ptr came from `device.alloc_bytes(self.size)` and is
        // freed exactly once (Drop runs once per Storage instance).
        unsafe { self.device.free_bytes(self.ptr, self.size.max(1)) };
    }
}

impl<D: MemoryPort> std::fmt::Debug for Storage<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("device", &self.device.name())
            .field("size", &self.size)
            .field("ptr", &self.ptr.as_ptr())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infra::cpu::Cpu;

    #[test]
    fn alloc_and_drop_is_safe() {
        let s = Storage::alloc(&Cpu, 1024).unwrap();
        assert_eq!(s.size(), 1024);
        // Drop runs at end of scope — must not panic / leak.
        drop(s);
    }

    #[test]
    fn arc_clone_shares_storage() {
        let s = Storage::alloc(&Cpu, 64).unwrap();
        let s2 = Arc::clone(&s);
        assert_eq!(s.ptr(), s2.ptr());
        assert_eq!(Arc::strong_count(&s), 2);
        drop(s);
        assert_eq!(Arc::strong_count(&s2), 1);
        // s2 drops here — should free exactly once.
    }

    #[test]
    fn zero_size_alloc_is_safe() {
        // We bump 0 → 1 internally so we always have a valid ptr.
        let s = Storage::alloc(&Cpu, 0).unwrap();
        assert_eq!(s.size(), 0);
    }
}
