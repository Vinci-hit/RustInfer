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

use crate::device::MemoryPort;
use crate::error::OpResult;

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
        // NOTE: we deliberately do NOT call `self.device.synchronize()` here.
        // A device/stream synchronize is ILLEGAL during CUDA-graph capture and
        // would invalidate the capture — every transient scratch tensor freed
        // inside a captured decode step would otherwise break the graph. It is
        // also unnecessary: on the CUDA backend `free_bytes` recycles the block
        // into a size-keyed pool (no `cudaFree`, no sync); a block is re-zeroed
        // with `cudaMemsetAsync` on the COMPUTE stream when it is next handed
        // out, which is program-ordered after the prior tenant's consumer
        // kernels ON THAT SAME STREAM — so no synchronize is needed. (This holds
        // only while all transient-tensor work stays on the compute stream, not
        // the copy_in/copy_out streams.) Arena-owned scratch (graph capture) is
        // not freed at all, so it needs no sync.
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
