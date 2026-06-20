use std::alloc::Layout;
use std::fmt::Debug;
use std::ptr::NonNull;

use crate::error::OpResult;

/// A compute device (CPU or CUDA). The TypeState axis for Tensor<T, D>.
pub trait Device: Clone + Send + Sync + Debug + 'static {
    /// Execution context (CUDA: handles+stream; CPU: ()).
    type ExecCtx: Send + Sync;
    /// Borrow the exec context.
    fn exec_ctx(&self) -> &Self::ExecCtx;
    /// Stable physical device id inside this process. CPU defaults to 0.
    fn device_id(&self) -> i32 {
        0
    }
    /// Human-readable name for errors.
    fn name(&self) -> &'static str;
}

/// Marker: "this device has host-accessible memory" (enables as_slice).
pub trait HostDevice: Device {}

/// Memory port — devices must implement raw allocation, free, and
/// host/device copy primitives. The domain `Storage` type uses this to
/// provide RAII; `Tensor::from_host_slice` / `to_host_vec` use it for I/O.
///
/// Implementations live in `infra/cpu` and `infra/cuda`. Domain code never
/// uses raw `cudaMalloc` or `std::alloc` directly — it always goes through
/// this port so device semantics stay correct.
pub trait MemoryPort: Device {
    /// Allocate `size` bytes of zero-initialized memory on this device.
    fn alloc_bytes(&self, size: usize) -> OpResult<NonNull<u8>>;

    /// Free memory previously returned by `alloc_bytes` on this device.
    ///
    /// # Safety
    /// `ptr` must have been returned by this device's `alloc_bytes` with
    /// the same `size`, and must not have been freed yet.
    unsafe fn free_bytes(&self, ptr: NonNull<u8>, size: usize);

    /// Copy `size` bytes from a host buffer to a device buffer. May be async
    /// on the device's stream — pair with `synchronize` if the host buffer
    /// will be reused or freed before completion.
    ///
    /// # Safety
    /// `dst` must be a valid device pointer with at least `size` bytes;
    /// `src` must be a valid host pointer with at least `size` bytes.
    unsafe fn upload(&self, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()>;

    /// Async H2D copy that does NOT synchronize the device stream.
    ///
    /// CPU impl: identical to `upload` (memcpy is already synchronous).
    /// CUDA impl: `cudaMemcpyAsync(H2D)` on the device's stream, no
    /// `cudaStreamSynchronize` afterwards. Subsequent kernels on the same
    /// stream see the data; cross-stream usage is undefined.
    ///
    /// # Safety
    /// Same preconditions as `upload`, plus: `src` must remain valid until
    /// the device stream consumes the copy.
    unsafe fn upload_async(&self, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()>;

    /// Copy `size` bytes from a device buffer to a host buffer. Synchronous:
    /// returns only after the copy completes.
    ///
    /// # Safety
    /// `dst` must be a valid host pointer with at least `size` bytes;
    /// `src` must be a valid device pointer with at least `size` bytes.
    unsafe fn download(&self, dst: *mut u8, src: NonNull<u8>, size: usize) -> OpResult<()>;

    /// Wait for all pending async operations on this device's stream to
    /// complete. CPU implementations may make this a no-op.
    fn synchronize(&self) -> OpResult<()>;

    /// Copy `size` bytes from one device buffer to another device buffer on
    /// the same device. The buffers must not overlap.
    ///
    /// # Safety
    /// `dst` and `src` must be valid device pointers with at least `size`
    /// bytes each. The memory regions must not overlap.
    unsafe fn copy_device_to_device(
        &self,
        dst: NonNull<u8>,
        src: NonNull<u8>,
        size: usize,
    ) -> OpResult<()>;
}

/// Layout-aware allocator port used by diffusion VAE buffer pools.
pub trait Allocator: Debug + Send + Sync {
    /// Allocate raw bytes.
    /// # Safety: caller must dealloc with same allocator + layout.
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError>;
    /// Deallocate.
    /// # Safety: ptr must come from this allocator with matching layout.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}

#[derive(Debug, thiserror::Error)]
#[error("allocation failed: {0}")]
pub struct AllocError(pub String);
