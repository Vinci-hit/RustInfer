//! Page-locked destinations for direct filesystem reads; no mmap-to-staging copy.
use crate::{error::allocation_error, ffi};
use infer_core::{device::HostBuffer, error::OpResult};

pub(crate) struct PinnedBuffer {
    ptr: *mut std::ffi::c_void,
    len: usize,
}
// Exclusive ownership moves between the file reader and the uploading thread.
unsafe impl Send for PinnedBuffer {}
impl PinnedBuffer {
    pub(crate) fn new(len: usize) -> OpResult<Self> {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            let code = ffi::cudaMallocHost(&mut ptr, len.max(1));
            if code != ffi::cudaError_cudaSuccess {
                return Err(allocation_error(code, "allocate layer read buffer"));
            }
            // HostBuffer exposes initialized bytes, including unused capacity.
            std::ptr::write_bytes(ptr.cast::<u8>(), 0, len);
        }
        Ok(Self { ptr, len })
    }
}
impl HostBuffer for PinnedBuffer {
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len) }
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast(), self.len) }
    }
}
impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        // The consumer's synchronous uploads finish before returning a buffer.
        // cudaFreeHost can release portable host allocations on either thread.
        unsafe {
            ffi::cudaFreeHost(self.ptr);
        }
    }
}
