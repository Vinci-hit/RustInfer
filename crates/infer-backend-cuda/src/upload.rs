//! Bounded, synchronous-at-return bulk uploads. CPU fills one pinned buffer
//! while the compute stream copies the other; events protect slot reuse.
use crate::error::{allocation_error, classify_sync_error, cuda_check};
use crate::ffi;
use infer_core::ports::{OpError, OpResult};

const MIB: usize = 1024 * 1024;

#[derive(Debug)]
struct Slot {
    host: *mut std::ffi::c_void,
    done: ffi::cudaEvent_t,
    pending: bool,
    safe_to_free: bool,
}
// Slots are only accessed under the owning CudaConfig's mutex.
unsafe impl Send for Slot {}
impl Slot {
    fn new(bytes: usize) -> OpResult<Self> {
        let mut slot = Self {
            host: std::ptr::null_mut(),
            done: std::ptr::null_mut(),
            pending: false,
            safe_to_free: true,
        };
        unsafe {
            let code = ffi::cudaMallocHost(&mut slot.host, bytes);
            if code != ffi::cudaError_cudaSuccess {
                return Err(allocation_error(code, "allocate bulk upload pinned buffer"));
            }
            // cudaEventDisableTiming = 2; completion only, not profiling.
            cuda_check!(ffi::cudaEventCreateWithFlags(&mut slot.done, 2));
        }
        Ok(slot)
    }
}
impl Drop for Slot {
    fn drop(&mut self) {
        // A failed drain leaves the CUDA context unusable. Keep the pinned
        // memory alive rather than risk freeing an in-flight DMA source.
        if self.safe_to_free {
            unsafe {
                if !self.done.is_null() {
                    ffi::cudaEventDestroy(self.done);
                }
                if !self.host.is_null() {
                    ffi::cudaFreeHost(self.host);
                }
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct BulkUpload {
    slots: Option<[Slot; 2]>,
    chunk_bytes: usize,
    failed: bool,
}
impl BulkUpload {
    pub(crate) fn from_env() -> OpResult<Self> {
        let mib = match std::env::var("RUSTINFER_BULK_UPLOAD_CHUNK_MIB") {
            Ok(value) => value.parse::<usize>().map_err(|_| {
                OpError::Shape("RUSTINFER_BULK_UPLOAD_CHUNK_MIB must be 0..=64".into())
            })?,
            Err(std::env::VarError::NotPresent) => 8,
            Err(error) => return Err(OpError::Shape(error.to_string())),
        };
        if mib > 64 {
            return Err(OpError::Shape(
                "RUSTINFER_BULK_UPLOAD_CHUNK_MIB must be 0..=64".into(),
            ));
        }
        Self::new(mib * MIB)
    }

    fn new(chunk_bytes: usize) -> OpResult<Self> {
        let slots = if chunk_bytes == 0 {
            None
        } else {
            Some([Slot::new(chunk_bytes)?, Slot::new(chunk_bytes)?])
        };
        Ok(Self {
            slots,
            chunk_bytes,
            failed: false,
        })
    }

    pub(crate) fn should_stage(&self, bytes: usize) -> bool {
        // Small transfers cannot overlap two chunks; avoid an extra host copy.
        self.chunk_bytes != 0 && bytes > self.chunk_bytes
    }

    /// Caller owns the source/destination spans, has selected the device and
    /// is outside CUDA graph capture. Always drains before returning an error.
    pub(crate) unsafe fn upload(
        &mut self,
        stream: ffi::cudaStream_t,
        dst: *mut u8,
        src: *const u8,
        bytes: usize,
    ) -> OpResult<()> {
        if self.failed {
            return Err(OpError::Fatal(
                "bulk upload stream previously failed to drain".into(),
            ));
        }
        let slots = self.slots.as_mut().expect("staging enabled");
        let submitted = (|| -> OpResult<()> {
            for (index, offset) in (0..bytes).step_by(self.chunk_bytes).enumerate() {
                let slot = &mut slots[index % 2];
                if slot.pending {
                    let status = unsafe { ffi::cudaEventSynchronize(slot.done) };
                    if status != ffi::cudaError_cudaSuccess {
                        return Err(classify_sync_error(status, "wait for bulk upload slot"));
                    }
                    slot.pending = false;
                }
                let count = self.chunk_bytes.min(bytes - offset);
                unsafe {
                    // With mmap-backed input this faults/reads the next file
                    // range while the previous chunk's H2D is in flight.
                    std::ptr::copy_nonoverlapping(src.add(offset), slot.host.cast(), count);
                    cuda_check!(ffi::cudaMemcpyAsync(
                        dst.add(offset).cast(),
                        slot.host,
                        count,
                        ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
                        stream
                    ));
                    cuda_check!(ffi::cudaEventRecord(slot.done, stream));
                }
                slot.pending = true;
            }
            Ok(())
        })();
        // Even if recording the completion event failed after enqueue, the
        // source and destination must remain alive until queued copies finish.
        let drained = unsafe { ffi::cudaStreamSynchronize(stream) };
        if drained != ffi::cudaError_cudaSuccess {
            self.failed = true;
            for slot in slots {
                slot.safe_to_free = false;
            }
            return Err(classify_sync_error(drained, "drain bulk upload"));
        }
        for slot in slots {
            slot.pending = false;
        }
        submitted?;
        crate::error::check_last_error("bulk upload observed prior kernel error")
    }
}
