//! Memory-only test device: can defer copies, fail them, and expose device IDs.
use std::ptr::NonNull;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use infer_backend_cpu::Cpu;
use infer_core::device::{Device, HostBuffer, MemoryPort};
use infer_core::dtype::quant::BlockQuantFormat;
use infer_core::error::{OpError, OpResult};
use infer_core::quantized::{BlockQuantLayout, BlockQuantView, BlockQuantWeight};
use infer_worker::components::block_quant_projection::BlockQuantProjection;

#[derive(Debug, Default)]
struct State {
    pending: Mutex<Vec<(usize, usize, usize)>>,
    host_drops: AtomicUsize,
    device_frees: AtomicUsize,
    syncs: AtomicUsize,
    fail_upload: AtomicBool,
    fail_sync: AtomicBool,
}

#[derive(Clone, Debug, Default)]
struct DeferredDevice {
    id: i32,
    state: Arc<State>,
}

impl Device for DeferredDevice {
    type ExecCtx = ();
    fn exec_ctx(&self) -> &() {
        &()
    }
    fn name(&self) -> &'static str {
        "deferred-test"
    }
    fn device_id(&self) -> i32 {
        self.id
    }
}

struct Buffer {
    bytes: Vec<u8>,
    state: Arc<State>,
}
impl HostBuffer for Buffer {
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        self.state.host_drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl MemoryPort for DeferredDevice {
    fn alloc_bytes(&self, size: usize) -> OpResult<NonNull<u8>> {
        Cpu.alloc_bytes(size)
    }
    unsafe fn free_bytes(&self, p: NonNull<u8>, size: usize) {
        self.state.device_frees.fetch_add(1, Ordering::SeqCst);
        unsafe {
            Cpu.free_bytes(p, size);
        }
    }
    unsafe fn upload(&self, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()> {
        self.state
            .pending
            .lock()
            .unwrap()
            .push((dst.as_ptr() as usize, src as usize, size));
        if self.state.fail_upload.load(Ordering::SeqCst) {
            Err(OpError::Kernel("injected upload error".into()))
        } else {
            Ok(())
        }
    }
    fn alloc_host_buffer(&self, size: usize) -> OpResult<Box<dyn HostBuffer>> {
        Ok(Box::new(Buffer {
            bytes: vec![0; size],
            state: self.state.clone(),
        }))
    }
    unsafe fn upload_async(&self, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()> {
        unsafe { self.upload(dst, src, size) }
    }
    unsafe fn download(&self, dst: *mut u8, src: NonNull<u8>, size: usize) -> OpResult<()> {
        unsafe { Cpu.download(dst, src, size) }
    }
    fn synchronize(&self) -> OpResult<()> {
        self.state.syncs.fetch_add(1, Ordering::SeqCst);
        if self.state.fail_sync.load(Ordering::SeqCst) {
            return Err(OpError::Kernel("injected sync error".into()));
        }
        for (dst, src, len) in self.state.pending.lock().unwrap().drain(..) {
            unsafe {
                Cpu.upload(NonNull::new(dst as *mut u8).unwrap(), src as *const u8, len)?;
            }
        }
        Ok(())
    }
    unsafe fn copy_device_to_device(
        &self,
        dst: NonNull<u8>,
        src: NonNull<u8>,
        size: usize,
    ) -> OpResult<()> {
        unsafe { Cpu.copy_device_to_device(dst, src, size) }
    }
}

fn upload(device: &DeferredDevice) -> OpResult<BlockQuantWeight<DeferredDevice>> {
    let layout = BlockQuantLayout::new(BlockQuantFormat::Q3_K, 1, 256).unwrap();
    let bytes = vec![37; layout.byte_len()];
    BlockQuantWeight::from_host(BlockQuantView::new(layout, &bytes).unwrap(), device)
}

#[test]
fn upload_waits_and_error_path_drains_before_freeing() {
    for fail in [false, true] {
        let device = DeferredDevice::default();
        device.state.fail_upload.store(fail, Ordering::SeqCst);
        let result = upload(&device);
        assert_eq!(device.state.syncs.load(Ordering::SeqCst), 1);
        assert!(device.state.pending.lock().unwrap().is_empty());
        assert_eq!(device.state.host_drops.load(Ordering::SeqCst), 1);
        if fail {
            assert!(result.is_err());
            assert_eq!(device.state.device_frees.load(Ordering::SeqCst), 1);
        } else {
            let weight = result.unwrap();
            assert_eq!(weight.bytes().to_host_vec().unwrap(), vec![37; 110]);
            drop(weight);
            assert_eq!(device.state.device_frees.load(Ordering::SeqCst), 1);
        }
    }
}

#[test]
fn failed_synchronization_is_fatal_and_does_not_free_pending_buffers() {
    let device = DeferredDevice::default();
    device.state.fail_sync.store(true, Ordering::SeqCst);
    assert!(upload(&device).unwrap_err().is_fatal());
    assert_eq!(device.state.host_drops.load(Ordering::SeqCst), 0);
    assert_eq!(device.state.device_frees.load(Ordering::SeqCst), 0);
    // The fatal path deliberately retains allocations. Simulate completion
    // after return to prove the source and destination have stayed valid.
    device.state.fail_sync.store(false, Ordering::SeqCst);
    device.synchronize().unwrap();
}

#[test]
fn projection_rejects_distinct_physical_devices() {
    let first = DeferredDevice::default();
    let second = DeferredDevice {
        id: 1,
        ..Default::default()
    };
    assert!(
        BlockQuantProjection::try_new(vec![upload(&first).unwrap(), upload(&second).unwrap()])
            .is_err()
    );
}
