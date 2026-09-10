//! Persistent event pairs for optional, nonblocking compute-stream sampling.
use crate::error::cuda_check;
use crate::{Cuda, CudaScope, ffi};
use infer_core::exec::{ExecScope, ScopeTimer};
use infer_core::ports::OpResult;

pub(crate) struct CudaTimer {
    // Keeps the stream/device alive until both events have been destroyed.
    scope: CudaScope,
    start: ffi::cudaEvent_t,
    stop: ffi::cudaEvent_t,
}
// All event use is serialized through &mut self; CUDA handles may move threads.
unsafe impl Send for CudaTimer {}
impl CudaTimer {
    pub(crate) fn new(device: Cuda) -> OpResult<Self> {
        let scope = CudaScope::new(device);
        let mut timer = Self {
            scope,
            start: std::ptr::null_mut(),
            stop: std::ptr::null_mut(),
        };
        {
            let _guard = timer.scope.enter();
            unsafe {
                cuda_check!(ffi::cudaEventCreate(&mut timer.start));
                cuda_check!(ffi::cudaEventCreate(&mut timer.stop));
            }
        }
        Ok(timer)
    }
}
impl ScopeTimer for CudaTimer {
    fn start(&mut self) -> OpResult<()> {
        let _guard = self.scope.enter();
        unsafe {
            cuda_check!(ffi::cudaEventRecord(self.start, self.scope.stream().0));
        }
        Ok(())
    }
    fn stop(&mut self) -> OpResult<()> {
        let _guard = self.scope.enter();
        unsafe {
            cuda_check!(ffi::cudaEventRecord(self.stop, self.scope.stream().0));
        }
        Ok(())
    }
    fn elapsed_ms(&mut self) -> OpResult<Option<f32>> {
        let _guard = self.scope.enter();
        let status = unsafe { ffi::cudaEventQuery(self.stop) };
        if status == ffi::cudaError_cudaErrorNotReady {
            return Ok(None);
        }
        cuda_check!(status);
        let mut elapsed = 0.0;
        unsafe {
            cuda_check!(ffi::cudaEventElapsedTime(
                &mut elapsed,
                self.start,
                self.stop
            ));
        }
        Ok(Some(elapsed))
    }
}
impl Drop for CudaTimer {
    fn drop(&mut self) {
        let _guard = self.scope.enter();
        unsafe {
            if !self.start.is_null() {
                ffi::cudaEventDestroy(self.start);
            }
            if !self.stop.is_null() {
                ffi::cudaEventDestroy(self.stop);
            }
        }
    }
}
