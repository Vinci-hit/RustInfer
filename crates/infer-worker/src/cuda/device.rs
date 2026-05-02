use crate::{base::error::Result, cuda_check};

use super::{ffi};
    
/// 获取当前活动的 CUDA 设备 ID.
pub fn current_device() -> Result<i32> {
    let mut device_id: i32 = -1;
    unsafe {
        cuda_check!(ffi::cudaGetDevice(&mut device_id))?;
    }
    Ok(device_id)
}

pub fn set_current_device(device_id: i32) -> Result<()> {
    unsafe {
        // 调用 FFI 函数，并用我们的宏来检查错误
        crate::cuda_check!(ffi::cudaSetDevice(device_id))?;
    }
    // 如果 cuda_check! 没有提前返回 Err，说明操作成功。
    // 我们返回 Ok(()) 表示成功且没有返回值。
    Ok(())
}

/// RAII guard: temporarily switch the current CUDA device and restore it on drop.
pub struct CudaDeviceGuard {
    prev_device: i32,
}

impl CudaDeviceGuard {
    pub fn new(device_id: i32) -> Result<Self> {
        let prev_device = current_device()?;
        if prev_device != device_id {
            set_current_device(device_id)?;
        }
        Ok(Self { prev_device })
    }
}

impl Drop for CudaDeviceGuard {
    fn drop(&mut self) {
        let _ = set_current_device(self.prev_device);
    }
}