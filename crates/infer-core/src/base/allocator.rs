#[cfg(feature = "cuda")]
use crate::cuda::{self,ffi::cudaError_cudaErrorMemoryAllocation};

use super::error::{Result, Error};
use std::alloc::Layout;
use std::ptr::NonNull;
use super::DeviceType;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::fmt::Debug;

const BIG_BUFFER_THRESHOLD: usize = 1024 * 1024; // 1MB
const GC_THRESHOLD: usize = 1024 * 1024 * 1024; // 1GB
// 用来描述池中一个内存块的状态
#[derive(Debug)]
struct CudaMemoryChunk {
    ptr: CudaPtr, // 使用 NonNull 保证指针非空
    size_bytes: usize,
    is_busy: bool,
}

/// 一个 newtype 包装，用于表示一个我们逻辑上保证了并发安全的 CUDA 指针。
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CudaPtr(NonNull<u8>);
// 这是我们向编译器做出的承诺。
// 我们说：“编译器请放心，虽然你看不懂 CudaPtr 内部的裸指针，
// 但我，作为开发者，通过外部的 Mutex 和 is_busy 逻辑，
// 保证了不会有两个线程同时操作同一个 CudaPtr 指向的内存。
// 因此，将 CudaPtr 在线程间‘发送’（转移所有权）是安全的。”
unsafe impl Send for CudaPtr {}

// 我们同样承诺：“...因此，在多个线程间‘共享’（通过 &CudaPtr 引用）也是安全的。”
unsafe impl Sync for CudaPtr {}

// 定义分配器的内部状态，由 Mutex 保护
#[derive(Debug)]
struct AllocatorState {
    /// Key: device_id, Value: 内存块列表
    small_pool: DashMap<i32, Vec<CudaMemoryChunk>>,
    large_pool: DashMap<i32, Vec<CudaMemoryChunk>>,
    /// Key: device_id, Value: 未使用内存的字节总数 (用于 GC)
    idle_bytes: DashMap<i32, usize>,
}

// CachingCudaAllocator 结构体本身只包含状态
#[derive(Debug)]
pub struct CachingCudaAllocator {
    state: AllocatorState,
}

// 使用 Lazy 创建一个线程安全的全局单例
static CACHING_ALLOCATOR: Lazy<CachingCudaAllocator> = Lazy::new(|| CachingCudaAllocator {
    state: AllocatorState {
        small_pool: DashMap::new(),
        large_pool: DashMap::new(),
        idle_bytes: DashMap::new(),
    },
});
impl CachingCudaAllocator {
     /// 获取全局唯一的分配器实例
    pub fn instance() -> &'static Self {
        &CACHING_ALLOCATOR
    }
    #[cfg(feature = "cuda")]
    /// 辅助函数：调用 cudaMalloc 并处理错误
    pub fn cuda_alloc_raw(size: usize) -> Result<CudaPtr> {
        use crate::cuda::{error::CudaError, ffi};
        if size == 0 {
            return Ok(CudaPtr(NonNull::dangling()));
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe { crate::cuda_check!(ffi::cudaMalloc(&mut ptr, size))? };
        NonNull::new(ptr as *mut u8)
            .map(CudaPtr)
            .ok_or(Error::CudaError(CudaError(cudaError_cudaErrorMemoryAllocation)).into())
    }
}
/// DeviceAllocator Trait 定义了内存分配器的通用行为.
pub trait DeviceAllocator: Debug {
    /// Allocate memory according to the given layout.
    /// 
    /// # Safety
    /// This function is unsafe because it allocates raw memory. The caller must ensure that:
    /// 1. The layout is valid (size >= 0, alignment is a power of two, etc.)
    /// 2. The returned pointer is properly aligned for the requested layout
    /// 3. The memory is deallocated using the corresponding deallocate method
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>>;

    /// 释放之前申请的内存.
    /// # Safety
    /// ptr 必须是由同一个分配器实例通过 `allocate` 分配的，且 layout 必须匹配.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// 返回此分配器关联的设备类型.
    fn device(&self) -> DeviceType;
}

// --- CPU 分配器的实现 (派生类) ---
#[derive(Debug)]
pub struct CpuAllocator;

impl DeviceAllocator for CpuAllocator {
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        let ptr = unsafe {std::alloc::alloc(layout)};
        NonNull::new(ptr).ok_or_else(|| Error::AllocationFailed("std::alloc::alloc returned null".into()).into())
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe {std::alloc::dealloc(ptr.as_ptr(), layout);}
    }

    fn device(&self) -> DeviceType {
        DeviceType::Cpu
    }
}
#[cfg(feature = "cuda")]
impl DeviceAllocator for &CachingCudaAllocator {
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        let device_id = cuda::device::current_device()?;
        let size = layout.size();

        let pool = if size > BIG_BUFFER_THRESHOLD {
            &self.state.large_pool
        } else {
            &self.state.small_pool
        };
        // 使用 DashMap 的 entry API 来获取或创建设备对应的 Vec
        let mut pool_for_device = pool.entry(device_id).or_insert_with(Vec::new);
        let best_fit_idx = if size > BIG_BUFFER_THRESHOLD {
            // 大块内存：使用 best-fit 策略（在 1MB 容忍度内）
            pool_for_device
                .iter()
                .enumerate()
                .filter(|(_, chunk)| !chunk.is_busy && chunk.size_bytes >= size && chunk.size_bytes - size < BIG_BUFFER_THRESHOLD)
                .min_by_key(|(_, chunk)| chunk.size_bytes)
                .map(|(i, _)| i)
        } else {
            // 小块内存：使用 first-fit 策略
            pool_for_device
                .iter()
                .position(|chunk| !chunk.is_busy && chunk.size_bytes >= size)
        };

        if let Some(idx) = best_fit_idx {
            let chunk = &mut pool_for_device[idx];
            chunk.is_busy = true;
            // 如果之前是空闲的，现在要从 GC 计数器中减去
            if !size > BIG_BUFFER_THRESHOLD && let Some(mut idle_bytes) = self.state.idle_bytes.get_mut(&device_id) {
                *idle_bytes -= chunk.size_bytes;
            }
            Ok(chunk.ptr.0)
        }else{
            let new_ptr = CachingCudaAllocator::cuda_alloc_raw(size).map(|p| p.0)?;
            pool_for_device.push(CudaMemoryChunk {
                ptr: CudaPtr(new_ptr),
                size_bytes: size,
                is_busy: true,
            });
            Ok(new_ptr)
        }
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        let target_ptr = CudaPtr(ptr);
        let device_id_res = cuda::device::current_device();
        if device_id_res.is_err() { return; } // 在 deallocate 中不应 panic
        let device_id = device_id_res.unwrap();
        
        // --- 在池中查找并标记为空闲 ---
        for pool in [&self.state.small_pool, &self.state.large_pool] {
            if let Some(mut pool_for_device) = pool.get_mut(&device_id)
                && let Some(chunk) = pool_for_device.iter_mut().find(|c| c.ptr == target_ptr) {

                chunk.is_busy = false;
                // 如果是小块内存，增加 GC 计数
                if chunk.size_bytes <= BIG_BUFFER_THRESHOLD {
                    let mut idle_bytes = self.state.idle_bytes.entry(device_id).or_insert(0);
                    *idle_bytes += chunk.size_bytes;
                    // --- 检查是否需要 GC ---
                    if *idle_bytes > GC_THRESHOLD {
                        CachingCudaAllocator::garbage_collect(&self.state, device_id);
                    }
                }
                return;
            }
        }
        // --- 如果在池中找不到，直接释放 ---
        let _ = unsafe {cuda::ffi::cudaFree(ptr.as_ptr() as *mut _)};
    }
    fn device(&self) -> DeviceType {
        super::DeviceType::Cuda(cuda::device::current_device().unwrap_or(0))
    }
}
#[cfg(feature = "cuda")]
// 为 CachingCudaAllocator 添加 GC 的辅助函数
impl CachingCudaAllocator {
    fn garbage_collect(state: &AllocatorState, device_id: i32) {
        // 设置正确的设备以执行 cudaFree
        if cuda::device::set_current_device(device_id).is_err() { return; };
        
        // 清理小内存池
        if let Some(mut small_pool) = state.small_pool.get_mut(&device_id) {
            // 使用 Vec::retain 高效地移除空闲块
            small_pool.retain(|chunk| {
                if !chunk.is_busy {
                    let _ = unsafe {cuda::ffi::cudaFree(chunk.ptr.0.as_ptr() as *mut _)};
                    false // 返回 false 表示从 Vec 中移除
                } else {
                    true // 返回 true 表示保留
                }
            });
        }
        
        // 重置 GC 计数器
        if let Some(mut idle_bytes) = state.idle_bytes.get_mut(&device_id) {
            *idle_bytes = 0;
        }
    }
}