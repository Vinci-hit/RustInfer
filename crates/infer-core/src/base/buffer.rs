use super::allocator::DeviceAllocator;
use super::error::{Result,Error};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;
use super::DeviceType;

// 这是 Buffer 的内部核心，真正持有资源的地方。
// Arc 指向的就是这个结构体。
#[derive(Debug)]
struct BufferInner {
    ptr: NonNull<u8>,             // “地契”上记录的地址
    len_bytes: usize,             // “地契”上记录的面积
    allocator: Arc<dyn DeviceAllocator + Send + Sync>,// “地契”上记录了是哪个“开发商”分配的
}

/// Buffer 是一个线程安全的、引用计数的内存缓冲区.
/// 它是内存的“所有者”。
#[derive(Clone,Debug)]
pub struct Buffer {
    /// 指向真正拥有内存的 Allocation 对象的共享指针。
    /// 当最后一个指向此 Allocation 的 Arc 被销毁时，内存才会被释放。
    inner: Option<Arc<BufferInner>>,
    /// 此 Buffer 视图可见的起始指针。
    ptr: NonNull<u8>,
    /// 此 Buffer 视图的字节长度。
    len_bytes: usize,
    /// 此 Buffer 视图所在的设备类型。
    device: DeviceType,
}

impl Buffer {
    /// 创建一个新的 Buffer. 这就是“获取资源”的时刻 (Acquisition).
    pub fn new(len_bytes: usize, allocator: Arc<dyn DeviceAllocator + Send + Sync>) -> Result<Self> {
        let layout = Self::calculate_layout(len_bytes)?;
        let ptr = if len_bytes > 0 {
            unsafe { allocator.allocate(layout)? }
        } else {
            // 对于零长度的分配，使用一个对齐的悬空指针
            NonNull::dangling()
        };
        
        let device = allocator.device();

        let inner = Arc::new(BufferInner {
            ptr,
            len_bytes,
            allocator : allocator,
        });

        Ok(Buffer {
            inner : Some(inner),
            ptr,
            len_bytes,
            device,
        })
    }

    /// 创建一个只读的 Buffer，用于包装一个外部的、已存在的内存切片。
    /// 这个 Buffer 不拥有内存的所有权，也不会在 drop 时释放它。
    /// 对应 C++ 中 `use_external = true` 的情况。
    /// # Safety
    /// 调用者必须保证 `data` 在此 Buffer 的整个生命周期内都有效。
    pub unsafe fn from_external_slice<T>(data: &[T]) -> Self {
        let len_bytes = std::mem::size_of_val(data);
        
        let ptr = if len_bytes > 0 {
            NonNull::new(data.as_ptr() as *mut u8).expect("Slice pointer cannot be null")
        } else {
            // 对于零长度的外部切片，也使用 dangling 指针
            NonNull::dangling()
        };

        Buffer {
            // 关键：设置为 None，表示此 Buffer 不拥有内存，也不会尝试释放它。
            inner: None,
            ptr,
            len_bytes,
            // 外部切片总是被假定为在 CPU 上。
            device: DeviceType::Cpu,
        }
    }

    /// 返回指向此视图内存的只读裸指针 (*const u8)。
    pub fn as_ptr(&self) -> *const u8 {
        // 直接返回视图自己的 ptr
        self.ptr.as_ptr()
    }
    /// 返回指向此视图内存的可变裸指针 (*mut u8)。
    /// 
    /// # Safety
    /// 调用者必须确保没有其他线程同时在访问或修改这块内存，
    /// 并且修改的内容不会违反数据类型的内部规则。
    /// 特别注意：对于从 `from_external_slice` 创建的只读 Buffer，
    /// 对其进行写操作可能会导致未定义行为。
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        // 直接返回视图自己的 ptr
        self.ptr.as_ptr()
    }

    /// 返回此 Buffer 视图的字节大小。
    pub fn len_bytes(&self) -> usize {
        // 直接返回视图自己的 len_bytes
        self.len_bytes
    }

    /// 返回 Buffer 关联的设备类型。
    pub fn device(&self) -> DeviceType {
        // 直接返回视图自己的 device 字段
        self.device
    }

    /// 返回对此 Buffer 底层内存分配器的共享引用 (如果存在)。
    ///
    /// 对于从 `from_external_slice` 创建的 Buffer，这将返回 `None`。
    pub fn allocator(&self) -> Option<&Arc<dyn DeviceAllocator + Send + Sync>> {
        // 1. self.allocation 是 Option<Arc<BufferAllocation>>
        // 2. .as_ref() 把它变成 Option<&Arc<BufferAllocation>>
        // 3. .map(|arc_alloc| &arc_alloc.allocator) 深入到 BufferAllocation 内部，
        //    取出对 allocator 字段的引用。
        //    最终结果是 Option<&Arc<dyn DeviceAllocator>>
        self.inner.as_deref().map(|arc_alloc| &arc_alloc.allocator)
    }

    /// 检查此 Buffer 是否是对外部内存的借用视图 (不拥有所有权)。
    ///
    /// 如果返回 `true`，则此 Buffer 在销毁时不会尝试释放其指向的内存。
    pub fn is_external(&self) -> bool {
        // `allocation` 字段为 `None` 就意味着它是外部的。
        self.inner.is_none()
    }
    
    /// 辅助函数，用于计算内存布局
    fn calculate_layout(len_bytes: usize) -> Result<Layout> {
        Layout::from_size_align(len_bytes, 16) // 假设 16 字节对齐
            .map_err(|_| Error::InvalidArgument("Invalid layout".into()).into())
    }

    /// 将 Buffer 的内容全部清零.
    pub fn zero_out(&mut self) -> Result<()> {
        match self.device() {
            DeviceType::Cpu => {
                // 对于 CPU 内存，使用安全的 Rust slice 操作
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len_bytes())
                };
                slice.fill(0);
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                // 对于 CUDA 内存，调用 cudaMemset
                unsafe {
                    use crate::cuda;

                    crate::cuda_check!(cuda::ffi::cudaMemset(
                        self.as_mut_ptr() as *mut _,
                        0, // 要设置的值
                        self.len_bytes()
                    ))?;
                }
            }
        }
        Ok(())
    }

    /// 从另一个 Buffer 拷贝数据到此 Buffer.
    /// 此方法会自动推断正确的 MemcpyKind.
    pub fn copy_from(&mut self, src: &Buffer) -> Result<()> {
        if self.len_bytes() != src.len_bytes() {
            return Err(Error::InvalidArgument(format!(
                "Buffer size mismatch: dst has {} bytes, src has {} bytes",
                self.len_bytes(),
                src.len_bytes()
            )).into());
        }
        if self.len_bytes() == 0 {
            return Ok(());
        }

        match (src.device(), self.device()) {
            (DeviceType::Cpu, DeviceType::Cpu) => {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.as_ptr(),
                        self.as_mut_ptr(),
                        self.len_bytes()
                    );
                }
            }

            #[cfg(feature = "cuda")]
            (DeviceType::Cpu, DeviceType::Cuda(_)) => {
                // ==================== 修正点 ====================
                // 之前: crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice
                // 现在:
                let kind = crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice;
                unsafe {
                    crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                        self.as_mut_ptr() as *mut _,
                        src.as_ptr() as *const _,
                        self.len_bytes(),
                        kind
                    ))?;
                }
                // ================================================
            }

            #[cfg(feature = "cuda")]
            (DeviceType::Cuda(_), DeviceType::Cpu) => {
                let kind = crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost;
                unsafe {
                    crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                        self.as_mut_ptr() as *mut _,
                        src.as_ptr() as *const _,
                        self.len_bytes(),
                        kind
                    ))?;
                }
            }
            
            #[cfg(feature = "cuda")]
            (DeviceType::Cuda(_), DeviceType::Cuda(_)) => {
                let kind = crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice;
                unsafe {
                    crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                        self.as_mut_ptr() as *mut _,
                        src.as_ptr() as *const _,
                        self.len_bytes(),
                        kind
                    ))?;
                }
            }
        }
        
        Ok(())
    }

    pub fn copy_from_host<T: Copy>(&mut self, host_slice: &[T]) -> Result<()> {
        // 计算 slice 的总字节数
        let size_bytes = std::mem::size_of_val(host_slice);

        // 检查 Buffer 是否有足够的空间容纳 slice
        if self.len_bytes() < size_bytes {
            return Err(Error::InvalidArgument(format!(
                "Buffer too small: buffer has {} bytes, but slice needs {} bytes",
                self.len_bytes(),
                size_bytes
            )).into());
        }

        // 这里不再需要 match 来获取 kind，因为 kind 是在 match 内部使用的
        match self.device() {
            DeviceType::Cpu => {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        host_slice.as_ptr() as *const u8,
                        self.as_mut_ptr(),
                        size_bytes
                    );
                }
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                let kind = crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice; // <--- 修正
                unsafe {
                    crate::cuda_check!(crate::cuda::ffi::cudaMemcpy(
                        self.as_mut_ptr() as *mut _,
                        host_slice.as_ptr() as *const _,
                        size_bytes,
                        kind
                    ))?;
                }
            }
        }
        Ok(())
    }
    /// 创建一个共享底层内存的、新的零拷贝 Buffer 视图（切片）。
    pub fn slice(&self, offset_bytes: usize, len_bytes: usize) -> Result<Self> {
        if offset_bytes + len_bytes > self.len_bytes() {
            return Err(Error::InvalidArgument(format!(
                "Slice is out of bounds: offset ({}) + len ({}) > view_len ({})",
                offset_bytes, len_bytes, self.len_bytes()
            )).into());
        }

        Ok(Buffer {
            // 关键：克隆 Option<Arc<...>>。如果原始的是 Some，我们共享所有权；
            // 如果原始的是 None (外部的)，我们继续保持 None。
            inner: self.inner.clone(),
            ptr: unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(offset_bytes)) },
            len_bytes,
            device: self.device,
        })
    }
}

// ======================= RAII 的核心在这里 =======================
// 这是 BufferInner 的“遗言”。
// 当最后一个指向 BufferInner 的 Arc 被销毁时，这个 drop 方法会被自动调用。
impl Drop for BufferInner {
    fn drop(&mut self) {
        if self.len_bytes > 0 {
                
            let layout = Buffer::calculate_layout(self.len_bytes).unwrap();
            unsafe {
                self.allocator.deallocate(self.ptr, layout);
            }
            
        }
    }
}
// =================================================================

#[cfg(test)]
mod tests {
    use super::*; // 导入父模块（也就是 buffer.rs）的所有公共项
    use crate::base::allocator::CpuAllocator;

    #[test]
    fn test_buffer_allocate() {
        let allocator = Arc::new(CpuAllocator);
        
        let buffer = Buffer::new(32, allocator).unwrap();
        assert!(!buffer.as_ptr().is_null());
        assert_eq!(buffer.len_bytes(), 32);
    }

    #[test]
    fn test_buffer_use_external() {
        let external_data: Vec<f32> = vec![0.0; 32];
        let buffer = unsafe { Buffer::from_external_slice(&external_data) };

        assert!(buffer.is_external());
        // 我们还可以检查指针地址是否一致
        assert_eq!(buffer.as_ptr(), external_data.as_ptr() as *const u8);

    }

    #[test]
    fn test_cpu_buffer_zero_out() {
        let allocator = Arc::new(CpuAllocator);
        let mut buffer = Buffer::new(4, allocator).unwrap();
        
        // 填充非零数据
        unsafe {
            let slice = std::slice::from_raw_parts_mut(buffer.as_mut_ptr(), 4);
            slice.copy_from_slice(&[1u8, 2, 3, 4]);
        }

        // 调用清零方法
        buffer.zero_out().unwrap();

        // 验证结果
        unsafe {
            let slice = std::slice::from_raw_parts(buffer.as_ptr(), 4);
            assert_eq!(slice, &[0u8, 0, 0, 0]);
        }
    }
    
    #[test]
    fn test_cpu_buffer_copy_from_host() {
        let allocator = Arc::new(CpuAllocator);
        let mut buffer = Buffer::new(16, allocator).unwrap();
        let data_to_copy: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        buffer.copy_from_host(&data_to_copy).unwrap();

        unsafe {
            // 将裸指针转换回 f32 slice 来验证内容
            let ptr = buffer.as_ptr() as *const f32;
            let slice = std::slice::from_raw_parts(ptr, 4);
            assert_eq!(slice, &data_to_copy[..]);
        }
    }
    
    // ========================================================================
    //  CUDA 测试 (需要启用 "cuda" feature)
    // ========================================================================
    // 这个属性表示，只有在 cargo test --features "cuda" 时才运行这个测试
    #[test]
    #[cfg(feature = "cuda")]
    fn test_buffer_cpu_to_cuda() -> Result<()> { // 让测试返回 Result
        use crate::base::allocator::CachingCudaAllocator;
        
        let size = 32;
        let byte_size = size * std::mem::size_of::<f32>();

        // 1. 准备 CPU 数据
        let host_data_in: Vec<f32> = (0..size).map(|i| i as f32).collect();

        // 2. 创建一个包装了 CPU 数据的外部 Buffer
        let cpu_buffer = unsafe { Buffer::from_external_slice(&host_data_in) };
        assert!(cpu_buffer.is_external());
        assert_eq!(cpu_buffer.device(), DeviceType::Cpu);

        // 3. 创建一个 CUDA Buffer
        let allocator_cu = Arc::new(CachingCudaAllocator::instance());
        let mut gpu_buffer = Buffer::new(byte_size, allocator_cu)?;
        assert_eq!(gpu_buffer.device(), DeviceType::Cuda(0)); // 假设 device 0

        // 4. 执行拷贝: CPU -> CUDA
        gpu_buffer.copy_from(&cpu_buffer)?;

        // 5. 验证数据：将数据从 CUDA 拷贝回一个新的 CPU 数组进行验证
        // Rust: 我们需要实现一个 copy_to_host 方法来让 API 更优雅
        // 为了先让测试跑通，我们直接用 copy_from 来模拟 D->H 拷贝
        let mut cpu_verify_buffer = Buffer::new(byte_size, Arc::new(CpuAllocator))?;
        cpu_verify_buffer.copy_from(&gpu_buffer)?;
        // 现在数据在 cpu_verify_buffer 里了，我们可以安全地访问它
        let result_slice = unsafe {
            std::slice::from_raw_parts(cpu_verify_buffer.as_ptr() as *const f32, size)
        };

        assert_eq!(result_slice, &host_data_in[..]);

        // 所有内存在函数结束时都会被 RAII 自动管理和释放，无需手动 delete
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_buffer_cuda_to_cuda() -> Result<()> {
        use crate::base::allocator::CachingCudaAllocator;

        let size = 32;
        let byte_size = size * std::mem::size_of::<f32>();
        let allocator_cu = Arc::new(CachingCudaAllocator::instance());

        // 1. 创建两个 CUDA Buffer
        let mut gpu_buffer1 = Buffer::new(byte_size, allocator_cu.clone())?;
        let mut gpu_buffer2 = Buffer::new(byte_size, allocator_cu.clone())?;
        assert_eq!(gpu_buffer1.device(), DeviceType::Cuda(0));
        assert_eq!(gpu_buffer2.device(), DeviceType::Cuda(0));

        // 2. 初始化 gpu_buffer2 的数据
        // C++: set_value_cu((float*)cu_buffer2.ptr(), size);
        // Rust: 我们先在 CPU 创建数据，再拷贝上去
        let host_data: Vec<f32> = vec![1.0; size];
        gpu_buffer2.copy_from_host(&host_data)?;

        // 3. 执行拷贝: CUDA -> CUDA
        gpu_buffer1.copy_from(&gpu_buffer2)?;

        // 4. 验证数据：将 gpu_buffer1 的数据拷贝回 CPU
        // 我们先用之前的模拟方式
        let mut cpu_verify_buffer = Buffer::new(byte_size, Arc::new(CpuAllocator))?;
        cpu_verify_buffer.copy_from(&gpu_buffer1)?;
        let result_slice = unsafe {
            std::slice::from_raw_parts(cpu_verify_buffer.as_ptr() as *const f32, size)
        };

        assert_eq!(result_slice, &host_data[..]);

        Ok(())
    }

    #[test]
    fn test_buffer_slice() {
        let allocator = Arc::new(CpuAllocator);
        let buffer = Buffer::new(100, allocator).unwrap();

        // 创建一个从字节 10 开始，长度为 20 的切片
        let slice = buffer.slice(10, 20).unwrap();
        
        assert_eq!(slice.len_bytes(), 20);
        
        // 检查切片的指针是否正确
        let expected_ptr = unsafe { buffer.as_ptr().add(10) };
        assert_eq!(slice.as_ptr(), expected_ptr);

        // 检查分配器是否被共享
        assert!(Arc::ptr_eq(slice.allocator().unwrap(), buffer.allocator().unwrap()));
    }

    #[test]
    fn test_buffer_slice_out_of_bounds() {
        let allocator = Arc::new(CpuAllocator);
        let buffer = Buffer::new(100, allocator).unwrap();

        // 尝试创建一个越界的切片
        let result = buffer.slice(90, 20);
        assert!(result.is_err());
        
        let result2 = buffer.slice(101, 1);
        assert!(result2.is_err());
    }
}