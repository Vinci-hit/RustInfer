use half::bf16;

use crate::base::allocator::{CpuAllocator, DeviceAllocator};
use crate::base::buffer::Buffer;
use crate::base::{DeviceType, error::{Error, Result}, DataType};
use std::ops::{Index, IndexMut};
use std::sync::Arc;
use safetensors::tensor::TensorView;
use safetensors::Dtype as SafetensorDtype;

use crate::op::kernels::cpu::cast_kernel;

// 1. 定义一个 trait 来约束合法的 Tensor 数据类型
//    Send + Sync + 'static 保证了类型可以安全地在线程间传递
//    Copy 保证了类型可以按位复制，这是底层内存操作的基础
pub trait Dtype: Send + Sync + Copy + 'static {
    // 可以在这里添加关联常量
    const DTYPE: DataType;
}

// 2. 为我们支持的类型实现这个 trait
impl Dtype for f32 { const DTYPE: DataType = DataType::F32; }
impl Dtype for i32 { const DTYPE: DataType = DataType::I32; }
impl Dtype for i8 { const DTYPE: DataType = DataType::I8; }
impl Dtype for bf16 { const DTYPE: DataType = DataType::BF16; }

#[derive(Clone,Debug)] // Clone 是廉价的，因为它只克隆 Arc
pub struct TypedTensor<T: Dtype> {
    /// 形状 (dimensions)
    dims: Arc<[usize]>,
    /// 元素总数，缓存起来避免重复计算
    num_elements: usize,
    /// 底层存储的 Buffer，Arc 实现了共享所有权
    buffer: Buffer,
    /// 占位符，告诉编译器我们“拥有”类型 T 的数据
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Dtype> TypedTensor<T> {
    pub fn from_buffer(buffer: Buffer, shape: &[usize]) -> Result<Self> {
        let num_elements: usize = shape.iter().product();
        let expected_size = num_elements * std::mem::size_of::<T>();
        if buffer.len_bytes() != expected_size {
            return Err(Error::InvalidArgument(format!(
                "Buffer size {} does not match expected size {} for shape {:?} and dtype {:?}",
                buffer.len_bytes(),
                expected_size,
                shape,
                T::DTYPE
            )).into());
        }

        Ok(Self {
            dims: Arc::from(shape),
            num_elements,
            buffer,
            _phantom: std::marker::PhantomData,
        })
    }
    pub fn new(shape: &[usize], device: DeviceType) -> Result<Self> {
        let num_elements: usize = shape.iter().product();
        let size_bytes = num_elements * std::mem::size_of::<T>();
        // 临时的分配器获取逻辑，未来可以替换为更复杂的分配器管理
        let allocator: Arc<dyn DeviceAllocator + Send + Sync> = match device {
            DeviceType::Cpu => Arc::new(CpuAllocator),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => Arc::new(crate::base::allocator::CachingCudaAllocator::instance()), //最后在当前线程的设备上分配。
        };

        let buffer = Buffer::new(size_bytes, allocator)?;
        
        Ok(Self {
            dims: Arc::from(shape),
            num_elements,
            buffer,
            _phantom: std::marker::PhantomData,
        })
    }

    pub(crate) fn shape(&self) -> &Arc<[usize]> { &self.dims }
    pub(crate) fn buffer(&self) -> &Buffer { &self.buffer }
    pub(crate) fn buffer_mut(&mut self) -> &mut Buffer { &mut self.buffer }

    pub fn as_slice(&self) -> Result<&[T]> {
        if self.buffer.device() != DeviceType::Cpu {
            return Err(Error::DeviceMismatch {
                expected: DeviceType::Cpu,
                actual: self.buffer.device(),
                in_method: "as_slice".to_string()
            }.into());
        }
        let num_elements = self.dims.iter().product();
        unsafe {
            let ptr = self.buffer.as_ptr() as *const T;
            Ok(std::slice::from_raw_parts(ptr, num_elements))
        }
    }

    pub fn as_slice_mut(&mut self) -> Result<&mut [T]> {
        if self.buffer.device() != DeviceType::Cpu {
            return Err(Error::DeviceMismatch {
                expected: DeviceType::Cpu,
                actual: self.buffer.device(),
                in_method: "as_slice".to_string()
            }.into());
        }
        let num_elements = self.dims.iter().product();
        unsafe {
            let ptr = self.buffer.as_mut_ptr() as *mut T;
            Ok(std::slice::from_raw_parts_mut(ptr, num_elements))
        }
    }

    pub fn num_elements(&self) -> usize {
        self.num_elements
    }
}


/// DynTensor 是一个动态类型的张量枚举。
/// 它可以持有任何实现了 Dtype 的具体类型的 Tensor<T>。
/// 这使得我们可以在一个集合中存储不同类型的张量。
#[derive(Clone,Debug)]
pub enum Tensor {
    F32(TypedTensor<f32>),
    I32(TypedTensor<i32>),
    I8(TypedTensor<i8>),
    BF16(TypedTensor<bf16>),
    // 未来可以轻松扩展，比如 F16, BF16 等
}
macro_rules! dispatch_on_tensor {
    ($self:expr, $method:ident($($args:expr),*)) => {
        // 宏的“展开体”
        match $self {
            Tensor::F32(t) => t.$method($($args),*),
            Tensor::I32(t) => t.$method($($args),*),
            Tensor::I8(t) => t.$method($($args),*),
            Tensor::BF16(t) => t.$method($($args),*),
        }
    };
}

impl Tensor {
    pub fn new(shape: &[usize], dtype: DataType, device: DeviceType) -> Result<Self> {
        match dtype {
            DataType::F32 => Ok(Tensor::F32(TypedTensor::<f32>::new(shape, device)?)),
            DataType::I32 => Ok(Tensor::I32(TypedTensor::<i32>::new(shape, device)?)),
            DataType::I8 => Ok(Tensor::I8(TypedTensor::<i8>::new(shape, device)?)),
            DataType::BF16 => Ok(Tensor::BF16(TypedTensor::<bf16>::new(shape, device)?)),
            _ => unimplemented!()
        }
    }
    pub fn from_buffer(buffer: Buffer, shape: &[usize], dtype: DataType) -> Result<Self> {
        match dtype {
            DataType::F32 => Ok(Tensor::F32(TypedTensor::<f32>::from_buffer(buffer, shape)?)),
            DataType::I32 => Ok(Tensor::I32(TypedTensor::<i32>::from_buffer(buffer, shape)?)),
            DataType::I8 => Ok(Tensor::I8(TypedTensor::<i8>::from_buffer(buffer, shape)?)),
            DataType::BF16 => Ok(Tensor::BF16(TypedTensor::<bf16>::from_buffer(buffer, shape)?)),
            _ => unimplemented!()
        }
    }

    /// 返回张量的形状
    pub fn shape(&self) -> &[usize] {
        dispatch_on_tensor!(self, shape())
    }
    
    /// 返回张量的数据类型
    pub fn dtype(&self) -> DataType {
        match self {
            Tensor::F32(_) => DataType::F32,
            Tensor::I32(_) => DataType::I32,
            Tensor::I8(_) => DataType::I8,
            Tensor::BF16(_) => DataType::BF16,
        }
    }

    /// 返回张量所在的设备
    pub fn device(&self) -> DeviceType {
        self.buffer().device()
    }

    /// 返回底层 Buffer 的一个引用
    pub fn buffer(&self) -> &Buffer {
        dispatch_on_tensor!(self, buffer())
    }

    /// 返回元素的总数
    pub fn num_elements(&self) -> usize {
        dispatch_on_tensor!(self, num_elements())
    }

    /// 创建一个新的 Tensor 视图，具有不同的形状 (零拷贝)。
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        if self.num_elements() != new_shape.iter().product::<usize>() {
            return Err(Error::InvalidArgument(
                "Cannot reshape to a different number of elements   ".into(),
            ).into());
        }
        
        let new_dims = Arc::from(new_shape);
        // 根据 self 的类型，创建同类型的新 Tensor，共享 Buffer
        match self {
            Tensor::F32(t) => Ok(Tensor::F32(TypedTensor {
                dims: new_dims,
                num_elements: self.num_elements(), // 新形状的元素数量与原形状相同
                buffer: t.buffer().clone(), // 廉价的 Arc clone
                _phantom: std::marker::PhantomData,
            })),
            Tensor::I32(t) => Ok(Tensor::I32(TypedTensor {
                dims: new_dims,
                num_elements: self.num_elements(),
                buffer: t.buffer().clone(),
                _phantom: std::marker::PhantomData,
            })),
            Tensor::I8(t) => Ok(Tensor::I8(TypedTensor {
                dims: new_dims,
                num_elements: self.num_elements(),
                buffer: t.buffer().clone(),
                _phantom: std::marker::PhantomData,
            })),
            Tensor::BF16(t) => Ok(Tensor::BF16(TypedTensor {
                dims: new_dims,
                num_elements: self.num_elements(),
                buffer: t.buffer().clone(),
                _phantom: std::marker::PhantomData,
            })),
        }
    }

    pub fn as_f32(&self) -> Result<&TypedTensor<f32>> {
        match self {
            Tensor::F32(t) => Ok(t),
            _ => Err(Error::InvalidArgument(format!(
                "Type mismatch: expected F32, found {:?}",
                self.dtype()
            )).into()),
        }
    }

    pub fn as_i32(&self) -> Result<&TypedTensor<i32>> {
        match self {
            Tensor::I32(t) => Ok(t),
            _ => Err(Error::InvalidArgument(format!(
                "Type mismatch: expected F32, found {:?}",
                self.dtype()
            )).into()),
        }
    }

    pub fn as_f32_mut(&mut self) -> Result<&mut TypedTensor<f32>> {
        match self {
            Tensor::F32(t) => Ok(t),
            _ => Err(Error::InvalidArgument(format!(
                "Type mismatch: expected F32, found {:?}",
                self.dtype()
            )).into()),
        }
    }

    pub fn as_i32_mut(&mut self) -> Result<&mut TypedTensor<i32>> {
        match self {
            Tensor::I32(t) => Ok(t),
            _ => Err(Error::InvalidArgument(format!(
                "Type mismatch: expected F32, found {:?}",
                self.dtype()
            )).into()),
        }
    }

    pub fn as_bf16_mut(&mut self) -> Result<&mut TypedTensor<bf16>> {
        match self {
            Tensor::BF16(t) => Ok(t),
            _ => Err(Error::InvalidArgument(format!(
                "Type mismatch: expected BF16, found {:?}",
                self.dtype()
            )).into()),
        }
    }

    pub fn as_bf16(&self) -> Result<&TypedTensor<bf16>> {
        match self {
            Tensor::BF16(t) => Ok(t),
            _ => Err(Error::InvalidArgument(format!(
                "Type mismatch: expected BF16, found {:?}",
                self.dtype()
            )).into()),
        }
    }

    pub fn to_cpu(&self) -> Result<Self> {
        if self.device() == DeviceType::Cpu {
            return Ok(self.clone()); // 已经在 CPU 上，廉价克隆
        }

        // 1. 在 CPU 上创建一个新的、同样大小和类型的 Buffer
        let allocator = Arc::new(CpuAllocator);
        let mut cpu_buffer = Buffer::new(self.buffer().len_bytes(), allocator)?;

        // 2. 执行设备到 Host (CPU) 的数据拷贝
        cpu_buffer.copy_from(self.buffer())?;

        // 3. 用新的 CPU buffer 创建一个新的 Tensor
        //    我们通过 match 来构造和 self 相同类型的 Tensor 变体
        let new_typed_tensor = match self {
            Tensor::F32(t) => Tensor::F32(TypedTensor {
                dims: t.shape().clone(),
                num_elements: t.num_elements(),
                buffer: cpu_buffer,
                _phantom: std::marker::PhantomData,
            }),
            Tensor::I32(t) => Tensor::I32(TypedTensor {
                dims: t.shape().clone(),
                num_elements: t.num_elements(),
                buffer: cpu_buffer,
                _phantom: std::marker::PhantomData,
            }),
            Tensor::I8(t) => Tensor::I8(TypedTensor {
                dims: t.shape().clone(),
                num_elements: t.num_elements(),
                buffer: cpu_buffer,
                _phantom: std::marker::PhantomData,
            }),
            Tensor::BF16(t) => Tensor::BF16(TypedTensor {
                dims: t.shape().clone(),
                num_elements: t.num_elements(),
                buffer: cpu_buffer,
                _phantom: std::marker::PhantomData,
            })
        };
        Ok(new_typed_tensor)
    }

    #[cfg(feature = "cuda")]
    pub fn to_cuda(&self, device_id: i32) -> Result<Self> {
        if self.device() == DeviceType::Cuda(device_id) {
            return Ok(self.clone());
        }

        // 1. 在目标 CUDA 设备上创建一个新的 Buffer
        let allocator = Arc::new(crate::base::allocator::CachingCudaAllocator::instance());
        // 注意：CUDADeviceAllocator 内部会使用当前的设备，
        // 我们需要先设置设备
        crate::cuda::device::set_current_device(device_id)?;
        let mut gpu_buffer = Buffer::new(self.buffer().len_bytes(), allocator)?;

        // 2. 执行数据拷贝
        gpu_buffer.copy_from(self.buffer())?;

        // 3. 用新的 GPU buffer 创建一个新的 Tensor
        let new_typed_tensor = match self {
            Tensor::F32(t) => Tensor::F32(TypedTensor {
                dims: t.shape().clone(),
                num_elements: t.num_elements(),
                buffer: gpu_buffer,
                _phantom: std::marker::PhantomData,
            }),
            Tensor::I32(t) => Tensor::I32(TypedTensor {
                dims: t.shape().clone(),
                num_elements: t.num_elements(),
                buffer: gpu_buffer,
                _phantom: std::marker::PhantomData,
            }),
            Tensor::I8(t) => Tensor::I8(TypedTensor {
                dims: t.shape().clone(),
                num_elements: t.num_elements(),
                buffer: gpu_buffer,
                _phantom: std::marker::PhantomData,
            }),
            Tensor::BF16(t) => Tensor::BF16(TypedTensor {
                dims: t.shape().clone(),
                num_elements: t.num_elements(),
                buffer: gpu_buffer,
                _phantom: std::marker::PhantomData,
            })
        };
        Ok(new_typed_tensor)
    }

    /// Asynchronously transfer tensor to CUDA device using a CUDA stream.
    /// This is a non-blocking operation - caller must synchronize the stream.
    ///
    /// # Arguments
    /// * `device_id` - Target CUDA device ID
    /// * `stream` - Optional CUDA stream. If None, uses default stream.
    #[cfg(feature = "cuda")]
    pub fn to_cuda_async(&self, device_id: i32, stream: Option<crate::cuda::ffi::cudaStream_t>) -> Result<Self> {
        if self.device() == DeviceType::Cuda(device_id) {
            return Ok(self.clone());
        }

        // 1. Create a new Buffer on the target CUDA device
        let allocator = Arc::new(crate::base::allocator::CachingCudaAllocator::instance());
        crate::cuda::device::set_current_device(device_id)?;
        let mut gpu_buffer = Buffer::new(self.buffer().len_bytes(), allocator)?;

        // 2. Execute async data copy
        gpu_buffer.async_copy_from(self.buffer(), stream)?;

        // 3. Create a new Tensor with the GPU buffer
        let new_typed_tensor = match self {
            Tensor::F32(t) => Tensor::F32(TypedTensor {
                dims: t.shape().clone(),
                num_elements: t.num_elements(),
                buffer: gpu_buffer,
                _phantom: std::marker::PhantomData,
            }),
            Tensor::I32(t) => Tensor::I32(TypedTensor {
                dims: t.shape().clone(),
                num_elements: t.num_elements(),
                buffer: gpu_buffer,
                _phantom: std::marker::PhantomData,
            }),
            Tensor::I8(t) => Tensor::I8(TypedTensor {
                dims: t.shape().clone(),
                num_elements: t.num_elements(),
                buffer: gpu_buffer,
                _phantom: std::marker::PhantomData,
            }),
            Tensor::BF16(t) => Tensor::BF16(TypedTensor {
                dims: t.shape().clone(),
                num_elements: t.num_elements(),
                buffer: gpu_buffer,
                _phantom: std::marker::PhantomData,
            })
        };
        Ok(new_typed_tensor)
    }

    pub fn from_view(view: &TensorView, device: DeviceType) -> Result<Self> {
        let shape = view.shape();
        let safetensor_dtype = view.dtype();
        let data_bytes = view.data();

        // 1. 将 safetensors 的 Dtype 映射到我们自己的 DataType 枚举
        let our_dtype = match safetensor_dtype {
            SafetensorDtype::F32 => DataType::F32,
            SafetensorDtype::BF16 => DataType::BF16,
            SafetensorDtype::I32 => DataType::I32,
            SafetensorDtype::I8 => DataType::I8,
            // 如果遇到不支持的类型，返回错误
            unsupported => return Err(Error::InvalidArgument(format!(
                "Unsupported tensor dtype {:?} from safetensors file", unsupported
            )).into()),
        };

        // 2. 根据目标设备，决定如何填充 Buffer
        match device {
            DeviceType::Cpu => {
                // --- 目标是 CPU ---
                // a. 创建一个正确大小和类型的 CPU Tensor
                let mut tensor = Tensor::new(shape, our_dtype, DeviceType::Cpu)?;
                tensor.buffer_mut().copy_from_host(data_bytes)?;
                Ok(tensor)
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(device_id) => {
                // --- 目标是 CUDA ---
                // a. 在 CPU 上临时创建一个 Tensor 来持有 mmap 的数据
                //    这是一个小技巧，避免了复杂的 FFI 指针操作
                let mut cpu_tensor = Tensor::new(shape, our_dtype, DeviceType::Cpu)?;
                cpu_tensor.buffer_mut().copy_from_host(data_bytes)?;

                // b. **调用我们已经写好的 to_cuda 方法**
                //    这会处理好 CUDA 设备的设置、内存分配和 Host-to-Device 的拷贝
                let gpu_tensor = cpu_tensor.to_cuda(device_id)?;

                Ok(gpu_tensor)
            }
        }
    }
    /// 从 safetensors 的 TensorView 创建一个 **非拥有** 的 Tensor 视图。
    ///
    /// 这是一个完全的零拷贝操作。返回的 Tensor **借用** 了 `view` 中的数据。
    ///
    /// # Safety
    /// 调用者必须保证 `view` (以及其底层的 mmap) 的生命周期
    /// 比返回的 Tensor 更长。
    pub unsafe fn from_view_borrowed(view: &TensorView) -> Result<Self> {
        let shape = view.shape();
        let safetensor_dtype = view.dtype();
        let data_bytes = view.data();

        // 1. 将 safetensors 的 Dtype 映射到我们自己的 DataType 枚举
        let our_dtype = match safetensor_dtype {
            SafetensorDtype::F32 => DataType::F32,
            SafetensorDtype::BF16 => DataType::BF16,
            SafetensorDtype::I32 => DataType::I32,
            SafetensorDtype::I8 => DataType::I8,
            // 如果遇到不支持的类型，返回错误
            unsupported => return Err(Error::InvalidArgument(format!(
                "Unsupported tensor dtype {:?} from safetensors file", unsupported
            )).into()),
        };
        let buffer = unsafe {Buffer::from_external_slice(data_bytes)};
        Tensor::from_buffer(buffer, shape, our_dtype)
    }

    // `buffer_mut` 需要在 Tensor 枚举上实现
    pub fn buffer_mut(&mut self) -> &mut Buffer {
        match self {
            Tensor::F32(t) => t.buffer_mut(),
            Tensor::I32(t) => t.buffer_mut(),
            Tensor::I8(t) => t.buffer_mut(),
            Tensor::BF16(t) => t.buffer_mut(),
        }
    }

    /// 返回张量每个维度的步长（strides），以元素数量为单位。
    ///
    /// 步长对于计算多维索引到一维内存地址的偏移至关重要。
    /// 例如，一个 C-contiguous (row-major) 张量，形状为 [D1, D2, D3]，
    /// 其步长为 [D2*D3, D3, 1]。
    ///
    /// # 返回
    /// - `Vec<usize>`: 包含每个维度步长的向量。
    fn strides(&self) -> Vec<usize> {
        let shape = self.shape();
        if shape.is_empty() {
            return vec![];
        }
        
        let mut strides = vec![1; shape.len()];
        // 从倒数第二个维度开始向前计算
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
    
    /// 从当前张量中创建一个零拷贝的切片（视图）。
    ///
    /// 此方法不会分配新的计算内存，返回的 `Tensor` 将共享原始 `Tensor` 的
    /// 底层内存分配。
    ///
    /// # 参数
    /// - `offsets`: 每个维度的起始偏移量（以元素为单位），例如 `&[0, 5, 0]`。
    /// - `shape`: 新切片的形状，例如 `&[1, 10, 32]`。
    ///
    /// # 返回
    /// - `Result<Self>`: 成功时返回一个新的 `Tensor` 实例，它指向原始数据的一个子区域。
    pub fn slice(&self, offsets: &[usize], new_shape: &[usize]) -> Result<Self> {
        let original_shape = self.shape();

        // --- 1. 验证输入的有效性 ---
        if offsets.len() != original_shape.len() || new_shape.len() != original_shape.len() {
            return Err(Error::InvalidArgument(format!(
                "Slice dimensions mismatch: original_dims={}, offsets_dims={}, new_shape_dims={}",
                original_shape.len(), offsets.len(), new_shape.len()
            )).into());
        }

        // 检查切片是否会越过每个维度的边界
        for i in 0..original_shape.len() {
            if offsets[i] + new_shape[i] > original_shape[i] {
                return Err(Error::InvalidArgument(format!(
                    "Slice is out of bounds on dimension {}: offset {} + shape {} > original_shape {}",
                    i, offsets[i], new_shape[i], original_shape[i]
                )).into());
            }
        }

        // --- 2. 计算字节偏移量和新视图的字节长度 ---
        let strides = self.strides();
        let element_size_bytes = self.dtype().size_in_bytes();
        
        // a. 起始字节偏移量
        // (offset_dim0 * stride_dim0 + offset_dim1 * stride_dim1 + ...) * element_size
        let start_byte_offset = offsets.iter().zip(strides.iter())
                                       .map(|(&offset, &stride)| offset * stride)
                                       .sum::<usize>() * element_size_bytes;
        
        // b. 新视图的总字节长度
        let new_len_bytes = new_shape.iter().product::<usize>() * element_size_bytes;

        // --- 3. 调用 Buffer::slice 来创建底层的内存视图 ---
        let sliced_buffer = self.buffer().slice(start_byte_offset, new_len_bytes)?;
        
        // --- 4. 用切片出的 Buffer 和新的 Shape 构建新的 Tensor ---
        // 我们需要通过 match 来构造和 self 相同类型的 Tensor 变体
        let new_tensor = match self {
            Tensor::F32(_) => Tensor::F32(TypedTensor::<f32>::from_buffer(sliced_buffer, new_shape)?),
            Tensor::I32(_) => Tensor::I32(TypedTensor::<i32>::from_buffer(sliced_buffer, new_shape)?),
            Tensor::I8(_) => Tensor::I8(TypedTensor::<i8>::from_buffer(sliced_buffer, new_shape)?),
            Tensor::BF16(_) => Tensor::BF16(TypedTensor::<bf16>::from_buffer(sliced_buffer, new_shape)?),
        };
        
        Ok(new_tensor)
    }

    /// 将另一个张量 (`src`) 的数据拷贝到此张量 (`self`)。
    ///
    /// 这个方法会执行一次内存拷贝操作。根据源张量和目标张量的设备，
    /// 这可能是一次 Host-to-Host, Host-to-Device, Device-to-Host,
    /// 或者 Device-to-Device 的拷贝。
    ///
    /// # 参数
    /// - `src`: 源张量，数据将从这里读取。
    ///
    /// # 返回
    /// - `Result<()>`: 如果拷贝成功则返回 Ok，否则返回一个错误。
    ///
    /// # Panics
    /// - 如果 `self` 是一个从外部切片创建的只读视图（is_external() == true），
    ///   并且 `self.buffer()` 返回一个不可变引用，那么尝试调用
    ///   `self.buffer_mut()` 可能会失败。
    ///   (注：根据您 Buffer 的 as_mut_ptr 实现，这里可能不会 panic，
    ///   但逻辑上不应向只读视图写入)
    pub fn copy_from(&mut self, src: &Tensor) -> Result<()> {
        // --- 1. 检查形状和数据类型是否匹配 ---
        if self.shape().iter().product::<usize>() != src.shape().iter().product::<usize>() {
            anyhow::bail!(
                "Tensor shape mismatch for copy_from: dst shape {:?}, src shape {:?}, 而且元素个数也不一样复制不了。",
                self.shape(),
                src.shape()
            );
        }
        if self.dtype() != src.dtype() {
            return Err(Error::InvalidArgument(format!(
                "Tensor dtype mismatch for copy_from: dst dtype {:?}, src dtype {:?}",
                self.dtype(),
                src.dtype()
            )).into());
        }
        
        // --- 2. 委托给底层的 Buffer::copy_from ---
        // a. 获取对目标 Buffer (self) 的可变引用
        let dst_buffer = self.buffer_mut();
        
        // b. 获取对源 Buffer (src) 的不可变引用
        let src_buffer = src.buffer();

        // c. 调用 Buffer 的 copy_from 方法来执行实际的内存拷贝
        dst_buffer.copy_from(src_buffer)?;

        Ok(())
    }
    pub fn from_view_on_cpu(view: &TensorView) -> Result<Self> {
        // from_view 已经可以将数据加载到 CPU 上了，我们只需要确保它这样做
        Tensor::from_view(view, DeviceType::Cpu)
    }
    
    /// 一个方便的辅助方法，用于将张量移动到指定的设备。
    /// 如果张量已经在目标设备上，它会执行一次廉价的克隆（只克隆 Arc）。
    /// 否则，它会在新设备上分配内存并执行一次数据拷贝。
    pub fn to_device(&self, device: DeviceType) -> Result<Self> {
        if self.device() == device {
            return Ok(self.clone());
        }

        match device {
            DeviceType::Cpu => self.to_cpu(),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(id) => self.to_cuda(id),
        }
    }

    /// Asynchronously transfer tensor to specified device using a CUDA stream.
    /// This is a non-blocking operation for GPU transfers - caller must synchronize the stream.
    ///
    /// # Arguments
    /// * `device` - Target device
    /// * `stream` - Optional CUDA stream. Only used for CUDA transfers. If None, uses default stream.
    #[cfg(feature = "cuda")]
    pub fn to_device_async(&self, device: DeviceType, stream: Option<crate::cuda::ffi::cudaStream_t>) -> Result<Self> {
        if self.device() == device {
            return Ok(self.clone());
        }

        match device {
            DeviceType::Cpu => self.to_cpu(), // CPU transfers are always synchronous
            DeviceType::Cuda(id) => self.to_cuda_async(id, stream),
        }
    }

    /// 将张量转换为指定的数据类型。
    ///
    /// 这个方法会创建一个具有新数据类型的新张量，并执行并行的数据转换。
    ///
    /// # Arguments
    /// * `target_dtype`: 要转换到的目标 `DataType`。
    ///
    /// # Returns
    /// - `Result<Self>`: 成功时返回一个包含转换后数据的新 `Tensor`。
    pub fn to_dtype(&self, target_dtype: DataType) -> Result<Self> {
        // 如果已经是目标类型，执行廉价克隆并直接返回
        if self.dtype() == target_dtype {
            return Ok(self.clone());
        }

        // 创建一个新的、具有目标类型的空张量
        let mut new_tensor = Tensor::new(self.shape(), target_dtype, self.device())?;

        // --- 类型转换分发 ---
        // 我们使用一个宏来减少 `match` 语句的重复代码
        macro_rules! dispatch_cast {
            ($from_ty:ty, $from_tensor:expr, $to_tensor:expr, $to_dtype:expr) => {
                match $to_dtype {
                    DataType::F32 => {
                        let from_slice = $from_tensor.as_slice()?;
                        let to_slice = $to_tensor.as_f32_mut()?.as_slice_mut()?;
                        cast_kernel(from_slice, to_slice);
                    }
                    DataType::BF16 => {
                        let from_slice = $from_tensor.as_slice()?;
                        let to_slice = $to_tensor.as_bf16_mut()?.as_slice_mut()?;
                        cast_kernel(from_slice, to_slice);
                    }
                    // ... 在这里添加其他支持的目标类型
                    _ => return Err(Error::InvalidArgument(format!(
                        "Casting from {:?} to {:?} is not supported.",
                        self.dtype(),
                        target_dtype
                    )).into()),
                }
            };
        }

        // 根据源张量 (self) 的类型，调用宏进行分发
        match self {
            Tensor::F32(t) => dispatch_cast!(f32, t, &mut new_tensor, target_dtype),
            Tensor::BF16(t) => dispatch_cast!(bf16, t, &mut new_tensor, target_dtype),
            // ... 在这里添加其他支持的源类型
            _ => return Err(Error::InvalidArgument(format!(
                "Casting from {:?} is not supported.",
                self.dtype()
            )).into()),
        }

        Ok(new_tensor)
    }
}

// --- 1. 实现不可变索引 (tensor[i]) ---
impl Index<usize> for Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        // 1. 检查设备/类型 (如果 Tensor::as_f32 没做，这里需要做)
        if self.device().is_cuda() || self.dtype() != DataType::F32 {
            panic!("Attempted to index non-f32 or non-CPU Tensor directly.");
        }
        
        // 2. 边界检查
        let total_size: usize = self.shape().iter().product();
        if index >= total_size {
            panic!("Tensor index out of bounds: index {} >= total size {}", index, total_size);
        }

        // 3. 获取底层切片并返回引用
        // **注意：由于 index 方法返回 &Self::Output，我们不能在这里使用 ? 或 Result。**
        let slice = self.as_f32().unwrap().as_slice().unwrap();
        &slice[index]
    }
}

// --- 2. 实现可变索引 (tensor[i] = value) ---
impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // 1. 检查设备/类型
        if self.device().is_cuda() || self.dtype() != DataType::F32 {
            panic!("Attempted to mutable index non-f32 or non-CPU Tensor directly.");
        }

        // 2. 边界检查
        let total_size: usize = self.shape().iter().product();
        if index >= total_size {
            panic!("Tensor index out of bounds: index {} >= total size {}", index, total_size);
        }
        
        // 3. 获取底层可变切片并返回可变引用
        let slice = self.as_f32_mut().unwrap().as_slice_mut().unwrap();
        &mut slice[index]
    }
}

#[cfg(test)]
mod tests {
    use crate::{base::{error::Result, DataType, DeviceType}, tensor::Tensor};

    #[test]
    fn test_tensor_copy_from_cpu_to_cpu() -> Result<()> {
        let device = DeviceType::Cpu;
        let dtype = DataType::F32;

        // 1. 创建源张量并填充数据
        let mut src_tensor = Tensor::new(&[2, 3], dtype, device)?;
        let src_data: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        src_tensor.as_f32_mut()?.as_slice_mut()?.copy_from_slice(src_data);

        // 2. 创建一个同样大小的目标张量（内容为零或未定义）
        let mut dst_tensor = Tensor::new(&[2, 3], dtype, device)?;

        // 3. 执行拷贝
        dst_tensor.copy_from(&src_tensor)?;

        // 4. 验证目标张量的内容
        let dst_slice = dst_tensor.as_f32()?.as_slice()?;
        assert_eq!(dst_slice, src_data);

        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_copy_from_gpu_to_gpu() -> Result<()> {
        use crate::base::DataType;

        let device = DeviceType::Cuda(0);
        let dtype = DataType::F32;

        // 1. 在 GPU 上创建源张量
        // (先在 CPU 创建，再移到 GPU)
        let mut src_cpu = Tensor::new(&[4], dtype, DeviceType::Cpu)?;
        src_cpu.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let src_gpu = src_cpu.to_cuda(0)?;
        
        // 2. 在 GPU 上创建目标张量
        let mut dst_gpu = Tensor::new(&[4], dtype, device)?;

        // 3. 执行 D2D 拷贝
        dst_gpu.copy_from(&src_gpu)?;

        // 4. 将目标张量拷回 CPU 以进行验证
        let dst_cpu = dst_gpu.to_cpu()?;
        let dst_slice = dst_cpu.as_f32()?.as_slice()?;
        assert_eq!(dst_slice, &[1.0, 2.0, 3.0, 4.0]);
        
        Ok(())
    }
}