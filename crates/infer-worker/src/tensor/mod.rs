use half::{bf16, f16};

use crate::base::allocator::{CpuAllocator, DeviceAllocator};
use crate::base::buffer::Buffer;
use crate::base::{DeviceType, error::{Error, Result}, DataType};
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg};
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
impl Dtype for f16 { const DTYPE: DataType = DataType::F16; }
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
                in_method: "as_slice_mut".to_string()
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
    F16(TypedTensor<f16>),
    BF16(TypedTensor<bf16>),
}
macro_rules! dispatch_on_tensor {
    ($self:expr, $method:ident($($args:expr),*)) => {
        match $self {
            Tensor::F32(t) => t.$method($($args),*),
            Tensor::I32(t) => t.$method($($args),*),
            Tensor::I8(t) => t.$method($($args),*),
            Tensor::F16(t) => t.$method($($args),*),
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
            DataType::F16 => Ok(Tensor::F16(TypedTensor::<f16>::new(shape, device)?)),
            DataType::BF16 => Ok(Tensor::BF16(TypedTensor::<bf16>::new(shape, device)?)),
            _ => unimplemented!()
        }
    }

    /// 生成标准正态分布随机张量 N(0,1)。
    /// - `seed`: Some(42) 可复现，None 随机
    /// - 先在 CPU 生成 f32，再按 dtype 转换，如果 device 是 CUDA 则上传
    pub fn randn(shape: &[usize], dtype: DataType, device: DeviceType, seed: Option<u64>) -> Result<Self> {
        use rand::prelude::*;

        let num_elements: usize = shape.iter().product();

        // 1. 用 Box-Muller 变换生成 N(0,1) 随机数
        let mut rng: Box<dyn RngCore> = match seed {
            Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };
        let mut f32_data = Vec::with_capacity(num_elements);
        while f32_data.len() < num_elements {
            let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE); // 避免 log(0)
            let u2: f32 = rng.random::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f32::consts::TAU * u2;
            f32_data.push(r * theta.cos());
            if f32_data.len() < num_elements {
                f32_data.push(r * theta.sin());
            }
        }

        // 2. 创建 CPU tensor 并填充
        let mut t = Tensor::new(shape, dtype, DeviceType::Cpu)?;
        match &mut t {
            Tensor::F32(typed) => {
                typed.as_slice_mut()?.copy_from_slice(&f32_data);
            }
            Tensor::BF16(typed) => {
                let bf16_data: Vec<bf16> = f32_data.iter().map(|&v| bf16::from_f32(v)).collect();
                typed.as_slice_mut()?.copy_from_slice(&bf16_data);
            }
            Tensor::F16(typed) => {
                let f16_data: Vec<f16> = f32_data.iter().map(|&v| f16::from_f32(v)).collect();
                typed.as_slice_mut()?.copy_from_slice(&f16_data);
            }
            _ => return Err(Error::InvalidArgument(format!(
                "randn: unsupported dtype {:?}", dtype
            )).into()),
        }

        // 3. 如果目标是 CUDA，上传
        match device {
            DeviceType::Cpu => Ok(t),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(id) => t.to_cuda(id),
        }
    }
    pub fn from_buffer(buffer: Buffer, shape: &[usize], dtype: DataType) -> Result<Self> {
        match dtype {
            DataType::F32 => Ok(Tensor::F32(TypedTensor::<f32>::from_buffer(buffer, shape)?)),
            DataType::I32 => Ok(Tensor::I32(TypedTensor::<i32>::from_buffer(buffer, shape)?)),
            DataType::I8 => Ok(Tensor::I8(TypedTensor::<i8>::from_buffer(buffer, shape)?)),
            DataType::F16 => Ok(Tensor::F16(TypedTensor::<f16>::from_buffer(buffer, shape)?)),
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
            Tensor::F16(_) => DataType::F16,
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
                num_elements: self.num_elements(),
                buffer: t.buffer().clone(),
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
            Tensor::F16(t) => Ok(Tensor::F16(TypedTensor {
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

    pub fn as_f16(&self) -> Result<&TypedTensor<f16>> {
        match self {
            Tensor::F16(t) => Ok(t),
            _ => Err(Error::InvalidArgument(format!(
                "Type mismatch: expected F16, found {:?}",
                self.dtype()
            )).into()),
        }
    }

    pub fn as_f16_mut(&mut self) -> Result<&mut TypedTensor<f16>> {
        match self {
            Tensor::F16(t) => Ok(t),
            _ => Err(Error::InvalidArgument(format!(
                "Type mismatch: expected F16, found {:?}",
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
            Tensor::F16(t) => Tensor::F16(TypedTensor {
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
            Tensor::F16(t) => Tensor::F16(TypedTensor {
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
            SafetensorDtype::F16 => DataType::F16,
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
            SafetensorDtype::F16 => DataType::F16,
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
            Tensor::F16(t) => t.buffer_mut(),
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
    pub fn strides(&self) -> Vec<usize> {
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

    /// 零拷贝 reshape：只改 shape，不动数据。要求新 shape 的总元素数不变。
    ///
    /// ```ignore
    /// let t = Tensor::new(&[16, 1, 512, 512], ...)?;
    /// let t2 = t.view(&[16, 1, 1, 256, 2, 256, 2])?; // 零拷贝
    /// ```
    pub fn view(&self, new_shape: &[usize]) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.num_elements() {
            return Err(Error::InvalidArgument(format!(
                "view: new shape {:?} has {} elements, but tensor has {}",
                new_shape, new_numel, self.num_elements()
            )).into());
        }
        // 共享 buffer，只改 dims
        macro_rules! view_typed {
            ($typed:expr) => {{
                let mut t = $typed.clone();
                t.dims = Arc::from(new_shape);
                // num_elements 不变
                t
            }};
        }
        Ok(match self {
            Tensor::F32(t) => Tensor::F32(view_typed!(t)),
            Tensor::I32(t) => Tensor::I32(view_typed!(t)),
            Tensor::I8(t) => Tensor::I8(view_typed!(t)),
            Tensor::F16(t) => Tensor::F16(view_typed!(t)),
            Tensor::BF16(t) => Tensor::BF16(view_typed!(t)),
        })
    }

    /// Eager-copy permute：按指定轴顺序重新排列数据，返回新的 contiguous tensor。
    ///
    /// `perm[i]` 表示新 tensor 的第 i 维来自旧 tensor 的第 `perm[i]` 维。
    ///
    /// ```ignore
    /// // [C, F, pF, H, pH, W, pW] → [F, H, W, pF, pH, pW, C]
    /// let t2 = t.permute(&[1, 3, 5, 2, 4, 6, 0])?;
    /// ```
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        let old_shape = self.shape();
        let ndim = old_shape.len();
        if perm.len() != ndim {
            return Err(Error::InvalidArgument(format!(
                "permute: perm length {} != ndim {}", perm.len(), ndim
            )).into());
        }
        // 验证 perm 是 0..ndim 的排列
        let mut seen = vec![false; ndim];
        for &p in perm {
            if p >= ndim || seen[p] {
                return Err(Error::InvalidArgument(format!(
                    "permute: invalid permutation {:?}", perm
                )).into());
            }
            seen[p] = true;
        }

        let new_shape: Vec<usize> = perm.iter().map(|&i| old_shape[i]).collect();
        let mut out = Tensor::new(&new_shape, self.dtype(), self.device())?;
        self.permute_into(perm, &mut out)?;
        Ok(out)
    }

    /// Dst-write permute: writes `src.permute(perm)` into `dst` without
    /// allocating. `dst` must already have shape
    /// `[src.shape[perm[0]], ..., src.shape[perm[ndim-1]]]`, matching dtype
    /// and device.
    ///
    /// This is the zero-allocation entry point used by the diffusion
    /// hot path; `Tensor::permute` is a convenience wrapper around it.
    pub fn permute_into(&self, perm: &[usize], dst: &mut Tensor) -> Result<()> {
        let old_shape = self.shape();
        let ndim = old_shape.len();
        if perm.len() != ndim {
            return Err(Error::InvalidArgument(format!(
                "permute_into: perm length {} != ndim {}", perm.len(), ndim
            )).into());
        }
        // 验证 perm 是 0..ndim 的排列
        let mut seen = vec![false; ndim];
        for &p in perm {
            if p >= ndim || seen[p] {
                return Err(Error::InvalidArgument(format!(
                    "permute_into: invalid permutation {:?}", perm
                )).into());
            }
            seen[p] = true;
        }
        let expected_shape: Vec<usize> = perm.iter().map(|&i| old_shape[i]).collect();
        if dst.shape() != expected_shape.as_slice() {
            return Err(Error::InvalidArgument(format!(
                "permute_into: dst shape {:?} does not match permuted shape {:?}",
                dst.shape(), expected_shape
            )).into());
        }
        if dst.dtype() != self.dtype() {
            return Err(Error::InvalidArgument(format!(
                "permute_into: dtype mismatch src={:?} dst={:?}", self.dtype(), dst.dtype()
            )).into());
        }
        if dst.device() != self.device() {
            return Err(Error::InvalidArgument(format!(
                "permute_into: device mismatch src={:?} dst={:?}", self.device(), dst.device()
            )).into());
        }

        // CUDA: 使用原生 CUDA permute kernel（支持 F32/BF16/F16/I32）
        #[cfg(feature = "cuda")]
        if self.device() != DeviceType::Cpu {
            return self.permute_into_cuda(perm, dst);
        }

        let old_strides = self.strides();
        let n = self.num_elements();
        let new_strides = {
            let mut s = vec![1usize; ndim];
            for i in (0..ndim.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * expected_shape[i + 1];
            }
            s
        };

        macro_rules! permute_copy {
            ($src_slice:expr, $dst_slice:expr) => {{
                for flat_new in 0..n {
                    let mut old_flat = 0usize;
                    let mut rem = flat_new;
                    for j in 0..ndim {
                        let coord = rem / new_strides[j];
                        rem %= new_strides[j];
                        old_flat += coord * old_strides[perm[j]];
                    }
                    $dst_slice[flat_new] = $src_slice[old_flat];
                }
            }};
        }

        match (self, dst) {
            (Tensor::F32(s), Tensor::F32(d)) => {
                let src = s.as_slice()?;
                let dst = d.as_slice_mut()?;
                permute_copy!(src, dst);
            }
            (Tensor::BF16(s), Tensor::BF16(d)) => {
                let src = s.as_slice()?;
                let dst = d.as_slice_mut()?;
                permute_copy!(src, dst);
            }
            (Tensor::F16(s), Tensor::F16(d)) => {
                let src = s.as_slice()?;
                let dst = d.as_slice_mut()?;
                permute_copy!(src, dst);
            }
            (Tensor::I32(s), Tensor::I32(d)) => {
                let src = s.as_slice()?;
                let dst = d.as_slice_mut()?;
                permute_copy!(src, dst);
            }
            (Tensor::I8(s), Tensor::I8(d)) => {
                let src = s.as_slice()?;
                let dst = d.as_slice_mut()?;
                permute_copy!(src, dst);
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// CUDA permute: 直接调用 CUDA kernel，不绕道 CPU。
    #[cfg(feature = "cuda")]
    fn permute_into_cuda(&self, perm: &[usize], dst: &mut Tensor) -> Result<()> {
        use crate::cuda::ffi::cudaStream_t;

        unsafe extern "C" {
            fn permute_f32_forward(dst: *mut f32, src: *const f32,
                ndim: i32, new_shape: *const i64, new_strides: *const i64,
                old_strides: *const i64, perm: *const i32,
                num_elements: i64, stream: cudaStream_t);
            fn permute_bf16_forward(dst: *mut half::bf16, src: *const half::bf16,
                ndim: i32, new_shape: *const i64, new_strides: *const i64,
                old_strides: *const i64, perm: *const i32,
                num_elements: i64, stream: cudaStream_t);
            fn permute_f16_forward(dst: *mut half::f16, src: *const half::f16,
                ndim: i32, new_shape: *const i64, new_strides: *const i64,
                old_strides: *const i64, perm: *const i32,
                num_elements: i64, stream: cudaStream_t);
            fn permute_i32_forward(dst: *mut i32, src: *const i32,
                ndim: i32, new_shape: *const i64, new_strides: *const i64,
                old_strides: *const i64, perm: *const i32,
                num_elements: i64, stream: cudaStream_t);
        }

        let src_shape = self.shape();
        let ndim = src_shape.len();

        // I8 等不支持的 dtype fallback 到 CPU round-trip（走分配路径，
        // 调用方已知这是慢路径）
        if !matches!(self.dtype(), DataType::F32 | DataType::BF16 | DataType::F16 | DataType::I32) {
            let cpu_tensor = self.to_cpu()?;
            let permuted_cpu = cpu_tensor.permute(perm)?;
            let uploaded = permuted_cpu.to_device(self.device())?;
            dst.copy_from(&uploaded)?;
            return Ok(());
        }

        let old_strides: Vec<i64> = self.strides().iter().map(|&s| s as i64).collect();
        let new_shape: Vec<i64> = perm.iter().map(|&i| src_shape[i] as i64).collect();
        let mut new_strides: Vec<i64> = vec![1; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }
        let perm_i32: Vec<i32> = perm.iter().map(|&p| p as i32).collect();

        let num_elements = self.num_elements() as i64;
        let stream = crate::cuda::get_current_cuda_stream();

        match self.dtype() {
            DataType::F32 => unsafe {
                permute_f32_forward(
                    dst.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32,
                    self.as_f32()?.buffer().as_ptr() as *const f32,
                    ndim as i32, new_shape.as_ptr(), new_strides.as_ptr(),
                    old_strides.as_ptr(), perm_i32.as_ptr(), num_elements, stream,
                );
            }
            DataType::BF16 => unsafe {
                permute_bf16_forward(
                    dst.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16,
                    self.as_bf16()?.buffer().as_ptr() as *const half::bf16,
                    ndim as i32, new_shape.as_ptr(), new_strides.as_ptr(),
                    old_strides.as_ptr(), perm_i32.as_ptr(), num_elements, stream,
                );
            }
            DataType::F16 => unsafe {
                permute_f16_forward(
                    dst.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16,
                    self.as_f16()?.buffer().as_ptr() as *const half::f16,
                    ndim as i32, new_shape.as_ptr(), new_strides.as_ptr(),
                    old_strides.as_ptr(), perm_i32.as_ptr(), num_elements, stream,
                );
            }
            DataType::I32 => unsafe {
                permute_i32_forward(
                    dst.as_i32_mut()?.buffer_mut().as_mut_ptr() as *mut i32,
                    self.as_i32()?.buffer().as_ptr() as *const i32,
                    ndim as i32, new_shape.as_ptr(), new_strides.as_ptr(),
                    old_strides.as_ptr(), perm_i32.as_ptr(), num_elements, stream,
                );
            }
            _ => unreachable!(), // guarded by matches! above
        }

        Ok(())
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
            Tensor::F16(_) => Tensor::F16(TypedTensor::<f16>::from_buffer(sliced_buffer, new_shape)?),
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

    /// Stream-ordered async 版本的 [`Tensor::copy_from`]。
    ///
    /// 把拷贝排到 `stream` 上并立即返回，host 不阻塞。跨阶段的
    /// 正确性依赖于调用方在适当的位置做 `cudaStreamSynchronize`
    /// / `cudaDeviceSynchronize`（本工程 `generate()` 每个阶段末尾
    /// 已经有 sync）。
    ///
    /// CPU→CPU 的场景 stream 被忽略，语义与 `copy_from` 一致。
    #[cfg(feature = "cuda")]
    pub fn copy_from_async(
        &mut self,
        src: &Tensor,
        stream: crate::cuda::ffi::cudaStream_t,
    ) -> Result<()> {
        if self.shape().iter().product::<usize>() != src.shape().iter().product::<usize>() {
            anyhow::bail!(
                "Tensor shape mismatch for copy_from_async: dst {:?}, src {:?}",
                self.shape(),
                src.shape()
            );
        }
        if self.dtype() != src.dtype() {
            return Err(Error::InvalidArgument(format!(
                "Tensor dtype mismatch for copy_from_async: dst {:?}, src {:?}",
                self.dtype(),
                src.dtype()
            )).into());
        }
        let dst_buffer = self.buffer_mut();
        let src_buffer = src.buffer();
        dst_buffer.copy_from_async(src_buffer, stream)?;
        Ok(())
    }

    /// 自动在"当前线程的 CUDA stream"上做 async 拷贝；CPU tensor
    /// 或未启用 cuda feature 时退化为同步 `copy_from`。
    ///
    /// 这是 diffusion 热路径的首选调用方式：调用点一行替换即可
    /// 把同步 `cudaMemcpy` 变成 stream-ordered `cudaMemcpyAsync`。
    #[inline]
    pub fn copy_from_on_current_stream(&mut self, src: &Tensor) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // 只要任意一端在 CUDA 上，就走 async 路径。两端都在 CPU
            // 时 buffer 层会忽略 stream，语义等同。
            if self.device().is_cuda() || src.device().is_cuda() {
                let stream = crate::cuda::get_current_cuda_stream();
                return self.copy_from_async(src, stream);
            }
        }
        self.copy_from(src)
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

// --- 3. 运算符重载: + 和 += ---

/// `&Tensor + &Tensor` → 新 `Tensor`，委托已有 Add Op（内含 device + dtype 分发）。
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        let mut out = Tensor::new(self.shape(), self.dtype(), self.device())
            .expect("Tensor + Tensor: allocation failed");
        crate::op::add::Add::new()
            .forward(self, rhs, &mut out, None)
            .expect("Tensor + Tensor: forward failed");
        out
    }
}

/// `Tensor += &Tensor`，委托已有 AddInplace Op（内含 device + dtype 分发）。
impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, rhs: &Tensor) {
        crate::op::add_inplace::AddInplace::new()
            .forward(rhs, self, None)
            .expect("Tensor += Tensor: forward failed");
    }
}

/// `-&Tensor` → 逐元素取负，复用 `Mul<f32>`。
impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self * (-1.0_f32)
    }
}

/// `&Tensor * f32` → 逐元素乘标量，委托 `kernels::scalar_mul`（内含 device 分发）。
impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Tensor {
        let mut out = Tensor::new(self.shape(), self.dtype(), self.device())
            .expect("Tensor * f32: allocation failed");
        crate::op::scalar::scalar_mul(self, &mut out, rhs)
            .expect("Tensor * f32: kernel failed");
        out
    }
}

/// `f32 * &Tensor` → 交换律。
impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        rhs * self
    }
}

/// `&Tensor / f32` → 逐元素除标量，复用 `Mul<f32>`。
impl Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Tensor {
        self * (1.0 / rhs)
    }
}

/// `&Tensor + f32` → 逐元素加标量，委托 `kernels::scalar_add`（内含 device 分发）。
impl Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Tensor {
        let mut out = Tensor::new(self.shape(), self.dtype(), self.device())
            .expect("Tensor + f32: allocation failed");
        crate::op::scalar::scalar_add(self, &mut out, rhs)
            .expect("Tensor + f32: kernel failed");
        out
    }
}

// --- 4. 逐元素激活函数 ---

impl Tensor {
    /// 原地 SiLU 激活: self[i] = self[i] * sigmoid(self[i])
    pub fn silu_(&mut self) {
        crate::op::scalar::silu_inplace(self)
            .expect("Tensor::silu_: kernel failed");
    }

    /// 原地 tanh: self[i] = tanh(self[i])
    pub fn tanh_(&mut self) {
        crate::op::scalar::tanh_inplace(self)
            .expect("Tensor::tanh_: kernel failed");
    }

    /// 沿最后一维切分为 n 段，返回 n 个 contiguous tensor。
    /// 要求最后一维能被 n 整除。
    pub fn chunk(&self, n: usize) -> Result<Vec<Tensor>> {
        let shape = self.shape();
        let last_dim = *shape.last().ok_or_else(|| Error::InvalidArgument("chunk: empty shape".into()))?;
        if last_dim % n != 0 {
            return Err(Error::InvalidArgument(format!(
                "chunk: last dim {} not divisible by {}", last_dim, n
            )).into());
        }
        let chunk_size = last_dim / n;
        let leading: usize = self.num_elements() / last_dim;

        // 构造每段的新 shape
        let mut new_shape: Vec<usize> = shape.to_vec();
        *new_shape.last_mut().unwrap() = chunk_size;

        let mut results = Vec::with_capacity(n);
        for c in 0..n {
            let mut out = Tensor::new(&new_shape, self.dtype(), self.device())?;
            let src_offset = c * chunk_size;

            macro_rules! chunk_copy {
                ($src_typed:expr, $dst_typed:expr) => {{
                    let s = $src_typed.as_slice()?;
                    let d = $dst_typed.as_slice_mut()?;
                    for row in 0..leading {
                        let src_base = row * last_dim + src_offset;
                        let dst_base = row * chunk_size;
                        d[dst_base..dst_base + chunk_size]
                            .copy_from_slice(&s[src_base..src_base + chunk_size]);
                    }
                }};
            }

            match (self, &mut out) {
                (Tensor::F32(s), Tensor::F32(d)) => chunk_copy!(s, d),
                (Tensor::BF16(s), Tensor::BF16(d)) => chunk_copy!(s, d),
                (Tensor::F16(s), Tensor::F16(d)) => chunk_copy!(s, d),
                (Tensor::I32(s), Tensor::I32(d)) => chunk_copy!(s, d),
                (Tensor::I8(s), Tensor::I8(d)) => chunk_copy!(s, d),
                _ => unreachable!(),
            }
            results.push(out);
        }
        Ok(results)
    }

    /// 广播逐元素乘法: self[..., j] * scale[j] → new tensor
    /// self: [..., D], scale: [D]
    pub fn broadcast_mul(&self, scale: &Tensor) -> Result<Tensor> {
        let mut out = Tensor::new(self.shape(), self.dtype(), self.device())?;
        crate::op::broadcast_mul::broadcast_mul(self, scale, &mut out)?;
        out.view(self.shape())
    }

    // ──────────────────────── In-place method API ────────────────────────
    //
    // Ergonomic wrappers around the underlying `*_inplace` kernels. All of
    // these overwrite `self` in place and return `Result` so invalid
    // shapes / dtypes surface as recoverable errors (unlike the `*Assign`
    // operator overloads below, which panic on mismatch).

    /// `self[i] = silu(self[i])`
    pub fn silu(&mut self) -> Result<()> {
        crate::op::scalar::silu_inplace(self)
    }

    /// `self[i] = tanh(self[i])`
    pub fn tanh(&mut self) -> Result<()> {
        crate::op::scalar::tanh_inplace(self)
    }

    /// `self[i, j] *= row[j]` with `row.shape == [D]` matching the last dim.
    pub fn mul_row(&mut self, row: &Tensor) -> Result<()> {
        crate::op::broadcast_mul::broadcast_mul_inplace(self, row)
    }

    /// Interleaved RoPE applied in place.
    ///
    /// - `self`: `[seq, n_heads, head_dim]` (dtype F32 or BF16 on device)
    /// - `cos`, `sin`: `[seq, head_dim/2]` F32
    pub fn rope_interleaved(&mut self, cos: &Tensor, sin: &Tensor, head_dim: usize) -> Result<()> {
        crate::op::tensor_utils::apply_rope_interleaved_dev(self, cos, sin, head_dim)
    }
}

// ─────────────────────── In-place operator overloads ────────────────────
//
// Rust's `+=`, `*=` correspond to the `*Assign` traits, which take
// `&mut self` and return `()`. That matches the semantics we want for
// in-place tensor math without allocating.
//
// Shape / dtype / device mismatches **panic** here — operator overloads
// cannot return `Result`. Callers that need graceful error handling must
// invoke the free functions (`scalar_add_inplace`, `AddInplace::forward`,
// …) directly. Diffusion hot paths feed only state-resident buffers with
// statically-matched shapes, so panics are reserved for actual bugs.
//
// Note: `AddAssign<&Tensor>` is declared above (legacy), so only the
// missing variants (`+= f32`, `*= &Tensor`, `*= f32`) are added here.

impl std::ops::AddAssign<f32> for Tensor {
    /// Scalar broadcast `self += rhs`.
    fn add_assign(&mut self, rhs: f32) {
        crate::op::scalar::scalar_add_inplace(self, rhs)
            .expect("Tensor += f32: kernel failed");
    }
}

impl std::ops::MulAssign<&Tensor> for Tensor {
    /// Element-wise `self *= rhs`. Shapes must match.
    fn mul_assign(&mut self, rhs: &Tensor) {
        crate::op::tensor_utils::ewise_mul_inplace(self, rhs)
            .expect("Tensor *= &Tensor: kernel failed");
    }
}

impl std::ops::MulAssign<f32> for Tensor {
    /// Scalar broadcast `self *= rhs`. Also the canonical idiom for
    /// negation: `x *= -1.0`.
    fn mul_assign(&mut self, rhs: f32) {
        crate::op::scalar::scalar_mul_inplace(self, rhs)
            .expect("Tensor *= f32: kernel failed");
    }
}

#[cfg(test)]
mod tests {
    use crate::{base::{error::Result, DataType, DeviceType}, tensor::Tensor};

    // ──────────── In-place operator overloads ────────────

    fn make_f32_cpu(data: &[f32]) -> Result<Tensor> {
        let mut t = Tensor::new(&[data.len()], DataType::F32, DeviceType::Cpu)?;
        t.as_f32_mut()?.as_slice_mut()?.copy_from_slice(data);
        Ok(t)
    }

    #[test]
    fn op_add_assign_scalar_cpu() -> Result<()> {
        let mut x = make_f32_cpu(&[1.0, 2.0, 3.0, 4.0])?;
        x += 1.5;
        assert_eq!(x.as_f32()?.as_slice()?, &[2.5, 3.5, 4.5, 5.5]);
        Ok(())
    }

    #[test]
    fn op_mul_assign_scalar_cpu_negate() -> Result<()> {
        // `x *= -1.0` is the canonical way to negate.
        let mut x = make_f32_cpu(&[1.0, -2.0, 3.0, -4.0])?;
        x *= -1.0;
        assert_eq!(x.as_f32()?.as_slice()?, &[-1.0, 2.0, -3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn op_mul_assign_tensor_cpu() -> Result<()> {
        let mut a = make_f32_cpu(&[1.0, 2.0, 3.0, 4.0])?;
        let b = make_f32_cpu(&[2.0, 3.0, 4.0, 5.0])?;
        a *= &b;
        assert_eq!(a.as_f32()?.as_slice()?, &[2.0, 6.0, 12.0, 20.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn op_add_assign_scalar_cuda() -> Result<()> {
        let mut x = make_f32_cpu(&[1.0, 2.0, 3.0, 4.0])?.to_cuda(0)?;
        x += 1.5;
        let x_cpu = x.to_cpu()?;
        assert_eq!(x_cpu.as_f32()?.as_slice()?, &[2.5, 3.5, 4.5, 5.5]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn op_mul_assign_tensor_cuda() -> Result<()> {
        let mut a = make_f32_cpu(&[1.0, 2.0, 3.0, 4.0])?.to_cuda(0)?;
        let b = make_f32_cpu(&[2.0, 3.0, 4.0, 5.0])?.to_cuda(0)?;
        a *= &b;
        let a_cpu = a.to_cpu()?;
        assert_eq!(a_cpu.as_f32()?.as_slice()?, &[2.0, 6.0, 12.0, 20.0]);
        Ok(())
    }

    #[test]
    fn method_mul_row_matches_broadcast_mul_cpu() -> Result<()> {
        // x: [2, 4], v: [4]
        let mut x = Tensor::new(&[2, 4], DataType::F32, DeviceType::Cpu)?;
        x.as_f32_mut()?.as_slice_mut()?.copy_from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut v = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        v.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.5, 1.0, 2.0, 4.0]);

        x.mul_row(&v)?;
        assert_eq!(
            x.as_f32()?.as_slice()?,
            &[0.5, 2.0, 6.0, 16.0, 2.5, 6.0, 14.0, 32.0]
        );
        Ok(())
    }

    #[test]
    fn method_silu_matches_scalar_silu_inplace_cpu() -> Result<()> {
        let data = vec![-2.0f32, -0.5, 0.0, 0.5, 2.0];
        let mut via_method = make_f32_cpu(&data)?;
        via_method.silu()?;

        let mut via_fn = make_f32_cpu(&data)?;
        crate::op::scalar::silu_inplace(&mut via_fn)?;

        assert_eq!(via_method.as_f32()?.as_slice()?, via_fn.as_f32()?.as_slice()?);
        Ok(())
    }

    // ────────────── existing tests below ──────────────

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

    #[test]
    fn test_tensor_add_operator_f32() -> Result<()> {
        let mut a = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let mut b = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        b.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);

        let c = &a + &b;
        assert_eq!(c.as_f32()?.as_slice()?, &[11.0, 22.0, 33.0, 44.0]);
        // a 和 b 未被修改
        assert_eq!(a.as_f32()?.as_slice()?, &[1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_add_assign_operator_f32() -> Result<()> {
        let mut a = Tensor::new(&[3], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0]);

        let mut b = Tensor::new(&[3], DataType::F32, DeviceType::Cpu)?;
        b.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[100.0, 200.0, 300.0]);

        a += &b;
        assert_eq!(a.as_f32()?.as_slice()?, &[101.0, 202.0, 303.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_add_operator_bf16() -> Result<()> {
        use half::bf16;
        let mut a = Tensor::new(&[3], DataType::BF16, DeviceType::Cpu)?;
        a.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&[
            bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0),
        ]);
        let mut b = Tensor::new(&[3], DataType::BF16, DeviceType::Cpu)?;
        b.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&[
            bf16::from_f32(10.0), bf16::from_f32(20.0), bf16::from_f32(30.0),
        ]);

        let c = &a + &b;
        let result: Vec<f32> = c.as_bf16()?.as_slice()?.iter().map(|x| x.to_f32()).collect();
        assert_eq!(result, vec![11.0, 22.0, 33.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_mul_scalar_f32() -> Result<()> {
        let mut a = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let c = &a * 3.0;
        assert_eq!(c.as_f32()?.as_slice()?, &[3.0, 6.0, 9.0, 12.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_scalar_mul_commutative() -> Result<()> {
        let mut a = Tensor::new(&[3], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[2.0, 4.0, 6.0]);

        let c = 0.5_f32 * &a;
        assert_eq!(c.as_f32()?.as_slice()?, &[1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_div_scalar_f32() -> Result<()> {
        let mut a = Tensor::new(&[3], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[10.0, 20.0, 30.0]);

        let c = &a / 10.0;
        assert_eq!(c.as_f32()?.as_slice()?, &[1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_neg_f32() -> Result<()> {
        let mut a = Tensor::new(&[3], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, -2.0, 3.0]);

        let c = -&a;
        assert_eq!(c.as_f32()?.as_slice()?, &[-1.0, 2.0, -3.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_add_scalar_f32() -> Result<()> {
        let mut a = Tensor::new(&[3], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0]);

        let c = &a + 10.0;
        assert_eq!(c.as_f32()?.as_slice()?, &[11.0, 12.0, 13.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_mul_scalar_bf16() -> Result<()> {
        use half::bf16;
        let mut a = Tensor::new(&[3], DataType::BF16, DeviceType::Cpu)?;
        a.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&[
            bf16::from_f32(2.0), bf16::from_f32(4.0), bf16::from_f32(6.0),
        ]);

        let c = &a * 3.0;
        let result: Vec<f32> = c.as_bf16()?.as_slice()?.iter().map(|x| x.to_f32()).collect();
        assert_eq!(result, vec![6.0, 12.0, 18.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_euler_step() -> Result<()> {
        // x_next = x + dt * v
        let mut x = Tensor::new(&[3], DataType::F32, DeviceType::Cpu)?;
        x.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.0, 1.0, 2.0]);

        let mut v = Tensor::new(&[3], DataType::F32, DeviceType::Cpu)?;
        v.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[10.0, 20.0, 30.0]);

        let dt = 0.5_f32;
        let x_next = &x + &(&v * dt);
        assert_eq!(x_next.as_f32()?.as_slice()?, &[5.0, 11.0, 17.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_mul_scalar_cuda_f32() -> Result<()> {
        let mut a = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let a_gpu = a.to_cuda(0)?;

        let c_gpu = &a_gpu * 3.0;
        let c = c_gpu.to_cpu()?;
        assert_eq!(c.as_f32()?.as_slice()?, &[3.0, 6.0, 9.0, 12.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_add_scalar_cuda_f32() -> Result<()> {
        let mut a = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let a_gpu = a.to_cuda(0)?;

        let c_gpu = &a_gpu + 10.0;
        let c = c_gpu.to_cpu()?;
        assert_eq!(c.as_f32()?.as_slice()?, &[11.0, 12.0, 13.0, 14.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_neg_cuda_bf16() -> Result<()> {
        use half::bf16;
        let mut a = Tensor::new(&[4], DataType::BF16, DeviceType::Cpu)?;
        a.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&[
            bf16::from_f32(1.0), bf16::from_f32(-2.0), bf16::from_f32(3.0), bf16::from_f32(-4.0),
        ]);
        let a_gpu = a.to_cuda(0)?;

        let c_gpu = -&a_gpu;
        let c = c_gpu.to_cpu()?;
        let result: Vec<f32> = c.as_bf16()?.as_slice()?.iter().map(|x| x.to_f32()).collect();
        assert_eq!(result, vec![-1.0, 2.0, -3.0, 4.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_euler_step_cuda() -> Result<()> {
        // x_next = x + dt * v  (全程 GPU)
        let mut x_cpu = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        x_cpu.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.0, 1.0, 2.0, 3.0]);
        let mut v_cpu = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        v_cpu.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);

        let x = x_cpu.to_cuda(0)?;
        let v = v_cpu.to_cuda(0)?;
        let dt = 0.5_f32;
        let x_next = &x + &(&v * dt);

        let result = x_next.to_cpu()?;
        assert_eq!(result.as_f32()?.as_slice()?, &[5.0, 11.0, 17.0, 23.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_randn_shape_and_dtype() -> Result<()> {
        let t = Tensor::randn(&[2, 3, 4], DataType::F32, DeviceType::Cpu, Some(42))?;
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.dtype(), DataType::F32);
        assert_eq!(t.num_elements(), 24);
        Ok(())
    }

    #[test]
    fn test_tensor_randn_deterministic() -> Result<()> {
        let a = Tensor::randn(&[100], DataType::F32, DeviceType::Cpu, Some(123))?;
        let b = Tensor::randn(&[100], DataType::F32, DeviceType::Cpu, Some(123))?;
        assert_eq!(a.as_f32()?.as_slice()?, b.as_f32()?.as_slice()?);
        Ok(())
    }

    #[test]
    fn test_tensor_randn_statistics() -> Result<()> {
        let t = Tensor::randn(&[10000], DataType::F32, DeviceType::Cpu, Some(0))?;
        let data = t.as_f32()?.as_slice()?;
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        // N(0,1): mean ≈ 0, var ≈ 1
        assert!(mean.abs() < 0.05, "mean = {mean}, expected ~0");
        assert!((var - 1.0).abs() < 0.1, "var = {var}, expected ~1");
        Ok(())
    }

    #[test]
    fn test_tensor_randn_bf16() -> Result<()> {
        let t = Tensor::randn(&[1000], DataType::BF16, DeviceType::Cpu, Some(7))?;
        assert_eq!(t.dtype(), DataType::BF16);
        let data = t.as_bf16()?.as_slice()?;
        let mean: f32 = data.iter().map(|x| x.to_f32()).sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.1, "bf16 mean = {mean}");
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_randn_cuda() -> Result<()> {
        let t = Tensor::randn(&[2, 16, 64, 64], DataType::F32, DeviceType::Cuda(0), Some(42))?;
        assert_eq!(t.shape(), &[2, 16, 64, 64]);
        assert_eq!(t.device(), DeviceType::Cuda(0));
        // 拷回 CPU 验证统计量
        let cpu = t.to_cpu()?;
        let data = cpu.as_f32()?.as_slice()?;
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.02, "cuda randn mean = {mean}");
        Ok(())
    }

    // ==================== tanh tests ====================

    #[test]
    fn test_tanh_cpu() -> Result<()> {
        let mut t = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        t.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.0, 1.0, -1.0, 100.0]);
        t.tanh_();
        let d = t.as_f32()?.as_slice()?;
        assert!((d[0] - 0.0).abs() < 1e-5);
        assert!((d[1] - 0.7616).abs() < 1e-3);
        assert!((d[2] - (-0.7616)).abs() < 1e-3);
        assert!((d[3] - 1.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tanh_cuda_vs_cpu() -> Result<()> {
        let n = 2048;
        let mut cpu = Tensor::randn(&[n], DataType::F32, DeviceType::Cpu, Some(42))?;
        let mut gpu = cpu.to_cuda(0)?;
        cpu.tanh_();
        gpu.tanh_();
        let gpu_back = gpu.to_cpu()?;
        let cd = cpu.as_f32()?.as_slice()?;
        let gd = gpu_back.as_f32()?.as_slice()?;
        for i in 0..n {
            assert!((cd[i] - gd[i]).abs() < 1e-5,
                "tanh mismatch at {}: cpu={}, gpu={}", i, cd[i], gd[i]);
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tanh_cuda_vs_cpu_bf16() -> Result<()> {
        let n = 1024;
        let mut cpu = Tensor::randn(&[n], DataType::BF16, DeviceType::Cpu, Some(7))?;
        let mut gpu = cpu.to_cuda(0)?;
        cpu.tanh_();
        gpu.tanh_();
        let gpu_back = gpu.to_cpu()?;
        let cd = cpu.as_bf16()?.as_slice()?;
        let gd = gpu_back.as_bf16()?.as_slice()?;
        for i in 0..n {
            assert!((cd[i].to_f32() - gd[i].to_f32()).abs() < 0.02,
                "bf16 tanh mismatch at {}", i);
        }
        Ok(())
    }

    // ==================== chunk tests ====================

    #[test]
    fn test_chunk_basic() -> Result<()> {
        let mut t = Tensor::new(&[2, 8], DataType::F32, DeviceType::Cpu)?;
        let d = t.as_f32_mut()?.as_slice_mut()?;
        for i in 0..16 { d[i] = i as f32; }

        let chunks = t.chunk(4)?;
        assert_eq!(chunks.len(), 4);
        for c in &chunks { assert_eq!(c.shape(), &[2, 2]); }

        // row 0: [0,1,2,3,4,5,6,7] → chunks of 2: [0,1], [2,3], [4,5], [6,7]
        assert_eq!(chunks[0].as_f32()?.as_slice()?, &[0.0, 1.0, 8.0, 9.0]);
        assert_eq!(chunks[1].as_f32()?.as_slice()?, &[2.0, 3.0, 10.0, 11.0]);
        Ok(())
    }

    // ==================== broadcast_mul tests ====================

    #[test]
    fn test_broadcast_mul_cpu() -> Result<()> {
        // a: [3, 4], b: [4] → dst: [3, 4]
        let mut a = Tensor::new(&[3, 4], DataType::F32, DeviceType::Cpu)?;
        let ad = a.as_f32_mut()?.as_slice_mut()?;
        for i in 0..12 { ad[i] = (i + 1) as f32; }

        let mut b = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        b.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let result = a.broadcast_mul(&b)?;
        let rd = result.as_f32()?.as_slice()?;
        // row 0: 1*1, 2*2, 3*3, 4*4 = 1, 4, 9, 16
        assert_eq!(&rd[0..4], &[1.0, 4.0, 9.0, 16.0]);
        // row 1: 5*1, 6*2, 7*3, 8*4 = 5, 12, 21, 32
        assert_eq!(&rd[4..8], &[5.0, 12.0, 21.0, 32.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_broadcast_mul_cuda_vs_cpu() -> Result<()> {
        let mut a = Tensor::randn(&[64, 128], DataType::F32, DeviceType::Cpu, Some(42))?;
        let b = Tensor::randn(&[128], DataType::F32, DeviceType::Cpu, Some(7))?;

        let cpu_out = a.broadcast_mul(&b)?;

        let a_gpu = a.to_cuda(0)?;
        let b_gpu = b.to_cuda(0)?;
        let gpu_out = a_gpu.broadcast_mul(&b_gpu)?.to_cpu()?;

        let cd = cpu_out.as_f32()?.as_slice()?;
        let gd = gpu_out.as_f32()?.as_slice()?;
        for i in 0..cd.len() {
            assert!((cd[i] - gd[i]).abs() < 1e-4,
                "broadcast_mul mismatch at {}: cpu={}, gpu={}", i, cd[i], gd[i]);
        }
        Ok(())
    }

    // ==================== layernorm tests ====================

    #[test]
    fn test_layernorm_cpu() -> Result<()> {
        // [2, 4] — each row normalized to mean≈0, var≈1
        let mut input = Tensor::new(&[2, 4], DataType::F32, DeviceType::Cpu)?;
        input.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        let mut output = Tensor::new(&[2, 4], DataType::F32, DeviceType::Cpu)?;

        crate::op::layernorm::layernorm(&input, &mut output, 1e-5)?;

        let d = output.as_f32()?.as_slice()?;
        // row 0: mean=2.5, var=1.25, rstd=1/sqrt(1.25+1e-5) ≈ 0.8944
        // (1-2.5)*0.8944 ≈ -1.3416
        assert!((d[0] - (-1.3416)).abs() < 1e-3, "row 0 elem 0: {}", d[0]);
        // row mean should be ≈ 0
        let mean: f32 = d[0..4].iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "row 0 mean: {}", mean);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_layernorm_cuda_vs_cpu() -> Result<()> {
        let input = Tensor::randn(&[32, 256], DataType::F32, DeviceType::Cpu, Some(42))?;
        let mut cpu_out = Tensor::new(&[32, 256], DataType::F32, DeviceType::Cpu)?;
        crate::op::layernorm::layernorm(&input, &mut cpu_out, 1e-5)?;

        let gpu_in = input.to_cuda(0)?;
        let mut gpu_out = Tensor::new(&[32, 256], DataType::F32, DeviceType::Cuda(0))?;
        crate::op::layernorm::layernorm(&gpu_in, &mut gpu_out, 1e-5)?;
        let gpu_back = gpu_out.to_cpu()?;

        let cd = cpu_out.as_f32()?.as_slice()?;
        let gd = gpu_back.as_f32()?.as_slice()?;
        for i in 0..cd.len() {
            assert!((cd[i] - gd[i]).abs() < 1e-4,
                "layernorm mismatch at {}: cpu={}, gpu={}", i, cd[i], gd[i]);
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_layernorm_cuda_vs_cpu_bf16() -> Result<()> {
        let input = Tensor::randn(&[16, 128], DataType::BF16, DeviceType::Cpu, Some(7))?;
        let mut cpu_out = Tensor::new(&[16, 128], DataType::BF16, DeviceType::Cpu)?;
        crate::op::layernorm::layernorm(&input, &mut cpu_out, 1e-5)?;

        let gpu_in = input.to_cuda(0)?;
        let mut gpu_out = Tensor::new(&[16, 128], DataType::BF16, DeviceType::Cuda(0))?;
        crate::op::layernorm::layernorm(&gpu_in, &mut gpu_out, 1e-5)?;
        let gpu_back = gpu_out.to_cpu()?;

        let cd = cpu_out.as_bf16()?.as_slice()?;
        let gd = gpu_back.as_bf16()?.as_slice()?;
        for i in 0..cd.len() {
            assert!((cd[i].to_f32() - gd[i].to_f32()).abs() < 0.05,
                "bf16 layernorm mismatch at {}: cpu={}, gpu={}", i, cd[i].to_f32(), gd[i].to_f32());
        }
        Ok(())
    }

    // ==================== Conv2d tests ====================

    #[test]
    fn test_conv2d_cpu_basic() -> Result<()> {
        // input: [1, 1, 4, 4], weight: [1, 1, 3, 3], stride=1, padding=1
        // output: [1, 1, 4, 4]
        let mut input = Tensor::new(&[1, 1, 4, 4], DataType::F32, DeviceType::Cpu)?;
        input.as_f32_mut()?.as_slice_mut()?.fill(1.0);

        let mut weight = Tensor::new(&[1, 1, 3, 3], DataType::F32, DeviceType::Cpu)?;
        weight.as_f32_mut()?.as_slice_mut()?.fill(1.0);

        let mut bias = Tensor::new(&[1], DataType::F32, DeviceType::Cpu)?;
        bias.as_f32_mut()?.as_slice_mut()?[0] = 0.5;

        let mut output = Tensor::new(&[1, 1, 4, 4], DataType::F32, DeviceType::Cpu)?;
        crate::op::conv2d::conv2d(&input, &weight, Some(&bias), &mut output, 1, 1, None)?;

        let d = output.as_f32()?.as_slice()?;
        // center pixel: 9*1 + 0.5 = 9.5
        assert!((d[5] - 9.5).abs() < 1e-4, "center={}", d[5]);
        // corner pixel (0,0): 4*1 + 0.5 = 4.5
        assert!((d[0] - 4.5).abs() < 1e-4, "corner={}", d[0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_conv2d_cuda_vs_cpu() -> Result<()> {
        use crate::cuda::CudaConfig;

        let mut input = Tensor::randn(&[1, 3, 8, 8], DataType::F32, DeviceType::Cpu, Some(42))?;
        let weight = Tensor::randn(&[16, 3, 3, 3], DataType::F32, DeviceType::Cpu, Some(7))?;
        let bias = Tensor::randn(&[16], DataType::F32, DeviceType::Cpu, Some(13))?;

        // CPU
        let (h_out, w_out) = crate::op::conv2d::conv2d_output_size(8, 8, 3, 3, 1, 1);
        let mut cpu_out = Tensor::new(&[1, 16, h_out, w_out], DataType::F32, DeviceType::Cpu)?;
        crate::op::conv2d::conv2d(&input, &weight, Some(&bias), &mut cpu_out, 1, 1, None)?;

        // CUDA
        let gpu_input = input.to_cuda(0)?;
        let gpu_weight = weight.to_cuda(0)?;
        let gpu_bias = bias.to_cuda(0)?;
        let mut gpu_out = Tensor::new(&[1, 16, h_out, w_out], DataType::F32, DeviceType::Cuda(0))?;
        let cfg = CudaConfig::new()?;
        crate::op::conv2d::conv2d(&gpu_input, &gpu_weight, Some(&gpu_bias), &mut gpu_out, 1, 1, Some(&cfg))?;

        let gpu_back = gpu_out.to_cpu()?;
        let cd = cpu_out.as_f32()?.as_slice()?;
        let gd = gpu_back.as_f32()?.as_slice()?;
        let mut max_diff: f32 = 0.0;
        for i in 0..cd.len() {
            max_diff = max_diff.max((cd[i] - gd[i]).abs());
        }
        println!("Conv2d CPU vs CUDA max diff: {}", max_diff);
        assert!(max_diff < 1e-3, "Conv2d max diff {} too large", max_diff);
        Ok(())
    }

    // ==================== GroupNorm tests ====================

    #[test]
    fn test_groupnorm_cpu() -> Result<()> {
        // [1, 4, 2, 2], 2 groups → channels_per_group=2
        let mut input = Tensor::new(&[1, 4, 2, 2], DataType::F32, DeviceType::Cpu)?;
        let d = input.as_f32_mut()?.as_slice_mut()?;
        for i in 0..16 { d[i] = (i + 1) as f32; }

        let mut weight = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        weight.as_f32_mut()?.as_slice_mut()?.fill(1.0);
        let mut bias = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        bias.as_f32_mut()?.as_slice_mut()?.fill(0.0);

        let mut output = Tensor::new(&[1, 4, 2, 2], DataType::F32, DeviceType::Cpu)?;
        crate::op::groupnorm::groupnorm(&input, &weight, &bias, &mut output, 2, 1e-5)?;

        let r = output.as_f32()?.as_slice()?;
        // group 0 的均值应 ≈ 0
        let group0_mean: f32 = r[0..8].iter().sum::<f32>() / 8.0;
        assert!(group0_mean.abs() < 1e-4, "group0 mean: {}", group0_mean);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_groupnorm_cuda_vs_cpu() -> Result<()> {
        let input = Tensor::randn(&[2, 32, 4, 4], DataType::F32, DeviceType::Cpu, Some(42))?;
        let weight = Tensor::randn(&[32], DataType::F32, DeviceType::Cpu, Some(7))?;
        let bias = Tensor::randn(&[32], DataType::F32, DeviceType::Cpu, Some(13))?;

        let mut cpu_out = Tensor::new(&[2, 32, 4, 4], DataType::F32, DeviceType::Cpu)?;
        crate::op::groupnorm::groupnorm(&input, &weight, &bias, &mut cpu_out, 8, 1e-5)?;

        let gpu_in = input.to_cuda(0)?;
        let gpu_w = weight.to_cuda(0)?;
        let gpu_b = bias.to_cuda(0)?;
        let mut gpu_out = Tensor::new(&[2, 32, 4, 4], DataType::F32, DeviceType::Cuda(0))?;
        crate::op::groupnorm::groupnorm(&gpu_in, &gpu_w, &gpu_b, &mut gpu_out, 8, 1e-5)?;

        let gpu_back = gpu_out.to_cpu()?;
        let cd = cpu_out.as_f32()?.as_slice()?;
        let gd = gpu_back.as_f32()?.as_slice()?;
        let mut max_diff: f32 = 0.0;
        for i in 0..cd.len() {
            max_diff = max_diff.max((cd[i] - gd[i]).abs());
        }
        println!("GroupNorm CPU vs CUDA max diff: {}", max_diff);
        assert!(max_diff < 1e-3, "GroupNorm max diff {} too large", max_diff);
        Ok(())
    }

    // ==================== Upsample tests ====================

    #[test]
    fn test_upsample_nearest_2x_cpu() -> Result<()> {
        let mut input = Tensor::new(&[1, 1, 2, 2], DataType::F32, DeviceType::Cpu)?;
        input.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let mut output = Tensor::new(&[1, 1, 4, 4], DataType::F32, DeviceType::Cpu)?;
        crate::op::upsample::upsample_nearest_2x(&input, &mut output)?;

        let d = output.as_f32()?.as_slice()?;
        // [1,1,2,2] → [1,2,1,2, 3,3,4,4, 3,3,4,4]
        assert_eq!(d[0], 1.0); assert_eq!(d[1], 1.0);
        assert_eq!(d[2], 2.0); assert_eq!(d[3], 2.0);
        assert_eq!(d[4], 1.0); assert_eq!(d[5], 1.0);
        assert_eq!(d[8], 3.0); assert_eq!(d[10], 4.0);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_upsample_nearest_2x_cuda_vs_cpu() -> Result<()> {
        let input = Tensor::randn(&[2, 16, 8, 8], DataType::F32, DeviceType::Cpu, Some(42))?;

        let mut cpu_out = Tensor::new(&[2, 16, 16, 16], DataType::F32, DeviceType::Cpu)?;
        crate::op::upsample::upsample_nearest_2x(&input, &mut cpu_out)?;

        let gpu_in = input.to_cuda(0)?;
        let mut gpu_out = Tensor::new(&[2, 16, 16, 16], DataType::F32, DeviceType::Cuda(0))?;
        crate::op::upsample::upsample_nearest_2x(&gpu_in, &mut gpu_out)?;

        let gpu_back = gpu_out.to_cpu()?;
        let cd = cpu_out.as_f32()?.as_slice()?;
        let gd = gpu_back.as_f32()?.as_slice()?;
        for i in 0..cd.len() {
            assert!((cd[i] - gd[i]).abs() < 1e-6,
                "upsample mismatch at {}: cpu={}, gpu={}", i, cd[i], gd[i]);
        }
        Ok(())
    }

    // ==================== Softmax tests ====================

    #[test]
    fn test_softmax_cpu() -> Result<()> {
        let mut input = Tensor::new(&[2, 4], DataType::F32, DeviceType::Cpu)?;
        input.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);

        let mut output = Tensor::new(&[2, 4], DataType::F32, DeviceType::Cpu)?;
        crate::op::softmax::softmax(&input, &mut output)?;

        let d = output.as_f32()?.as_slice()?;
        // row 0 sum should be 1.0
        let sum0: f32 = d[0..4].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5, "row0 sum: {}", sum0);
        // row 1: uniform → all 0.25
        assert!((d[4] - 0.25).abs() < 1e-5, "uniform: {}", d[4]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_softmax_cuda_vs_cpu() -> Result<()> {
        let input = Tensor::randn(&[32, 64], DataType::F32, DeviceType::Cpu, Some(42))?;

        let mut cpu_out = Tensor::new(&[32, 64], DataType::F32, DeviceType::Cpu)?;
        crate::op::softmax::softmax(&input, &mut cpu_out)?;

        let gpu_in = input.to_cuda(0)?;
        let mut gpu_out = Tensor::new(&[32, 64], DataType::F32, DeviceType::Cuda(0))?;
        crate::op::softmax::softmax(&gpu_in, &mut gpu_out)?;

        let gpu_back = gpu_out.to_cpu()?;
        let cd = cpu_out.as_f32()?.as_slice()?;
        let gd = gpu_back.as_f32()?.as_slice()?;
        let mut max_diff: f32 = 0.0;
        for i in 0..cd.len() {
            max_diff = max_diff.max((cd[i] - gd[i]).abs());
        }
        println!("Softmax CPU vs CUDA max diff: {}", max_diff);
        assert!(max_diff < 1e-5, "Softmax max diff {} too large", max_diff);
        Ok(())
    }

    // ==================== SDPA tests ====================

    #[test]
    fn test_sdpa_cpu_basic() -> Result<()> {
        // B=1, heads=1, S=4, D=8
        let q = Tensor::randn(&[1, 1, 4, 8], DataType::F32, DeviceType::Cpu, Some(42))?;
        let k = Tensor::randn(&[1, 1, 4, 8], DataType::F32, DeviceType::Cpu, Some(7))?;
        let v = Tensor::randn(&[1, 1, 4, 8], DataType::F32, DeviceType::Cpu, Some(13))?;
        let mut output = Tensor::new(&[1, 1, 4, 8], DataType::F32, DeviceType::Cpu)?;

        crate::op::sdpa::scaled_dot_product_attention(&q, &k, &v, &mut output, None)?;

        let d = output.as_f32()?.as_slice()?;
        // output 应该是有限值，不是 NaN/Inf
        for &val in d {
            assert!(val.is_finite(), "sdpa output contains non-finite: {}", val);
        }
        Ok(())
    }

    #[test]
    fn test_sdpa_cpu_large_head_dim() -> Result<()> {
        // VAE 场景: B=1, heads=1, S=16, D=512
        let q = Tensor::randn(&[1, 1, 16, 512], DataType::F32, DeviceType::Cpu, Some(42))?;
        let k = Tensor::randn(&[1, 1, 16, 512], DataType::F32, DeviceType::Cpu, Some(7))?;
        let v = Tensor::randn(&[1, 1, 16, 512], DataType::F32, DeviceType::Cpu, Some(13))?;
        let mut output = Tensor::new(&[1, 1, 16, 512], DataType::F32, DeviceType::Cpu)?;

        crate::op::sdpa::scaled_dot_product_attention(&q, &k, &v, &mut output, None)?;

        let d = output.as_f32()?.as_slice()?;
        for &val in d {
            assert!(val.is_finite(), "sdpa output non-finite: {}", val);
        }
        Ok(())
    }

    #[test]
    fn test_sdpa_cpu_multi_head() -> Result<()> {
        // B=2, heads=4, S=8, D=64
        let q = Tensor::randn(&[2, 4, 8, 64], DataType::F32, DeviceType::Cpu, Some(42))?;
        let k = Tensor::randn(&[2, 4, 8, 64], DataType::F32, DeviceType::Cpu, Some(7))?;
        let v = Tensor::randn(&[2, 4, 8, 64], DataType::F32, DeviceType::Cpu, Some(13))?;
        let mut output = Tensor::new(&[2, 4, 8, 64], DataType::F32, DeviceType::Cpu)?;

        crate::op::sdpa::scaled_dot_product_attention(&q, &k, &v, &mut output, None)?;

        let d = output.as_f32()?.as_slice()?;
        for &val in d {
            assert!(val.is_finite(), "sdpa output non-finite: {}", val);
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_sdpa_cuda_vs_cpu() -> Result<()> {
        // B=1, heads=2, S=16, D=32
        let q = Tensor::randn(&[1, 2, 16, 32], DataType::F32, DeviceType::Cpu, Some(42))?;
        let k = Tensor::randn(&[1, 2, 16, 32], DataType::F32, DeviceType::Cpu, Some(7))?;
        let v = Tensor::randn(&[1, 2, 16, 32], DataType::F32, DeviceType::Cpu, Some(13))?;

        // CPU
        let mut cpu_out = Tensor::new(&[1, 2, 16, 32], DataType::F32, DeviceType::Cpu)?;
        crate::op::sdpa::scaled_dot_product_attention(&q, &k, &v, &mut cpu_out, None)?;

        // CUDA
        let q_gpu = q.to_cuda(0)?;
        let k_gpu = k.to_cuda(0)?;
        let v_gpu = v.to_cuda(0)?;
        let mut gpu_out = Tensor::new(&[1, 2, 16, 32], DataType::F32, DeviceType::Cuda(0))?;
        crate::op::sdpa::scaled_dot_product_attention(&q_gpu, &k_gpu, &v_gpu, &mut gpu_out, None)?;

        let gpu_back = gpu_out.to_cpu()?;
        let cd = cpu_out.as_f32()?.as_slice()?;
        let gd = gpu_back.as_f32()?.as_slice()?;
        let mut max_diff: f32 = 0.0;
        for i in 0..cd.len() {
            max_diff = max_diff.max((cd[i] - gd[i]).abs());
        }
        println!("SDPA CPU vs CUDA max diff: {}", max_diff);
        assert!(max_diff < 1e-3, "SDPA max diff {} too large", max_diff);
        Ok(())
    }
}