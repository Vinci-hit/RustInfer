pub mod error;
pub mod allocator;
pub mod buffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    // 包含一个 i32 来表示 GPU 的设备 ID
    #[cfg(feature = "cuda")]
    Cuda(i32),
}

impl DeviceType {
    /// 检查设备类型是否是 CPU。
    ///
    /// # Returns
    /// 如果设备是 DeviceType::Cpu，则返回 true；否则返回 false。
    pub fn is_cpu(&self) -> bool {
        // 使用模式匹配来检查变体
        match self {
            DeviceType::Cpu => true,
            _ => false,
        }
    }
    
    /// 检查设备类型是否是 CUDA GPU。
    pub fn is_cuda(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => true,
            _ => false,
        }

    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    UnKnown,
    F32,
    F16,
    I8,
    I16,
    I32,
    BF16,
}

impl DataType {
    /// 返回数据类型的字节大小。
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 => 2,
            DataType::I8 => 1,
            DataType::I16 => 2,
            DataType::I32 => 4,
            DataType::BF16 => 2,
            DataType::UnKnown => 0,
        }
    }
}