//! Worker Configuration Module
//!
//! Configuration structures for initializing and managing Worker instances.

use crate::base::DeviceType;

/// Worker configuration
///
/// Contains all the settings needed to initialize a Worker instance.
/// Note: model_path is NOT stored here - it comes from Scheduler via ModelLoadParams.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Worker unique identifier
    pub worker_id: String,
    /// Device ID (GPU index or 0 for CPU)
    pub device_id: u32,
    /// Device type (CPU or CUDA)
    pub device_type: DeviceType,
    /// Data type for inference ("bf16", "fp16", "fp32")
    pub dtype: String,
    /// Whether to use quantized model
    pub is_quant_model: bool,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Tensor Parallelism rank (for distributed inference)
    pub tp_rank: u32,
    /// Tensor Parallelism world size
    pub tp_world_size: u32,
    /// Pipeline Parallelism rank (reserved for future)
    pub pp_rank: u32,
    /// Pipeline Parallelism world size (reserved for future)
    pub pp_world_size: u32,
    /// Whether to enable Flash Attention
    pub enable_flash_attn: bool,
}

impl WorkerConfig {
    /// Create a new WorkerConfig for a CUDA device
    ///
    /// Note: model_path is NOT required here. The Scheduler will send LoadModel
    /// command with the model path to load.
    pub fn cuda(device_id: u32) -> Self {
        Self {
            worker_id: format!("worker-cuda-{}", device_id),
            device_id,
            device_type: DeviceType::Cuda(device_id as i32),
            dtype: "bf16".to_string(),
            is_quant_model: false,
            max_seq_len: 4096,
            tp_rank: 0,
            tp_world_size: 1,
            pp_rank: 0,
            pp_world_size: 1,
            enable_flash_attn: true,
        }
    }

    /// Create a new WorkerConfig for CPU
    ///
    /// Note: model_path is NOT required here. The Scheduler will send LoadModel
    /// command with the model path to load.
    pub fn cpu() -> Self {
        Self {
            worker_id: "worker-cpu-0".to_string(),
            device_id: 0,
            device_type: DeviceType::Cpu,
            dtype: "fp32".to_string(),
            is_quant_model: false,
            max_seq_len: 2048,
            tp_rank: 0,
            tp_world_size: 1,
            pp_rank: 0,
            pp_world_size: 1,
            enable_flash_attn: false,
        }
    }

    /// Create WorkerConfig from protocol ModelLoadParams
    ///
    /// This is the correct way to create config after receiving LoadModel command.
    #[cfg(feature = "protocol")]
    pub fn from_protocol_params(params: &infer_protocol::ModelLoadParams) -> Self {
        let device_type = if params.dtype.contains("cuda") || params.device_id > 0 {
            DeviceType::Cuda(params.device_id as i32)
        } else {
            DeviceType::Cpu
        };

        Self {
            worker_id: format!("worker-{}", params.device_id),
            device_id: params.device_id,
            device_type,
            dtype: params.dtype.clone(),
            is_quant_model: params.dtype.contains("int"),
            max_seq_len: 4096, // Default, will be overridden by model config
            tp_rank: params.tp_rank,
            tp_world_size: params.tp_world_size,
            pp_rank: params.pp_rank,
            pp_world_size: params.pp_world_size,
            enable_flash_attn: params.enable_flash_attn,
        }
    }

    /// Set the data type
    pub fn with_dtype(mut self, dtype: impl Into<String>) -> Self {
        self.dtype = dtype.into();
        self
    }

    /// Set whether to use quantized model
    pub fn with_quant(mut self, is_quant: bool) -> Self {
        self.is_quant_model = is_quant;
        self
    }

    /// Set maximum sequence length
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Set Tensor Parallelism configuration
    pub fn with_tp(mut self, rank: u32, world_size: u32) -> Self {
        self.tp_rank = rank;
        self.tp_world_size = world_size;
        self
    }

    /// Set worker ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.worker_id = id.into();
        self
    }

    /// Set Flash Attention
    pub fn with_flash_attn(mut self, enable: bool) -> Self {
        self.enable_flash_attn = enable;
        self
    }
}
