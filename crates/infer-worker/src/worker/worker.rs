//! Worker Implementation
//!
//! The Worker is the core inference execution unit that manages:
//! - Device resources (GPU/CPU)
//! - Model instance (any implementation of the Model trait)
//! - KV Cache memory pool
//! - CUDA configuration (streams, handles, etc.)
//!
//! Note: Sampler is managed internally by the Model, not by the Worker.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                              Worker                                      │
//! │                                                                          │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
//! │  │  DeviceInfo  │  │    Model     │  │  KVCachePool │                   │
//! │  │              │  │ (Box<dyn>)   │  │              │                   │
//! │  │ - device_id  │  │              │  │ - data       │                   │
//! │  │ - memory     │  │ - layers     │  │ - config     │                   │
//! │  └──────────────┘  │ - workspace  │  └──────────────┘                   │
//! │                    │ - sampler    │                                      │
//! │                    └──────────────┘                                      │
//! │                                                                          │
//! │  ┌──────────────┐                                                        │
//! │  │  CudaConfig  │                                                        │
//! │  │              │                                                        │
//! │  │ - stream     │                                                        │
//! │  │ - handles    │                                                        │
//! │  └──────────────┘                                                        │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Extensibility
//!
//! The Worker uses `Box<dyn Model>` to support any model that implements
//! the `Model` trait. This allows for:
//! - Multiple model architectures (Llama, Qwen, Mistral, etc.)
//! - Easy addition of new model types without changing Worker code
//! - Runtime model selection via ModelFactory

use std::time::Instant;

use crate::base::error::{Error, Result};
use crate::base::DeviceType;
use crate::model::kvcache::{KVCacheConfig, KVCachePool};
use crate::model::Model;
use crate::op::Op;
use crate::tensor::Tensor;
use infer_protocol::SamplingOutput;

#[cfg(feature = "cuda")]
use crate::cuda::CudaConfig;

use super::config::WorkerConfig;
use super::device_info::DeviceInfo;

/// Worker state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// Initial state, not yet initialized
    Uninitialized,
    /// Model loaded, ready for inference
    Ready,
    /// Currently processing a request
    Processing,
    /// Error state
    Error,
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total device memory in bytes
    pub total: u64,
    /// Used device memory in bytes
    pub used: u64,
    /// Free device memory in bytes
    pub free: u64,
    /// Memory used by model weights
    pub model_memory: u64,
    /// Memory used by KV cache
    pub kv_cache_memory: u64,
    /// Memory used by workspace/activations
    pub workspace_memory: u64,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            total: 0,
            used: 0,
            free: 0,
            model_memory: 0,
            kv_cache_memory: 0,
            workspace_memory: 0,
        }
    }
}

/// Performance statistics
#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    /// Total number of requests processed
    pub total_requests: u64,
    /// Total number of tokens generated
    pub total_tokens: u64,
    /// Total prefill time in milliseconds
    pub total_prefill_ms: u64,
    /// Total decode time in milliseconds
    pub total_decode_ms: u64,
}

impl PerformanceStats {
    /// Calculate average prefill time
    pub fn avg_prefill_ms(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_prefill_ms as f64 / self.total_requests as f64
        }
    }

    /// Calculate average decode time per token
    pub fn avg_decode_per_token_ms(&self) -> f64 {
        if self.total_tokens == 0 {
            0.0
        } else {
            self.total_decode_ms as f64 / self.total_tokens as f64
        }
    }

    /// Calculate throughput in tokens per second
    pub fn throughput_tps(&self) -> f64 {
        let total_time_s = (self.total_prefill_ms + self.total_decode_ms) as f64 / 1000.0;
        if total_time_s == 0.0 {
            0.0
        } else {
            self.total_tokens as f64 / total_time_s
        }
    }
}

/// Inference Worker
///
/// Manages device resources, model instance, KV cache, and executes inference.
/// The model is stored as `Box<dyn Model>` for extensibility, allowing any
/// model implementation (Llama3, Qwen2, etc.) to be used.
pub struct Worker {
    /// Worker configuration
    config: WorkerConfig,
    /// Current state
    state: WorkerState,
    /// Device information
    device_info: DeviceInfo,
    /// Model instance - uses trait object for extensibility
    model: Option<Box<dyn Model>>,
    /// KV Cache pool (optional, initialized after model load)
    kv_cache: Option<KVCachePool>,
    /// CUDA configuration (streams, handles, etc.)
    #[cfg(feature = "cuda")]
    cuda_config: Option<CudaConfig>,
    /// Pre-allocated output tensor for sampling [max_batch_size] (I32)
    /// Initialized during init_kv_cache, reused on every forward call
    output_token_ids: Option<Tensor>,
    /// Memory statistics
    memory_stats: MemoryStats,
    /// Performance statistics
    perf_stats: PerformanceStats,
    /// Model load time in milliseconds
    model_load_time_ms: u64,
}

impl Worker {
    /// Create a new Worker with the given configuration
    ///
    /// This does not load the model - call `load_model()` to initialize.
    pub fn new(config: WorkerConfig) -> Result<Self> {
        // Initialize device info
        let device_info = match config.device_type {
            DeviceType::Cpu => DeviceInfo::cpu(),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(device_id) => DeviceInfo::cuda(device_id)?,
        };

        println!("Worker {} initialized on device: {}", config.worker_id, device_info);

        Ok(Self {
            config,
            state: WorkerState::Uninitialized,
            device_info,
            model: None,
            kv_cache: None,
            #[cfg(feature = "cuda")]
            cuda_config: None,
            output_token_ids: None,
            memory_stats: MemoryStats::default(),
            perf_stats: PerformanceStats::default(),
            model_load_time_ms: 0,
        })
    }

    /// Load a model onto the device
    ///
    /// This method accepts a pre-built model that implements the Model trait.
    /// Use ModelFactory to create models from path and type.
    ///
    /// # Example
    /// ```ignore
    /// use infer_worker::model::{ModelFactory, ModelType};
    ///
    /// let model = ModelFactory::create(
    ///     ModelType::Llama3,
    ///     &config.model_path,
    ///     config.device_type,
    ///     config.is_quant_model,
    /// )?;
    /// worker.load_model(model)?;
    /// ```
    pub fn load_model(&mut self, model: Box<dyn Model>) -> Result<()> {
        if self.model.is_some() {
            return Err(Error::InvalidArgument("Model already loaded".to_string()).into());
        }

        println!("Loading model onto Worker {}...", self.config.worker_id);
        let start_time = Instant::now();

        // Verify device type matches
        if model.device_type() != self.config.device_type {
            return Err(Error::InvalidArgument(format!(
                "Model device type {:?} does not match worker device type {:?}",
                model.device_type(),
                self.config.device_type
            )).into());
        }

        self.model_load_time_ms = start_time.elapsed().as_millis() as u64;

        // Initialize CUDA config if needed
        #[cfg(feature = "cuda")]
        if self.config.device_type.is_cuda() {
            self.cuda_config = Some(CudaConfig::new()?);
        }

        self.model = Some(model);
        self.state = WorkerState::Ready;

        // Update memory stats
        self.refresh_memory_stats()?;

        println!(
            "Model loaded on Worker {} in {} ms. Memory used: {:.2} GB",
            self.config.worker_id,
            self.model_load_time_ms,
            self.memory_stats.used as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        Ok(())
    }

    pub fn device(&self) -> DeviceType {
        self.config.device_type
    }

    /// Initialize the KV cache pool from Scheduler parameters
    ///
    /// The Scheduler provides all necessary KV cache configuration parameters.
    /// This should be called after model loading.
    /// Also pre-allocates the output tensor for batch sampling (reused on every forward call).
    #[cfg(feature = "protocol")]
    pub fn init_kv_cache(&mut self, params: &infer_protocol::InitKVCacheParams) -> Result<()> {
        let _ = self.model.as_ref()
            .ok_or_else(|| Error::InvalidArgument("Model not loaded".to_string()))?;

        // Create KV cache config from scheduler parameters (source of truth)
        let kv_config = KVCacheConfig::from_protocol_params(params);

        println!(
            "Initializing KV Cache: {} blocks x {} = {} max tokens, {:.2} GB",
            params.num_blocks,
            params.block_size,

            kv_config.max_seq_len(),
            kv_config.total_memory_gb()
        );

        let start_time = Instant::now();
        let kv_cache = KVCachePool::new(kv_config, self.config.device_type)?;
        let init_time_ms = start_time.elapsed().as_millis() as u64;

        self.memory_stats.kv_cache_memory = kv_cache.memory_bytes() as u64;
        self.kv_cache = Some(kv_cache);

        // Pre-allocate output tensor for batch sampling [max_batch_size] (I32)
        // Max batch size = num_blocks (each request occupies at least 1 block)
        let max_batch_size = params.num_blocks;
        self.output_token_ids = Some(Tensor::new(
            &[max_batch_size],
            crate::base::DataType::I32,
            self.config.device_type,
        )?);

        // Update memory stats
        self.refresh_memory_stats()?;

        println!("KV Cache initialized in {} ms, output tensor pre-allocated for batch_size={}",
            init_time_ms, max_batch_size);

        Ok(())
    }

    pub fn forward(
        &mut self,
        input_tokens: &Tensor,
        positions: &Tensor,
        block_tables: &[u32],
        max_blocks_per_req: usize,
        slot_mapping: &Tensor,
        context_lens: &[u32],
        is_prefill: bool,
    ) -> Result<SamplingOutput> {
        let model = self.model.as_mut()
            .ok_or_else(|| Error::InvalidArgument("Model not loaded".to_string()))?;

        // Step 1: Get logits from model [batch_size, vocab_size]
        let logits = model.forward_paged(
            input_tokens,
            positions,
            block_tables,
            max_blocks_per_req,
            slot_mapping,
            context_lens,
            is_prefill,
        )?;

        // Step 2: Extract batch size from logits
        let batch_size = logits.shape()[0];

        // Step 3: Get pre-allocated output tensor
        let output_token_ids = self.output_token_ids.as_mut()
            .ok_or_else(|| Error::InvalidArgument(
                "Output tensor not initialized. Call init_kv_cache() first.".to_string()
            ))?;

        // Step 4: Create sampler (using default ArgmaxSampler for now)
        let sampler = Box::new(crate::op::sampler::ArgmaxSampler::new(self.config.device_type));
        let sampler_op = crate::op::sampler::SamplerOp::new(sampler);

        // Step 5: Execute sampling on GPU/CPU
        let inputs = [&logits];
        let mut outputs = [output_token_ids];
        let mut ctx = crate::op::OpContext::new(&inputs, &mut outputs, None);
        sampler_op.forward(&mut ctx)?;

        // Step 6: Copy sampled token IDs from device to CPU (only first batch_size elements)
        let next_token_ids = match output_token_ids.as_i32() {
            Ok(token_tensor) => {
                let all_tokens = token_tensor.as_slice()?;
                all_tokens[..batch_size].to_vec()
            }
            Err(e) => return Err(Error::InvalidArgument(
                format!("Failed to get token tensor: {}", e)
            ).into()),
        };

        // Step 7: Return SamplingOutput
        // TODO: Determine finish_reasons based on token IDs and sampling parameters
        Ok(SamplingOutput {
            next_token_ids,
            is_stopped: vec![false; batch_size],
            finish_reasons: vec![infer_protocol::FinishReason::Stop; batch_size],
        })
    }

    /// Reset KV cache state
    pub fn reset_kv_cache(&mut self) -> Result<()> {
        if let Some(model) = self.model.as_mut() {
            model.reset_kv_cache()?;
        }
        Ok(())
    }

    /// Refresh memory statistics
    #[cfg(feature = "cuda")]
    pub fn refresh_memory_stats(&mut self) -> Result<()> {
        self.device_info.refresh_memory()?;

        self.memory_stats.total = self.device_info.total_memory;
        self.memory_stats.free = self.device_info.free_memory;
        self.memory_stats.used = self.device_info.used_memory;

        Ok(())
    }

    /// Get worker configuration
    pub fn config(&self) -> &WorkerConfig {
        &self.config
    }

    /// Get current state
    pub fn state(&self) -> WorkerState {
        self.state
    }

    /// Get device information
    pub fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> &MemoryStats {
        &self.memory_stats
    }

    /// Get performance statistics
    pub fn perf_stats(&self) -> &PerformanceStats {
        &self.perf_stats
    }

    /// Get model load time in milliseconds
    pub fn model_load_time_ms(&self) -> u64 {
        self.model_load_time_ms
    }

    /// Check if model is loaded
    pub fn is_model_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Check if KV cache is initialized
    pub fn is_kv_cache_initialized(&self) -> bool {
        self.kv_cache.is_some()
    }

    /// Get worker ID
    pub fn worker_id(&self) -> &str {
        &self.config.worker_id
    }

    /// Get device type
    pub fn device_type(&self) -> DeviceType {
        self.config.device_type
    }

    /// Get mutable reference to KV cache (if initialized)
    pub fn kv_cache_mut(&mut self) -> Option<&mut KVCachePool> {
        self.kv_cache.as_mut()
    }

    /// Get reference to KV cache (if initialized)
    pub fn kv_cache(&self) -> Option<&KVCachePool> {
        self.kv_cache.as_ref()
    }

    /// Get reference to model (if loaded)
    pub fn model(&self) -> Option<&(dyn Model + 'static)> {
        self.model.as_ref().map(|m| &**m)
    }

    /// Get mutable reference to model (if loaded)
    pub fn model_mut(&mut self) -> Option<&mut (dyn Model + 'static)> {
        self.model.as_mut().map(|m| &mut **m)
    }

    /// Build WorkerStatus for protocol response
    #[cfg(feature = "protocol")]
    pub fn to_worker_status(&self) -> infer_protocol::WorkerStatus {
        use infer_protocol::WorkerState as ProtocolWorkerState;
        use infer_protocol::MemoryStats as ProtocolMemoryStats;
        use infer_protocol::PerformanceStats as ProtocolPerformanceStats;

        infer_protocol::WorkerStatus {
            worker_id: self.config.worker_id.clone(),
            device_id: self.config.device_id,
            state: match self.state {
                WorkerState::Uninitialized => ProtocolWorkerState::Initializing,
                WorkerState::Ready => ProtocolWorkerState::Idle,
                WorkerState::Processing => ProtocolWorkerState::Inferencing,
                WorkerState::Error => ProtocolWorkerState::Error,
            },
            model_loaded: self.model.is_some(),
            kv_cache_initialized: self.kv_cache.is_some(),
            memory_stats: ProtocolMemoryStats {
                total: self.memory_stats.total,
                used: self.memory_stats.used,
                free: self.memory_stats.free,
                model_memory: self.memory_stats.model_memory,
                kv_cache_memory: self.memory_stats.kv_cache_memory,
                activation_memory: self.memory_stats.workspace_memory,
            },
            performance_stats: ProtocolPerformanceStats {
                total_requests: self.perf_stats.total_requests,
                total_tokens: self.perf_stats.total_tokens,
                avg_prefill_time_ms: self.perf_stats.avg_prefill_ms(),
                avg_decode_time_ms: self.perf_stats.avg_decode_per_token_ms(),
                throughput_tokens_per_sec: self.perf_stats.throughput_tps(),
                gpu_utilization: 0.0, // TODO: implement GPU utilization tracking
            },
            tp_rank: Some(self.config.tp_rank),
            tp_world_size: Some(self.config.tp_world_size),
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        println!("Worker {} shutting down...", self.config.worker_id);
        // Resources are automatically cleaned up via Drop impls
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_performance_stats() {
        let mut stats = PerformanceStats::default();
        stats.total_requests = 10;
        stats.total_tokens = 1000;
        stats.total_prefill_ms = 500;
        stats.total_decode_ms = 10000;

        assert_eq!(stats.avg_prefill_ms(), 50.0);
        assert_eq!(stats.avg_decode_per_token_ms(), 10.0);
        // 1000 tokens in 10.5 seconds = ~95.24 tps
        assert!((stats.throughput_tps() - 95.24).abs() < 0.1);
    }
}
