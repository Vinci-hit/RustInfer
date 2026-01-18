//! KV Cache Module - Physical KV Cache Memory Management for LLM Inference
//!
//! This module manages the physical GPU/CPU memory for KV cache tensors.
//! It works in conjunction with the Scheduler's RadixTree which handles
//! logical block allocation and prefix sharing.
//!
//! # Architecture
//!
//! The overall system follows a separation of concerns:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Scheduler (infer-scheduler)                 │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │  RadixCache (Logical Layer)                              │   │
//! │  │  - Prefix sharing via radix tree                         │   │
//! │  │  - Token index allocation (TokenPool)                    │   │
//! │  │  - LRU eviction policy                                   │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └────────────────────────┬────────────────────────────────────────┘
//!                          │ WorkerCommand::Forward(block_ids)
//!                          ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Worker (infer-worker)                       │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │  KVCachePool (Physical Layer)                            │   │
//! │  │  - Pre-allocated GPU/CPU memory                          │   │
//! │  │  - Per-layer (K, V) tensor pairs                        │   │
//! │  │  - Block-based access for PagedAttention                │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use infer_protocol::InitKVCacheParams;
//!
//! let params = InitKVCacheParams {
//!     num_blocks: 1000,
//!     block_size: 16,
//!     num_layers: 32,
//!     num_heads: 8,
//!     head_dim: 128,
//!     dtype: "bf16".to_string(),
//!     use_unified_memory_pool: true,
//! };
//!
//! let kv_cache = KVCachePool::from_protocol_params(&params, DeviceType::Cuda(0))?;
//! ```

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;

/// Parse data type string from protocol to internal DataType
fn parse_dtype(dtype_str: &str) -> DataType {
    match dtype_str.to_lowercase().as_str() {
        "bf16" | "bfloat16" => DataType::BF16,
        "fp16" | "float16" | "f16" => DataType::F16,
        "fp32" | "float32" | "f32" => DataType::F32,
        "int8" | "i8" => DataType::I8,
        _ => DataType::BF16, // Default to BF16
    }
}

/// Configuration for KV Cache initialization
///
/// This can be created from:
/// - Direct construction with `KVCacheConfig::new()`
/// - From model config with `KVCacheConfig::from_model_config()`
/// - From protocol params with `KVCacheConfig::from_protocol_params()`
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of KV heads (for GQA, this may differ from num_attention_heads)
    pub num_kv_heads: usize,
    /// Size of each attention head
    pub head_size: usize,
    /// Data type for the cache (typically BF16 or F32)
    pub dtype: DataType,
    /// Number of tokens per block (typically 16, 32, 64, or 256)
    pub block_size: usize,
    /// Total number of blocks to allocate
    pub num_blocks: usize,
}

impl KVCacheConfig {
    /// Create a new KVCacheConfig
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_size: usize,
        dtype: DataType,
        block_size: usize,
        num_blocks: usize,
    ) -> Self {
        Self {
            num_layers,
            num_kv_heads,
            head_size,
            dtype,
            block_size,
            num_blocks,
        }
    }

    /// Create KVCacheConfig from RuntimeModelConfig with custom block settings
    pub fn from_model_config(
        model_config: &crate::model::config::RuntimeModelConfig,
        block_size: usize,
        num_blocks: usize,
    ) -> Self {
        let dtype = match model_config.torch_dtype.as_str() {
            "bfloat16" => DataType::BF16,
            "float16" => DataType::F16,
            _ => DataType::F32,
        };

        Self {
            num_layers: model_config.layer_num,
            num_kv_heads: model_config.kv_head_num,
            head_size: model_config.head_size,
            dtype,
            block_size,
            num_blocks,
        }
    }

    /// Create KVCacheConfig from infer_protocol::InitKVCacheParams
    ///
    /// This is the preferred method when initializing from Scheduler commands.
    #[cfg(feature = "protocol")]
    pub fn from_protocol_params(params: &infer_protocol::InitKVCacheParams) -> Self {
        Self {
            num_layers: params.num_layers as usize,
            num_kv_heads: params.num_heads as usize,
            head_size: params.head_dim as usize,
            dtype: parse_dtype(&params.dtype),
            block_size: params.block_size,
            num_blocks: params.num_blocks,
        }
    }

    /// Calculate the KV dimension (kv_head_num * head_size)
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_size
    }

    /// Calculate the maximum sequence length this cache can hold
    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.num_blocks * self.block_size
    }

    /// Calculate total memory required in bytes (for both K and V caches)
    pub fn total_memory_bytes(&self) -> usize {
        let elements_per_layer = self.num_blocks * self.block_size * self.kv_dim();
        let bytes_per_element = self.dtype.size_in_bytes();
        // K + V caches for all layers
        2 * self.num_layers * elements_per_layer * bytes_per_element
    }

    /// Calculate memory in GB
    pub fn total_memory_gb(&self) -> f64 {
        self.total_memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Block-based KV Cache Pool
///
/// This structure manages the entire KV cache memory for all transformer layers.
/// Each layer has its own (K, V) tensor pair, pre-allocated as a contiguous
/// memory region divided into fixed-size blocks.
///
/// # Memory Layout
///
/// For each layer, K and V caches are stored as separate tensors with shape:
/// `[num_blocks * block_size, kv_dim]` where `kv_dim = num_kv_heads * head_size`
///
/// This layout supports efficient:
/// - Sequential access during prefill
/// - Block-based access for PagedAttention during decode
/// - Zero-copy slicing for both operations
pub struct KVCachePool {
    /// Configuration used to create this pool
    config: KVCacheConfig,
    /// KV cache tensors for each layer: Vec<(K_cache, V_cache)>
    /// Shape of each tensor: [num_blocks * block_size, kv_dim]
    data: Vec<(Tensor, Tensor)>,
    /// Device where the cache is allocated
    device: DeviceType,
    /// Actual allocated number of blocks
    allocated_blocks: usize,
}

impl KVCachePool {
    /// Create a new KV Cache Pool with the given configuration
    ///
    /// This will allocate a large contiguous memory region on the specified device.
    /// For GPU, this typically consumes several GB of VRAM.
    ///
    /// # Arguments
    /// * `config` - KV cache configuration
    /// * `device` - Device to allocate on (CPU or CUDA)
    ///
    /// # Returns
    /// A new KVCachePool with pre-allocated memory
    pub fn new(config: KVCacheConfig, device: DeviceType) -> Result<Self> {
        // Validate configuration
        if config.num_layers == 0 {
            return Err(Error::InvalidArgument("num_layers must be > 0".to_string()).into());
        }
        if config.num_blocks == 0 {
            return Err(Error::InvalidArgument("num_blocks must be > 0".to_string()).into());
        }
        if config.block_size == 0 {
            return Err(Error::InvalidArgument("block_size must be > 0".to_string()).into());
        }

        // For CPU, force F32 for better precision
        let dtype = if device.is_cpu() {
            DataType::F32
        } else {
            config.dtype
        };

        // Calculate cache shape for PagedAttention
        // Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        // This layout allows efficient block-based access and GPU memory coalescing
        let cache_shape = vec![
            config.num_blocks,      // Total physical blocks
            config.block_size,      // Tokens per block (e.g., 16)
            config.num_kv_heads,    // Number of KV heads (GQA)
            config.head_size,       // Head dimension (e.g., 128)
        ];

        let total_tokens = config.num_blocks * config.block_size;

        println!(
            "Initializing KV Cache Pool: {} layers, {} blocks x {} = {} max tokens",
            config.num_layers,
            config.num_blocks,
            config.block_size,
            total_tokens
        );
        println!(
            "  Cache shape per layer: {:?}, dtype: {:?}",
            cache_shape, dtype
        );
        println!(
            "  Total memory: {:.2} GB",
            config.total_memory_gb()
        );

        // Allocate tensors for each layer
        let mut data = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let k_cache = Tensor::new(&cache_shape, dtype, device)?;
            let v_cache = Tensor::new(&cache_shape, dtype, device)?;

            if layer_idx == 0 {
                println!(
                    "  Layer 0 K cache allocated: {:?} on {:?}",
                    k_cache.shape(),
                    k_cache.device()
                );
            }

            data.push((k_cache, v_cache));
        }

        println!("KV Cache Pool initialization complete.");

        let allocated_blocks = config.num_blocks;
        Ok(Self {
            config,
            data,
            device,
            allocated_blocks,
        })
    }

    /// Create KVCachePool from infer_protocol::InitKVCacheParams
    #[cfg(feature = "protocol")]
    pub fn from_protocol_params(
        params: &infer_protocol::InitKVCacheParams,
        device: DeviceType,
    ) -> Result<Self> {
        let config = KVCacheConfig::from_protocol_params(params);
        Self::new(config, device)
    }

    /// Get immutable reference to (K, V) cache for a specific layer
    pub fn get(&self, layer_idx: usize) -> Result<(&Tensor, &Tensor)> {
        let (k_cache, v_cache) = self.data.get(layer_idx).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "Layer index {} out of bounds (max: {})",
                layer_idx,
                self.config.num_layers
            ))
        })?;
        Ok((k_cache, v_cache))
    }

    /// Get mutable reference to (K, V) cache for a specific layer
    pub fn get_mut(&mut self, layer_idx: usize) -> Result<(&mut Tensor, &mut Tensor)> {
        let num_layers = self.config.num_layers;
        let (k_cache, v_cache) = self.data.get_mut(layer_idx).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "Layer index {} out of bounds (max: {})",
                layer_idx,
                num_layers
            ))
        })?;
        Ok((k_cache, v_cache))
    }

    /// Slice the KV cache for a specific layer and position range
    ///
    /// NOTE: This method is for backward compatibility. For new code using
    /// the 4D layout [num_blocks, block_size, num_kv_heads, head_dim],
    /// use `get_block()` or `get_kv_for_sequence()` instead.
    ///
    /// This method computes which blocks the position range spans and
    /// returns a view that reshapes the data.
    ///
    /// # Arguments
    /// * `layer_idx` - The transformer layer index
    /// * `start_pos` - Starting position in the sequence
    /// * `len` - Number of tokens to slice
    ///
    /// # Returns
    /// Tuple of (K_slice, V_slice) tensors
    pub fn slice(
        &mut self,
        layer_idx: usize,
        start_pos: usize,
        len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let block_size = self.config.block_size;
        let max_seq_len = self.config.max_seq_len();
        let num_kv_heads = self.config.num_kv_heads;
        let head_size = self.config.head_size;

        // Bounds check
        if start_pos + len > max_seq_len {
            return Err(Error::InvalidArgument(format!(
                "KV cache slice out of bounds: pos {} + len {} > max_seq_len {}",
                start_pos, len, max_seq_len
            )).into());
        }

        // Calculate which blocks this range spans
        let start_block = start_pos / block_size;
        let start_slot = start_pos % block_size;
        let end_block = (start_pos + len - 1) / block_size;

        let (k_cache, v_cache) = self.get_mut(layer_idx)?;

        // For PagedAttention 4D layout: [num_blocks, block_size, num_kv_heads, head_dim]
        // Single block case - can return a simple slice
        if start_block == end_block {
            // Slice within a single block: [1, len, num_kv_heads, head_dim]
            let k_slice = k_cache.slice(
                &[start_block, start_slot, 0, 0],
                &[1, len, num_kv_heads, head_size]
            )?;
            let v_slice = v_cache.slice(
                &[start_block, start_slot, 0, 0],
                &[1, len, num_kv_heads, head_size]
            )?;
            Ok((k_slice, v_slice))
        } else {
            // Cross-block access - for now, just return the first block's portion
            // TODO: Implement proper gather for cross-block access
            let remaining_in_first = block_size - start_slot;
            let k_slice = k_cache.slice(
                &[start_block, start_slot, 0, 0],
                &[1, remaining_in_first, num_kv_heads, head_size]
            )?;
            let v_slice = v_cache.slice(
                &[start_block, start_slot, 0, 0],
                &[1, remaining_in_first, num_kv_heads, head_size]
            )?;
            Ok((k_slice, v_slice))
        }
    }

    /// Get slice by block index (for PagedAttention-style access)
    ///
    /// Returns the full block as a tensor with shape [block_size, num_kv_heads, head_dim]
    ///
    /// # Arguments
    /// * `layer_idx` - The transformer layer index
    /// * `block_idx` - The block index
    ///
    /// # Returns
    /// Tuple of (K_block, V_block) tensors with shape [block_size, num_kv_heads, head_dim]
    pub fn get_block(
        &mut self,
        layer_idx: usize,
        block_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        if block_idx >= self.allocated_blocks {
            return Err(Error::InvalidArgument(format!(
                "Block index {} out of bounds (max: {})",
                block_idx,
                self.allocated_blocks
            )).into());
        }

        let block_size = self.config.block_size;
        let num_kv_heads = self.config.num_kv_heads;
        let head_size = self.config.head_size;

        let (k_cache, v_cache) = self.get_mut(layer_idx)?;

        // For 4D layout [num_blocks, block_size, num_kv_heads, head_dim]
        // Slice a single block: [1, block_size, num_kv_heads, head_dim]
        let k_block = k_cache.slice(
            &[block_idx, 0, 0, 0],
            &[1, block_size, num_kv_heads, head_size]
        )?;
        let v_block = v_cache.slice(
            &[block_idx, 0, 0, 0],
            &[1, block_size, num_kv_heads, head_size]
        )?;

        Ok((k_block, v_block))
    }

    /// Get slices for multiple block indices (for batched PagedAttention)
    ///
    /// This is more efficient than calling get_block multiple times when
    /// processing multiple blocks for a single layer.
    ///
    /// # Arguments
    /// * `layer_idx` - The transformer layer index
    /// * `block_indices` - List of block indices to access
    ///
    /// # Returns
    /// Vector of (K_block, V_block) tensor pairs
    pub fn get_blocks(
        &mut self,
        layer_idx: usize,
        block_indices: &[u32],
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let mut result = Vec::with_capacity(block_indices.len());
        for &block_idx in block_indices {
            result.push(self.get_block(layer_idx, block_idx as usize)?);
        }
        Ok(result)
    }

    /// Write KV values to a specific position within a block
    ///
    /// This is used during token generation to write new KV values
    /// to the allocated block positions.
    ///
    /// # Arguments
    /// * `layer_idx` - The transformer layer index
    /// * `block_idx` - The block index to write to
    /// * `slot_idx` - The slot index within the block (0..block_size)
    /// * `k_value` - Key tensor to write, shape [num_kv_heads, head_dim]
    /// * `v_value` - Value tensor to write, shape [num_kv_heads, head_dim]
    pub fn write_kv_to_slot(
        &mut self,
        layer_idx: usize,
        block_idx: usize,
        slot_idx: usize,
        k_value: &Tensor,
        v_value: &Tensor,
    ) -> Result<()> {
        if block_idx >= self.allocated_blocks {
            return Err(Error::InvalidArgument(format!(
                "Block index {} out of bounds (max: {})",
                block_idx,
                self.allocated_blocks
            )).into());
        }
        if slot_idx >= self.config.block_size {
            return Err(Error::InvalidArgument(format!(
                "Slot index {} out of bounds (block_size: {})",
                slot_idx,
                self.config.block_size
            )).into());
        }

        let num_kv_heads = self.config.num_kv_heads;
        let head_size = self.config.head_size;

        let (k_cache, v_cache) = self.get_mut(layer_idx)?;

        // For 4D layout [num_blocks, block_size, num_kv_heads, head_dim]
        // Write to position [block_idx, slot_idx, :, :]
        let mut k_target = k_cache.slice(
            &[block_idx, slot_idx, 0, 0],
            &[1, 1, num_kv_heads, head_size]
        )?;
        let mut v_target = v_cache.slice(
            &[block_idx, slot_idx, 0, 0],
            &[1, 1, num_kv_heads, head_size]
        )?;

        // Copy values
        k_target.copy_from(k_value)?;
        v_target.copy_from(v_value)?;

        Ok(())
    }

    /// Get KV data for a sequence defined by block table
    ///
    /// This is the core method for PagedAttention. Given a list of block indices
    /// that make up a sequence, it returns information needed for attention.
    ///
    /// # Arguments
    /// * `layer_idx` - The transformer layer index
    /// * `block_table` - List of block indices for this sequence
    /// * `context_len` - Total context length (may be less than blocks * block_size)
    ///
    /// # Returns
    /// For contiguous blocks: Tuple of (K, V) tensor slices
    /// Note: For efficient PagedAttention, the kernel should work directly
    /// with block tables rather than copying data.
    pub fn get_kv_for_sequence(
        &self,
        layer_idx: usize,
        block_table: &[u32],
        context_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        if block_table.is_empty() {
            return Err(Error::InvalidArgument("Block table is empty".to_string()).into());
        }

        let block_size = self.config.block_size;
        let (k_cache, v_cache) = self.data.get(layer_idx).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "Layer index {} out of bounds (max: {})",
                layer_idx,
                self.config.num_layers
            ))
        })?;

        // Check if blocks are sequential (contiguous allocation)
        let first_block = block_table[0] as usize;
        let is_contiguous = block_table.iter().enumerate().all(|(i, &b)| {
            b as usize == first_block + i
        });

        let num_blocks_needed = (context_len + block_size - 1) / block_size;

        if is_contiguous && block_table.len() >= num_blocks_needed {
            // Contiguous blocks - can return a single slice
            // Calculate how many complete blocks and remaining slots
            let complete_blocks = context_len / block_size;
            let remaining_slots = context_len % block_size;

            // For 4D layout: slice [first_block:first_block+num_blocks, 0:block_size, :, :]
            // But we need context_len tokens, not full blocks
            if remaining_slots == 0 {
                // Exactly fills complete blocks
                let k_slice = k_cache.slice(
                    &[first_block, 0, 0, 0],
                    &[complete_blocks, block_size, self.config.num_kv_heads, self.config.head_size]
                )?;
                let v_slice = v_cache.slice(
                    &[first_block, 0, 0, 0],
                    &[complete_blocks, block_size, self.config.num_kv_heads, self.config.head_size]
                )?;
                Ok((k_slice, v_slice))
            } else {
                // Partial last block - for simplicity, return complete blocks only
                // The caller should handle the partial block separately
                let k_slice = k_cache.slice(
                    &[first_block, 0, 0, 0],
                    &[complete_blocks + 1, block_size, self.config.num_kv_heads, self.config.head_size]
                )?;
                let v_slice = v_cache.slice(
                    &[first_block, 0, 0, 0],
                    &[complete_blocks + 1, block_size, self.config.num_kv_heads, self.config.head_size]
                )?;
                Ok((k_slice, v_slice))
            }
        } else {
            // Non-contiguous blocks - PagedAttention kernel handles this directly
            // For compatibility, return just the first block
            let k_slice = k_cache.slice(
                &[first_block, 0, 0, 0],
                &[1, block_size, self.config.num_kv_heads, self.config.head_size]
            )?;
            let v_slice = v_cache.slice(
                &[first_block, 0, 0, 0],
                &[1, block_size, self.config.num_kv_heads, self.config.head_size]
            )?;
            Ok((k_slice, v_slice))
        }
    }

    /// Get block table information for PagedAttention kernels
    ///
    /// Returns the information needed for PagedAttention CUDA kernels
    /// to access KV cache blocks directly.
    ///
    /// # Arguments
    /// * `layer_idx` - The transformer layer index
    /// * `block_indices` - List of block indices for a sequence
    ///
    /// # Returns
    /// Reference to the (K, V) cache tensors for this layer
    /// The kernel uses block_indices to compute offsets into these tensors.
    pub fn get_cache_for_paged_attn(
        &self,
        layer_idx: usize,
    ) -> Result<(&Tensor, &Tensor)> {
        self.get(layer_idx)
    }

    /// Get the configuration
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }

    /// Get the device
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Get the KV dimension
    pub fn kv_dim(&self) -> usize {
        self.config.kv_dim()
    }

    /// Get the maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len()
    }

    /// Get the block size
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Get the number of allocated blocks
    pub fn num_blocks(&self) -> usize {
        self.allocated_blocks
    }

    /// Calculate memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.config.total_memory_bytes()
    }

    /// Calculate memory usage in GB
    pub fn memory_gb(&self) -> f64 {
        self.config.total_memory_gb()
    }

    /// Build KVCacheInfo for protocol response
    #[cfg(feature = "protocol")]
    pub fn to_kv_cache_info(&self, init_time_ms: u64) -> infer_protocol::KVCacheInfo {
        infer_protocol::KVCacheInfo {
            allocated_blocks: self.allocated_blocks,
            memory_used: self.memory_bytes() as u64,
            bytes_per_block: (self.config.block_size * self.config.kv_dim() * self.config.dtype.size_in_bytes() * 2) as u64,
            total_capacity_tokens: self.max_seq_len(),
            init_time_ms,
        }
    }
}

/// Simple KV Cache (for backward compatibility with existing code)
///
/// This is a simpler version that doesn't use block-based management,
/// maintaining compatibility with the existing llama3.rs implementation.
pub struct KvCache {
    cache: Vec<(Tensor, Tensor)>,
}

impl KvCache {
    /// Initialize KV cache from RuntimeModelConfig
    pub fn init_kv_cache(
        config: &crate::model::config::RuntimeModelConfig,
        device: &DeviceType,
    ) -> Result<Self> {
        let cache_shape = vec![config.seq_len, config.kv_head_num * config.head_size];

        // Use F32 for CPU, otherwise follow model dtype
        let float_type = if device.is_cpu() {
            println!("  -> Initializing F32 KV Cache for CPU device.");
            DataType::F32
        } else {
            match config.torch_dtype.as_str() {
                "float32" => {
                    println!("  -> torch_dtype is float32.");
                    DataType::F32
                }
                "bfloat16" => {
                    println!("  -> torch_dtype is bfloat16.");
                    DataType::BF16
                }
                _ => {
                    return Err(Error::InvalidArgument(format!(
                        "Unsupported torch_dtype: {}",
                        config.torch_dtype
                    )).into());
                }
            }
        };

        println!(
            "Initializing KV Cache for {} layers with shape {:?}...",
            config.layer_num, cache_shape
        );

        let mut kv_cache = Vec::with_capacity(config.layer_num);
        for _ in 0..config.layer_num {
            let k_cache = Tensor::new(&cache_shape, float_type, *device)?;
            let v_cache = Tensor::new(&cache_shape, float_type, *device)?;
            kv_cache.push((k_cache, v_cache));
        }

        Ok(KvCache { cache: kv_cache })
    }

    /// Slice KV cache for a specific layer and position range
    pub fn slice_kv_cache(
        &mut self,
        layer_idx: usize,
        start_pos: i32,
        len: usize,
        kv_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (k_cache_full, v_cache_full) = self.get_mut(layer_idx)?;

        let max_seq_len = k_cache_full.shape()[0];
        if start_pos as usize + len > max_seq_len {
            return Err(anyhow::anyhow!(
                "KV cache slice out of bounds: pos {} + len {} > max_seq_len {}",
                start_pos,
                len,
                max_seq_len
            ));
        }

        let k_slice = k_cache_full.slice(&[start_pos as usize, 0], &[len, kv_dim])?;
        let v_slice = v_cache_full.slice(&[start_pos as usize, 0], &[len, kv_dim])?;

        Ok((k_slice, v_slice))
    }

    /// Get immutable reference to (K, V) cache for a layer
    pub fn get(&self, layer_idx: usize) -> Result<(&Tensor, &Tensor)> {
        let (k_cache, v_cache) = self.cache.get(layer_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "Layer index {} out of bounds for KV cache",
                layer_idx
            )
        })?;
        Ok((k_cache, v_cache))
    }

    /// Get mutable reference to (K, V) cache for a layer
    pub fn get_mut(&mut self, layer_idx: usize) -> Result<(&mut Tensor, &mut Tensor)> {
        let (k_cache, v_cache) = self.cache.get_mut(layer_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "Layer index {} out of bounds for KV cache",
                layer_idx
            )
        })?;
        Ok((k_cache, v_cache))
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_config() {
        let config = KVCacheConfig::new(
            32,  // num_layers
            8,   // num_kv_heads
            128, // head_size
            DataType::BF16,
            256, // block_size
            100, // num_blocks
        );

        assert_eq!(config.kv_dim(), 8 * 128);
        assert_eq!(config.max_seq_len(), 256 * 100);

        // Memory calculation: 2 (K+V) * 32 layers * 25600 seq * 1024 kv_dim * 2 bytes
        let expected_bytes = 2 * 32 * 25600 * 1024 * 2;
        assert_eq!(config.total_memory_bytes(), expected_bytes);
    }

    #[test]
    fn test_kv_cache_pool_creation() -> Result<()> {
        let config = KVCacheConfig::new(
            2,   // num_layers (small for testing)
            4,   // num_kv_heads
            64,  // head_size
            DataType::F32,
            16,  // block_size
            10,  // num_blocks
        );

        let pool = KVCachePool::new(config, DeviceType::Cpu)?;

        assert_eq!(pool.num_layers(), 2);
        assert_eq!(pool.kv_dim(), 4 * 64);
        assert_eq!(pool.max_seq_len(), 16 * 10);
        assert_eq!(pool.block_size(), 16);
        assert_eq!(pool.num_blocks(), 10);

        // Verify tensor shapes - 4D layout: [num_blocks, block_size, num_kv_heads, head_dim]
        let (k, v) = pool.get(0)?;
        assert_eq!(k.shape(), &[10, 16, 4, 64]);
        assert_eq!(v.shape(), &[10, 16, 4, 64]);

        Ok(())
    }

    #[test]
    fn test_kv_cache_pool_slice() -> Result<()> {
        let config = KVCacheConfig::new(
            2,
            4,
            64,
            DataType::F32,
            16,
            10,
        );

        let mut pool = KVCachePool::new(config, DeviceType::Cpu)?;

        // Test slicing within a single block (first 10 tokens of block 0)
        // With 4D layout, slice returns [1, len, num_kv_heads, head_dim]
        let (k_slice, v_slice) = pool.slice(0, 0, 10)?;
        assert_eq!(k_slice.shape(), &[1, 10, 4, 64]);
        assert_eq!(v_slice.shape(), &[1, 10, 4, 64]);

        // Test block access - returns [1, block_size, num_kv_heads, head_dim]
        let (k_block, v_block) = pool.get_block(0, 0)?;
        assert_eq!(k_block.shape(), &[1, 16, 4, 64]);
        assert_eq!(v_block.shape(), &[1, 16, 4, 64]);

        Ok(())
    }

    #[test]
    fn test_kv_cache_pool_bounds_check() {
        let config = KVCacheConfig::new(
            2,
            4,
            64,
            DataType::F32,
            16,
            10,
        );

        let mut pool = KVCachePool::new(config, DeviceType::Cpu).unwrap();

        // Should fail: out of bounds layer
        assert!(pool.get(5).is_err());

        // Should fail: out of bounds slice
        assert!(pool.slice(0, 150, 20).is_err());

        // Should fail: out of bounds block
        assert!(pool.get_block(0, 20).is_err());
    }

    #[test]
    fn test_parse_dtype() {
        assert_eq!(parse_dtype("bf16"), DataType::BF16);
        assert_eq!(parse_dtype("BF16"), DataType::BF16);
        assert_eq!(parse_dtype("bfloat16"), DataType::BF16);
        assert_eq!(parse_dtype("fp16"), DataType::F16);
        assert_eq!(parse_dtype("float16"), DataType::F16);
        assert_eq!(parse_dtype("fp32"), DataType::F32);
        assert_eq!(parse_dtype("float32"), DataType::F32);
        assert_eq!(parse_dtype("int8"), DataType::I8);
        assert_eq!(parse_dtype("unknown"), DataType::BF16); // default
    }
}
