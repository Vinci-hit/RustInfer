// Core model infrastructure
use std::collections::HashMap;

pub mod config;
pub mod kvcache;
pub mod loader;
pub mod architectures;

// Layer abstractions
pub mod layers;

// Re-exports
pub use loader::registry::{ModelRegistry, ModelId, ModelBuilderFn, GLOBAL_REGISTRY};
pub use layers::{DecoderLayers, WeightMappingAdapter};
pub use kvcache::{KVCachePool, KVCacheConfig};
pub use loader::model_loader::{ModelLoader, TensorLocation, DType};




use crate::{base::DeviceType, tensor::Tensor};
use crate::base::error::Result;

/// Model trait for inference execution in Worker
///
/// This trait defines the core interface for running model inference with PagedAttention.
/// All inference MUST use forward_paged() for continuous batching support.
///
/// # Design Philosophy
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────┐
/// │                     Server (infer-server)                       │
/// │  - Text encoding/decoding (Tokenizer)                          │
/// │  - Request handling and response formatting                    │
/// │  - EOS token detection                                         │
/// └────────────────────────┬────────────────────────────────────────┘
///                          │ token_ids: Vec<i32>
///                          ▼
/// ┌─────────────────────────────────────────────────────────────────┐
/// │                     Scheduler (infer-scheduler)                 │
/// │  - Request batching and scheduling                             │
/// │  - Block allocation via RadixCache                             │
/// │  - Token pool management                                        │
/// └────────────────────────┬────────────────────────────────────────┘
///                          │ ForwardRequest with block_table
///                          ▼
/// ┌─────────────────────────────────────────────────────────────────┐
/// │                     Worker (infer-worker)                       │
/// │  - Model forward pass: forward_paged() ONLY (this trait)       │
/// │  - KV cache management via KVCachePool                         │
/// │  - Block-based attention (PagedAttention)                      │
/// └─────────────────────────────────────────────────────────────────┘
/// ```
///
/// # PagedAttention Architecture
///
/// For continuous batching with PagedAttention:
/// - `block_table`: List of physical block indices for each sequence
/// - `slot_mapping`: Mapping from logical positions to physical slots
/// - `context_lens`: Context length for each sequence in the batch
///
/// The model uses these to access the correct KV cache blocks.
pub trait Model: Send + Sync {
    /// Get the model configuration
    fn config(&self) -> &config::RuntimeModelConfig;

    /// Execute a forward pass with PagedAttention (the ONLY forward method)
    ///
    /// This is the exclusive entry point for inference with continuous batching.
    /// The Scheduler provides block tables that map logical positions to
    /// physical blocks in the KVCachePool.
    ///
    /// # Arguments
    /// * `input_tokens` - Input token IDs tensor, shape [num_tokens]
    /// * `positions` - Position tensor for each token
    /// * `block_tables` - Flattened block table [batch_size * max_blocks_per_req]
    /// * `max_blocks_per_req` - Stride for block table access (blocks per request)
    /// * `slot_mapping` - Mapping from token positions to physical slots
    /// * `context_lens` - Context length for each sequence
    /// * `is_prefill` - True if this is prefill phase, false for decode
    ///
    /// # Returns
    /// Logits tensor for sampled positions
    fn forward_paged(
        &mut self,
        input_tokens: &Tensor,
        positions: &Tensor,
        block_tables: &[u32],
        max_blocks_per_req: usize,
        slot_mapping: &Tensor,
        context_lens: &[u32],
        is_prefill: bool,
    ) -> Result<Tensor>;

    /// Reset KV cache state (e.g., for new sequence)
    fn reset_kv_cache(&mut self) -> Result<()>;

    /// Get the device type this model is running on
    fn device_type(&self) -> DeviceType;

    // ======================= Helper Methods =======================

    /// Get the number of layers in this model
    fn num_layers(&self) -> usize {
        self.config().layer_num
    }

    /// Get the hidden dimension
    fn hidden_dim(&self) -> usize {
        self.config().dim
    }

    /// Get vocabulary size
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }
}



#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferType {
    // Input and Embedding
    InputTokens,      // (CPU)
    InputEmbeddings,
    InputPos,         // (CPU)

    // Caches for RoPE
    SinCache,
    CosCache,

    // Buffers for Attention block
    Query,
    AttnScores,       // Storage for QK^T scores
    AttnOutput,       // Output of the attention mechanism (can reuse Query buffer)

    // Buffers for FFN block
    W1Output,
    W3Output,

    // Reusable general-purpose buffers of specific shapes
    RmsOutput,        // General buffer for RMSNorm outputs, shape [dim]
    
    // Final model output
    ForwardOutput,
    ForwardOutputCpu, // (CPU, only for CUDA execution)

    KeyCache,
    ValueCache,

    IntermediateBuffer1,
    BlockTable,
    CurrentKVLen,
}

pub type Workspace = HashMap<BufferType, Tensor>;