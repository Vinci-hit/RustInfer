// Core model infrastructure
use std::collections::HashMap;

pub mod config;
pub mod kvcache;
pub mod loader;
pub mod architectures;

// Layer abstractions
pub mod layers;

// Re-exports
pub use loader::registry::ModelRegistry;
pub use layers::{DecoderLayers, WeightMappingAdapter};
pub use kvcache::{KVCachePool, KVCacheConfig};
pub use loader::model_loader::{ModelLoader, TensorLocation, DType};

use crate::cuda::config::CudaGraph;
use crate::{base::DeviceType, tensor::Tensor};
use crate::base::error::Result;

pub trait Model: Send + Sync {
    /// Get the model configuration
    fn config(&self) -> &config::RuntimeModelConfig;

    fn forward_paged(
        &mut self,
        input_tokens: &Tensor,
        positions: &Tensor,
        kv_indices: &Tensor,         // [nnz_tokens] CSR-style slot indices for attention
        kv_indptr: &Tensor,          // [batch_size + 1] CSR row pointers
        new_slots: &Tensor,          // [total_tokens] slots for new K/V writes
        decode_tokens: usize,
        kv_cache: &mut KVCachePool,
    ) -> Result<Tensor>;
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

#[derive(Debug)]
pub struct Workspace {
    // 必需 buffer（推理核心路径）
    pub attn_output: Tensor,

    // QKV融合输出buffer
    pub qkv_output: Tensor,

    // 可选 buffer（按需分配，避免浪费）
    pub sin_cache: Tensor,
    pub cos_cache: Tensor,
    pub rms_output: Tensor,

    // FFN intermediate buffers
    pub w1_output: Tensor,        // [max_bs, intermediate_size] gate proj
    pub w3_output: Tensor,        // [max_bs, intermediate_size] up proj

    // Output buffer
    pub forward_output: Tensor,   // [max_bs, vocab_size] logits

    // 预定义的桶大小，例如 [1, 2, 4, 8, ..., 128]，当请求到来时进行二分查找，寻找最接近的桶
    ordered_buckets: Vec<usize>,

    // 静态分配的显存（Static Workspace）
    // 所有的 Graph 都绑定到这些内存地址上
    // [max_bs]
    static_input_ids: Tensor,
    // [max_bs]
    static_positions: Tensor,
    // [max_bs, max_blocks_per_req]
    static_block_tables: Tensor,
    // [max_bs]
    static_slot_mapping: Tensor,
    // [max_bs,vocab_size]
    static_output: Tensor,

    // Static buffers for attention (CudaGraph 绑定地址)
    // [max_nnz] I32 - kv_indices for CSR-style attention
    static_kv_indices: Tensor,
    // [max_bs + 1] I32 - kv_indptr for CSR row pointers
    static_kv_indptr: Tensor,

    graphs: Option<HashMap<usize, CudaGraph>>,
}

impl Workspace {
    /// 创建并初始化 Decode 专属推理工作区
    ///
    /// 专为 decoding 阶段设计，每个请求处理 1 个 token。
    ///
    /// # Arguments
    /// * `config` - 模型运行时配置
    /// * `device` - 目标设备 (CPU/CUDA)
    /// * `max_batch_size` - 最大批处理大小（决定 static buffer 和桶上限）
    /// * `max_blocks_per_req` - 每个请求最大块数（用于 block_tables 形状）
    pub fn new(
        config: &config::RuntimeModelConfig,
        device: DeviceType,
        max_batch_size: usize,
        max_blocks_per_req: usize,
    ) -> Result<Self> {
        use crate::base::DataType;

        println!("Initializing Decode Workspace: max_bs={}, max_blocks_per_req={}, device={:?}",
            max_batch_size, max_blocks_per_req, device);

        // 根据设备和模型配置选择浮点精度
        let float_dtype = if device.is_cpu() {
            DataType::F32
        } else {
            match config.torch_dtype.as_str() {
                "float32" => DataType::F32,
                "bfloat16" => DataType::BF16,
                _ => DataType::BF16, // 默认 BF16
            }
        };

        // Decode 专属: 每个请求只有 1 个 token
        // Buffer 形状: [max_batch_size, hidden_dim]

        // ---- 必需 buffer（推理核心路径）----
        // qkv_output: 融合QKV输出 [max_bs, dim + 2*kv_dim]
        let qkv_dim = config.dim + 2 * config.kv_dim;
        let qkv_output = Tensor::new(&[max_batch_size, qkv_dim], float_dtype, device)?;

        // attn_output: [max_bs, dim]
        let attn_output = Tensor::new(&[max_batch_size, config.dim], float_dtype, device)?;

        // ---- 可选 buffer ----
        // RoPE sin/cos cache: [max_seq_len, head_size] - 仍然需要完整序列长度用于 RoPE 计算
        let sin_cache = Tensor::new(&[config.seq_len, config.head_size], float_dtype, device)?;
        let cos_cache = Tensor::new(&[config.seq_len, config.head_size], float_dtype, device)?;
        // rms_output: [max_bs, dim]
        let rms_output = Tensor::new(&[max_batch_size, config.dim], float_dtype, device)?;

        // ---- FFN intermediate buffers ----
        // w1_output: [max_bs, intermediate_size] gate proj
        let w1_output = Tensor::new(&[max_batch_size, config.intermediate_size], float_dtype, device)?;
        // w3_output: [max_bs, intermediate_size] up proj
        let w3_output = Tensor::new(&[max_batch_size, config.intermediate_size], float_dtype, device)?;

        // ---- Output buffer ----
        // forward_output: [max_bs, vocab_size] logits
        let forward_output = Tensor::new(&[max_batch_size, config.vocab_size], float_dtype, device)?;

        // ---- 预定义桶大小 (powers of 2 up to max_batch_size) ----
        let ordered_buckets = Self::generate_buckets(max_batch_size);
        println!("  CudaGraph buckets: {:?}", ordered_buckets);

        // ---- 静态显存（CudaGraph 绑定地址）----
        // 使用最大桶大小作为静态 buffer 大小
        let max_bs = *ordered_buckets.last().unwrap_or(&max_batch_size);

        let static_input_ids = Tensor::new(&[max_bs], DataType::I32, device)?;
        let static_positions = Tensor::new(&[max_bs], DataType::I32, device)?;
        let static_block_tables = Tensor::new(&[max_bs, max_blocks_per_req], DataType::I32, device)?;
        let static_slot_mapping = Tensor::new(&[max_bs], DataType::I32, device)?;
        let static_output = Tensor::new(&[max_bs, config.vocab_size], float_dtype, device)?;

        // static_kv_indices: [max_nnz] for CSR-style attention indices
        // max_nnz = max_bs * seq_len (worst case: each request has full context)
        let max_nnz = max_bs * config.seq_len;
        let static_kv_indices = Tensor::new(&[max_nnz], DataType::I32, device)?;
        // static_kv_indptr: [max_bs + 1] CSR row pointers
        let static_kv_indptr = Tensor::new(&[max_bs + 1], DataType::I32, device)?;

        println!("Decode Workspace initialized successfully.");

        Ok(Self {
            qkv_output,
            attn_output,
            sin_cache,
            cos_cache,
            rms_output,
            w1_output,
            w3_output,
            forward_output,
            ordered_buckets,
            static_input_ids,
            static_positions,
            static_block_tables,
            static_slot_mapping,
            static_output,
            static_kv_indices,
            static_kv_indptr,
            graphs: None,
        })
    }

    /// 生成 2 的幂次桶大小序列: [1, 2, 4, 8, ..., max_batch_size 向上取整到 2 的幂]
    fn generate_buckets(max_batch_size: usize) -> Vec<usize> {
        let mut buckets = Vec::new();
        let mut size = 1;
        while size <= max_batch_size {
            buckets.push(size);
            size *= 2;
        }
        // 如果 max_batch_size 不是 2 的幂，把它也加上
        if let Some(&last) = buckets.last() {
            if last < max_batch_size {
                buckets.push(max_batch_size);
            }
        }
        buckets
    }

    /// 根据实际 batch_size 查找最接近的桶大小（>= batch_size 的最小桶）
    pub fn find_bucket(&self, batch_size: usize) -> Option<usize> {
        match self.ordered_buckets.binary_search(&batch_size) {
            Ok(idx) => Some(self.ordered_buckets[idx]),
            Err(idx) => self.ordered_buckets.get(idx).copied(),
        }
    }

    /// 获取对应桶大小的 CudaGraph（如果已捕获）
    pub fn get_graph(&self, bucket_size: usize) -> Option<&CudaGraph> {
        self.graphs.as_ref()?.get(&bucket_size)
    }

    /// 插入已捕获的 CudaGraph
    pub fn insert_graph(&mut self, bucket_size: usize, graph: CudaGraph) {
        self.graphs
            .get_or_insert_with(HashMap::new)
            .insert(bucket_size, graph);
    }

    /// 获取静态 input_ids buffer（用于 CudaGraph 的地址绑定）
    pub fn static_input_ids(&self) -> &Tensor { &self.static_input_ids }
    pub fn static_positions(&self) -> &Tensor { &self.static_positions }
    pub fn static_block_tables(&self) -> &Tensor { &self.static_block_tables }
    pub fn static_slot_mapping(&self) -> &Tensor { &self.static_slot_mapping }
    pub fn static_output(&self) -> &Tensor { &self.static_output }
    pub fn static_output_mut(&mut self) -> &mut Tensor { &mut self.static_output }
    pub fn static_kv_indices(&self) -> &Tensor { &self.static_kv_indices }
    pub fn static_kv_indptr(&self) -> &Tensor { &self.static_kv_indptr }

    /// 将运行时数据拷贝到静态 buffer（CudaGraph 执行前调用）
    pub fn copy_to_static(
        &mut self,
        input_ids: &Tensor,
        positions: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        self.static_input_ids.copy_from(input_ids)?;
        self.static_positions.copy_from(positions)?;
        self.static_slot_mapping.copy_from(slot_mapping)?;
        Ok(())
    }

    /// 返回所有桶大小
    pub fn buckets(&self) -> &[usize] {
        &self.ordered_buckets
    }
}