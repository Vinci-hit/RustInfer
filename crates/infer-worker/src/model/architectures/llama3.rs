use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
use std::time::Instant;
use crate::cuda::CudaConfig;
use crate::op::{Op, OpContext};
use crate::model::config::RuntimeModelConfig;
use crate::model::ModelLoader;
use crate::tensor::Tensor;
// use super::tokenizer::Tokenizer; // TODO: Tokenizer module was removed
use std::boxed::Box;
use crate::model::{BufferType, Workspace, Model};
use crate::op::sampler::{Sampler, ArgmaxSampler};
use crate::model::layers::DecoderLayers;
// Weight mapping is managed internally by this module
use crate::model::WeightMappingAdapter;
// Import KvCache from the kvcache module
use crate::model::kvcache::{KVCachePool, KVCacheConfig};

pub struct Llama3 {
    config: Arc<RuntimeModelConfig>,
    device_type: DeviceType,
    layers: DecoderLayers,
    kv_cache_pool: KVCachePool,
    current_kv_len: usize,
    block_table: Tensor,
    block_size: usize,
    num_total_blocks: usize,
    workspace: Workspace,
    output_token: Tensor,
    input_pos: Tensor,
    sampler: Box<dyn Sampler>,
    cuda_config: Option<CudaConfig>,
}

// ======================= Llama3 Weight Mapping =======================

/// Llama3 model's weight mapping implementation
///
/// Defines how Llama3 maps safetensor weight names to internal layers.
/// Llama3 uses the following naming convention:
/// - model.embed_tokens.weight
/// - model.layers.{layer}.self_attn.q_proj.weight
/// - model.layers.{layer}.mlp.gate_proj.weight
/// - model.norm.weight
/// - lm_head.weight
#[derive(Debug, Clone)]
struct Llama3WeightMapping;

impl WeightMappingAdapter for Llama3WeightMapping {
    fn embedding(&self) -> &'static str {
        "model.embed_tokens.weight"
    }

    fn rmsnorm_final(&self) -> &'static str {
        "model.norm.weight"
    }

    fn cls(&self) -> &'static str {
        "lm_head.weight"
    }

    fn format_layer_weight(&self, layer_idx: usize, weight_name: &str) -> String {
        format!("model.layers.{}.{}", layer_idx, weight_name)
    }

    fn attn_q(&self) -> &'static str {
        "self_attn.q_proj.weight"
    }

    fn attn_k(&self) -> &'static str {
        "self_attn.k_proj.weight"
    }

    fn attn_v(&self) -> &'static str {
        "self_attn.v_proj.weight"
    }

    fn attn_o(&self) -> &'static str {
        "self_attn.o_proj.weight"
    }

    fn ffn_gate(&self) -> &'static str {
        "mlp.gate_proj.weight"
    }

    fn ffn_up(&self) -> &'static str {
        "mlp.up_proj.weight"
    }

    fn ffn_down(&self) -> &'static str {
        "mlp.down_proj.weight"
    }

    fn rmsnorm_attn(&self) -> &'static str {
        "input_layernorm.weight"
    }

    fn rmsnorm_ffn(&self) -> &'static str {
        "post_attention_layernorm.weight"
    }
}

// ======================= Llama3 Implementation =======================

// Llama3 implementation

impl Llama3 {
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        device_type: DeviceType,
        is_quant_model: bool
    ) -> Result<Self> {
        let start_time = Instant::now();
        println!("Start calculate time, Loading Llama3 model from directory: {:?}", model_dir.as_ref());
        let mut loader = ModelLoader::load(model_dir.as_ref())?;
        let config = loader.config.clone();
        let cuda_config = CudaConfig::new()?;

        // Initialize KV Cache Pool (Paged 模式）
        let block_size = 16;  // 每个 block 16 个 token
        let num_total_blocks = (config.seq_len + block_size - 1) / block_size;

        println!("Creating Llama3 decoder layers...");

        // Use DecoderLayers with Llama3 weight mapping
        let weight_mapping = Llama3WeightMapping;
        let layers = DecoderLayers::from_loader(
            &loader,
            &config,
            &weight_mapping,
            device_type,
            is_quant_model,
            block_size,
            num_total_blocks,
        )?;
        let kv_cache_config = KVCacheConfig::new(
            config.layer_num,
            config.kv_head_num,
            config.head_size,
            if device_type.is_cpu() { DataType::F32 } else { DataType::BF16 },
            block_size,
            num_total_blocks,
        );
        let kv_cache_pool = KVCachePool::new(kv_cache_config, device_type)?;

        // --- 初始化工作区 ---
        let workspace = Self::init_workspace(&config, &device_type, num_total_blocks)?;
        let sampler = Box::new(ArgmaxSampler::new(device_type));
        let output_token = Tensor::new(&[1], DataType::I32, device_type)?;
        let input_pos = Tensor::new(&[1], DataType::I32, device_type)?;

        // 初始化 block table（连续分配，顺序使用）
        let max_blocks_per_seq = (config.seq_len + block_size - 1) / block_size;
        let mut block_table = Tensor::new(&[max_blocks_per_seq], DataType::I32, device_type)?;
        // 初始化 block table 为顺序索引 [0, 1, 2, ...]
        for i in 0..max_blocks_per_seq {
            block_table.as_i32_mut()?.as_slice_mut()?[i] = i as i32;
        }

        let mut model = Self {
            config: config.into(),
            device_type,
            layers,
            kv_cache_pool,
            current_kv_len: 0,
            block_table,
            block_size,
            num_total_blocks,
            workspace,
            sampler,
            cuda_config: Some(cuda_config),
            output_token,
            input_pos,
        };
        println!("Calculating RoPE sin/cos cache...");
        Self::calculate_rope_cache(&model.config, &mut model.workspace)?;
        let duration = start_time.elapsed();
        println!("Model successfully initialized. Loading time: {:.2} seconds", duration.as_secs_f64());
        Ok(model)
    }

    fn calculate_rope_cache(
        config: &RuntimeModelConfig,
        workspace: &mut Workspace,
    ) -> Result<()> {
        // 从工作区获取 sin 和 cos 缓存张量的可变引用
        // 注意：这里我们同时借用两个可变的元素，这在 Rust 中是安全的，
        // 因为 HashMap::get_mut 每次只返回一个元素的生命周期。
        // 为了避免借用冲突，我们分开获取。
        let caches = workspace.get_disjoint_mut([
            &BufferType::SinCache,
            &BufferType::CosCache,
        ]);

        // 使用 if let 来安全地解构元组
        if let [Some(sin_cache), Some(cos_cache)] = caches {
            // ---- 成功获取了两个可变的张量，现在可以安全地使用它们 ----
            match sin_cache.device() {
                DeviceType::Cpu => {
                    crate::op::kernels::cpu::rope_sin_cos_cache_calc(
                        config.head_size,
                        config.seq_len,
                        500000.0, // Llama3 rope_base
                        sin_cache, // 直接传递 &mut Tensor
                        cos_cache, // 直接传递 &mut Tensor
                    )?;
                }
                #[cfg(feature = "cuda")]
                DeviceType::Cuda(_) => {
                    let stream = crate::cuda::config::CudaConfig::new()?;
                    crate::op::kernels::cuda::rope_sin_cos_cache_calc_cuda(
                        config.head_size,
                        config.seq_len,
                        500000.0, // Llama3 rope_base
                        sin_cache, // 直接传递 &mut Tensor
                        cos_cache, // 直接传递 &mut Tensor
                        Some(&stream),
                    )?;
                }
            }
            Ok(())
        } else {
            // 如果有一个或多个键不存在，返回错误
            Err(Error::InternalError("SinCache or CosCache not found in workspace".to_string()).into())
        }
    }

    /// 预分配前向传播所需的所有中间张量。
    fn init_workspace(
        config: &RuntimeModelConfig,
        device: &DeviceType,
        num_total_blocks: usize,
    ) -> Result<Workspace> {
        println!("Initializing inference workspace for device {:?}...", device);
        let mut buffers = HashMap::new();

        // 遵循您的代码风格，大部分浮点运算使用 BF16
        // 但如果设备是CPU，则使用F32以获得更好的精度和兼容性
        let float_dtype = if device.is_cpu() {
            println!("  -> Using F32 for CPU device.");
            DataType::F32
        } else {
            match config.torch_dtype.as_str() {
                "float32" => {
                    println!("  -> torch_dtype is float32.");
                    DataType::F32
                },
                "bfloat16" => {
                    println!("  -> torch_dtype is bfloat16.");
                    DataType::BF16
                },
                _ => {
                    return Err(Error::InvalidArgument(format!(
                        "Unsupported torch_dtype: {}", config.torch_dtype
                    )).into());
                }
            }
        };
        let int_dtype = DataType::I32;
        
        // 为了支持批处理，许多缓冲区的大小需要基于 max_seq_len
        let max_seq_len = config.seq_len;

        // ---- 分配缓冲区 ----

        // 1. 输入张量
        // InputTokens 需要足够大以容纳整个批处理的 prompt
        buffers.insert(
            BufferType::InputTokens,
            Tensor::new(&[max_seq_len], int_dtype, *device)?, // << 尺寸变更为 device
        );
        buffers.insert(
            BufferType::InputPos,
            Tensor::new(&[1], int_dtype, DeviceType::Cpu)?,
        );

        // 2. 词嵌入层的输出 (现在是批处理形状)
        buffers.insert(
            BufferType::InputEmbeddings,
            Tensor::new(&[max_seq_len, config.dim], float_dtype, *device)?,
        );
        
        // 3. RoPE 缓存 (形状不变)
        buffers.insert(
            BufferType::SinCache,
            Tensor::new(&[max_seq_len, config.head_size], float_dtype, *device)?,
        );
        buffers.insert(
            BufferType::CosCache,
            Tensor::new(&[max_seq_len, config.head_size], float_dtype, *device)?,
        );
        
        // 4. 关键的、可复用的缓冲区 (现在是批处理形状)
        // 这个缓冲区将被 RMSNorm, MHA output 等轮流使用
        buffers.insert(
            BufferType::RmsOutput,
            Tensor::new(&[max_seq_len, config.dim], float_dtype, *device)?,
        );

        // 5. Query 缓冲区 (批处理形状)
        buffers.insert(
            BufferType::Query,
            Tensor::new(&[max_seq_len, config.dim], float_dtype, *device)?,
        );
        
        // 6. FFN (SwiGLU) 的输入缓冲区 (批处理形状)
        buffers.insert(
            BufferType::W1Output,
            Tensor::new(&[max_seq_len, config.intermediate_size], float_dtype, *device)?,
        );
        buffers.insert(
            BufferType::W3Output,
            Tensor::new(&[max_seq_len, config.intermediate_size], float_dtype, *device)?,
        );
        
        // 7. Attention scores (QK^T) 的缓冲区 (批处理形状)
        buffers.insert(
            BufferType::AttnScores,
            Tensor::new(&[config.head_num, max_seq_len, max_seq_len], float_dtype, *device)?,
        );

        // ======================= 新增的缓冲区 =======================
        // 8. 用于 forward_batch 的临时 K 和 V 缓冲区
        // 它们的形状与 Q 类似，都是 [max_seq_len, dim]
        buffers.insert(
            BufferType::KeyCache, // 临时 K 缓冲区
            Tensor::new(&[max_seq_len, config.kv_dim], float_dtype, *device)?,
        );
        buffers.insert(
            BufferType::ValueCache, // 临时 V 缓冲区
            Tensor::new(&[max_seq_len, config.kv_dim], float_dtype, *device)?,
        );

        buffers.insert(
            BufferType::IntermediateBuffer1,
            Tensor::new(&[max_seq_len, config.dim], float_dtype, *device)?,
        );

        // Block table buffer
        let max_blocks_per_seq = (max_seq_len + 16 - 1) / 16;  // block_size=16
        buffers.insert(
            BufferType::BlockTable,
            Tensor::new(&[max_blocks_per_seq], DataType::I32, *device)?
        );

        // Current KV length buffer
        buffers.insert(
            BufferType::CurrentKVLen,
            Tensor::new(&[1], DataType::I32, *device)?
        );

        // ==========================================================

        // 10. 模型的最终 logits 输出 (用于单 token)
        let forward_output = Tensor::new(&[config.vocab_size], float_dtype, *device)?;
        buffers.insert(BufferType::ForwardOutput, forward_output);

        println!("Workspace (batch-enabled) initialized with {} buffers.", buffers.len());
        Ok(buffers)
    }

    fn forward_decoding(&mut self, _tokens: &Tensor, pos_cpu: &Tensor) -> Result<i32> {
        self.input_pos.copy_from(pos_cpu)?;
        let input_tokens_view = &self.output_token;
        let cuda_config_ref = if self.device_type.is_cuda() { self.cuda_config.as_ref() } else { None };

        // Token embedding
        let x_buffer = self.workspace.get_mut(&BufferType::InputEmbeddings).unwrap();
        let mut x = x_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        self.layers.embedding_layer.forward(
            &mut OpContext::new(&[input_tokens_view], &mut [&mut x], cuda_config_ref)
        )?;

        // Get current position and update current_kv_len
        let pos = self.input_pos.as_i32()?.as_slice()?[0] as usize;

        // Process all transformer layers
        for i in 0..self.config.layer_num {
            let attn_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out = attn_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[i].forward(&mut OpContext::new(&[&x], &mut [&mut attn_norm_out], cuda_config_ref))?;

            let q_buffer = self.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buffer.slice(&[0, 0], &[1, self.config.dim])?;

            // Q projection
            self.layers.wq_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut q], cuda_config_ref))?;

            // K, V projections to temporary buffers
            let k_buffer = self.workspace.get_mut(&BufferType::KeyCache).unwrap();
            let mut k_temp = k_buffer.slice(&[0, 0], &[1, self.config.kv_dim])?;
            let v_buffer = self.workspace.get_mut(&BufferType::ValueCache).unwrap();
            let mut v_temp = v_buffer.slice(&[0, 0], &[1, self.config.kv_dim])?;

            self.layers.wk_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut k_temp], cuda_config_ref))?;
            self.layers.wv_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut v_temp], cuda_config_ref))?;

            // RoPE on K only (Q doesn't need RoPE in decode mode since pos is just new token)
            let sin_cache = self.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = self.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[i].forward(&mut OpContext::new(&[&self.input_pos, sin_cache, cos_cache], &mut [&mut k_temp], cuda_config_ref))?;

            // Write K, V to KV cache pool at current position
            let block_idx = pos / self.block_size;
            let slot_idx = pos % self.block_size;

            // Get K, V tensors from pool
            let (k_pool_base, v_pool_base) = self.kv_cache_pool.get(i)?;
            let k_pool_shape = k_pool_base.shape().to_vec();
            let v_pool_shape = v_pool_base.shape().to_vec();

            // Write K, V to pool
            let k_write_view = k_temp.slice(&[0, 0], &[1, self.config.kv_dim])?;
            let v_write_view = v_temp.slice(&[0, 0], &[1, self.config.kv_dim])?;
            self.kv_cache_pool.write_kv_to_slot(i, block_idx, slot_idx, &k_write_view, &v_write_view)?;

            // Get K, V references after write
            let (k_pool_base, v_pool_base) = self.kv_cache_pool.get(i)?;

            // Get current KV length buffer
            let kv_len_buffer = self.workspace.get_mut(&BufferType::CurrentKVLen).unwrap();
            kv_len_buffer.as_i32_mut()?.as_slice_mut()?[0] = self.current_kv_len as i32;

            // Paged Attention using 4D KV cache
            let mut attn_out = attn_norm_out; // Reuse buffer
            self.layers.mha_layers[i].forward(&mut OpContext::new(&[
                &q,
                k_pool_base,  // 4D: [num_blocks, block_size, num_kv_heads, head_dim]
                v_pool_base,
                &self.block_table,
                kv_len_buffer,
            ], &mut [&mut attn_out], cuda_config_ref))?;

            // Wo projection
            let mut wo_out = q; // Reuse buffer
            self.layers.wo_layers[i].forward(&mut OpContext::new(&[&attn_out], &mut [&mut wo_out], cuda_config_ref))?;
            self.layers.add_layers.forward(&mut OpContext::new(&[&wo_out], &mut [&mut x], cuda_config_ref))?;

            // FFN Block
            let mut ffn_norm_out = attn_out; // Reuse buffer
            self.layers.rmsnorm_ffn_layers[i].forward(&mut OpContext::new(&[&x], &mut [&mut ffn_norm_out], cuda_config_ref))?;
            let w1_buffer = self.workspace.get_mut(&BufferType::W1Output).unwrap();
            let mut w1_out = w1_buffer.slice(&[0, 0], &[1, self.config.intermediate_size])?;
            let w3_buffer = self.workspace.get_mut(&BufferType::W3Output).unwrap();
            let mut w3_out = w3_buffer.slice(&[0, 0], &[1, self.config.intermediate_size])?;
            self.layers.w1_layers[i].forward(&mut OpContext::new(&[&ffn_norm_out], &mut [&mut w1_out], cuda_config_ref))?;
            self.layers.w3_layers[i].forward(&mut OpContext::new(&[&ffn_norm_out], &mut [&mut w3_out], cuda_config_ref))?;
            self.layers.swiglu_layers[i].forward(&mut OpContext::new(&[&w3_out], &mut [&mut w1_out], cuda_config_ref))?;

            let mut w2_out = ffn_norm_out; // Reuse buffer
            self.layers.w2_layers[i].forward(&mut OpContext::new(&[&w1_out], &mut [&mut w2_out], cuda_config_ref))?;
            self.layers.add_layers.forward(&mut OpContext::new(&[&w2_out], &mut [&mut x], cuda_config_ref))?;
        }

        // Extract last token's hidden state
        let last_hidden_state_view = x.slice(&[0, 0], &[1, self.config.dim])?;

        // Final Norm and classifier
        let final_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
        let mut final_norm_out = final_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;

        self.layers.rmsnorm_final_layer.forward(
            &mut OpContext::new(&[&last_hidden_state_view], &mut [&mut final_norm_out], cuda_config_ref)
        )?;

        let logits = self.workspace.get_mut(&BufferType::ForwardOutput).unwrap();

        self.layers.cls_layer.forward(
            &mut OpContext::new(&[&final_norm_out], &mut [logits], cuda_config_ref)
        )?;
        let logits_ref = self.workspace.get(&BufferType::ForwardOutput).unwrap();
        self.sampler.sample(logits_ref, &mut self.output_token, cuda_config_ref)?;

        // Update KV length after processing all layers
        self.current_kv_len += 1;

        let next_token = self.output_token.to_cpu()?.as_i32()?.as_slice()?[0];
        Ok(next_token)
    }
    /// Forward prefill with TRUE batch processing
    /// Processes multiple users in a single forward pass
    ///
    /// # Arguments
    /// * `tokens` - Batched tokens with shape [batch_size * seq_len]
    /// * `pos_cpu` - Starting position (should be 0 for prefill)
    /// * `batch_size` - Number of users in the batch
    /// * `seq_len` - Sequence length per user
    ///
    /// # Returns
    /// Vec<i32> - First generated token for each user
    fn forward_prefill_batch(&mut self, tokens: &Tensor, pos_cpu: &Tensor, batch_size: usize, seq_len: usize) -> Result<Vec<i32>> {
        let pos = pos_cpu.as_i32()?.as_slice()?[0] as usize;
        let cuda_config_ref = if self.device_type.is_cuda() { self.cuda_config.as_ref() } else { None };
        self.input_pos.copy_from(pos_cpu)?;
        let total_tokens = batch_size * seq_len;

        // Prepare batch input tokens [batch_size * seq_len]
        let input_tokens_buffer = self.workspace.get_mut(&BufferType::InputTokens).unwrap();
        let mut input_tokens_view = input_tokens_buffer.slice(&[0], &[total_tokens])?;
        input_tokens_view.copy_from(tokens)?;
        // Token embedding: [batch_size * seq_len] -> [batch_size * seq_len, dim]
        let x_buffer = self.workspace.get_mut(&BufferType::InputEmbeddings).unwrap();
        let mut x = x_buffer.slice(&[0, 0], &[total_tokens, self.config.dim])?;
        self.layers.embedding_layer.forward(
            &mut OpContext::new(&[&input_tokens_view], &mut [&mut x], cuda_config_ref)
        )?;
        // Process all transformer layers with batched input
        for i in 0..self.config.layer_num {
            // Attention Block
            let attn_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out = attn_norm_out_buffer.slice(&[0, 0], &[total_tokens, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[i].forward(&mut OpContext::new(&[&x], &mut [&mut attn_norm_out], cuda_config_ref))?;

            let q_buffer = self.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buffer.slice(&[0, 0], &[total_tokens, self.config.dim])?;

            // For batch processing, we need to handle KV cache for each sequence separately
            // For now, process each user's KV cache sequentially within the batch
            // Temporary: K, V cache slicing not yet implemented
            let (k_cache, v_cache) = self.kv_cache_pool.get_mut(i)?;
            let mut k = k_cache.slice(&[0, 0], &[total_tokens, self.config.kv_dim])?;
            let mut v = v_cache.slice(&[0, 0], &[total_tokens, self.config.kv_dim])?;

            self.layers.wq_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut q], cuda_config_ref))?;
            self.layers.wk_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut k], cuda_config_ref))?;
            self.layers.wv_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut v], cuda_config_ref))?;

            let sin_cache = self.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = self.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[i].forward(&mut OpContext::new(&[&self.input_pos, sin_cache, cos_cache], &mut [&mut q, &mut k], cuda_config_ref))?;

            let (k_cache_history, v_cache_history) = self.kv_cache_pool.get(i).unwrap();
            let mut attn_out = attn_norm_out; // Reuse buffer
            self.layers.mha_layers[i].forward(&mut OpContext::new(&[&q, k_cache_history, v_cache_history, pos_cpu], &mut [&mut attn_out], cuda_config_ref))?;

            let mut wo_out = q; // Reuse buffer
            self.layers.wo_layers[i].forward(&mut OpContext::new(&[&attn_out], &mut [&mut wo_out], cuda_config_ref))?;
            self.layers.add_layers.forward(&mut OpContext::new(&[&wo_out], &mut [&mut x], cuda_config_ref))?;

            // FFN Block
            let mut ffn_norm_out = attn_out; // Reuse buffer
            self.layers.rmsnorm_ffn_layers[i].forward(&mut OpContext::new(&[&x], &mut [&mut ffn_norm_out], cuda_config_ref))?;

            let w1_buffer = self.workspace.get_mut(&BufferType::W1Output).unwrap();
            let mut w1_out = w1_buffer.slice(&[0, 0], &[total_tokens, self.config.intermediate_size])?;
            let w3_buffer = self.workspace.get_mut(&BufferType::W3Output).unwrap();
            let mut w3_out = w3_buffer.slice(&[0, 0], &[total_tokens, self.config.intermediate_size])?;

            self.layers.w1_layers[i].forward(&mut OpContext::new(&[&ffn_norm_out], &mut [&mut w1_out], cuda_config_ref))?;
            self.layers.w3_layers[i].forward(&mut OpContext::new(&[&ffn_norm_out], &mut [&mut w3_out], cuda_config_ref))?;
            self.layers.swiglu_layers[i].forward(&mut OpContext::new(&[&w3_out], &mut [&mut w1_out], cuda_config_ref))?;

            let mut w2_out = ffn_norm_out; // Reuse buffer
            self.layers.w2_layers[i].forward(&mut OpContext::new(&[&w1_out], &mut [&mut w2_out], cuda_config_ref))?;
            self.layers.add_layers.forward(&mut OpContext::new(&[&w2_out], &mut [&mut x], cuda_config_ref))?;
        }

        // Extract last token's hidden state for each user in the batch
        // For batch_size users with seq_len tokens each:
        // User 0: position seq_len-1
        // User 1: position 2*seq_len-1
        // User 2: position 3*seq_len-1, etc.
        let mut first_tokens = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let last_token_pos = (batch_idx + 1) * seq_len - 1;

            // Extract this user's last token hidden state
            let last_hidden_state_view = x.slice(&[last_token_pos, 0], &[1, self.config.dim])?;

            // Prepare for final norm and classifier
            let final_norm_input_buffer = self.workspace.get_mut(&BufferType::IntermediateBuffer1).unwrap();
            let mut final_norm_input = final_norm_input_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            final_norm_input.copy_from(&last_hidden_state_view)?;

            // Final Norm
            let final_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut final_norm_out = final_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            self.layers.rmsnorm_final_layer.forward(
                &mut OpContext::new(&[&final_norm_input], &mut [&mut final_norm_out], cuda_config_ref)
            )?;

            // Classifier
            let logits = self.workspace.get_mut(&BufferType::ForwardOutput).unwrap();
            self.layers.cls_layer.forward(
                &mut OpContext::new(&[&final_norm_out], &mut [logits], cuda_config_ref)
            )?;

            // Sample token for this user
            let logits_ref = self.workspace.get(&BufferType::ForwardOutput).unwrap();
            self.sampler.sample(logits_ref, &mut self.output_token, cuda_config_ref)?;
            let next_token = self.output_token.to_cpu()?.as_i32()?.as_slice()?[0];

            first_tokens.push(next_token);
        }

        Ok(first_tokens)
    }

    fn forward_prefill(&mut self, tokens: &Tensor, pos_cpu: &Tensor, seq_len: usize) -> Result<i32> {
        let pos = pos_cpu.as_i32()?.as_slice()?[0] as usize;
        let cuda_config_ref = if self.device_type.is_cuda() { self.cuda_config.as_ref() } else { None };
        self.input_pos.copy_from(pos_cpu)?;
        // Prepare batch input tokens
        let input_tokens_buffer = self.workspace.get_mut(&BufferType::InputTokens).unwrap();
        let mut input_tokens_view = input_tokens_buffer.slice(&[0], &[seq_len])?;
        input_tokens_view.copy_from(tokens)?;

        // Token embedding
        let x_buffer = self.workspace.get_mut(&BufferType::InputEmbeddings).unwrap();
        let mut x = x_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
        self.layers.embedding_layer.forward(
            &mut OpContext::new(&[&input_tokens_view], &mut [&mut x], cuda_config_ref)
        )?;

        // Process all transformer layers
        for i in 0..self.config.layer_num {
            // Attention Block
            let attn_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out = attn_norm_out_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[i].forward(&mut OpContext::new(&[&x], &mut [&mut attn_norm_out], cuda_config_ref))?;

            let q_buffer = self.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;

            // Compute Q, K, V for all prompt tokens
            self.layers.wq_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut q], cuda_config_ref))?;

            // K, V projections to temporary buffers
            let k_buffer = self.workspace.get_mut(&BufferType::KeyCache).unwrap();
            let mut k_temp = k_buffer.slice(&[0, 0], &[seq_len, self.config.kv_dim])?;
            let v_buffer = self.workspace.get_mut(&BufferType::ValueCache).unwrap();
            let mut v_temp = v_buffer.slice(&[0, 0], &[seq_len, self.config.kv_dim])?;

            self.layers.wk_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut k_temp], cuda_config_ref))?;
            self.layers.wv_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut v_temp], cuda_config_ref))?;

            // Apply RoPE to K (Q uses full position range for prefill)
            let sin_cache = self.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = self.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[i].forward(&mut OpContext::new(&[&self.input_pos, sin_cache, cos_cache], &mut [&mut k_temp], cuda_config_ref))?;

            // Write K, V values to KV cache pool for each position
            // For paged KV cache, we write each token to its slot
            for token_pos in pos..(pos + seq_len) {
                let block_idx = token_pos / self.block_size;
                let slot_idx = token_pos % self.block_size;

                // Get K, V for this specific token position
                // K, V shape: [seq_len, kv_dim], need to get token_pos-pos row
                let k_single = k_temp.slice(&[token_pos - pos, 0], &[1, self.config.kv_dim])?;
                let v_single = v_temp.slice(&[token_pos - pos, 0], &[1, self.config.kv_dim])?;

                // Write to pool
                self.kv_cache_pool.write_kv_to_slot(i, block_idx, slot_idx, &k_single, &v_single)?;
            }

            // Get KV cache for paged attention
            let (k_pool_base, v_pool_base) = self.kv_cache_pool.get(i)?;

            // Get current KV length buffer
            let kv_len_buffer = self.workspace.get_mut(&BufferType::CurrentKVLen).unwrap();
            kv_len_buffer.as_i32_mut()?.as_slice_mut()?[0] = (pos + seq_len) as i32;

            // Paged Attention using 4D KV cache
            let mut attn_out = attn_norm_out; // Reuse buffer
            self.layers.mha_layers[i].forward(&mut OpContext::new(&[
                &q,
                k_pool_base,  // 4D: [num_blocks, block_size, num_kv_heads, head_dim]
                v_pool_base,
                &self.block_table,
                kv_len_buffer,
            ], &mut [&mut attn_out], cuda_config_ref))?;

            let mut wo_out = q; // Reuse buffer
            self.layers.wo_layers[i].forward(&mut OpContext::new(&[&attn_out], &mut [&mut wo_out], cuda_config_ref))?;
            self.layers.add_layers.forward(&mut OpContext::new(&[&wo_out], &mut [&mut x], cuda_config_ref))?;

            // FFN Block
            let mut ffn_norm_out = attn_out; // Reuse buffer
            self.layers.rmsnorm_ffn_layers[i].forward(&mut OpContext::new(&[&x], &mut [&mut ffn_norm_out], cuda_config_ref))?;
            let w1_buffer = self.workspace.get_mut(&BufferType::W1Output).unwrap();
            let mut w1_out = w1_buffer.slice(&[0, 0], &[seq_len, self.config.intermediate_size])?;
            let w3_buffer = self.workspace.get_mut(&BufferType::W3Output).unwrap();
            let mut w3_out = w3_buffer.slice(&[0, 0], &[seq_len, self.config.intermediate_size])?;
            self.layers.w1_layers[i].forward(&mut OpContext::new(&[&ffn_norm_out], &mut [&mut w1_out], cuda_config_ref))?;
            self.layers.w3_layers[i].forward(&mut OpContext::new(&[&ffn_norm_out], &mut [&mut w3_out], cuda_config_ref))?;
            self.layers.swiglu_layers[i].forward(&mut OpContext::new(&[&w3_out], &mut [&mut w1_out], cuda_config_ref))?;

            let mut w2_out = ffn_norm_out; // Reuse buffer
            self.layers.w2_layers[i].forward(&mut OpContext::new(&[&w1_out], &mut [&mut w2_out], cuda_config_ref))?;
            self.layers.add_layers.forward(&mut OpContext::new(&[&w2_out], &mut [&mut x], cuda_config_ref))?;
        }

        // Update KV length after prefill
        self.current_kv_len = pos + seq_len;

        // Extract last token's hidden state
        let last_hidden_state_view = x.slice(&[seq_len - 1, 0], &[1, self.config.dim])?;

        // Prepare for final norm and classifier
        let final_norm_input_buffer = self.workspace.get_mut(&BufferType::IntermediateBuffer1).unwrap();
        let mut final_norm_input = final_norm_input_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        final_norm_input.copy_from(&last_hidden_state_view)?;

        // Final Norm and classifier
        let final_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
        let mut final_norm_out = final_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;

        self.layers.rmsnorm_final_layer.forward(
            &mut OpContext::new(&[&final_norm_input], &mut [&mut final_norm_out], cuda_config_ref)
        )?;
        let logits = self.workspace.get_mut(&BufferType::ForwardOutput).unwrap();

        self.layers.cls_layer.forward(
            &mut OpContext::new(&[&final_norm_out], &mut [logits], cuda_config_ref)
        )?;
        let logits_ref = self.workspace.get(&BufferType::ForwardOutput).unwrap();
        self.sampler.sample(logits_ref, &mut self.output_token, cuda_config_ref)?;
        let next_token = self.output_token.to_cpu()?.as_i32()?.as_slice()?[0];

        Ok(next_token)
    }

    pub fn get_buffer(&self, buffer_type: BufferType) -> Result<&Tensor> {
        self.workspace.get(&buffer_type).ok_or_else(|| {
            Error::InternalError(format!(
                "Buffer {:?} not found in workspace.",
                buffer_type
            )).into()
        })
    }

    pub fn get_buffer_mut(&mut self, buffer_type: BufferType) -> Result<&mut Tensor> {
        self.workspace.get_mut(&buffer_type).ok_or_else(|| {
            Error::InternalError(format!(
                "Mutable buffer {:?} not found in workspace.",
                buffer_type
            )).into()
        })
    }

    /// Get the device type used by this model
    pub fn device(&self) -> DeviceType {
        self.device_type
    }
}

/// Builder function for Llama3 models
/// 
/// This function creates a new Llama3 model instance and returns it as a Box<dyn Model>
/// which can be used with the ModelRegistry.
/// 
/// # Arguments
/// * `model_dir` - Path to the model directory containing config.json and weights
/// * `device_type` - Device to run the model on (CPU or CUDA)
/// * `is_quant_model` - Whether the model is quantized
/// 
/// # Returns
/// A Box containing the model instance implementing the Model trait
pub fn builder(
    model_dir: &Path,
    device_type: DeviceType,
    is_quant_model: bool,
) -> Result<Box<dyn Model>> {
    let model = Llama3::new(model_dir, device_type, is_quant_model)?;
    Ok(Box::new(model))
}

// ============================================================================ //  Model Trait Implementation // ============================================================================

impl Model for Llama3 {
    fn config(&self) -> &RuntimeModelConfig {
        &self.config
    }

    fn forward_paged(
            &mut self,
            input_tokens: &Tensor,
            positions: &Tensor,
            block_tables: &[Vec<u32>],
            slot_mapping: &Tensor,
            context_lens: &[usize],
            is_prefill: bool,
        ) -> Result<Tensor> {
        unimplemented!("Paged forward not yet implemented for Llama3.");
    }

    fn reset_kv_cache(&mut self) -> Result<()> {
        // Reset is handled by re-initializing - for now just return Ok
        // In a full implementation, you might zero out the cache or
        // reset position tracking
        Ok(())
    }

    fn device_type(&self) -> DeviceType {
        self.device_type
    }
}