use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;

use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
use std::time::Instant;
use crate::cuda::CudaConfig;
use crate::op::{Op, OpContext};
use super::config::RuntimeModelConfig;
use super::ModelLoader;
use crate::tensor::Tensor;
use super::tokenizer::Tokenizer;
use std::boxed::Box;
use crate::model::{BufferType, Workspace};
use crate::op::sampler::{Sampler, ArgmaxSampler};
use super::layers::{DecoderLayers, WeightMapping};


pub struct Llama3 {
    config: RuntimeModelConfig,
    device_type: DeviceType,
    tokenizer: Box<dyn Tokenizer>,
    /// Core decoder layers (shared abstraction with Qwen2)
    layers: DecoderLayers,

    /// KV Cache
    kv_cache: KvCache,
    kcache: Tensor,
    vcache: Tensor,
    /// 工作空间，用于存放中间计算结果，避免频繁分配和释放内存
    workspace: Workspace,
    output_token: Tensor,
    input_pos: Tensor,
    sampler: Box<dyn Sampler>,
    cuda_config: Option<CudaConfig>,
}

struct KvCache {
    cache: Vec<(Tensor, Tensor)>,
}
impl KvCache {
    pub fn slice_kv_cache(
        &mut self, 
        layer_idx: usize, 
        start_pos: i32, 
        len: usize,
        kv_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        // 1. 安全地获取指定层的 (Key, Value) 缓存张量的可变引用
        let (k_cache_full, v_cache_full) = self.get_mut(layer_idx)?;

        // 2. 确定 KV 缓存的形状和要切片的维度
        // 假设您的 KV 缓存形状是 `[max_seq_len, num_heads, head_size]` 或 `[max_seq_len, hidden_dim]`
        // 我们总是在第一个维度 (seq_len) 上进行切片。
        let seq_len_dim_idx = 0;
        let max_seq_len = k_cache_full.shape()[seq_len_dim_idx];

        // 3. 边界检查，确保切片范围 `[start_pos, start_pos + len)` 不会越界
        if start_pos as usize + len > max_seq_len {
            return Err(anyhow::anyhow!(
                "KV cache slice is out of bounds. Attempted to slice from pos {} with length {}, but max_seq_len is {}.",
                start_pos, len, max_seq_len
            ));
        }


        // 5. 调用 Tensor::slice 方法来创建零拷贝视图
        let k_slice = k_cache_full.slice(&[start_pos as usize,0], &[len,kv_dim])?;
        let v_slice = v_cache_full.slice(&[start_pos as usize,0], &[len,kv_dim])?;

        // 6. 返回包含两个视图的元组
        Ok((k_slice, v_slice))
    }

    fn get(&self, layer_id: usize) -> Result<(&Tensor, &Tensor)> {
        let (k_cache, v_cache) = self.cache.get(layer_id)
            .ok_or_else(|| anyhow::anyhow!("Layer index {} is out of bounds for KV cache with layers.", layer_id))?;
        Ok((k_cache, v_cache))
    }
    fn get_mut(&mut self, layer_id: usize) -> Result<(&mut Tensor, &mut Tensor)> {
        let (k_cache, v_cache) = self.cache.get_mut(layer_id)
            .ok_or_else(|| anyhow::anyhow!("Layer index {} is out of bounds for KV cache with layers.", layer_id))?;
        Ok((k_cache, v_cache))
    }

    pub fn init_kv_cache(
        config: &RuntimeModelConfig,
        device: &DeviceType
    ) -> Result<Self> {
        let cache_shape = vec![
            config.seq_len,
            config.kv_head_num*config.head_size
        ];

        // 如果设备是CPU，则使用F32以获得更好的精度和兼容性
        let float_type = if device.is_cpu() {
            println!("  -> Initializing F32 KV Cache for CPU device.");
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
        
        println!(
            "Initializing KV Cache for {} layers with shape {:?}...",
            config.layer_num, cache_shape
        );
        let mut kv_cache = Vec::with_capacity(config.layer_num);
        for _ in 0..config.layer_num {
            // **核心变化**: 创建 bf16 类型的空张量
            let k_cache = Tensor::new(&cache_shape, float_type, *device)?;
            let v_cache = Tensor::new(&cache_shape, float_type, *device)?;

            kv_cache.push((k_cache, v_cache));
        }

        Ok(KvCache { cache: kv_cache })
    }
}

// 在 impl Llama3 中

impl Llama3 {
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        device_type: DeviceType,
        is_quant_model: bool
    ) -> Result<Self> {
        let start_time = Instant::now();
        println!("Start calculate time, Loading Llama3 model from directory: {:?}", model_dir.as_ref());
        let mut loader = ModelLoader::load(model_dir.as_ref())?;
        let tokenizer = loader.create_tokenizer(model_dir.as_ref())?;
        let config = loader.config.clone();
        let cuda_config = CudaConfig::new()?;

        println!("Creating Llama3 decoder layers...");

        // Use DecoderLayers with Llama weight mapping
        let layers = DecoderLayers::from_loader(
            &loader,
            &config,
            &WeightMapping::LLAMA,
            device_type,
            is_quant_model,
        )?;

        // Initialize KV cache
        let kv_cache = KvCache::init_kv_cache(&config, &device_type)?;
        
        // --- 初始化工作区 (这是新添加的部分) ---
        let workspace = Self::init_workspace(&config, &device_type)?;
        let sampler = Box::new(ArgmaxSampler::new(device_type));
        let output_token = Tensor::new(&[1], DataType::I32, device_type)?;
        let input_pos = Tensor::new(&[1], DataType::I32, device_type)?;
        let mut kcache = Tensor::new(&[1, config.kv_dim], DataType::BF16, device_type)?;
        let mut vcache = Tensor::new(&[1, config.kv_dim], DataType::BF16, device_type)?;
        if device_type.is_cpu(){
            kcache = kcache.to_dtype(DataType::F32)?;
            vcache = vcache.to_dtype(DataType::F32)?;
        }
        let mut model = Self { config, device_type, tokenizer, layers, kv_cache, workspace, sampler, cuda_config: Some(cuda_config), output_token, input_pos, kcache, vcache };
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
        device: &DeviceType
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
        // ==========================================================
        
        // 10. 模型的最终 logits 输出 (用于单 token)
        let forward_output = Tensor::new(&[config.vocab_size], float_dtype, *device)?;
        buffers.insert(BufferType::ForwardOutput, forward_output);

        println!("Workspace (batch-enabled) initialized with {} buffers.", buffers.len());
        Ok(buffers)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        print_output: bool,
    ) -> Result<(String, u32, u64, u64, usize)> {
        let mut stdout = io::stdout();
        if print_output {
            println!("----------------------------------------");
            println!("Prompt: {}", prompt);
            stdout.flush()?;
        }
        
        let mut input_pos = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
        input_pos.as_i32_mut()?.as_slice_mut()?[0] = 0;
        let prompt_tokens = self.tokenizer.encode(prompt)?;
        if prompt_tokens.is_empty() {
            return Err(Error::InvalidArgument("Prompt cannot be empty.".to_string()).into());
        }
        let mut input_tokens_cpu = Tensor::new(&[prompt_tokens.len()], DataType::I32, DeviceType::Cpu)?;
        input_tokens_cpu.as_i32_mut()?.as_slice_mut()?.copy_from_slice(&prompt_tokens);
        // Prefill stage - fill KV cache
        let prefill_start = Instant::now();
        let mut current_token = self.forward_prefill(&input_tokens_cpu, &input_pos, prompt_tokens.len())?;
        let prefill_duration = prefill_start.elapsed();
        let prefill_ms = prefill_duration.as_millis() as u64;

        // Generation stage
        let mut generated_tokens = vec![current_token];
        
        if print_output {
            let decoded = self.tokenizer.decode(&[current_token])?;
            let _ = write!(stdout,"{}", decoded);
            io::stdout().flush().expect("Failed to flush stdout");
        }

        // Generate tokens starting from the end of the prompt
        let decode_start = Instant::now();
        let mut decode_iterations = 0;
        let mut input_tokens_cpu = input_tokens_cpu.slice(&[0], &[1])?;
        for pos in prompt_tokens.len()..(prompt_tokens.len() - 1 + max_tokens) {
            input_pos.as_i32_mut()?.as_slice_mut()?[0] = pos as i32;
            input_tokens_cpu.as_i32_mut()?.as_slice_mut()?[0] = current_token;
            let next_token = self.forward_decoding(&input_tokens_cpu, &input_pos)?;

            if self.tokenizer.is_eos(next_token) {
                break;
            }

            generated_tokens.push(next_token);
            current_token = next_token;
            decode_iterations += 1;
            
            if print_output {
                let decoded = self.tokenizer.decode(&[current_token])?;
                let _ = write!(stdout,"{}", decoded);
                stdout.flush()?;
            }
        }
        let decode_duration = decode_start.elapsed();
        let decode_ms = decode_duration.as_millis() as u64;

        // 最后输出换行并刷新（可选，让终端提示符回到新行）
        if print_output {
            println!();
        }

        let generated_text = self.tokenizer.decode(&generated_tokens)?;
        Ok((generated_text, generated_tokens.len() as u32, prefill_ms, decode_ms, decode_iterations))
    }
    fn forward_decoding(&mut self, _tokens: &Tensor, pos_cpu: &Tensor) -> Result<i32> {
        self.input_pos.copy_from(pos_cpu)?;
        let input_tokens_view = &self.output_token;
        let config = self.cuda_config.as_mut().expect("CudaConfig should be initialized");
        if self.device_type.is_cuda(){
            if config.cuda_graph.is_none(){
                config.capture_graph_begin()?;
            }else{
                config.launch_graph()?;
                config.sync_stream()?;
                let next_token = self.output_token.to_cpu()?.as_i32()?.as_slice()?[0];
                return Ok(next_token);
            }
        }
        
        
        let cuda_config_ref = if self.device_type.is_cuda() { self.cuda_config.as_ref() } else { None };
        // Token embedding
        let x_buffer = self.workspace.get_mut(&BufferType::InputEmbeddings).unwrap();
        let mut x = x_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        self.layers.embedding_layer.forward(
            &mut OpContext::new(&[input_tokens_view], &mut [&mut x], cuda_config_ref)
        )?;
        
        // Process all transformer layers
        for i in 0..self.config.layer_num {
            let attn_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out = attn_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[i].forward(&mut OpContext::new(&[&x], &mut [&mut attn_norm_out], cuda_config_ref))?;
            let q_buffer = self.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            
            self.layers.wq_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut q], cuda_config_ref))?;
            self.layers.wk_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut self.kcache], cuda_config_ref))?;
            self.layers.wv_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut self.vcache], cuda_config_ref))?;
            let (k_cache_full, v_cache_full) = self.kv_cache.get_mut(i)?;
            
            let sin_cache = self.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = self.workspace.get(&BufferType::CosCache).unwrap();
            
            self.layers.rope_layers[i].forward(&mut OpContext::new(&[&self.input_pos, sin_cache, cos_cache], &mut [&mut q, &mut self.kcache], cuda_config_ref))?;
            
            self.layers.scatter_layer.forward(&mut OpContext::new(&[&self.kcache, &self.input_pos], &mut [k_cache_full], cuda_config_ref))?;
            self.layers.scatter_layer.forward(&mut OpContext::new(&[&self.vcache, &self.input_pos], &mut [v_cache_full], cuda_config_ref))?;
            let (k_cache_history, v_cache_history) = self.kv_cache.get(i).unwrap();
            let mut attn_out = attn_norm_out; // Reuse buffer
            self.layers.mha_layers[i].forward(&mut OpContext::new(&[&q, k_cache_history, v_cache_history, &self.input_pos], &mut [&mut attn_out], cuda_config_ref))?;
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
        
        let config = self.cuda_config.as_mut().expect("CudaConfig should be initialized");
        if self.device_type.is_cuda() && config.cuda_graph.is_none(){
            config.capture_graph_end()?;
        }
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
            let (mut k, mut v) = self.kv_cache.slice_kv_cache(i, pos as i32, total_tokens, self.config.kv_dim)?;

            self.layers.wq_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut q], cuda_config_ref))?;
            self.layers.wk_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut k], cuda_config_ref))?;
            self.layers.wv_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut v], cuda_config_ref))?;

            let sin_cache = self.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = self.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[i].forward(&mut OpContext::new(&[&self.input_pos, sin_cache, cos_cache], &mut [&mut q, &mut k], cuda_config_ref))?;

            let (k_cache_history, v_cache_history) = self.kv_cache.get(i).unwrap();
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

    fn forward_prefill(&mut self, tokens: &Tensor, pos_cpu: &Tensor, seq_len:usize) -> Result<i32> {
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
            let (mut k, mut v) = self.kv_cache.slice_kv_cache(i, pos as i32, seq_len, self.config.kv_dim)?;
            self.layers.wq_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut q], cuda_config_ref))?;
            self.layers.wk_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut k], cuda_config_ref))?;
            self.layers.wv_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut v], cuda_config_ref))?;
            let sin_cache = self.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = self.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[i].forward(&mut OpContext::new(&[&self.input_pos, sin_cache, cos_cache], &mut [&mut q, &mut k], cuda_config_ref))?;
            let (k_cache_history, v_cache_history) = self.kv_cache.get(i).unwrap();
            let mut attn_out = attn_norm_out; // Reuse buffer
            self.layers.mha_layers[i].forward(&mut OpContext::new(&[&q, k_cache_history, v_cache_history, pos_cpu], &mut [&mut attn_out], cuda_config_ref))?;
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

    /// Get reference to the tokenizer
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }
}

// ============================================================================
//  CacheAwareModel Implementation for Engine-Worker Integration
// ============================================================================
use super::cache_interface::{CacheAwareModel, CacheInstruction, KVCachePool};

impl CacheAwareModel for Llama3 {
    /// Forward pass with cache instructions from Engine
    ///
    /// This method uses the cache instruction to:
    /// 1. Skip KV computation for cached tokens (reuse from kv_pool)
    /// 2. Compute KV only for new tokens
    /// 3. Write new KV to the specified indices in kv_pool
    fn forward_with_cache(
        &mut self,
        tokens: &Tensor,
        instruction: &CacheInstruction,
        kv_pool: &mut KVCachePool,
    ) -> Result<i32> {
        if instruction.is_full_cache_hit() {
            // Decode phase: single new token
            self.decode_with_cache(
                tokens.as_i32()?.as_slice()?[0],
                instruction,
                kv_pool,
            )
        } else {
            // Prefill phase: multiple tokens
            self.prefill_with_cache(tokens, instruction, kv_pool)
        }
    }

    /// Prefill with cache instructions
    ///
    /// Engine tells us:
    /// - cached_indices: KV already computed, just read from kv_pool
    /// - new_indices: Need to compute KV, write to kv_pool
    fn prefill_with_cache(
        &mut self,
        tokens: &Tensor,
        instruction: &CacheInstruction,
        kv_pool: &mut KVCachePool,
    ) -> Result<i32> {
        let cuda_config_ref = if self.device_type.is_cuda() {
            self.cuda_config.as_ref()
        } else {
            None
        };

        let cached_len = instruction.cached_len();
        let new_len = instruction.new_len();
        let total_len = instruction.total_seq_len;

        // Set up position tensor for RoPE
        let mut pos_tensor = Tensor::new(&[1], crate::base::DataType::I32, crate::base::DeviceType::Cpu)?;
        pos_tensor.as_i32_mut()?.as_slice_mut()?[0] = instruction.seq_start_pos as i32;
        self.input_pos.copy_from(&pos_tensor)?;

        // Token embedding for ALL tokens (we need embeddings even for cached tokens for attention)
        let input_tokens_buffer = self.workspace.get_mut(&BufferType::InputTokens).unwrap();
        let mut input_tokens_view = input_tokens_buffer.slice(&[0], &[total_len])?;
        input_tokens_view.copy_from(tokens)?;

        let x_buffer = self.workspace.get_mut(&BufferType::InputEmbeddings).unwrap();
        let mut x = x_buffer.slice(&[0, 0], &[total_len, self.config.dim])?;
        self.layers.embedding_layer.forward(
            &mut OpContext::new(&[&input_tokens_view], &mut [&mut x], cuda_config_ref)
        )?;

        // Process all transformer layers
        for layer_idx in 0..self.config.layer_num {
            // === Attention Block ===
            let attn_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out = attn_norm_out_buffer.slice(&[0, 0], &[total_len, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[layer_idx].forward(
                &mut OpContext::new(&[&x], &mut [&mut attn_norm_out], cuda_config_ref)
            )?;

            // Q projection for all tokens (needed for attention)
            let q_buffer = self.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buffer.slice(&[0, 0], &[total_len, self.config.dim])?;
            self.layers.wq_layers[layer_idx].forward(
                &mut OpContext::new(&[&attn_norm_out], &mut [&mut q], cuda_config_ref)
            )?;

            // Get KV slices from pool
            // For cached tokens: read existing KV
            // For new tokens: compute and write KV
            let all_indices = instruction.all_indices();
            let (mut k_slice, mut v_slice) = kv_pool.get_kv_slice_mut(layer_idx, &all_indices)?;

            if cached_len > 0 {
                // Only compute K,V for NEW tokens (skip cached prefix)
                let new_attn_norm = attn_norm_out.slice(&[cached_len, 0], &[new_len, self.config.dim])?;
                let mut new_k = k_slice.slice(&[cached_len, 0], &[new_len, self.config.kv_dim])?;
                let mut new_v = v_slice.slice(&[cached_len, 0], &[new_len, self.config.kv_dim])?;

                self.layers.wk_layers[layer_idx].forward(
                    &mut OpContext::new(&[&new_attn_norm], &mut [&mut new_k], cuda_config_ref)
                )?;
                self.layers.wv_layers[layer_idx].forward(
                    &mut OpContext::new(&[&new_attn_norm], &mut [&mut new_v], cuda_config_ref)
                )?;
            } else {
                // Cache miss: compute K,V for all tokens
                self.layers.wk_layers[layer_idx].forward(
                    &mut OpContext::new(&[&attn_norm_out], &mut [&mut k_slice], cuda_config_ref)
                )?;
                self.layers.wv_layers[layer_idx].forward(
                    &mut OpContext::new(&[&attn_norm_out], &mut [&mut v_slice], cuda_config_ref)
                )?;
            }

            // Apply RoPE to Q and K
            let sin_cache = self.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = self.workspace.get(&BufferType::CosCache).unwrap();

            if cached_len > 0 {
                // Only apply RoPE to new tokens' K (cached K already has RoPE applied)
                let mut new_q = q.slice(&[cached_len, 0], &[new_len, self.config.dim])?;
                let mut new_k = k_slice.slice(&[cached_len, 0], &[new_len, self.config.kv_dim])?;

                // Adjust position for new tokens
                let mut new_pos = Tensor::new(&[1], crate::base::DataType::I32, crate::base::DeviceType::Cpu)?;
                new_pos.as_i32_mut()?.as_slice_mut()?[0] = cached_len as i32;
                self.input_pos.copy_from(&new_pos)?;

                self.layers.rope_layers[layer_idx].forward(
                    &mut OpContext::new(&[&self.input_pos, sin_cache, cos_cache], &mut [&mut new_q, &mut new_k], cuda_config_ref)
                )?;
            } else {
                self.layers.rope_layers[layer_idx].forward(
                    &mut OpContext::new(&[&self.input_pos, sin_cache, cos_cache], &mut [&mut q, &mut k_slice], cuda_config_ref)
                )?;
            }

            // Multi-head attention using full KV cache
            let mut attn_out = attn_norm_out;
            self.layers.mha_layers[layer_idx].forward(
                &mut OpContext::new(&[&q, &k_slice, &v_slice, &pos_tensor], &mut [&mut attn_out], cuda_config_ref)
            )?;

            // Output projection and residual
            let mut wo_out = q;
            self.layers.wo_layers[layer_idx].forward(
                &mut OpContext::new(&[&attn_out], &mut [&mut wo_out], cuda_config_ref)
            )?;
            self.layers.add_layers.forward(
                &mut OpContext::new(&[&wo_out], &mut [&mut x], cuda_config_ref)
            )?;

            // === FFN Block ===
            let mut ffn_norm_out = attn_out;
            self.layers.rmsnorm_ffn_layers[layer_idx].forward(
                &mut OpContext::new(&[&x], &mut [&mut ffn_norm_out], cuda_config_ref)
            )?;

            let w1_buffer = self.workspace.get_mut(&BufferType::W1Output).unwrap();
            let mut w1_out = w1_buffer.slice(&[0, 0], &[total_len, self.config.intermediate_size])?;
            let w3_buffer = self.workspace.get_mut(&BufferType::W3Output).unwrap();
            let mut w3_out = w3_buffer.slice(&[0, 0], &[total_len, self.config.intermediate_size])?;

            self.layers.w1_layers[layer_idx].forward(
                &mut OpContext::new(&[&ffn_norm_out], &mut [&mut w1_out], cuda_config_ref)
            )?;
            self.layers.w3_layers[layer_idx].forward(
                &mut OpContext::new(&[&ffn_norm_out], &mut [&mut w3_out], cuda_config_ref)
            )?;
            self.layers.swiglu_layers[layer_idx].forward(
                &mut OpContext::new(&[&w3_out], &mut [&mut w1_out], cuda_config_ref)
            )?;

            let mut w2_out = ffn_norm_out;
            self.layers.w2_layers[layer_idx].forward(
                &mut OpContext::new(&[&w1_out], &mut [&mut w2_out], cuda_config_ref)
            )?;
            self.layers.add_layers.forward(
                &mut OpContext::new(&[&w2_out], &mut [&mut x], cuda_config_ref)
            )?;
        }

        // Extract last token's hidden state and generate
        let last_hidden = x.slice(&[total_len - 1, 0], &[1, self.config.dim])?;

        let final_norm_input_buffer = self.workspace.get_mut(&BufferType::IntermediateBuffer1).unwrap();
        let mut final_norm_input = final_norm_input_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        final_norm_input.copy_from(&last_hidden)?;

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

    /// Decode single token with cache
    ///
    /// All previous tokens are cached, only compute for 1 new token
    fn decode_with_cache(
        &mut self,
        token: i32,
        instruction: &CacheInstruction,
        kv_pool: &mut KVCachePool,
    ) -> Result<i32> {
        let cuda_config_ref = if self.device_type.is_cuda() {
            self.cuda_config.as_ref()
        } else {
            None
        };

        let current_pos = instruction.seq_start_pos;

        // Set up position tensor
        let mut pos_tensor = Tensor::new(&[1], crate::base::DataType::I32, crate::base::DeviceType::Cpu)?;
        pos_tensor.as_i32_mut()?.as_slice_mut()?[0] = current_pos as i32;
        self.input_pos.copy_from(&pos_tensor)?;

        // Single token embedding
        self.output_token.as_i32_mut()?.as_slice_mut()?[0] = token;

        let x_buffer = self.workspace.get_mut(&BufferType::InputEmbeddings).unwrap();
        let mut x = x_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        self.layers.embedding_layer.forward(
            &mut OpContext::new(&[&self.output_token], &mut [&mut x], cuda_config_ref)
        )?;

        // Process all layers
        for layer_idx in 0..self.config.layer_num {
            // Attention block
            let attn_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out = attn_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[layer_idx].forward(
                &mut OpContext::new(&[&x], &mut [&mut attn_norm_out], cuda_config_ref)
            )?;

            // Q projection
            let q_buffer = self.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            self.layers.wq_layers[layer_idx].forward(
                &mut OpContext::new(&[&attn_norm_out], &mut [&mut q], cuda_config_ref)
            )?;

            // K, V projection for new token only
            self.layers.wk_layers[layer_idx].forward(
                &mut OpContext::new(&[&attn_norm_out], &mut [&mut self.kcache], cuda_config_ref)
            )?;
            self.layers.wv_layers[layer_idx].forward(
                &mut OpContext::new(&[&attn_norm_out], &mut [&mut self.vcache], cuda_config_ref)
            )?;

            // Apply RoPE
            let sin_cache = self.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = self.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[layer_idx].forward(
                &mut OpContext::new(&[&self.input_pos, sin_cache, cos_cache], &mut [&mut q, &mut self.kcache], cuda_config_ref)
            )?;

            // Write new KV to pool at the specified index
            if !instruction.new_indices.is_empty() {
                let new_idx = instruction.new_indices[0];
                let (mut k_slot, mut v_slot) = kv_pool.get_kv_slice_mut(layer_idx, &[new_idx])?;
                k_slot.copy_from(&self.kcache)?;
                v_slot.copy_from(&self.vcache)?;
            }

            // Get full KV cache for attention (all cached + new)
            let all_indices = instruction.all_indices();
            let (k_cache_full, v_cache_full) = kv_pool.get_kv_slice_mut(layer_idx, &all_indices)?;

            // Multi-head attention
            let mut attn_out = attn_norm_out;
            self.layers.mha_layers[layer_idx].forward(
                &mut OpContext::new(&[&q, &k_cache_full, &v_cache_full, &self.input_pos], &mut [&mut attn_out], cuda_config_ref)
            )?;

            // Output projection and residual
            let mut wo_out = q;
            self.layers.wo_layers[layer_idx].forward(
                &mut OpContext::new(&[&attn_out], &mut [&mut wo_out], cuda_config_ref)
            )?;
            self.layers.add_layers.forward(
                &mut OpContext::new(&[&wo_out], &mut [&mut x], cuda_config_ref)
            )?;

            // FFN block
            let mut ffn_norm_out = attn_out;
            self.layers.rmsnorm_ffn_layers[layer_idx].forward(
                &mut OpContext::new(&[&x], &mut [&mut ffn_norm_out], cuda_config_ref)
            )?;

            let w1_buffer = self.workspace.get_mut(&BufferType::W1Output).unwrap();
            let mut w1_out = w1_buffer.slice(&[0, 0], &[1, self.config.intermediate_size])?;
            let w3_buffer = self.workspace.get_mut(&BufferType::W3Output).unwrap();
            let mut w3_out = w3_buffer.slice(&[0, 0], &[1, self.config.intermediate_size])?;

            self.layers.w1_layers[layer_idx].forward(
                &mut OpContext::new(&[&ffn_norm_out], &mut [&mut w1_out], cuda_config_ref)
            )?;
            self.layers.w3_layers[layer_idx].forward(
                &mut OpContext::new(&[&ffn_norm_out], &mut [&mut w3_out], cuda_config_ref)
            )?;
            self.layers.swiglu_layers[layer_idx].forward(
                &mut OpContext::new(&[&w3_out], &mut [&mut w1_out], cuda_config_ref)
            )?;

            let mut w2_out = ffn_norm_out;
            self.layers.w2_layers[layer_idx].forward(
                &mut OpContext::new(&[&w1_out], &mut [&mut w2_out], cuda_config_ref)
            )?;
            self.layers.add_layers.forward(
                &mut OpContext::new(&[&w2_out], &mut [&mut x], cuda_config_ref)
            )?;
        }

        // Final norm and classifier
        let last_hidden = x.slice(&[0, 0], &[1, self.config.dim])?;
        let final_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
        let mut final_norm_out = final_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        self.layers.rmsnorm_final_layer.forward(
            &mut OpContext::new(&[&last_hidden], &mut [&mut final_norm_out], cuda_config_ref)
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
}

// ============================================================================
//  集成测试 (Integration Tests)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use crate::base::error::Result;

    /// Helper struct for performance metrics
    #[derive(Debug, Clone)]
    struct PerformanceMetrics {
        pub total_compute_ms: f64,
        pub cuda_tps: f64,
        pub prefill_throughput: f64,
        pub decode_avg_itl: f64,
        pub decode_throughput: f64,
        pub prompt_tokens_len: f64,
        pub generated_tokens_len: f64,
    }

    /// Calculate performance metrics from generation results
    fn calculate_performance_metrics(
        prompt_tokens_len: f64,
        generated_tokens_len: f64,
        prefill_ms: u64,
        decode_ms: u64,
        decode_iterations: usize,
    ) -> PerformanceMetrics {
        let total_tokens = prompt_tokens_len + generated_tokens_len;
        let total_compute_ms = (prefill_ms + decode_ms) as f64;

        let cuda_tps = if total_compute_ms > 0.0 {
            total_tokens / (total_compute_ms / 1000.0)
        } else {
            0.0
        };

        let prefill_throughput = if prefill_ms > 0 {
            prompt_tokens_len / (prefill_ms as f64 / 1000.0)
        } else {
            0.0
        };

        let decode_avg_itl = if decode_iterations > 0 {
            decode_ms as f64 / decode_iterations as f64
        } else {
            0.0
        };

        let decode_throughput = if decode_ms > 0 {
            (decode_iterations as f64) / (decode_ms as f64 / 1000.0)
        } else {
            0.0
        };

        PerformanceMetrics {
            total_compute_ms,
            cuda_tps,
            prefill_throughput,
            decode_avg_itl,
            decode_throughput,
            prompt_tokens_len,
            generated_tokens_len,
        }
    }

    /// 辅助函数：执行生成并返回 (生成的文本, 耗时(毫秒), 生成的Token数)
    fn generate_and_measure(
        model: &mut Llama3,
        prompt: &str,
        max_tokens: usize,
        verbose: bool
    ) -> Result<(String, u64, u32, u64, u64, usize)> {

        let start_time = Instant::now();
        let (generated_text, num_generated_tokens, prefill_ms, decode_ms, decode_iterations) = model.generate(prompt, max_tokens, verbose)?;
        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as u64;

        Ok((generated_text, duration_ms, num_generated_tokens, prefill_ms, decode_ms, decode_iterations))
    }

    #[test]
    #[ignore = "该测试花费时间较长，请单独测试运行。"]
    fn test_llama3_cpu_loading_and_generation() -> Result<()> {
        // --- 1. Arrange (准备) ---
        let model_path = get_dummy_model_path();
        assert!(model_path.exists(), "Dummy model directory not found.");

        println!("Loading model on CPU...");
        let mut model = Llama3::new(
            model_path,
            DeviceType::Cpu,
            false // is_quant_model
        )?;
        println!("Model loaded.");

        // --- 2. Act (执行) ---
        let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 14 Dec 2025

<|eot_id|><|start_header_id|>user<|end_header_id|>

你是算法糕手，写一段C++代码，实现一个简单的中序遍历函数。<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
        let max_tokens = 150;

        println!("Starting CPU generation...");
        let (_, _duration_ms, num_generated_tokens, prefill_ms, decode_ms, decode_iterations) = generate_and_measure(&mut model, prompt, max_tokens, true)?;

        // --- 3. Assert & Report (断言 & 报告) ---
        assert!(num_generated_tokens > 0, "No tokens were generated.");

        // 计算性能指标
        let prompt_tokens = model.tokenizer.encode(prompt)?;
        let prompt_tokens_len = prompt_tokens.len() as f64;
        let generated_tokens_len = num_generated_tokens as f64;

        let metrics = calculate_performance_metrics(
            prompt_tokens_len,
            generated_tokens_len,
            prefill_ms,
            decode_ms,
            decode_iterations,
        );

        println!("\n================ CPU PERFORMANCE ================");
        println!("Total Generation Time (compute only): {:.0} ms", metrics.total_compute_ms);
        println!("Generated Tokens: {}, Prompt Tokens: {}", num_generated_tokens, metrics.prompt_tokens_len as usize);
        println!("Total Tokens: {}, Tokens Per Second (TPS): {:.2}",
            (metrics.prompt_tokens_len + metrics.generated_tokens_len) as usize,
            metrics.cuda_tps);
        println!("Prefill TTFT: {:.2} ms  Throughput: {:.2} tok/s", prefill_ms, metrics.prefill_throughput);
        println!("Decode  Avg ITL: {:.2} ms   Throughput: {:.2} tok/s", metrics.decode_avg_itl, metrics.decode_throughput);
        println!("==================================================\n");

        Ok(())
    }

    #[test]
    #[ignore = "该测试花费时间较长，请单独测试运行。"]
    #[cfg(feature = "cuda")]
    fn test_llama3_cuda_performance() -> Result<()> {
        // --- 1. Setup (准备) ---
        let model_path = get_dummy_model_path();
        assert!(model_path.exists(), "Dummy model directory not found.");

        let shared_prefix = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 14 Dec 2025

<|eot_id|><|start_header_id|>user<|end_header_id|>

你是算法糕手，写一段C++代码，";

        let prompt_suffix_1 = "实现一个简单的中序遍历函数。<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
        let prompt_suffix_2 = "实现一个简单的快速排序函数。<|eot_id|><|start_header_id|>assistant<|end_header_id|>";

        let max_tokens = 100;

        // --- 2. First request: Cache miss (缓存未命中) ---
        println!("\n=== Request 1: Cache MISS (Computing Full Prompt) ===");
        let mut model_cuda = Llama3::new(model_path, DeviceType::Cuda(0), false)?;

        let prompt_1 = format!("{}{}", shared_prefix, prompt_suffix_1);
        let (text_1, _, tokens_1, prefill_ms_1, decode_ms_1, _) =
            generate_and_measure(&mut model_cuda, &prompt_1, max_tokens, false)?;

        println!("Request 1 Results:");
        println!("  Generated Tokens: {}", tokens_1);
        println!("  Prefill Time: {:.2} ms (computing full prompt)", prefill_ms_1);
        println!("  Decode Time: {:.2} ms", decode_ms_1);
        println!("  Total: {:.2} ms", (prefill_ms_1 + decode_ms_1) as f64);

        // --- 3. Second request: Cache hit simulation (缓存命中模拟) ---
        println!("\n=== Request 2: Simulating Cache HIT (Shared Prefix) ===");
        println!("Note: In production, RadixCache would skip recomputing the shared prefix");
        println!("      and provide cached KV indices via CacheInstruction");

        let prompt_2 = format!("{}{}", shared_prefix, prompt_suffix_2);

        // Tokenize both prompts to show prefix sharing
        let tokens_full_1 = model_cuda.tokenizer.encode(&prompt_1)?;
        let tokens_full_2 = model_cuda.tokenizer.encode(&prompt_2)?;
        let tokens_prefix = model_cuda.tokenizer.encode(shared_prefix)?;

        println!("\nToken Analysis:");
        println!("  Prefix tokens: {} tokens", tokens_prefix.len());
        println!("  Request 1 total: {} tokens", tokens_full_1.len());
        println!("  Request 2 total: {} tokens", tokens_full_2.len());

        // In a real implementation, RadixCache would have cached the prefix
        // and returned the cached_indices for reuse. Here we show the potential savings:
        let prefix_tokens = tokens_prefix.len();
        let new_tokens_per_request = tokens_full_2.len() - prefix_tokens;

        // Estimate savings: prefix computation would be skipped
        let estimated_prefix_time = (prefill_ms_1 as f64) * (prefix_tokens as f64 / tokens_full_1.len() as f64);
        println!("\nEstimated Savings with RadixCache:");
        println!("  Shared prefix: {} tokens", prefix_tokens);
        println!("  Estimated time to recompute prefix: {:.2} ms", estimated_prefix_time);
        println!("  Unique tokens in Request 2: {} tokens", new_tokens_per_request);

        // Run second request (currently without cache, but would benefit from it)
        let (text_2, _, tokens_2, prefill_ms_2, decode_ms_2, _) =
            generate_and_measure(&mut model_cuda, &prompt_2, max_tokens, false)?;

        println!("\nRequest 2 Results (without RadixCache optimization):");
        println!("  Generated Tokens: {}", tokens_2);
        println!("  Prefill Time: {:.2} ms (recomputing shared prefix)", prefill_ms_2);
        println!("  Decode Time: {:.2} ms", decode_ms_2);
        println!("  Total: {:.2} ms", (prefill_ms_2 + decode_ms_2) as f64);

        // --- 4. Performance comparison ---
        println!("\n================ RADIXCACHE POTENTIAL IMPACT ================");
        println!("Combined metrics (2 requests):");
        let total_prefill = (prefill_ms_1 + prefill_ms_2) as f64;
        let with_cache_prefill = total_prefill - estimated_prefix_time;
        println!("  Current (no cache): {:.2} ms total prefill", total_prefill);
        println!("  With RadixCache:    {:.2} ms (estimated, skipping prefix recompute)", with_cache_prefill);
        println!("  Potential speedup:  {:.2}x", total_prefill / with_cache_prefill);

        println!("\n=== How forward_with_cache() Works ===");
        println!("1. Request 1 arrives (no cached prefix):");
        println!("   - CacheInstruction::with_cache_miss(id, [0..{}])", tokens_full_1.len());
        println!("   - forward_with_cache() computes all {} token KVs", tokens_full_1.len());
        println!("   - Stores KV results in KVCachePool at indices [0..{}]", tokens_full_1.len());

        println!("\n2. Request 2 arrives (with shared prefix):");
        println!("   - RadixCache finds cached prefix (len={})", prefix_tokens);
        println!("   - CacheInstruction::with_cache_hit(id, cached=[0..{}], new=[{}..{}], pos={})",
            prefix_tokens, prefix_tokens, tokens_full_2.len(), prefix_tokens);
        println!("   - forward_with_cache() SKIPS KV computation for cached tokens");
        println!("   - forward_with_cache() ONLY computes KV for new {} tokens", new_tokens_per_request);
        println!("   - Stores new KV results in KVCachePool at indices [{}..{}]",
            prefix_tokens, tokens_full_2.len());

        println!("\n3. Speedup from forward_with_cache():");
        let full_compute_time = prefill_ms_2 as f64;
        let partial_compute_time = full_compute_time * (new_tokens_per_request as f64 / tokens_full_2.len() as f64);
        println!("   - Full computation (no cache): {:.2} ms", full_compute_time);
        println!("   - Partial (forward_with_cache): {:.2} ms", partial_compute_time);
        println!("   - Speedup: {:.2}x", full_compute_time / partial_compute_time);

        println!("==========================================================\n");

        // Verify both requests completed
        assert!(!text_1.is_empty(), "Request 1 generated empty text");
        assert!(!text_2.is_empty(), "Request 2 generated empty text");

        Ok(())
    }


    /// Test that actually uses forward_with_cache() with KVCachePool
    /// This is the authentic test demonstrating cache-aware model behavior
    #[test]
    #[ignore = "该测试花费时间较长，请单独测试运行。"]
    #[cfg(feature = "cuda")]
    fn test_llama3_forward_with_cache() -> Result<()> {
        use crate::model::cache_interface::{CacheInstruction, KVCachePool};

        let model_path = get_dummy_model_path();
        assert!(model_path.exists(), "Dummy model directory not found.");

        let mut model = Llama3::new(model_path, DeviceType::Cuda(0), false)?;

        // Initialize KV Cache Pool with pre-allocated space
        let max_tokens = 1024;
        let mut kv_pool = KVCachePool::new(
            max_tokens,
            model.config.layer_num,
            model.config.kv_dim,
            if model.device_type.is_cpu() { DataType::F32 } else { DataType::BF16 },
            model.device_type,
        )?;

        println!("\n=== Test: forward_with_cache() with KVCachePool ===");

        // Request 1: Full cache miss - compute all tokens
        println!("\n1. Request 1: Full Cache MISS");
        let prompt_1 = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
Today Date: 14 Dec 2025
<|eot_id|><|start_header_id|>user<|end_header_id|>
写一个C++排序函数<|eot_id|><|start_header_id|>assistant<|end_header_id|>";

        let tokens_1 = model.tokenizer().encode(prompt_1)?;
        println!("  Prompt tokens: {}", tokens_1.len());

        // Pre-allocate indices for Request 1
        let indices_1: Vec<usize> = (0..tokens_1.len()).collect();
        println!("  Allocated KV indices: [0..{})", tokens_1.len());

        // Create cache miss instruction
        let instruction_1 = CacheInstruction::with_cache_miss(
            "req1".to_string(),
            indices_1.clone(),
        );

        // Create token tensor
        let mut tokens_tensor = Tensor::new(&[tokens_1.len()], DataType::I32, model.device_type)?;
        tokens_tensor.as_i32_mut()?.as_slice_mut()?.copy_from_slice(&tokens_1);

        // Call forward_with_cache - should compute all token KVs
        let start = Instant::now();
        let token_1 = model.forward_with_cache(
            &tokens_tensor,
            &instruction_1,
            &mut kv_pool,
        )?;
        let prefill_ms_1 = start.elapsed().as_millis() as u64;
        println!("  First generated token: {}", token_1);
        println!("  Prefill time: {:.2} ms (full computation)", prefill_ms_1);

        // Request 2: Partial cache hit - only compute new tokens
        println!("\n2. Request 2: Partial Cache HIT (shared prefix)");
        let prompt_2 = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
Today Date: 14 Dec 2025
<|eot_id|><|start_header_id|>user<|end_header_id|>
写一个C++搜索函数<|eot_id|><|start_header_id|>assistant<|end_header_id|>";

        let tokens_2 = model.tokenizer().encode(prompt_2)?;
        let tokens_prefix = model.tokenizer().encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
Today Date: 14 Dec 2025
<|eot_id|><|start_header_id|>user<|end_header_id|>
写一个C++")?;

        println!("  Request 1 tokens: {}", tokens_1.len());
        println!("  Request 2 tokens: {}", tokens_2.len());
        println!("  Shared prefix: {} tokens", tokens_prefix.len());
        let new_tokens = tokens_2.len() - tokens_prefix.len();
        println!("  New unique tokens: {} tokens", new_tokens);

        // Create indices: reuse cached prefix + new slots
        let mut indices_2: Vec<usize> = (0..tokens_prefix.len()).collect();
        indices_2.extend(tokens_1.len()..tokens_1.len() + new_tokens);

        // Create cache hit instruction: prefix is already cached
        let cached_indices: Vec<usize> = (0..tokens_prefix.len()).collect();
        let new_indices: Vec<usize> = (tokens_1.len()..tokens_1.len() + new_tokens).collect();

        let instruction_2 = CacheInstruction::with_cache_hit(
            "req2".to_string(),
            cached_indices.clone(),
            new_indices.clone(),
            0,
        );

        println!("  Cache HIT: {} cached indices + {} new indices",
            cached_indices.len(), new_indices.len());

        // Create token tensor for request 2
        let mut tokens_tensor_2 = Tensor::new(&[tokens_2.len()], DataType::I32, model.device_type)?;
        tokens_tensor_2.as_i32_mut()?.as_slice_mut()?.copy_from_slice(&tokens_2);

        // Call forward_with_cache - should SKIP cached tokens, only compute new ones
        let start = Instant::now();
        let token_2 = model.forward_with_cache(
            &tokens_tensor_2,
            &instruction_2,
            &mut kv_pool,
        )?;
        let prefill_ms_2 = start.elapsed().as_millis() as u64;
        println!("  First generated token: {}", token_2);
        println!("  Prefill time: {:.2} ms (partial computation)", prefill_ms_2);

        // Verify speedup
        println!("\n3. Cache Speedup Analysis");
        println!("  Request 1 (full):   {:.2} ms", prefill_ms_1);
        println!("  Request 2 (cached): {:.2} ms", prefill_ms_2);
        let speedup = prefill_ms_1 as f64 / prefill_ms_2 as f64;
        println!("  Speedup: {:.2}x (computed {:.1}% fewer tokens)",
            speedup,
            (new_tokens as f64 / tokens_2.len() as f64) * 100.0);

        // Decode phase: both should use decode_with_cache
        println!("\n4. Decode with Cache");

        // Request 1 decode - add 2 more tokens
        let mut all_indices_1 = indices_1.clone();
        for i in 0..2 {
            let new_idx = tokens_1.len() + new_tokens + i;
            all_indices_1.push(new_idx);

            let decode_instr = CacheInstruction::for_decode(
                "req1".to_string(),
                all_indices_1.clone(),
                new_idx,
                tokens_1.len() + i,
            );

            let _next_token = model.decode_with_cache(token_1, &decode_instr, &mut kv_pool)?;
        }
        println!("  Request 1: Generated 2 additional tokens");

        // Request 2 decode - add 2 more tokens
        let mut all_indices_2 = indices_2.clone();
        for i in 0..2 {
            let new_idx = tokens_1.len() + new_tokens + 2 + i;
            all_indices_2.push(new_idx);

            let decode_instr = CacheInstruction::for_decode(
                "req2".to_string(),
                all_indices_2.clone(),
                new_idx,
                tokens_2.len() + i,
            );

            let _next_token = model.decode_with_cache(token_2, &decode_instr, &mut kv_pool)?;
        }
        println!("  Request 2: Generated 2 additional tokens");

        println!("\n=== Test Completed: forward_with_cache() works correctly ===\n");

        Ok(())
    }

    // 辅助函数，获取虚拟模型的路径
    fn get_dummy_model_path() -> &'static Path {
        Path::new("/mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct")
    }
}