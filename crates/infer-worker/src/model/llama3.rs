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
use crate::model::{BufferType, Workspace, Model};
use crate::op::sampler::{Sampler, ArgmaxSampler};
use super::layers::{DecoderLayers, WeightMapping};
// Import KvCache from the kvcache module
use super::kvcache::KvCache;


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

// KvCache is now imported from super::kvcache::KvCache

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
        // On CPU, convert KV cache to F32 for better precision and compatibility
        
        if device_type.is_cpu() {
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

    /// Get the device type used by this model
    pub fn device(&self) -> DeviceType {
        self.device_type
    }
}

// ============================================================================
//  Model Trait Implementation
// ============================================================================

impl Model for Llama3 {
    fn config(&self) -> &RuntimeModelConfig {
        &self.config
    }

    fn forward(&mut self, input: &Tensor, pos: &Tensor) -> Result<Tensor> {
        // Determine if this is a single token (decoding) or multiple tokens (prefill)
        let seq_len = input.shape()[0];

        if seq_len == 1 {
            // Single token - decoding mode
            let token_id = self.forward_decoding(input, pos)?;
            // Return the token ID as a tensor
            let mut output = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
            output.as_i32_mut()?.as_slice_mut()?[0] = token_id;
            Ok(output)
        } else {
            // Multiple tokens - prefill mode
            let token_id = self.forward_prefill(input, pos, seq_len)?;
            let mut output = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
            output.as_i32_mut()?.as_slice_mut()?[0] = token_id;
            Ok(output)
        }
    }

    fn forward_with_cache(
        &mut self,
        input: &Tensor,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        // Create position tensor from start_pos
        let mut pos = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
        pos.as_i32_mut()?.as_slice_mut()?[0] = start_pos as i32;

        if seq_len == 1 {
            let token_id = self.forward_decoding(input, &pos)?;
            let mut output = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
            output.as_i32_mut()?.as_slice_mut()?[0] = token_id;
            Ok(output)
        } else {
            let token_id = self.forward_prefill(input, &pos, seq_len)?;
            let mut output = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
            output.as_i32_mut()?.as_slice_mut()?[0] = token_id;
            Ok(output)
        }
    }

    fn reset_kv_cache(&mut self) -> Result<()> {
        // Reset is handled by re-initializing - for now just return Ok
        // In a full implementation, you might zero out the cache or
        // reset position tracking
        Ok(())
    }

    fn slice_kv_cache(
        &mut self,
        layer_idx: usize,
        start_pos: usize,
        len: usize,
    ) -> Result<(Tensor, Tensor)> {
        self.kv_cache.slice_kv_cache(layer_idx, start_pos as i32, len, self.config.kv_dim)
    }

    fn device_type(&self) -> DeviceType {
        self.device_type
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
    // 1. 准备工作
    let model_path = get_dummy_model_path();
    assert!(model_path.exists(), "Dummy model directory not found.");

    let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 14 Dec 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n你是算法糕手，写一段C++代码，实现一个简单的中序遍历函数。<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
    let max_tokens = 2000;

    // 2. 仅初始化一次模型
    println!("\n=== Initializing Model on CUDA (Only Once) ===");
    let mut model_cuda = Llama3::new(model_path, DeviceType::Cuda(0), false)?;

    // 预计算 Prompt Tokens
    let prompt_tokens = model_cuda.tokenizer.encode(prompt)?;
    let prompt_tokens_len = prompt_tokens.len() as f64;

    // 3. 循环运行三次测试
    for i in 1..=3 {
        println!("\n>>> Running Test Round {}/3", i);

        // 执行推理并测量
        // 假设 prefill_ms 和 decode_ms 返回的是 u64
        let (_cuda_generated_text, _cuda_duration_ms, cuda_num_tokens, prefill_ms, decode_ms, decode_iterations) = 
            generate_and_measure(&mut model_cuda, prompt, max_tokens, true)?;

        // --- 修正后的计算逻辑：显式转换类型 ---
        let generated_tokens_len = cuda_num_tokens as f64;
        let total_tokens = prompt_tokens_len + generated_tokens_len;
        
        // 将 u64 转换为 f64 进行计算
        let prefill_ms_f = prefill_ms as f64;
        let decode_ms_f = decode_ms as f64;
        let total_compute_ms_f = prefill_ms_f + decode_ms_f;
        
        let cuda_tps = if total_compute_ms_f > 0.0 {
            total_tokens / (total_compute_ms_f / 1000.0)
        } else {
            0.0
        };
        
        let prefill_throughput = if prefill_ms > 0 { // u64 比较用 0
            prompt_tokens_len / (prefill_ms_f / 1000.0)
        } else {
            0.0
        };
        
        let decode_avg_itl = if decode_iterations > 0 {
            decode_ms_f / (decode_iterations as f64)
        } else {
            0.0
        };
        
        let decode_throughput = if decode_ms > 0 { // u64 比较用 0
            (decode_iterations as f64) / (decode_ms_f / 1000.0)
        } else {
            0.0
        };

        // 4. 输出单次成绩
        println!("================ PERFORMANCE ROUND {} ================", i);
        println!("CUDA - Total Time (compute): {} ms, Total Tokens: {}, TPS: {:.2}", total_compute_ms_f as u64, total_tokens as usize, cuda_tps);
        println!("CUDA - Generated Tokens: {}, Prompt Tokens: {}", cuda_num_tokens, prompt_tokens_len as usize);
        println!("CUDA - Prefill TTFT: {:.2} ms  Throughput: {:.2} tok/s", prefill_ms_f, prefill_throughput);
        println!("CUDA - Decode  Avg ITL: {:.2} ms   Throughput: {:.2} tok/s", decode_avg_itl, decode_throughput);
        println!("======================================================\n");
    }

    Ok(())
}
    // 辅助函数，获取虚拟模型的路径
    fn get_dummy_model_path() -> &'static Path {
        Path::new("/mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct")
    }
}