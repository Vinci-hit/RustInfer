use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;

use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
use crate::cuda::CudaConfig;
use crate::op::{Op, OpContext};
use super::config::RuntimeModelConfig;
use super::ModelLoader;
use crate::tensor::Tensor;
use super::tokenizer::Tokenizer;
use crate::base::error::Error::InternalError;
use std::boxed::Box;
use crate::op::add::Add;
use crate::op::embedding::Embedding;
use crate::op::flash_gqa::FlashAttnGQA;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::rope::RoPEOp;
use crate::op::swiglu::SwiGLU;
use crate::model::{BufferType, Workspace};
use crate::op::sampler::{Sampler, ArgmaxSampler};

/// LlamaLayers 结构体，用于持有模型的所有算子。
pub struct LlamaLayers {
    // ---- Token Embedding ----
    pub embedding_layer: Embedding,
    pub rmsnorm_final_layer: RMSNorm, // 对应 model.norm.weight
    pub cls_layer: Matmul, // 对应 lm_head.weight

    // ---- Transformer Blocks (按类型组织) ----
    pub rmsnorm_attn_layers: Vec<RMSNorm>,
    pub rmsnorm_ffn_layers: Vec<RMSNorm>,
    pub wq_layers: Vec<Matmul>,
    pub wk_layers: Vec<Matmul>,
    pub wv_layers: Vec<Matmul>,
    pub wo_layers: Vec<Matmul>,
    pub mha_layers: Vec<FlashAttnGQA>,
    pub rope_layers: Vec<RoPEOp>,
    pub add_layers: Add,
    
    // FFN layers
    pub w1_layers: Vec<Matmul>, // gate_proj
    pub w2_layers: Vec<Matmul>, // down_proj
    pub w3_layers: Vec<Matmul>, // up_proj
    pub swiglu_layers: Vec<SwiGLU>,
}

impl LlamaLayers {
    /// 将所有层和它们的权重移动到指定的 CUDA 设备。
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        // ---- Token Embedding, Final Norm, and Classifier ----
        self.embedding_layer.to_cuda(device_id)?;
        self.rmsnorm_final_layer.to_cuda(device_id)?;
        self.cls_layer.to_cuda(device_id)?;

        // ---- Transformer Blocks (按类型遍历) ----
        
        // --- RMSNorm Layers ---
        // 为了提高可读性，我们可以用 for_each 结合闭包
        self.rmsnorm_attn_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.rmsnorm_ffn_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;

        // --- Attention Matmul Layers ---
        self.wq_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.wk_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.wv_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.wo_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;

        // --- FFN Matmul Layers ---
        self.w1_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.w2_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.w3_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;

        // --- Non-parameterized Layers (如果它们有 to_cuda 实现) ---
        // 这种写法等同于您参考代码中的 for 循环，但更符合 Rust 风格
        self.mha_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.rope_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.add_layers.to_cuda(device_id)?;
        self.swiglu_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        
        println!("All layers successfully moved to CUDA device {}.", device_id);
        Ok(())
    }
}


pub struct Llama3 {
    config: RuntimeModelConfig,
    device_type: DeviceType,
    tokenizer: Box<dyn Tokenizer>,
    
    /// 核心算子和权重容器
    layers: LlamaLayers,

    /// KV Cache
    kv_cache: Vec<(Tensor, Tensor)>,
    /// 工作空间，用于存放中间计算结果，避免频繁分配和释放内存
    workspace: Workspace,
    sampler: Box<dyn Sampler>,
}

// 在 impl Llama3 中

impl Llama3 {
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        device_type: DeviceType,
        is_quant_model: bool
    ) -> Result<Self> {
        println!("Loading model from directory: {:?}", model_dir.as_ref());
        let mut loader = ModelLoader::load(model_dir.as_ref())?;
        let tensor_names: std::collections::HashSet<String> = loader.tensor_names().into_iter().collect();
        let tokenizer = loader.create_tokenizer(model_dir.as_ref())?;
        let config = loader.config.clone();

        println!("Creating model layers...");
        
        let (
            embedding_layer,
            rmsnorm_final_layer,
            cls_layer,
            rmsnorm_attn_layers,
            rmsnorm_ffn_layers,
            wq_layers,
            wk_layers,
            wv_layers,
            wo_layers,
            w1_layers,
            w2_layers,
            w3_layers,
        ) = if !is_quant_model {
            // === 非量化路径 (Normal Path) ===
            let layer_num = config.layer_num;
            let mut rmsnorm_attn_layers = Vec::with_capacity(layer_num);
            let mut rmsnorm_ffn_layers = Vec::with_capacity(layer_num);
            let mut wq_layers = Vec::with_capacity(layer_num);
            let mut wk_layers = Vec::with_capacity(layer_num);
            let mut wv_layers = Vec::with_capacity(layer_num);
            let mut wo_layers = Vec::with_capacity(layer_num);
            let mut w1_layers = Vec::with_capacity(layer_num);
            let mut w2_layers = Vec::with_capacity(layer_num);
            let mut w3_layers = Vec::with_capacity(layer_num);
            
            for i in 0..layer_num {
                wq_layers.push(Self::load_matmul(&format!("model.layers.{}.self_attn.q_proj.weight", i), &loader, device_type)?);
                wk_layers.push(Self::load_matmul(&format!("model.layers.{}.self_attn.k_proj.weight", i), &loader, device_type)?);
                wv_layers.push(Self::load_matmul(&format!("model.layers.{}.self_attn.v_proj.weight", i), &loader, device_type)?);
                wo_layers.push(Self::load_matmul(&format!("model.layers.{}.self_attn.o_proj.weight", i), &loader, device_type)?);
                w1_layers.push(Self::load_matmul(&format!("model.layers.{}.mlp.gate_proj.weight", i), &loader, device_type)?);
                w3_layers.push(Self::load_matmul(&format!("model.layers.{}.mlp.up_proj.weight", i), &loader, device_type)?);
                w2_layers.push(Self::load_matmul(&format!("model.layers.{}.mlp.down_proj.weight", i), &loader, device_type)?);
                rmsnorm_attn_layers.push(Self::load_rmsnorm(&format!("model.layers.{}.input_layernorm.weight", i), &loader, device_type)?);
                rmsnorm_ffn_layers.push(Self::load_rmsnorm(&format!("model.layers.{}.post_attention_layernorm.weight", i), &loader, device_type)?);
            }

            let embedding_layer = Self::load_embedding("model.embed_tokens.weight", &loader, device_type)?;
            let rmsnorm_final_layer = Self::load_rmsnorm("model.norm.weight", &loader, device_type)?;
            // 通常是通过检查 `lm_head.weight` 是否存在来隐式判断的。
            let cls_layer = if tensor_names.contains("lm_head.weight") {
                // --- 情况 B：权重是独立的 ---
                // 如果 `lm_head.weight` 存在，我们就加载它。
                println!("Found independent 'lm_head.weight'. Loading it.");
                Self::load_matmul("lm_head.weight", &loader, device_type)?

            } else {
                // --- 情况 A：权重是共享/绑定的 (Tied Weights) ---
                // 如果 `lm_head.weight` 不存在，我们就复用词嵌入层的权重。
                println!("'lm_head.weight' not found. Reusing token embedding weights (tied weights).");
                
                // `Matmul::from` 接收一个拥有的 Tensor 和一个可选的 bias。
                // `embedding_layer.weight` 是一个 Tensor，它的 `clone()` 是廉价的 Arc clone，
                // 这实现了权重的共享，而不是数据的复制。
                Matmul::from(embedding_layer.weight.clone(), None)
            };

            (
                embedding_layer,
                rmsnorm_final_layer,
                cls_layer,
                rmsnorm_attn_layers,
                rmsnorm_ffn_layers,
                wq_layers,
                wk_layers,
                wv_layers,
                wo_layers,
                w1_layers,
                w2_layers,
                w3_layers,
            )
        } else {
            // === 量化路径 (Quantized Path) ===
            return Err(Error::InvalidArgument("Quantized models are not yet supported.".to_string()).into());
        };


        let layer_num = config.layer_num;
        let mha_layers: Result<Vec<FlashAttnGQA>> = (0..layer_num)
            .map(|_| {
                FlashAttnGQA::new(config.head_num, config.kv_head_num, config.head_size)
            })
            .collect();
        let mha_layers = mha_layers?; 
        let rope_layers: Result<Vec<RoPEOp>> = (0..layer_num)
            .map(|_| RoPEOp::new(config.dim, config.kv_dim, config.head_size))
            .collect();
        let rope_layers = rope_layers?;
        let add_layers:Add = Add::new(); 
        let swiglu_layers:Vec<SwiGLU> = (0..layer_num).map(|_| SwiGLU::new()).collect();

        // 在 Rust 中，许多检查是隐式的。`load_*` 函数返回 Result，如果失败 `?` 会立即返回错误。
        // 但我们可以添加对集合长度的显式检查，这是一种很好的防御性编程。
        if rmsnorm_attn_layers.len() != layer_num || rmsnorm_ffn_layers.len() != layer_num {
             return Err(InternalError("Incorrect number of RMSNorm layers created.".to_string()).into());
        }

        if wq_layers.len() != layer_num || wk_layers.len() != layer_num || 
           wv_layers.len() != layer_num || wo_layers.len() != layer_num {
            return Err(InternalError("Incorrect number of attention Matmul layers created.".to_string()).into());
        }
        
        if w1_layers.len() != layer_num || w2_layers.len() != layer_num || w3_layers.len() != layer_num {
            return Err(InternalError("Incorrect number of FFN Matmul layers created.".to_string()).into());
        }
        // 对于无参数算子，`.collect()` 保证了长度，但检查一下也无妨。
        if mha_layers.len() != layer_num || rope_layers.len() != layer_num ||
           swiglu_layers.len() != layer_num{
            return Err(InternalError("Incorrect number of non-parameterized layers created.".to_string()).into());
        }
        
        println!("All layers created and checked successfully.");
        
        // --- 组装 LlamaLayers ---
        let layers = LlamaLayers {
            embedding_layer,
            rmsnorm_final_layer,
            cls_layer,
            rmsnorm_attn_layers,
            rmsnorm_ffn_layers,
            wq_layers,
            wk_layers,
            wv_layers,
            wo_layers,
            mha_layers,
            rope_layers,
            add_layers,
            w1_layers,
            w2_layers,
            w3_layers,
            swiglu_layers,
        };
        
        let kv_cache = Self::init_kv_cache(&config, &device_type)?;
        
        // --- 初始化工作区 (这是新添加的部分) ---
        let workspace = Self::init_workspace(&config, &device_type)?;
        let sampler = Box::new(ArgmaxSampler::new(device_type));
        let mut model = Self { config, device_type, tokenizer, layers, kv_cache, workspace, sampler };
        println!("Calculating RoPE sin/cos cache...");
        Self::calculate_rope_cache(&model.config, &mut model.workspace)?;
        println!("Model successfully initialized.");
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
        let float_dtype = if *device == DeviceType::Cpu {
            println!("  -> Using F32 for CPU workspace buffers.");
            DataType::F32
        } else {
            println!("  -> Using BF16 for CUDA workspace buffers.");
            DataType::F32
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
            Tensor::new(&[max_seq_len, config.hidden_dim], float_dtype, *device)?,
        );
        buffers.insert(
            BufferType::W3Output,
            Tensor::new(&[max_seq_len, config.hidden_dim], float_dtype, *device)?,
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

        // 11. CPU 上的 logits 副本
        if device.is_cuda() {
            let forward_output_cpu = Tensor::new(&[config.vocab_size], float_dtype, DeviceType::Cpu)?;
            buffers.insert(BufferType::ForwardOutputCpu, forward_output_cpu);
        }

        println!("Workspace (batch-enabled) initialized with {} buffers.", buffers.len());
        Ok(buffers)
    }

    // --- 创建一系列辅助加载函数，提高代码可读性 ---
    fn load_matmul(name: &str, loader: &ModelLoader, device: DeviceType) -> Result<Matmul> {
        println!("Loading Matmul weight: {}", name);
        let tensor_view = loader.get_tensor(name)?;
        
        // 1. 在 CPU 上从 TensorView 加载，保留原始 dtype (通常是 bf16)
        let mut weight = Tensor::from_view_on_cpu(&tensor_view)?;

        // 2. 如果目标设备是 CPU，则将权重转换为 F32
        if  weight.dtype() == DataType::BF16 { // device == DeviceType::Cpu &&
            println!("  -> Converting weight '{}' to F32 for CPU execution.", name);
            weight = weight.to_dtype(DataType::F32)?;
        }
        
        // 3. 将最终确定类型的权重移动到目标设备
        let weight_on_device = weight.to_device(device)?;

        Ok(Matmul::from(weight_on_device, None))
    }

    // ======================= 已修改的函数 =======================
    fn load_rmsnorm(name: &str, loader: &ModelLoader, device: DeviceType) -> Result<RMSNorm> {
        println!("Loading RMSNorm weight: {}", name);
        let tensor_view = loader.get_tensor(name)?;
        
        // 1. 在 CPU 上加载 bf16/f32 权重
        let mut weight = Tensor::from_view_on_cpu(&tensor_view)?;

        // 2. 如果目标是 CPU，确保权重是 F32
        //    RMSNorm 的权重总是需要 f32，无论是在 CPU 还是 GPU (为了精度)
        //    这是一个可以简化的假设，但现在我们先保持一致
        if weight.dtype() == DataType::BF16 { // device == DeviceType::Cpu &&
            println!("  -> Converting RMSNorm weight '{}' to F32 for CPU execution.", name);
            weight = weight.to_dtype(DataType::F32)?;
        }
        
        // 3. 将权重移动到目标设备
        let weight_on_device = weight.to_device(device)?;

        // RMSNorm::from 接收一个最终的、位于正确设备和类型的权重
        Ok(RMSNorm::from(weight_on_device))
    }

    fn load_embedding(name: &str, loader: &ModelLoader, device: DeviceType) -> Result<Embedding> {
        println!("Loading Embedding weight: {}", name);
        let tensor_view = loader.get_tensor(name)?;

        // 1. 在 CPU 上加载 bf16 权重
        let mut weight = Tensor::from_view_on_cpu(&tensor_view)?;

        // 2. **重要**: Embedding 层本身不进行计算，只是查表（内存拷贝）。
        //    所以，如果整个流水线是 bf16 (在 CUDA 上)，它的权重也应该是 bf16。
        //    只有当 CPU 流水线被确定为 f32 时，我们才转换它。
        if weight.dtype() == DataType::BF16 { // device == DeviceType::Cpu &&
            println!("  -> Converting Embedding weight '{}' to F32 for CPU execution.", name);
            weight = weight.to_dtype(DataType::F32)?;
        }
        
        // 3. 将权重移动到目标设备
        let weight_on_device = weight.to_device(device)?;
        
        Ok(Embedding::from(weight_on_device))
    }

    /// 根据运行时配置，初始化 KV Cache。
    /// KV Cache 通常也使用半精度来节省显存。
    fn init_kv_cache(
        config: &RuntimeModelConfig,
        device: &DeviceType
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let cache_shape = vec![
            config.seq_len,
            config.kv_head_num*config.head_size
        ];

        println!(
            "Initializing F16 KV Cache for {} layers with shape {:?}...",
            config.layer_num, cache_shape
        );
        let float_type = if *device == DeviceType::Cpu {
            println!("  -> Using F32 for CPU KV Cache.");
            DataType::F32
        } else {
            println!("  -> Using BF16 for CUDA KV Cache.");
            DataType::F32
        };
        let mut kv_cache = Vec::with_capacity(config.layer_num);
        for _ in 0..config.layer_num {
            // **核心变化**: 创建 bf16 类型的空张量
            let k_cache = Tensor::new(&cache_shape, float_type, *device)?;
            let v_cache = Tensor::new(&cache_shape, float_type, *device)?;

            kv_cache.push((k_cache, v_cache));
        }

        Ok(kv_cache)
    }

    /// 使用当前模型生成文本。
    ///
    /// # 参数
    /// - `prompt`: 输入的提示字符串。
    /// - `max_tokens`: 要生成的最大 token 数量。
    /// - `print_output`: 是否在生成时将 token 流式打印到控制台。
    ///
    /// # 返回
    /// - `Result<String>`: 成功时返回完整的生成文本（不包括提示）。
    /// 使用当前模型生成文本，支持对长提示进行批处理优化。
    pub fn generate(&mut self, prompt: &str, max_tokens: usize, print_output: bool) -> Result<String> {
        // --- 1. Tokenize 输入提示 ---
        let prompt_tokens = self.tokenizer.encode(prompt)?;

        if prompt_tokens.is_empty() {
            return Err(Error::InvalidArgument("Prompt cannot be empty.".to_string()).into());
        }
        
        // --- 2. 提示处理阶段 (使用批处理) ---
        if print_output {
            print!("{}", self.tokenizer.decode(prompt_tokens.as_slice())?);
            io::stdout().flush()?;
        }
        // 这个方法会填充 KV 缓存，并返回整个序列的最后一个。
        let mut current_token = self.forward_batch(&prompt_tokens,0)?;

        // --- 3. 生成阶段 ---
        let mut generated_tokens = vec![current_token];
        if print_output {
            let decoded = self.tokenizer.decode(&[current_token])?;
            print!("{}", decoded);
            io::stdout().flush()?;
        }

        // 生成循环从提示的末尾开始
        for pos in prompt_tokens.len()..(prompt_tokens.len() - 1 + max_tokens) {
            // 后续的 token 生成仍然是逐个进行的
            let next_token = self.forward_batch(&[current_token], pos)?;

            if self.tokenizer.is_eos(next_token) {
                break;
            }

            generated_tokens.push(next_token);
            current_token = next_token;

            if print_output {
                let decoded = self.tokenizer.decode(&[current_token])?;
                print!("{}", decoded);
                io::stdout().flush()?;
            }
        }
        if print_output {
            println!();
        }

        // --- 4. 解码并返回完整结果 ---
        self.tokenizer.decode(&generated_tokens)
    }

    /// 从 KV 缓存中获取指定层和位置的 Key 和 Value 的零拷贝视图。
    ///
    /// 这个函数在自回归生成时（`forward_one_token` 中）被调用，
    /// 用于定位当前 token 应该写入的 KV 缓存位置。
    ///
    /// # 参数
    /// - `layer_idx`: 目标 Transformer 层的索引。
    /// - `token_pos`: 序列中的令牌位置（通常是当前生成的 token 的索引）。
    ///
    /// # 返回
    /// - `Result<(Tensor, Tensor)>`: 成功时返回一个元组，包含 Key 和 Value 的张量视图。
    ///                              这些视图是可变的，因为它们将被写入。
    ///
    /// 注意：这个方法返回的是拥有的 Tensor 视图，而不是引用，
    /// 这使得调用方的生命周期管理更简单。
    pub fn slice_kv_cache(
        &mut self, 
        layer_idx: usize, 
        start_pos: i32, 
        len: usize
    ) -> Result<(Tensor, Tensor)> {
        // 1. 安全地获取指定层的 (Key, Value) 缓存张量的可变引用
        let (k_cache_full, v_cache_full) = self.kv_cache.get_mut(layer_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "Layer index {} is out of bounds for KV cache with {} layers.",
                layer_idx, self.config.layer_num
            )
        })?;

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
        let k_slice = k_cache_full.slice(&[start_pos as usize,0], &[len,self.config.kv_dim])?;
        let v_slice = v_cache_full.slice(&[start_pos as usize,0], &[len,self.config.kv_dim])?;

        // 6. 返回包含两个视图的元组
        Ok((k_slice, v_slice))
    }
    /// 对一个 token 序列进行批处理前向传播。
    ///
    /// 这个方法用于高效处理输入提示。它会填充 KV 缓存，并返回
    /// **序列中最后一个 token** 对应的 logits。
    ///
    /// # 参数
    /// - `tokens`: 输入的 token ID 切片 `&[i32]`。
    ///
    /// # 返回
    /// - `Result<&Tensor>`: 成功时返回一个对工作区中最终 logits 张量的不可变引用。
    /// 对一个 token 序列进行批处理前向传播。
    ///
    /// 这个方法用于高效处理输入提示。它会并行计算所有 token，
    /// 填充 KV 缓存，并返回序列中**最后一个 token**对应的 logits。
    ///
    /// # 参数
    /// - `tokens`: 输入的 token ID 切片 `&[i32]`。
    ///
    /// # 返回
    /// - `Result<&Tensor>`: 成功时返回一个对工作区中最终 logits 张量的不可变引用。
    fn forward_batch(&mut self, tokens: &[i32], pos: usize) -> Result<i32> {
        let seq_len = tokens.len();
        
        let cuda_config = if self.device_type.is_cuda() { Some(CudaConfig::new()?) } else { None };
        let cuda_config_ref = cuda_config.as_ref();

        // --- 1. 准备批处理输入张量 ---
        // a. 获取工作区中的输入 token 缓冲区，并切出所需长度的视图
        let input_tokens_buffer = self.workspace.get_mut(&BufferType::InputTokens).unwrap();
        let mut input_tokens_view = input_tokens_buffer.slice(&[0], &[seq_len])?;
        // b. 在 CPU 上创建临时 token 数据，并拷贝到最终的设备视图中
        let mut input_tokens_cpu = Tensor::new(&[seq_len], DataType::I32, DeviceType::Cpu)?;
        input_tokens_cpu.as_i32_mut()?.as_slice_mut()?.copy_from_slice(tokens);
        input_tokens_view.copy_from(&input_tokens_cpu.to_device(self.device_type)?)?;

        // --- 2. 词嵌入 ---
        // `x` 是当前计算流中的隐藏状态，形状为 [seq_len, dim]
        let x_buffer = self.workspace.get_mut(&BufferType::InputEmbeddings).unwrap();
        let mut x = x_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
        self.layers.embedding_layer.forward(
            &mut OpContext::new(&[&input_tokens_view], &mut [&mut x], cuda_config_ref)
        )?;
        // a. 准备一个 pos 张量，值为 0，表示批处理从位置 0 开始
        let mut pos_tensor = self.workspace.remove(&BufferType::InputPos).unwrap();
        pos_tensor.as_i32_mut()?.as_slice_mut()?[0] = pos as i32; // 表示起始位置
        // --- 3. 遍历所有 Transformer 层 ---
        for i in 0..self.config.layer_num {
            // 残差连接：将当前的 x 拷贝一份作为残差输入
            let residual_buffer = self.workspace.get_mut(&BufferType::IntermediateBuffer1).unwrap();
            let mut residual = residual_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
            residual.copy_from(&x)?;
            

            // ==================== a. Attention Block ====================
            let attn_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out = attn_norm_out_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[i].forward(&mut OpContext::new(&[&residual], &mut [&mut attn_norm_out], cuda_config_ref))?;
            
            let q_buffer = self.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
            let (mut k,mut v) = self.slice_kv_cache(i, pos as i32, seq_len)?;
            
            self.layers.wq_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut q], cuda_config_ref))?;
            self.layers.wk_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut k], cuda_config_ref))?;
            self.layers.wv_layers[i].forward(&mut OpContext::new(&[&attn_norm_out], &mut [&mut v], cuda_config_ref))?;
            
            let sin_cache = self.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = self.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[i].forward(&mut OpContext::new(&[&pos_tensor, sin_cache, cos_cache], &mut [&mut q, &mut k], cuda_config_ref))?;
            let (k_cache_history, v_cache_history) = self.kv_cache.get(i).unwrap();
            let mut attn_out = attn_norm_out; // 复用缓冲区
            self.layers.mha_layers[i].forward(&mut OpContext::new(&[&q, k_cache_history, v_cache_history,&pos_tensor], &mut [&mut attn_out], cuda_config_ref))?;
            let mut wo_out = q; // 复用缓冲区
            self.layers.wo_layers[i].forward(&mut OpContext::new(&[&attn_out], &mut [&mut wo_out], cuda_config_ref))?;
            
            self.layers.add_layers.forward(&mut OpContext::new(&[&residual, &wo_out], &mut [&mut x], cuda_config_ref))?;
            
            // ==================== b. FFN Block ====================
            residual.copy_from(&x)?; // 更新残差
            let mut ffn_norm_out = attn_out; // 复用缓冲区
            self.layers.rmsnorm_ffn_layers[i].forward(&mut OpContext::new(&[&residual], &mut [&mut ffn_norm_out], cuda_config_ref))?;
            
            let w1_buffer = self.workspace.get_mut(&BufferType::W1Output).unwrap();
            let mut w1_out = w1_buffer.slice(&[0, 0], &[seq_len, self.config.hidden_dim])?;
            let w3_buffer = self.workspace.get_mut(&BufferType::W3Output).unwrap();
            let mut w3_out = w3_buffer.slice(&[0, 0], &[seq_len, self.config.hidden_dim])?;

            self.layers.w1_layers[i].forward(&mut OpContext::new(&[&ffn_norm_out], &mut [&mut w1_out], cuda_config_ref))?;
            self.layers.w3_layers[i].forward(&mut OpContext::new(&[&ffn_norm_out], &mut [&mut w3_out], cuda_config_ref))?;
            self.layers.swiglu_layers[i].forward(&mut OpContext::new(&[&w3_out], &mut [&mut w1_out], cuda_config_ref))?;
            
            let mut w2_out = ffn_norm_out; // 复用缓冲区
            self.layers.w2_layers[i].forward(&mut OpContext::new(&[&w1_out], &mut [&mut w2_out], cuda_config_ref))?;

            self.layers.add_layers.forward(&mut OpContext::new(&[&residual, &w2_out], &mut [&mut x], cuda_config_ref))?;
            
            
        }
        self.workspace.insert(BufferType::InputPos, pos_tensor); // 还回去供下次使用

        // --- 4. 提取最后一个 token 的隐藏状态 ---
        let last_hidden_state_view = x.slice(&[seq_len - 1, 0], &[1, self.config.dim])?;
        // 我们需要一个单 token 形状的缓冲区来执行 final norm 和 classifier
        let final_norm_input_buffer = self.workspace.get_mut(&BufferType::IntermediateBuffer1).unwrap();
        let mut final_norm_input = final_norm_input_buffer.slice(&[0,0], &[1,self.config.dim])?;
        final_norm_input.copy_from(&last_hidden_state_view)?;
        
        // --- 5. Final Norm 和分类器 (只对最后一个 token) ---
        let final_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
        let mut final_norm_out = final_norm_out_buffer.slice(&[0,0], &[1,self.config.dim])?;
        
        self.layers.rmsnorm_final_layer.forward(
            &mut OpContext::new(&[&final_norm_input], &mut [&mut final_norm_out], cuda_config_ref)
        )?;
        
        let logits = self.workspace.get_mut(&BufferType::ForwardOutput).unwrap();
        self.layers.cls_layer.forward(
            &mut OpContext::new(&[&final_norm_out], &mut [logits], cuda_config_ref)
        )?;
        
        let logits_ref = self.workspace.get(&BufferType::ForwardOutput).unwrap();
        let next_token = self.sampler.sample(logits_ref)?;
        Ok(next_token)
    }

    /// 从工作区中检索一个对张量的不可变引用。
    ///
    /// 这是在前向传播中为算子获取输入张量的主要方式。
    /// 如果请求的缓冲区不存在，将返回一个错误。
    ///
    /// # 参数
    /// - `buffer_type`: 需要检索的缓冲区的 `BufferType` 枚举键。
    ///
    /// # 返回
    /// - `Result<&Tensor>`: 成功时返回对张量的引用，失败时返回错误。
    pub fn get_buffer(&self, buffer_type: BufferType) -> Result<&Tensor> {
        // HashMap::get 返回一个 Option<&Tensor>
        // 我们使用 ok_or_else 将 None 的情况转换为一个具体的错误。
        // 这种方式比 .unwrap() 或 .expect() 更健壮，因为它允许错误被优雅地传播。
        self.workspace.get(&buffer_type).ok_or_else(|| {
            Error::InternalError(format!(
                "Buffer {:?} not found in workspace.",
                buffer_type
            )).into()
        })
    }

    /// 从工作区中检索一个对张量的可变引用。
    ///
    /// 这主要用于为算子获取输出缓冲区，以便将计算结果写入其中。
    /// 如果请求的缓冲区不存在，将返回一个错误。
    ///
    /// # 参数
    /// - `buffer_type`: 需要检索的缓冲区的 `BufferType` 枚举键。
    ///
    /// # 返回
    /// - `Result<&mut Tensor>`: 成功时返回对张量的可变引用，失败时返回错误。
    pub fn get_buffer_mut(&mut self, buffer_type: BufferType) -> Result<&mut Tensor> {
        // HashMap::get_mut 返回一个 Option<&mut Tensor>
        self.workspace.get_mut(&buffer_type).ok_or_else(|| {
            Error::InternalError(format!(
                "Mutable buffer {:?} not found in workspace.",
                buffer_type
            )).into()
        })
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

    /// 辅助函数：执行生成并返回 (生成的文本, 耗时(毫秒), 生成的Token数)
    fn generate_and_measure(
        model: &mut Llama3,
        prompt: &str,
        max_tokens: usize,
        verbose: bool
    ) -> Result<(String, u64, usize)> {
        // 如果需要捕获输出以计算Token数，这里可以进行设置
        // 假设 model.generate 有一个变体或者可以通过其他方式获取生成的文本
        // 为了简化，我们先专注于计时
        
        let start_time = Instant::now();
        
        // 执行生成，verbose设为false以减少IO干扰
        let generated_text = model.generate(prompt, max_tokens, false)?;
        
        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as u64;
        
        // 简单估算生成的Token数（这取决于你的Tokenizer实现）
        // 这里用一个简单的split_whitespace来估算，实际应用中应该用真实的Tokenizer
        let num_generated_tokens = generated_text.split_whitespace().count();

        if verbose {
            println!("----------------------------------------");
            println!("Prompt: {}", prompt);
            println!("Generated Text: {}", generated_text);
            println!("----------------------------------------");
        }
        
        Ok((generated_text, duration_ms, num_generated_tokens))
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
        let prompt = "Write a short email to a colleague declining a meeting invitation next Monday, giving a polite reason and suggesting an alternative time.";
        let max_tokens = 50;
        
        println!("Starting CPU generation...");
        let (_, duration_ms, num_generated_tokens) = generate_and_measure(&mut model, prompt, max_tokens, true)?;
        
        // --- 3. Assert & Report (断言 & 报告) ---
        assert!(num_generated_tokens > 0, "No tokens were generated.");
        
        let tps = if duration_ms > 0 {
            (num_generated_tokens as f64) / (duration_ms as f64 / 1000.0)
        } else {
            0.0
        };

        println!("\n================ CPU PERFORMANCE ================");
        println!("Total Generation Time: {} ms", duration_ms);
        println!("Generated Tokens: {}", num_generated_tokens);
        println!("Tokens Per Second (TPS): {:.2}", tps);
        println!("==================================================\n");
        
        Ok(())
    }
    
    #[test]
    #[ignore = "该测试花费时间较长，请单独测试运行。"]
    #[cfg(feature = "cuda")]
    fn test_llama3_cuda_vs_cpu_performance() -> Result<()> {
        // 这个测试现在同时验证一致性和性能
        
        let model_path = get_dummy_model_path();
        assert!(model_path.exists(), "Dummy model directory not found.");

        let prompt = "Write a short email to a colleague declining a meeting invitation next Monday, giving a polite reason and suggesting an alternative time.";
        let max_tokens = 50;

        // --- 1. 在 CPU 上运行并获取黄金结果和性能 ---
        println!("=== Step 1: Running on CPU ===");
        let mut model_cpu = Llama3::new(model_path, DeviceType::Cpu, false)?;
        let (cpu_generated_text, cpu_duration_ms, cpu_num_tokens) = generate_and_measure(&mut model_cpu, prompt, max_tokens, false)?;
        
        let cpu_tps = if cpu_duration_ms > 0 {
            (cpu_num_tokens as f64) / (cpu_duration_ms as f64 / 1000.0)
        } else {
            0.0
        };

        // --- 2. 在 CUDA 上运行并获取结果和性能 ---
        println!("\n=== Step 2: Running on CUDA ===");
        let mut model_cuda = Llama3::new(model_path, DeviceType::Cuda(0), false)?;
        let (cuda_generated_text, cuda_duration_ms, cuda_num_tokens) = generate_and_measure(&mut model_cuda, prompt, max_tokens, false)?;

        let cuda_tps = if cuda_duration_ms > 0 {
            (cuda_num_tokens as f64) / (cuda_duration_ms as f64 / 1000.0)
        } else {
            0.0
        };

        // --- 4. 报告性能 ---
        println!("\n================ PERFORMANCE COMPARISON ================");
        println!("CPU - Total Time: {} ms, Tokens: {}, TPS: {:.2}", cpu_duration_ms, cpu_num_tokens, cpu_tps);
        println!("CUDA - Total Time: {} ms, Tokens: {}, TPS: {:.2}", cuda_duration_ms, cuda_num_tokens, cuda_tps);
        if cpu_duration_ms > 0 {
             println!("CUDA is {:.2}x faster than CPU", cpu_duration_ms as f64 / cuda_duration_ms as f64);
        }
        println!("=========================================================\n");
        
        Ok(())
    }
    #[test]
    #[ignore = "该测试花费时间较长，请单独测试运行。"]
    #[cfg(feature = "cuda")]
    fn test_llama3_cuda_performance() -> Result<()> {
        // 这个测试现在同时验证一致性和性能
        
        let model_path = get_dummy_model_path();
        assert!(model_path.exists(), "Dummy model directory not found.");

        let prompt = "Write a short email to a colleague declining a meeting invitation next Monday, giving a polite reason and suggesting an alternative time.";
        let max_tokens = 50;

        // --- 2. 在 CUDA 上运行并获取结果和性能 ---
        println!("\n=== Step 2: Running on CUDA ===");
        let mut model_cuda = Llama3::new(model_path, DeviceType::Cuda(0), false)?;
        let (cuda_generated_text, cuda_duration_ms, cuda_num_tokens) = generate_and_measure(&mut model_cuda, prompt, max_tokens, false)?;

        let cuda_tps = if cuda_duration_ms > 0 {
            (cuda_num_tokens as f64) / (cuda_duration_ms as f64 / 1000.0)
        } else {
            0.0
        };

        // --- 4. 报告性能 ---
        println!("\n================ PERFORMANCE COMPARISON ================");
        println!("CUDA - Total Time: {} ms, Tokens: {}, TPS: {:.2}", cuda_duration_ms, cuda_num_tokens, cuda_tps);
        println!("=========================================================\n");
        panic!("!");
        Ok(())
    }

    #[test]
    #[ignore = "该测试花费时间较长，请单独测试运行。"]
    #[cfg(feature = "cuda")]
    fn test_llama3_cuda_only() -> Result<()> {
        // 这个测试的核心是确保 CPU 和 GPU 的输出完全一致
        
        let model_path = get_dummy_model_path();
        assert!(model_path.exists(), "Dummy model directory not found.");

        // --- 1. 在 CPU 上运行并获取黄金结果 ---
        let prompt = "Write a short email to a colleague declining a meeting invitation next Monday, giving a polite reason and suggesting an alternative time.";
        let max_tokens = 50;

        // --- 2. 在 CUDA 上运行 ---
        let mut model_cuda = Llama3::new(model_path, DeviceType::Cuda(0), false)?;
        model_cuda.generate(prompt, max_tokens, true)?;

        Ok(())
    }

    // 辅助函数，获取虚拟模型的路径
    fn get_dummy_model_path() -> &'static Path {
        Path::new("/mnt/d/llama3.2_1B_Instruct/Llama-3.2-1B-Instruct")
    }
}