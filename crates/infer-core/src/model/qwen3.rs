use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;

use super::config::RuntimeModelConfig;
use super::tokenizer::Tokenizer;
use super::ModelLoader;
use crate::base::error::Error::InternalError;
use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::cuda::CudaConfig;
use crate::model::{BufferType, Workspace};
use crate::op::add_inplace::AddInplace;
use crate::op::embedding::Embedding;
use crate::op::flash_gqa::FlashAttnGQA;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::rope::RoPEOp;
use crate::op::sampler::{ArgmaxSampler, Sampler};
use crate::op::scatter::Scatter;
use crate::op::swiglu::SwiGLU;
use crate::op::{Op, OpContext};
use crate::tensor::Tensor;
use std::boxed::Box;
use std::time::Instant;

/// Qwen3Layers 结构体，用于持有模型的所有算子。
/// 相比 Llama3，Qwen3 添加了 QK-norm 支持
pub struct Qwen3Layers {
    // ---- Token Embedding ----
    pub embedding_layer: Embedding,
    pub rmsnorm_final_layer: RMSNorm, // 对应 model.norm.weight
    pub cls_layer: Matmul,            // 对应 lm_head.weight

    // ---- Transformer Blocks (按类型组织) ----
    pub rmsnorm_attn_layers: Vec<RMSNorm>,
    pub rmsnorm_ffn_layers: Vec<RMSNorm>,

    // ---- Qwen3 特定：QK-norm 层 ----
    // Qwen3 支持对 Query 和 Key 进行 RMSNorm，有两种方式：
    // 1. 可选的 Q-norm 和 K-norm 层（如果 qk_norm 特性启用）
    pub qnorm_layers: Option<Vec<RMSNorm>>, // Query normalization
    pub knorm_layers: Option<Vec<RMSNorm>>, // Key normalization

    pub wq_layers: Vec<Matmul>,
    pub wk_layers: Vec<Matmul>,
    pub wv_layers: Vec<Matmul>,
    pub wo_layers: Vec<Matmul>,
    pub mha_layers: Vec<FlashAttnGQA>,
    pub rope_layers: Vec<RoPEOp>,
    pub add_layers: AddInplace,
    pub scatter_layer: Scatter,

    // FFN layers
    // Qwen3 使用 SwiGLU，但某些配置可能使用标准的 MLP
    pub w1_layers: Vec<Matmul>, // gate_proj
    pub w2_layers: Vec<Matmul>, // down_proj
    pub w3_layers: Vec<Matmul>, // up_proj
    pub swiglu_layers: Vec<SwiGLU>,
}

impl Qwen3Layers {
    /// 将所有层和它们的权重移动到指定的 CUDA 设备。
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        // ---- Token Embedding, Final Norm, and Classifier ----
        self.embedding_layer.to_cuda(device_id)?;
        self.rmsnorm_final_layer.to_cuda(device_id)?;
        self.cls_layer.to_cuda(device_id)?;

        // ---- Transformer Blocks (按类型遍历) ----

        // --- RMSNorm Layers ---
        self.rmsnorm_attn_layers
            .iter_mut()
            .try_for_each(|layer| layer.to_cuda(device_id))?;
        self.rmsnorm_ffn_layers
            .iter_mut()
            .try_for_each(|layer| layer.to_cuda(device_id))?;

        // --- Qwen3 特定：QK-norm 层 ---
        if let Some(ref mut qnorm_layers) = self.qnorm_layers {
            qnorm_layers
                .iter_mut()
                .try_for_each(|layer| layer.to_cuda(device_id))?;
        }
        if let Some(ref mut knorm_layers) = self.knorm_layers {
            knorm_layers
                .iter_mut()
                .try_for_each(|layer| layer.to_cuda(device_id))?;
        }

        // --- Attention Matmul Layers ---
        self.wq_layers
            .iter_mut()
            .try_for_each(|layer| layer.to_cuda(device_id))?;
        self.wk_layers
            .iter_mut()
            .try_for_each(|layer| layer.to_cuda(device_id))?;
        self.wv_layers
            .iter_mut()
            .try_for_each(|layer| layer.to_cuda(device_id))?;
        self.wo_layers
            .iter_mut()
            .try_for_each(|layer| layer.to_cuda(device_id))?;

        // --- FFN Matmul Layers ---
        self.w1_layers
            .iter_mut()
            .try_for_each(|layer| layer.to_cuda(device_id))?;
        self.w2_layers
            .iter_mut()
            .try_for_each(|layer| layer.to_cuda(device_id))?;
        self.w3_layers
            .iter_mut()
            .try_for_each(|layer| layer.to_cuda(device_id))?;

        // --- Non-parameterized Layers ---
        self.mha_layers
            .iter_mut()
            .try_for_each(|layer| layer.to_cuda(device_id))?;
        self.rope_layers
            .iter_mut()
            .try_for_each(|layer| layer.to_cuda(device_id))?;
        self.add_layers.to_cuda(device_id)?;
        self.swiglu_layers
            .iter_mut()
            .try_for_each(|layer| layer.to_cuda(device_id))?;

        Ok(())
    }
}

pub struct Qwen3 {
    config: RuntimeModelConfig,
    device_type: DeviceType,
    tokenizer: Box<dyn Tokenizer>,
    /// 核心算子和权重容器
    layers: Qwen3Layers,

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
        let seq_len_dim_idx = 0;
        let max_seq_len = k_cache_full.shape()[seq_len_dim_idx];

        // 3. 边界检查
        if start_pos as usize + len > max_seq_len {
            return Err(anyhow::anyhow!(
                "KV cache slice is out of bounds. Attempted to slice from pos {} with length {}, but max_seq_len is {}.",
                start_pos, len, max_seq_len
            ));
        }

        // 5. 调用 Tensor::slice 方法来创建零拷贝视图
        let k_slice = k_cache_full.slice(&[start_pos as usize, 0], &[len, kv_dim])?;
        let v_slice = v_cache_full.slice(&[start_pos as usize, 0], &[len, kv_dim])?;

        // 6. 返回包含两个视图的元组
        Ok((k_slice, v_slice))
    }

    fn get(&self, layer_id: usize) -> Result<(&Tensor, &Tensor)> {
        let (k_cache, v_cache) = self.cache.get(layer_id).ok_or_else(|| {
            anyhow::anyhow!(
                "Layer index {} is out of bounds for KV cache with layers.",
                layer_id
            )
        })?;
        Ok((k_cache, v_cache))
    }
    fn get_mut(&mut self, layer_id: usize) -> Result<(&mut Tensor, &mut Tensor)> {
        let (k_cache, v_cache) = self.cache.get_mut(layer_id).ok_or_else(|| {
            anyhow::anyhow!(
                "Layer index {} is out of bounds for KV cache with layers.",
                layer_id
            )
        })?;
        Ok((k_cache, v_cache))
    }

    pub fn init_kv_cache(config: &RuntimeModelConfig, device: &DeviceType) -> Result<Self> {
        let cache_shape = vec![config.seq_len, config.kv_head_num * config.head_size];

        // 根据设备类型选择数据类型
        let float_type = if device.is_cpu() {
            DataType::F32
        } else {
            match config.torch_dtype.as_str() {
                "float32" => DataType::F32,
                "bfloat16" => DataType::BF16,
                _ => {
                    return Err(Error::InvalidArgument(format!(
                        "Unsupported torch_dtype: {}",
                        config.torch_dtype
                    ))
                    .into());
                }
            }
        };

        let mut kv_cache = Vec::with_capacity(config.layer_num);
        for _ in 0..config.layer_num {
            let k_cache = Tensor::new(&cache_shape, float_type, *device)?;
            let v_cache = Tensor::new(&cache_shape, float_type, *device)?;

            kv_cache.push((k_cache, v_cache));
        }

        Ok(KvCache { cache: kv_cache })
    }
}

// 在 impl Qwen3 中

impl Qwen3 {
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        device_type: DeviceType,
        is_quant_model: bool,
    ) -> Result<Self> {
        let mut loader = ModelLoader::load(model_dir.as_ref())?;
        let tensor_names: std::collections::HashSet<String> =
            loader.tensor_names().into_iter().collect();
        let tokenizer = loader.create_tokenizer(model_dir.as_ref())?;
        let config = loader.config.clone();
        let cuda_config = CudaConfig::new()?;

        let (
            embedding_layer,
            rmsnorm_final_layer,
            cls_layer,
            rmsnorm_attn_layers,
            rmsnorm_ffn_layers,
            qnorm_layers,
            knorm_layers,
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
            let mut qnorm_layers_opt = Vec::with_capacity(layer_num);
            let mut knorm_layers_opt = Vec::with_capacity(layer_num);
            let mut wq_layers = Vec::with_capacity(layer_num);
            let mut wk_layers = Vec::with_capacity(layer_num);
            let mut wv_layers = Vec::with_capacity(layer_num);
            let mut wo_layers = Vec::with_capacity(layer_num);
            let mut w1_layers = Vec::with_capacity(layer_num);
            let mut w2_layers = Vec::with_capacity(layer_num);
            let mut w3_layers = Vec::with_capacity(layer_num);

            // 标记是否存在 QK-norm 层
            let has_qnorm = tensor_names.iter().any(|name| name.contains("q_norm"));
            let has_knorm = tensor_names.iter().any(|name| name.contains("k_norm"));

            for i in 0..layer_num {
                // ---- Attention Layers ----
                wq_layers.push(Self::load_matmul(
                    &format!("model.layers.{}.self_attn.q_proj.weight", i),
                    &loader,
                    device_type,
                )?);
                wk_layers.push(Self::load_matmul(
                    &format!("model.layers.{}.self_attn.k_proj.weight", i),
                    &loader,
                    device_type,
                )?);
                wv_layers.push(Self::load_matmul(
                    &format!("model.layers.{}.self_attn.v_proj.weight", i),
                    &loader,
                    device_type,
                )?);
                wo_layers.push(Self::load_matmul(
                    &format!("model.layers.{}.self_attn.o_proj.weight", i),
                    &loader,
                    device_type,
                )?);

                // ---- FFN Layers ----
                w1_layers.push(Self::load_matmul(
                    &format!("model.layers.{}.mlp.gate_proj.weight", i),
                    &loader,
                    device_type,
                )?);
                w3_layers.push(Self::load_matmul(
                    &format!("model.layers.{}.mlp.up_proj.weight", i),
                    &loader,
                    device_type,
                )?);
                w2_layers.push(Self::load_matmul(
                    &format!("model.layers.{}.mlp.down_proj.weight", i),
                    &loader,
                    device_type,
                )?);

                // ---- RMSNorm Layers (Pre/Post) ----
                rmsnorm_attn_layers.push(Self::load_rmsnorm(
                    &format!("model.layers.{}.input_layernorm.weight", i),
                    &loader,
                    device_type,
                )?);
                rmsnorm_ffn_layers.push(Self::load_rmsnorm(
                    &format!("model.layers.{}.post_attention_layernorm.weight", i),
                    &loader,
                    device_type,
                )?);

                // ---- Qwen3 特定：QK-norm 层 ----
                if has_qnorm {
                    qnorm_layers_opt.push(Self::load_rmsnorm(
                        &format!("model.layers.{}.self_attn.q_norm.weight", i),
                        &loader,
                        device_type,
                    )?);
                }
                if has_knorm {
                    knorm_layers_opt.push(Self::load_rmsnorm(
                        &format!("model.layers.{}.self_attn.k_norm.weight", i),
                        &loader,
                        device_type,
                    )?);
                }
            }

            let embedding_layer =
                Self::load_embedding("model.embed_tokens.weight", &loader, device_type)?;
            let rmsnorm_final_layer =
                Self::load_rmsnorm("model.norm.weight", &loader, device_type)?;

            // 检查是否有独立的 lm_head 权重
            let cls_layer = if tensor_names.contains("lm_head.weight") {
                Self::load_matmul("lm_head.weight", &loader, device_type)?
            } else {
                Matmul::from(embedding_layer.weight.clone(), None)
            };

            let qnorm_layers_result = if has_qnorm {
                Some(qnorm_layers_opt)
            } else {
                None
            };
            let knorm_layers_result = if has_knorm {
                Some(knorm_layers_opt)
            } else {
                None
            };

            (
                embedding_layer,
                rmsnorm_final_layer,
                cls_layer,
                rmsnorm_attn_layers,
                rmsnorm_ffn_layers,
                qnorm_layers_result,
                knorm_layers_result,
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
            return Err(Error::InvalidArgument(
                "Quantized models are not yet supported.".to_string(),
            )
            .into());
        };

        let layer_num = config.layer_num;
        let mha_layers: Result<Vec<FlashAttnGQA>> = (0..layer_num)
            .map(|_| FlashAttnGQA::new(config.head_num, config.kv_head_num, config.head_size))
            .collect();
        let mha_layers = mha_layers?;
        let rope_layers: Result<Vec<RoPEOp>> = (0..layer_num)
            .map(|_| RoPEOp::new(config.q_dim, config.kv_dim, config.head_size))
            .collect();
        let rope_layers = rope_layers?;
        let add_layers = AddInplace::new();
        let swiglu_layers: Vec<SwiGLU> = (0..layer_num).map(|_| SwiGLU::new()).collect();

        // 验证层数量
        if rmsnorm_attn_layers.len() != layer_num || rmsnorm_ffn_layers.len() != layer_num {
            return Err(
                InternalError("Incorrect number of RMSNorm layers created.".to_string()).into(),
            );
        }

        if wq_layers.len() != layer_num
            || wk_layers.len() != layer_num
            || wv_layers.len() != layer_num
            || wo_layers.len() != layer_num
        {
            return Err(InternalError(
                "Incorrect number of attention Matmul layers created.".to_string(),
            )
            .into());
        }

        if w1_layers.len() != layer_num
            || w2_layers.len() != layer_num
            || w3_layers.len() != layer_num
        {
            return Err(InternalError(
                "Incorrect number of FFN Matmul layers created.".to_string(),
            )
            .into());
        }

        // 验证 QK-norm 层数量
        if let Some(ref qnorm) = qnorm_layers {
            if qnorm.len() != layer_num {
                return Err(InternalError(
                    "Incorrect number of Q-norm layers created.".to_string(),
                )
                .into());
            }
        }
        if let Some(ref knorm) = knorm_layers {
            if knorm.len() != layer_num {
                return Err(InternalError(
                    "Incorrect number of K-norm layers created.".to_string(),
                )
                .into());
            }
        }

        if mha_layers.len() != layer_num
            || rope_layers.len() != layer_num
            || swiglu_layers.len() != layer_num
        {
            return Err(InternalError(
                "Incorrect number of non-parameterized layers created.".to_string(),
            )
            .into());
        }

        // --- 组装 Qwen3Layers ---
        let layers = Qwen3Layers {
            embedding_layer,
            rmsnorm_final_layer,
            cls_layer,
            rmsnorm_attn_layers,
            rmsnorm_ffn_layers,
            qnorm_layers,
            knorm_layers,
            wq_layers,
            wk_layers,
            wv_layers,
            wo_layers,
            mha_layers,
            rope_layers,
            add_layers,
            scatter_layer: Scatter::new(),
            w1_layers,
            w2_layers,
            w3_layers,
            swiglu_layers,
        };

        let kv_cache = KvCache::init_kv_cache(&config, &device_type)?;

        // --- 初始化工作区 ---
        let workspace = Self::init_workspace(&config, &device_type)?;
        let sampler = Box::new(ArgmaxSampler::new(device_type));
        let output_token = Tensor::new(&[1], DataType::I32, device_type)?;
        let input_pos = Tensor::new(&[1], DataType::I32, device_type)?;
        // 临时 k/v cache 的 dtype 需与其他张量保持一致：CPU 用 F32，GPU 用模型配置的 dtype
        let float_dtype = if device_type.is_cpu() {
            DataType::F32
        } else {
            match config.torch_dtype.as_str() {
                "bfloat16" => DataType::BF16,
                "float32" => DataType::F32,
                _ => DataType::BF16,
            }
        };
        let kcache = Tensor::new(&[1, config.kv_dim], float_dtype, device_type)?;
        let vcache = Tensor::new(&[1, config.kv_dim], float_dtype, device_type)?;
        let mut model = Self {
            config,
            device_type,
            tokenizer,
            layers,
            kv_cache,
            workspace,
            sampler,
            cuda_config: Some(cuda_config),
            output_token,
            input_pos,
            kcache,
            vcache,
        };

        Self::calculate_rope_cache(&model.config, &mut model.workspace)?;
        Ok(model)
    }

    fn calculate_rope_cache(config: &RuntimeModelConfig, workspace: &mut Workspace) -> Result<()> {
        let caches = workspace.get_disjoint_mut([&BufferType::SinCache, &BufferType::CosCache]);

        if let [Some(sin_cache), Some(cos_cache)] = caches {
            match sin_cache.device() {
                DeviceType::Cpu => {
                    crate::op::kernels::cpu::rope_sin_cos_cache_calc(
                        config.head_size,
                        config.seq_len,
                        config.rope_theta,
                        sin_cache,
                        cos_cache,
                    )?;
                }
                #[cfg(feature = "cuda")]
                DeviceType::Cuda(_) => {
                    let stream = crate::cuda::config::CudaConfig::new()?;
                    crate::op::kernels::cuda::rope_sin_cos_cache_calc_cuda(
                        config.head_size,
                        config.seq_len,
                        config.rope_theta,
                        sin_cache,
                        cos_cache,
                        Some(&stream),
                    )?;
                }
            }
            Ok(())
        } else {
            Err(
                Error::InternalError("SinCache or CosCache not found in workspace".to_string())
                    .into(),
            )
        }
    }

    /// 预分配前向传播所需的所有中间张量。
    fn init_workspace(config: &RuntimeModelConfig, device: &DeviceType) -> Result<Workspace> {
        let mut buffers = HashMap::new();

        let float_dtype = if device.is_cpu() {
            DataType::F32
        } else {
            match config.torch_dtype.as_str() {
                "float32" => DataType::F32,
                "bfloat16" => DataType::BF16,
                _ => {
                    return Err(Error::InvalidArgument(format!(
                        "Unsupported torch_dtype: {}",
                        config.torch_dtype
                    ))
                    .into());
                }
            }
        };
        let int_dtype = DataType::I32;

        let max_seq_len = config.seq_len;

        // ---- 分配缓冲区 ----

        // 1. 输入张量
        buffers.insert(
            BufferType::InputTokens,
            Tensor::new(&[max_seq_len], int_dtype, *device)?,
        );
        buffers.insert(
            BufferType::InputPos,
            Tensor::new(&[1], int_dtype, DeviceType::Cpu)?,
        );

        // 2. 词嵌入层的输出
        buffers.insert(
            BufferType::InputEmbeddings,
            Tensor::new(&[max_seq_len, config.dim], float_dtype, *device)?,
        );

        // 3. RoPE 缓存
        buffers.insert(
            BufferType::SinCache,
            Tensor::new(&[max_seq_len, config.head_size], float_dtype, *device)?,
        );
        buffers.insert(
            BufferType::CosCache,
            Tensor::new(&[max_seq_len, config.head_size], float_dtype, *device)?,
        );

        // 4. 可复用的缓冲区
        buffers.insert(
            BufferType::RmsOutput,
            Tensor::new(&[max_seq_len, config.dim], float_dtype, *device)?,
        );

        // 4.5 Attention output 缓冲区 (q_dim = num_heads * head_size)
        buffers.insert(
            BufferType::AttnOutput,
            Tensor::new(&[max_seq_len, config.q_dim], float_dtype, *device)?,
        );

        // 5. Query 缓冲区 (q_dim = num_heads * head_size, 可能不等于 dim)
        buffers.insert(
            BufferType::Query,
            Tensor::new(&[max_seq_len, config.q_dim], float_dtype, *device)?,
        );

        // 6. FFN 的输入缓冲区
        buffers.insert(
            BufferType::W1Output,
            Tensor::new(
                &[max_seq_len, config.intermediate_size],
                float_dtype,
                *device,
            )?,
        );
        buffers.insert(
            BufferType::W3Output,
            Tensor::new(
                &[max_seq_len, config.intermediate_size],
                float_dtype,
                *device,
            )?,
        );

        // 7. (AttnScores 在 Flash Attention 下不需要，已移除)

        // 8. 用于前向传播的临时 K 和 V 缓冲区
        buffers.insert(
            BufferType::KeyCache,
            Tensor::new(&[max_seq_len, config.kv_dim], float_dtype, *device)?,
        );
        buffers.insert(
            BufferType::ValueCache,
            Tensor::new(&[max_seq_len, config.kv_dim], float_dtype, *device)?,
        );

        buffers.insert(
            BufferType::IntermediateBuffer1,
            Tensor::new(&[max_seq_len, config.dim], float_dtype, *device)?,
        );

        // QK-norm buffers for per-head normalization
        // Q reshaped: [max_seq_len * head_num, head_size]
        buffers.insert(
            BufferType::QNormBuffer,
            Tensor::new(
                &[max_seq_len * config.head_num, config.head_size],
                float_dtype,
                *device,
            )?,
        );
        // K reshaped: [max_seq_len * kv_head_num, head_size]
        buffers.insert(
            BufferType::KNormBuffer,
            Tensor::new(
                &[max_seq_len * config.kv_head_num, config.head_size],
                float_dtype,
                *device,
            )?,
        );

        // 10. 模型的最终 logits 输出
        let forward_output = Tensor::new(&[config.vocab_size], float_dtype, *device)?;
        buffers.insert(BufferType::ForwardOutput, forward_output);

        Ok(buffers)
    }

    // --- 创建一系列辅助加载函数 ---
    fn load_matmul(name: &str, loader: &ModelLoader, device: DeviceType) -> Result<Matmul> {
        let tensor_view = loader.get_tensor(name)?;
        let weight = Tensor::from_view_on_cpu(&tensor_view)?;

        let weight_converted = if device.is_cpu() && weight.dtype() != DataType::F32 {
            weight.to_dtype(DataType::F32)?
        } else {
            weight
        };

        let weight_on_device = weight_converted.to_device(device)?;

        Ok(Matmul::from(weight_on_device, None))
    }

    fn load_rmsnorm(name: &str, loader: &ModelLoader, device: DeviceType) -> Result<RMSNorm> {
        let tensor_view = loader.get_tensor(name)?;
        let weight = Tensor::from_view_on_cpu(&tensor_view)?;

        let weight_converted = if device.is_cpu() && weight.dtype() != DataType::F32 {
            weight.to_dtype(DataType::F32)?
        } else {
            weight
        };

        let weight_on_device = weight_converted.to_device(device)?;

        Ok(RMSNorm::from(weight_on_device))
    }

    fn load_embedding(name: &str, loader: &ModelLoader, device: DeviceType) -> Result<Embedding> {
        let tensor_view = loader.get_tensor(name)?;
        let weight = Tensor::from_view_on_cpu(&tensor_view)?;

        let weight_converted = if device.is_cpu() && weight.dtype() != DataType::F32 {
            weight.to_dtype(DataType::F32)?
        } else {
            weight
        };

        let weight_on_device = weight_converted.to_device(device)?;

        Ok(Embedding::from(weight_on_device))
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
        let mut input_tokens_cpu =
            Tensor::new(&[prompt_tokens.len()], DataType::I32, DeviceType::Cpu)?;
        input_tokens_cpu
            .as_i32_mut()?
            .as_slice_mut()?
            .copy_from_slice(&prompt_tokens);

        // Prefill stage - fill KV cache
        let prefill_start = Instant::now();
        let mut current_token =
            self.forward_prefill(&input_tokens_cpu, &input_pos, prompt_tokens.len())?;
        let prefill_duration = prefill_start.elapsed();
        let prefill_ms = prefill_duration.as_millis() as u64;

        // Generation stage
        let mut generated_tokens = vec![current_token];
        let mut printed_len = 0usize; // 已打印的字符数

        if print_output {
            let decoded = self.tokenizer.decode(&generated_tokens)?;
            let _ = write!(stdout, "{}", &decoded[printed_len..]);
            printed_len = decoded.len();
            stdout.flush()?;
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
                let decoded = self.tokenizer.decode(&generated_tokens)?;
                if decoded.len() > printed_len {
                    let new_text = &decoded[printed_len..];
                    // 只输出不含 replacement character 的部分
                    if !new_text.contains('\u{FFFD}') {
                        let _ = write!(stdout, "{}", new_text);
                        printed_len = decoded.len();
                        stdout.flush()?;
                    }
                }
            }
        }
        let decode_duration = decode_start.elapsed();
        let decode_ms = decode_duration.as_millis() as u64;

        if print_output {
            println!();
        }

        let generated_text = self.tokenizer.decode(&generated_tokens)?;
        Ok((
            generated_text,
            generated_tokens.len() as u32,
            prefill_ms,
            decode_ms,
            decode_iterations,
        ))
    }

    fn forward_decoding(&mut self, _tokens: &Tensor, pos_cpu: &Tensor) -> Result<i32> {
        self.input_pos.copy_from(pos_cpu)?;

        // CUDA Graph 逻辑仅在 CUDA 设备上启用
        if self.device_type.is_cuda() {
            let config = self
                .cuda_config
                .as_mut()
                .expect("CudaConfig should be initialized");
            if config.cuda_graph.is_none() {
                config.capture_graph_begin()?;
            } else {
                config.launch_graph()?;
                config.sync_stream()?;
                let next_token = self.output_token.to_cpu()?.as_i32()?.as_slice()?[0];
                return Ok(next_token);
            }
        }

        let cuda_config_ref = if self.device_type.is_cuda() {
            self.cuda_config.as_ref()
        } else {
            None
        };
        // Token embedding
        let x_buffer = self
            .workspace
            .get_mut(&BufferType::InputEmbeddings)
            .unwrap();
        let mut x = x_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        self.layers.embedding_layer.forward(&mut OpContext::new(
            &[&self.output_token],
            &mut [&mut x],
            cuda_config_ref,
        ))?;
        // Process all transformer layers
        for i in 0..self.config.layer_num {
            let attn_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out = attn_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[i].forward(&mut OpContext::new(
                &[&x],
                &mut [&mut attn_norm_out],
                cuda_config_ref,
            ))?;

            let q_buffer = self.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buffer.slice(&[0, 0], &[1, self.config.q_dim])?;
            self.layers.wq_layers[i].forward(&mut OpContext::new(
                &[&attn_norm_out],
                &mut [&mut q],
                cuda_config_ref,
            ))?;
            self.layers.wk_layers[i].forward(&mut OpContext::new(
                &[&attn_norm_out],
                &mut [&mut self.kcache],
                cuda_config_ref,
            ))?;
            self.layers.wv_layers[i].forward(&mut OpContext::new(
                &[&attn_norm_out],
                &mut [&mut self.vcache],
                cuda_config_ref,
            ))?;

            // ---- Qwen3 特定：Per-head QK-norm ----
            let mut q = if let Some(ref qnorm_layers) = self.layers.qnorm_layers {
                let q_reshaped = q.reshape(&[self.config.head_num, self.config.head_size])?;
                let qnorm_buffer = self.workspace.get_mut(&BufferType::QNormBuffer).unwrap();
                let mut qnorm_out = qnorm_buffer.slice(&[0, 0], &[self.config.head_num, self.config.head_size])?;
                qnorm_layers[i].forward(&mut OpContext::new(
                    &[&q_reshaped],
                    &mut [&mut qnorm_out],
                    cuda_config_ref,
                ))?;
                qnorm_out.reshape(&[1, self.config.q_dim])?
            } else {
                q
            };

            let mut k_active = if let Some(ref knorm_layers) = self.layers.knorm_layers {
                let k_reshaped = self.kcache.reshape(&[self.config.kv_head_num, self.config.head_size])?;
                let knorm_buffer = self.workspace.get_mut(&BufferType::KNormBuffer).unwrap();
                let mut knorm_out = knorm_buffer.slice(&[0, 0], &[self.config.kv_head_num, self.config.head_size])?;
                knorm_layers[i].forward(&mut OpContext::new(
                    &[&k_reshaped],
                    &mut [&mut knorm_out],
                    cuda_config_ref,
                ))?;
                knorm_out.reshape(&[1, self.config.kv_dim])?
            } else {
                self.kcache.reshape(&[1, self.config.kv_dim])?
            };

            let (k_cache_full, v_cache_full) = self.kv_cache.get_mut(i)?;

            let sin_cache = self.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = self.workspace.get(&BufferType::CosCache).unwrap();

            self.layers.rope_layers[i].forward(&mut OpContext::new(
                &[&self.input_pos, sin_cache, cos_cache],
                &mut [&mut q, &mut k_active],
                cuda_config_ref,
            ))?;
            self.layers.scatter_layer.forward(&mut OpContext::new(
                &[&k_active, &self.input_pos],
                &mut [k_cache_full],
                cuda_config_ref,
            ))?;
            self.layers.scatter_layer.forward(&mut OpContext::new(
                &[&self.vcache, &self.input_pos],
                &mut [v_cache_full],
                cuda_config_ref,
            ))?;
            let (k_cache_history, v_cache_history) = self.kv_cache.get(i).unwrap();
            let attn_out_buffer = self.workspace.get_mut(&BufferType::AttnOutput).unwrap();
            let mut attn_out = attn_out_buffer.slice(&[0, 0], &[1, self.config.q_dim])?;
            self.layers.mha_layers[i].forward(&mut OpContext::new(
                &[&q, k_cache_history, v_cache_history, &self.input_pos],
                &mut [&mut attn_out],
                cuda_config_ref,
            ))?;
            // wo_proj: [1, q_dim] -> [1, dim], 不能复用 q 因为 q_dim != dim
            let wo_buffer = self.workspace.get_mut(&BufferType::IntermediateBuffer1).unwrap();
            let mut wo_out = wo_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            self.layers.wo_layers[i].forward(&mut OpContext::new(
                &[&attn_out],
                &mut [&mut wo_out],
                cuda_config_ref,
            ))?;
            self.layers.add_layers.forward(&mut OpContext::new(
                &[&wo_out],
                &mut [&mut x],
                cuda_config_ref,
            ))?;
            // FFN Block - 用 RmsOutput 缓冲区，它是 [1, dim]
            let ffn_norm_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut ffn_norm_out = ffn_norm_buffer.slice(&[0, 0], &[1, self.config.dim])?;
            self.layers.rmsnorm_ffn_layers[i].forward(&mut OpContext::new(
                &[&x],
                &mut [&mut ffn_norm_out],
                cuda_config_ref,
            ))?;
            let w1_buffer = self.workspace.get_mut(&BufferType::W1Output).unwrap();
            let mut w1_out = w1_buffer.slice(&[0, 0], &[1, self.config.intermediate_size])?;
            let w3_buffer = self.workspace.get_mut(&BufferType::W3Output).unwrap();
            let mut w3_out = w3_buffer.slice(&[0, 0], &[1, self.config.intermediate_size])?;
            self.layers.w1_layers[i].forward(&mut OpContext::new(
                &[&ffn_norm_out],
                &mut [&mut w1_out],
                cuda_config_ref,
            ))?;
            self.layers.w3_layers[i].forward(&mut OpContext::new(
                &[&ffn_norm_out],
                &mut [&mut w3_out],
                cuda_config_ref,
            ))?;
            self.layers.swiglu_layers[i].forward(&mut OpContext::new(
                &[&w3_out],
                &mut [&mut w1_out],
                cuda_config_ref,
            ))?;

            let mut w2_out = ffn_norm_out; // Reuse buffer
            self.layers.w2_layers[i].forward(&mut OpContext::new(
                &[&w1_out],
                &mut [&mut w2_out],
                cuda_config_ref,
            ))?;
            self.layers.add_layers.forward(&mut OpContext::new(
                &[&w2_out],
                &mut [&mut x],
                cuda_config_ref,
            ))?;
        }
        // Extract last token's hidden state
        let last_hidden_state_view = x.slice(&[0, 0], &[1, self.config.dim])?;

        // Final Norm and classifier
        let final_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
        let mut final_norm_out = final_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;

        self.layers
            .rmsnorm_final_layer
            .forward(&mut OpContext::new(
                &[&last_hidden_state_view],
                &mut [&mut final_norm_out],
                cuda_config_ref,
            ))?;

        let logits = self.workspace.get_mut(&BufferType::ForwardOutput).unwrap();

        self.layers.cls_layer.forward(&mut OpContext::new(
            &[&final_norm_out],
            &mut [logits],
            cuda_config_ref,
        ))?;
        let logits_full = self.workspace.get(&BufferType::ForwardOutput).unwrap();
        // 截断到 tokenizer 的实际 vocab_size，避免采样到 padding 区域的 token
        let logits_ref = logits_full.slice(&[0], &[self.config.tokenizer_vocab_size])?;
        self.sampler
            .sample(&logits_ref, &mut self.output_token, cuda_config_ref)?;

        // CUDA Graph: 结束捕获（仅 CUDA 设备）
        if self.device_type.is_cuda() {
            let config = self
                .cuda_config
                .as_mut()
                .expect("CudaConfig should be initialized");
            if config.cuda_graph.is_none() {
                config.capture_graph_end()?;
            }
        }

        let next_token = self.output_token.to_cpu()?.as_i32()?.as_slice()?[0];
        Ok(next_token)
    }

    fn forward_prefill(
        &mut self,
        tokens: &Tensor,
        pos_cpu: &Tensor,
        seq_len: usize,
    ) -> Result<i32> {
        let pos = pos_cpu.as_i32()?.as_slice()?[0] as usize;
        let cuda_config_ref = if self.device_type.is_cuda() {
            self.cuda_config.as_ref()
        } else {
            None
        };
        self.input_pos.copy_from(pos_cpu)?;
        // Prepare batch input tokens
        let input_tokens_buffer = self.workspace.get_mut(&BufferType::InputTokens).unwrap();
        let mut input_tokens_view = input_tokens_buffer.slice(&[0], &[seq_len])?;
        input_tokens_view.copy_from(tokens)?;

        // Token embedding
        let x_buffer = self
            .workspace
            .get_mut(&BufferType::InputEmbeddings)
            .unwrap();
        let mut x = x_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
        self.layers.embedding_layer.forward(&mut OpContext::new(
            &[&input_tokens_view],
            &mut [&mut x],
            cuda_config_ref,
        ))?;
        // Process all transformer layers
        for i in 0..self.config.layer_num {
            // Attention Block
            let attn_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut attn_norm_out =
                attn_norm_out_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
            self.layers.rmsnorm_attn_layers[i].forward(&mut OpContext::new(
                &[&x],
                &mut [&mut attn_norm_out],
                cuda_config_ref,
            ))?;

            let q_buffer = self.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buffer.slice(&[0, 0], &[seq_len, self.config.q_dim])?;
            let (mut k, mut v) =
                self.kv_cache
                    .slice_kv_cache(i, pos as i32, seq_len, self.config.kv_dim)?;
            self.layers.wq_layers[i].forward(&mut OpContext::new(
                &[&attn_norm_out],
                &mut [&mut q],
                cuda_config_ref,
            ))?;
            self.layers.wk_layers[i].forward(&mut OpContext::new(
                &[&attn_norm_out],
                &mut [&mut k],
                cuda_config_ref,
            ))?;
            self.layers.wv_layers[i].forward(&mut OpContext::new(
                &[&attn_norm_out],
                &mut [&mut v],
                cuda_config_ref,
            ))?;

            // ---- Qwen3 Per-head QK-norm ----
            if let Some(ref qnorm_layers) = self.layers.qnorm_layers {
                let q_reshaped = q.reshape(&[seq_len * self.config.head_num, self.config.head_size])?;
                let qnorm_buffer = self.workspace.get_mut(&BufferType::QNormBuffer).unwrap();
                let mut qnorm_out = qnorm_buffer.slice(&[0, 0], &[seq_len * self.config.head_num, self.config.head_size])?;
                qnorm_layers[i].forward(&mut OpContext::new(
                    &[&q_reshaped],
                    &mut [&mut qnorm_out],
                    cuda_config_ref,
                ))?;
                let qnorm_flat = qnorm_out.reshape(&[seq_len, self.config.q_dim])?;
                q.copy_from(&qnorm_flat)?;
            }

            if let Some(ref knorm_layers) = self.layers.knorm_layers {
                let k_reshaped = k.reshape(&[seq_len * self.config.kv_head_num, self.config.head_size])?;
                let knorm_buffer = self.workspace.get_mut(&BufferType::KNormBuffer).unwrap();
                let mut knorm_out = knorm_buffer.slice(&[0, 0], &[seq_len * self.config.kv_head_num, self.config.head_size])?;
                knorm_layers[i].forward(&mut OpContext::new(
                    &[&k_reshaped],
                    &mut [&mut knorm_out],
                    cuda_config_ref,
                ))?;
                let knorm_flat = knorm_out.reshape(&[seq_len, self.config.kv_dim])?;
                k.copy_from(&knorm_flat)?;
            }

            let sin_cache = self.workspace.get(&BufferType::SinCache).unwrap();
            let cos_cache = self.workspace.get(&BufferType::CosCache).unwrap();
            self.layers.rope_layers[i].forward(&mut OpContext::new(
                &[&self.input_pos, sin_cache, cos_cache],
                &mut [&mut q, &mut k],
                cuda_config_ref,
            ))?;
            let (k_cache_history, v_cache_history) = self.kv_cache.get(i).unwrap();
            let attn_out_buffer = self.workspace.get_mut(&BufferType::AttnOutput).unwrap();
            let mut attn_out = attn_out_buffer.slice(&[0, 0], &[seq_len, self.config.q_dim])?;
            self.layers.mha_layers[i].forward(&mut OpContext::new(
                &[&q, k_cache_history, v_cache_history, pos_cpu],
                &mut [&mut attn_out],
                cuda_config_ref,
            ))?;
            // wo_proj: [seq_len, q_dim] -> [seq_len, dim], 不能复用 q 因为 q_dim != dim
            let wo_buffer = self.workspace.get_mut(&BufferType::IntermediateBuffer1).unwrap();
            let mut wo_out = wo_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
            self.layers.wo_layers[i].forward(&mut OpContext::new(
                &[&attn_out],
                &mut [&mut wo_out],
                cuda_config_ref,
            ))?;
            self.layers.add_layers.forward(&mut OpContext::new(
                &[&wo_out],
                &mut [&mut x],
                cuda_config_ref,
            ))?;
            // FFN Block - 用 RmsOutput 缓冲区
            let ffn_norm_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut ffn_norm_out = ffn_norm_buffer.slice(&[0, 0], &[seq_len, self.config.dim])?;
            self.layers.rmsnorm_ffn_layers[i].forward(&mut OpContext::new(
                &[&x],
                &mut [&mut ffn_norm_out],
                cuda_config_ref,
            ))?;
            let w1_buffer = self.workspace.get_mut(&BufferType::W1Output).unwrap();
            let mut w1_out = w1_buffer.slice(&[0, 0], &[seq_len, self.config.intermediate_size])?;
            let w3_buffer = self.workspace.get_mut(&BufferType::W3Output).unwrap();
            let mut w3_out = w3_buffer.slice(&[0, 0], &[seq_len, self.config.intermediate_size])?;
            self.layers.w1_layers[i].forward(&mut OpContext::new(
                &[&ffn_norm_out],
                &mut [&mut w1_out],
                cuda_config_ref,
            ))?;
            self.layers.w3_layers[i].forward(&mut OpContext::new(
                &[&ffn_norm_out],
                &mut [&mut w3_out],
                cuda_config_ref,
            ))?;
            self.layers.swiglu_layers[i].forward(&mut OpContext::new(
                &[&w3_out],
                &mut [&mut w1_out],
                cuda_config_ref,
            ))?;

            let mut w2_out = ffn_norm_out; // Reuse buffer
            self.layers.w2_layers[i].forward(&mut OpContext::new(
                &[&w1_out],
                &mut [&mut w2_out],
                cuda_config_ref,
            ))?;
            self.layers.add_layers.forward(&mut OpContext::new(
                &[&w2_out],
                &mut [&mut x],
                cuda_config_ref,
            ))?;
        }
        // Extract last token's hidden state
        let last_hidden_state_view = x.slice(&[seq_len - 1, 0], &[1, self.config.dim])?;

        // Prepare for final norm and classifier
        let final_norm_input_buffer = self
            .workspace
            .get_mut(&BufferType::IntermediateBuffer1)
            .unwrap();
        let mut final_norm_input = final_norm_input_buffer.slice(&[0, 0], &[1, self.config.dim])?;
        final_norm_input.copy_from(&last_hidden_state_view)?;

        // Final Norm and classifier
        let final_norm_out_buffer = self.workspace.get_mut(&BufferType::RmsOutput).unwrap();
        let mut final_norm_out = final_norm_out_buffer.slice(&[0, 0], &[1, self.config.dim])?;

        self.layers
            .rmsnorm_final_layer
            .forward(&mut OpContext::new(
                &[&final_norm_input],
                &mut [&mut final_norm_out],
                cuda_config_ref,
            ))?;

        let logits = self.workspace.get_mut(&BufferType::ForwardOutput).unwrap();

        self.layers.cls_layer.forward(&mut OpContext::new(
            &[&final_norm_out],
            &mut [logits],
            cuda_config_ref,
        ))?;
        let logits_full = self.workspace.get(&BufferType::ForwardOutput).unwrap();
        // 截断到 tokenizer 的实际 vocab_size，避免采样到 padding 区域的 token
        let logits_ref = logits_full.slice(&[0], &[self.config.tokenizer_vocab_size])?;
        self.sampler
            .sample(&logits_ref, &mut self.output_token, cuda_config_ref)?;
        let next_token = self.output_token.to_cpu()?.as_i32()?.as_slice()?[0];
        Ok(next_token)
    }
}

// 实现 Model 特性
use super::Model;

impl Model for Qwen3 {
    fn init(&mut self, _device_type: DeviceType) -> Result<()> {
        #[cfg(feature = "cuda")]
        if !_device_type.is_cpu() {
            if let DeviceType::Cuda(device_id) = _device_type {
                self.layers.to_cuda(device_id)?;
            }
        }
        Ok(())
    }

    fn forward(&mut self, _input: &Tensor, _pos: &Tensor) -> Result<Tensor> {
        // Placeholder for full forward implementation
        Err(Error::InvalidArgument("forward not yet implemented for Qwen3".to_string()).into())
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    fn is_eos_token(&self, token_id: u32) -> bool {
        self.tokenizer.is_eos(token_id as i32)
    }

    fn slice_kv_cache(
        &self,
        _layer_idx: usize,
        _start_pos: usize,
        _end_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        // Placeholder implementation
        Err(
            Error::InvalidArgument("slice_kv_cache not yet implemented for Qwen3".to_string())
                .into(),
        )
    }
}

// ============================================================================
//  集成测试 (Integration Tests)
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Instant;
    use crate::base::error::Result;

    /// 辅助函数：执行生成并返回性能指标
    fn generate_and_measure(
        model: &mut Qwen3,
        prompt: &str,
        max_tokens: usize,
        verbose: bool,
    ) -> Result<(String, u64, u32, u64, u64, usize)> {
        let start_time = Instant::now();
        let (generated_text, num_generated_tokens, prefill_ms, decode_ms, decode_iterations) =
            model.generate(prompt, max_tokens, verbose)?;
        let duration = start_time.elapsed();
        let duration_ms = duration.as_millis() as u64;

        Ok((
            generated_text,
            duration_ms,
            num_generated_tokens,
            prefill_ms,
            decode_ms,
            decode_iterations,
        ))
    }

    fn get_qwen3_model_path() -> &'static Path {
        Path::new("/apdcephfs_qy2/share_303432435/vinciiliu/models/qwen3-4b-instruct")
    }

    #[test]
    #[ignore = "需要 Qwen3 模型权重，请单独运行。"]
    #[cfg(feature = "cuda")]
    fn test_qwen3_cuda_performance() -> Result<()> {
        let model_path = get_qwen3_model_path();
        assert!(model_path.exists(), "Qwen3 model directory not found at {:?}", model_path);

        let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n写一段C++代码，实现一个简单的中序遍历函数。<|im_end|>\n<|im_start|>assistant\n";
        let max_tokens = 2000;

        println!("\n=== Running Qwen3 on CUDA ===");
        let mut model = Qwen3::new(model_path, DeviceType::Cuda(0), false)?;
        let (_text, _duration_ms, num_tokens, prefill_ms, decode_ms, decode_iterations) =
            generate_and_measure(&mut model, prompt, max_tokens, true)?;

        // 性能指标计算
        let prompt_tokens = model.tokenizer.encode(prompt)?;
        let prompt_tokens_len = prompt_tokens.len() as f64;
        let generated_tokens_len = num_tokens as f64;
        let total_tokens = prompt_tokens_len + generated_tokens_len;
        let total_compute_ms = (prefill_ms + decode_ms) as f64;

        let tps = if total_compute_ms > 0.0 {
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

        println!("\n================ QWEN3 CUDA PERFORMANCE ================");
        println!(
            "Total Time (compute only): {} ms, Total Tokens: {}, TPS: {:.2}",
            total_compute_ms as u64, total_tokens as usize, tps
        );
        println!(
            "Generated Tokens: {}, Prompt Tokens: {}",
            num_tokens, prompt_tokens_len as usize
        );
        println!(
            "Prefill TTFT: {:.2} ms  Throughput: {:.2} tok/s",
            prefill_ms, prefill_throughput
        );
        println!(
            "Decode  Avg ITL: {:.2} ms   Throughput: {:.2} tok/s",
            decode_avg_itl, decode_throughput
        );
        println!("=========================================================\n");
        Ok(())
    }

    #[test]
    #[ignore = "需要 Qwen3 模型权重，请单独运行。"]
    fn test_qwen3_cpu_loading_and_generation() -> Result<()> {
        let model_path = get_qwen3_model_path();
        assert!(model_path.exists(), "Qwen3 model directory not found at {:?}", model_path);

        let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n写一段C++代码，实现一个简单的中序遍历函数。<|im_end|>\n<|im_start|>assistant\n";
        let max_tokens = 150;

        println!("\n=== Running Qwen3 on CPU ===");
        let mut model = Qwen3::new(model_path, DeviceType::Cpu, false)?;
        let (_text, _duration_ms, num_tokens, prefill_ms, decode_ms, decode_iterations) =
            generate_and_measure(&mut model, prompt, max_tokens, true)?;

        let prompt_tokens = model.tokenizer.encode(prompt)?;
        let prompt_tokens_len = prompt_tokens.len() as f64;
        let generated_tokens_len = num_tokens as f64;
        let total_tokens = prompt_tokens_len + generated_tokens_len;
        let total_compute_ms = (prefill_ms + decode_ms) as f64;

        let tps = if total_compute_ms > 0.0 {
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

        println!("\n================ QWEN3 CPU PERFORMANCE ================");
        println!(
            "Total Time (compute only): {} ms, Total Tokens: {}, TPS: {:.2}",
            total_compute_ms as u64, total_tokens as usize, tps
        );
        println!(
            "Generated Tokens: {}, Prompt Tokens: {}",
            num_tokens, prompt_tokens_len as usize
        );
        println!(
            "Prefill TTFT: {:.2} ms  Throughput: {:.2} tok/s",
            prefill_ms, prefill_throughput
        );
        println!(
            "Decode  Avg ITL: {:.2} ms   Throughput: {:.2} tok/s",
            decode_avg_itl, decode_throughput
        );
        println!("=========================================================\n");
        Ok(())
    }

    /// 二分对比 CPU vs GPU 的逐算子输出，定位第一个产生差异的算子。
    /// 只跑第 0 层的 prefill 第一步。
    #[test]
    #[ignore = "需要 Qwen3 模型权重和 CUDA，请单独运行。"]
    #[cfg(feature = "cuda")]
    fn test_qwen3_cpu_vs_gpu_layer0() -> Result<()> {
        let model_path = get_qwen3_model_path();
        assert!(model_path.exists());

        // 短 prompt 方便对比
        let prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";

        println!("=== Loading CPU model ===");
        let mut cpu_model = Qwen3::new(model_path, DeviceType::Cpu, false)?;
        println!("=== Loading GPU model ===");
        let mut gpu_model = Qwen3::new(model_path, DeviceType::Cuda(0), false)?;

        // Tokenize
        let prompt_tokens = cpu_model.tokenizer.encode(prompt)?;
        let seq_len = prompt_tokens.len();
        println!("Prompt tokens: {:?} (len={})", &prompt_tokens[..seq_len.min(20)], seq_len);

        // 准备输入
        let mut input_tokens = Tensor::new(&[seq_len], DataType::I32, DeviceType::Cpu)?;
        input_tokens.as_i32_mut()?.as_slice_mut()?.copy_from_slice(&prompt_tokens);
        let mut input_pos = Tensor::new(&[1], DataType::I32, DeviceType::Cpu)?;
        input_pos.as_i32_mut()?.as_slice_mut()?[0] = 0;

        // ============ CPU forward (手动第0层) ============
        let cpu_cfg = None;
        {
            let m = &mut cpu_model;
            m.input_pos.copy_from(&input_pos)?;
            // embedding
            let input_buf = m.workspace.get_mut(&BufferType::InputTokens).unwrap();
            let mut itv = input_buf.slice(&[0], &[seq_len])?;
            itv.copy_from(&input_tokens)?;

            let x_buf = m.workspace.get_mut(&BufferType::InputEmbeddings).unwrap();
            let mut x = x_buf.slice(&[0, 0], &[seq_len, m.config.dim])?;
            m.layers.embedding_layer.forward(&mut OpContext::new(&[&itv], &mut [&mut x], cpu_cfg))?;
            let cpu_embed = x.as_f32()?.as_slice()?.to_vec();
            println!("[CPU] embedding first 8: {:?}", &cpu_embed[..8]);

            // attn_norm
            let norm_buf = m.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut norm_out = norm_buf.slice(&[0, 0], &[seq_len, m.config.dim])?;
            m.layers.rmsnorm_attn_layers[0].forward(&mut OpContext::new(&[&x], &mut [&mut norm_out], cpu_cfg))?;
            let cpu_norm = norm_out.as_f32()?.as_slice()?.to_vec();
            println!("[CPU] attn_norm first 8: {:?}", &cpu_norm[..8]);

            // wq
            let q_buf = m.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buf.slice(&[0, 0], &[seq_len, m.config.q_dim])?;
            m.layers.wq_layers[0].forward(&mut OpContext::new(&[&norm_out], &mut [&mut q], cpu_cfg))?;
            let cpu_q = q.as_f32()?.as_slice()?.to_vec();
            println!("[CPU] wq first 8: {:?}", &cpu_q[..8]);

            // wk, wv
            let (mut k, mut v) = m.kv_cache.slice_kv_cache(0, 0, seq_len, m.config.kv_dim)?;
            m.layers.wk_layers[0].forward(&mut OpContext::new(&[&norm_out], &mut [&mut k], cpu_cfg))?;
            m.layers.wv_layers[0].forward(&mut OpContext::new(&[&norm_out], &mut [&mut v], cpu_cfg))?;
            let cpu_k = k.as_f32()?.as_slice()?.to_vec();
            println!("[CPU] wk first 8: {:?}", &cpu_k[..8]);

            // qk-norm
            if let Some(ref qnorm) = m.layers.qnorm_layers {
                let q_reshaped = q.reshape(&[seq_len * m.config.head_num, m.config.head_size])?;
                let qn_buf = m.workspace.get_mut(&BufferType::QNormBuffer).unwrap();
                let mut qn_out = qn_buf.slice(&[0, 0], &[seq_len * m.config.head_num, m.config.head_size])?;
                qnorm[0].forward(&mut OpContext::new(&[&q_reshaped], &mut [&mut qn_out], cpu_cfg))?;
                let qn_flat = qn_out.reshape(&[seq_len, m.config.q_dim])?;
                q.copy_from(&qn_flat)?;
                let cpu_qn = q.as_f32()?.as_slice()?.to_vec();
                println!("[CPU] q after qk-norm first 8: {:?}", &cpu_qn[..8]);
            }

            if let Some(ref knorm) = m.layers.knorm_layers {
                let k_reshaped = k.reshape(&[seq_len * m.config.kv_head_num, m.config.head_size])?;
                let kn_buf = m.workspace.get_mut(&BufferType::KNormBuffer).unwrap();
                let mut kn_out = kn_buf.slice(&[0, 0], &[seq_len * m.config.kv_head_num, m.config.head_size])?;
                knorm[0].forward(&mut OpContext::new(&[&k_reshaped], &mut [&mut kn_out], cpu_cfg))?;
                let kn_flat = kn_out.reshape(&[seq_len, m.config.kv_dim])?;
                k.copy_from(&kn_flat)?;
                let cpu_kn = k.as_f32()?.as_slice()?.to_vec();
                println!("[CPU] k after kk-norm first 8: {:?}", &cpu_kn[..8]);
            }

            // RoPE
            let sin = m.workspace.get(&BufferType::SinCache).unwrap();
            let cos = m.workspace.get(&BufferType::CosCache).unwrap();
            m.layers.rope_layers[0].forward(&mut OpContext::new(
                &[&m.input_pos, sin, cos], &mut [&mut q, &mut k], cpu_cfg
            ))?;
            let cpu_q_rope = q.as_f32()?.as_slice()?.to_vec();
            let cpu_k_rope = k.as_f32()?.as_slice()?.to_vec();
            println!("[CPU] q after RoPE first 8: {:?}", &cpu_q_rope[..8]);
            println!("[CPU] k after RoPE first 8: {:?}", &cpu_k_rope[..8]);

            // Attention
            let (k_hist, v_hist) = m.kv_cache.get(0).unwrap();
            let attn_buf = m.workspace.get_mut(&BufferType::AttnOutput).unwrap();
            let mut attn_out = attn_buf.slice(&[0, 0], &[seq_len, m.config.q_dim])?;
            m.layers.mha_layers[0].forward(&mut OpContext::new(
                &[&q, k_hist, v_hist, &input_pos], &mut [&mut attn_out], cpu_cfg
            ))?;
            let cpu_attn = attn_out.as_f32()?.as_slice()?.to_vec();
            println!("[CPU] attn_out first 8: {:?}", &cpu_attn[..8]);
        }

        // ============ GPU forward (手动第0层) ============
        {
            let m = &mut gpu_model;
            let gpu_cfg = m.cuda_config.as_ref();

            let input_tokens_gpu = input_tokens.to_cuda(0)?;
            let input_pos_gpu = input_pos.to_cuda(0)?;
            m.input_pos.copy_from(&input_pos_gpu)?;

            // embedding
            let input_buf = m.workspace.get_mut(&BufferType::InputTokens).unwrap();
            let mut itv = input_buf.slice(&[0], &[seq_len])?;
            itv.copy_from(&input_tokens_gpu)?;

            let x_buf = m.workspace.get_mut(&BufferType::InputEmbeddings).unwrap();
            let mut x = x_buf.slice(&[0, 0], &[seq_len, m.config.dim])?;
            m.layers.embedding_layer.forward(&mut OpContext::new(&[&itv], &mut [&mut x], gpu_cfg))?;
            let gpu_embed: Vec<f32> = {
                let t = x.to_cpu()?;
                t.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect()
            };
            println!("[GPU] embedding first 8: {:?}", &gpu_embed[..8]);

            // attn_norm
            let norm_buf = m.workspace.get_mut(&BufferType::RmsOutput).unwrap();
            let mut norm_out = norm_buf.slice(&[0, 0], &[seq_len, m.config.dim])?;
            m.layers.rmsnorm_attn_layers[0].forward(&mut OpContext::new(&[&x], &mut [&mut norm_out], gpu_cfg))?;
            let gpu_norm: Vec<f32> = {
                let t = norm_out.to_cpu()?;
                t.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect()
            };
            println!("[GPU] attn_norm first 8: {:?}", &gpu_norm[..8]);

            // wq
            let q_buf = m.workspace.get_mut(&BufferType::Query).unwrap();
            let mut q = q_buf.slice(&[0, 0], &[seq_len, m.config.q_dim])?;
            m.layers.wq_layers[0].forward(&mut OpContext::new(&[&norm_out], &mut [&mut q], gpu_cfg))?;
            let gpu_q: Vec<f32> = {
                let t = q.to_cpu()?;
                t.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect()
            };
            println!("[GPU] wq first 8: {:?}", &gpu_q[..8]);

            // wk
            let (mut k, mut v) = m.kv_cache.slice_kv_cache(0, 0, seq_len, m.config.kv_dim)?;
            m.layers.wk_layers[0].forward(&mut OpContext::new(&[&norm_out], &mut [&mut k], gpu_cfg))?;
            let gpu_k: Vec<f32> = {
                let t = k.to_cpu()?;
                t.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect()
            };
            println!("[GPU] wk first 8: {:?}", &gpu_k[..8]);

            // wv
            m.layers.wv_layers[0].forward(&mut OpContext::new(&[&norm_out], &mut [&mut v], gpu_cfg))?;

            // qk-norm
            if let Some(ref qnorm) = m.layers.qnorm_layers {
                let q_reshaped = q.reshape(&[seq_len * m.config.head_num, m.config.head_size])?;
                let qn_buf = m.workspace.get_mut(&BufferType::QNormBuffer).unwrap();
                let mut qn_out = qn_buf.slice(&[0, 0], &[seq_len * m.config.head_num, m.config.head_size])?;
                qnorm[0].forward(&mut OpContext::new(&[&q_reshaped], &mut [&mut qn_out], gpu_cfg))?;
                let qn_flat = qn_out.reshape(&[seq_len, m.config.q_dim])?;
                q.copy_from(&qn_flat)?;
                let gpu_qn: Vec<f32> = {
                    let t = q.to_cpu()?;
                    t.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect()
                };
                println!("[GPU] q after qk-norm first 8: {:?}", &gpu_qn[..8]);
            }

            if let Some(ref knorm) = m.layers.knorm_layers {
                let k_reshaped = k.reshape(&[seq_len * m.config.kv_head_num, m.config.head_size])?;
                let kn_buf = m.workspace.get_mut(&BufferType::KNormBuffer).unwrap();
                let mut kn_out = kn_buf.slice(&[0, 0], &[seq_len * m.config.kv_head_num, m.config.head_size])?;
                knorm[0].forward(&mut OpContext::new(&[&k_reshaped], &mut [&mut kn_out], gpu_cfg))?;
                let kn_flat = kn_out.reshape(&[seq_len, m.config.kv_dim])?;
                k.copy_from(&kn_flat)?;
                let gpu_kn: Vec<f32> = {
                    let t = k.to_cpu()?;
                    t.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect()
                };
                println!("[GPU] k after kk-norm first 8: {:?}", &gpu_kn[..8]);
            }

            // RoPE
            let sin = m.workspace.get(&BufferType::SinCache).unwrap();
            let cos = m.workspace.get(&BufferType::CosCache).unwrap();
            m.layers.rope_layers[0].forward(&mut OpContext::new(
                &[&m.input_pos, sin, cos], &mut [&mut q, &mut k], gpu_cfg
            ))?;
            let gpu_q_rope: Vec<f32> = {
                let t = q.to_cpu()?;
                t.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect()
            };
            let gpu_k_rope: Vec<f32> = {
                let t = k.to_cpu()?;
                t.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect()
            };
            println!("[GPU] q after RoPE first 8: {:?}", &gpu_q_rope[..8]);
            println!("[GPU] k after RoPE first 8: {:?}", &gpu_k_rope[..8]);

            // Attention
            let (k_hist, v_hist) = m.kv_cache.get(0).unwrap();
            let attn_buf = m.workspace.get_mut(&BufferType::AttnOutput).unwrap();
            let mut attn_out = attn_buf.slice(&[0, 0], &[seq_len, m.config.q_dim])?;
            m.layers.mha_layers[0].forward(&mut OpContext::new(
                &[&q, k_hist, v_hist, &input_pos], &mut [&mut attn_out], gpu_cfg
            ))?;
            let gpu_attn: Vec<f32> = {
                let t = attn_out.to_cpu()?;
                t.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect()
            };
            println!("[GPU] attn_out first 8: {:?}", &gpu_attn[..8]);
        }

        println!("\n=== Compare above CPU vs GPU values to find first divergence ===");
        Ok(())
    }
}
