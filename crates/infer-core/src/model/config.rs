// src/model/config.rs

use std::io::{Cursor, Read};

use serde::Deserialize;
use crate::base::error::{Error, Result};

const MAX_SEQ_LEN: usize = 2048; // 根据调整此值

// ======================= 新增的顶层结构体 =======================
/// 这个结构体匹配 config.json 的顶层结构。
/// 它的唯一作用就是提取出我们关心的 "text_config" 部分。
#[derive(Deserialize)]
struct TopLevelConfig {
    text_config: ModelFileConfig,
    // 如果未来需要，也可以在这里添加 vision_config 等字段
}

/// 这个函数会处理嵌套的 "text_config" 结构。
pub fn load_config_from_json<R: Read>(mut json_reader: R) -> Result<ModelFileConfig> {
    // 1. 先将整个 JSON 解析到顶层结构体中
// 步骤1：将 reader 所有内容读取到内存（Vec<u8>）
    let mut json_bytes = Vec::new();
    json_reader.read_to_end(&mut json_bytes)?; // 读取所有字节（消耗原始 reader，但只消耗一次）

    // 步骤2：基于字节数组创建可重复读取的 Cursor（Cursor 实现了 Read，且支持 seek 到起始位置）
    let mut cursor = Cursor::new(json_bytes);

    // 步骤3：第一次尝试解析为 TopLevelConfig
    if let Ok(top_level) = serde_json::from_reader::<_, TopLevelConfig>(&mut cursor) {
        return Ok(top_level.text_config);
    }

    // 步骤4：解析失败，将 cursor 重置到起始位置（关键！）
    cursor.set_position(0);

    // 步骤5：第二次尝试解析为 ModelFileConfig
    if let Ok(file_config) = serde_json::from_reader::<_, ModelFileConfig>(&mut cursor) {
        return Ok(file_config);
    }

    // 步骤6：两次解析都失败
    Err(Error::InvalidArgument("Failed to parse top-level config".into()).into())
    
}

/// 直接从 config.json 文件反序列化的原始模型配置。
/// 字段名必须与 JSON 中的键完全匹配。
#[derive(Debug, Clone, Deserialize)]
pub struct ModelFileConfig {
    // 基础参数
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: i32, // 使用 i32 来处理可能的负值
    
    // 模型元数据
    pub torch_dtype: String,
    
    // 可能存在的、模型特有的参数 (使用 Option)
    pub immediate_dim: Option<usize>,

    /// 显式的 head_dim（Qwen3 等模型使用，可能不等于 hidden_size/num_attention_heads）
    pub head_dim: Option<usize>,

    /// RoPE theta (base frequency)
    pub rope_theta: f64,

    /// RMSNorm epsilon from model config
    pub rms_norm_eps: f32,
}

/// 模型在运行时实际使用的配置，包含所有直接和派生参数。
#[derive(Debug, Clone)] // Clone is useful for passing config around
pub struct RuntimeModelConfig {
    // === 直接参数 ===
    pub dim: usize,
    pub intermediate_size: usize,
    pub layer_num: usize,
    pub head_num: usize,
    pub kv_head_num: usize,
    pub seq_len: usize,
    pub vocab_size: usize,

    // === 派生参数 ===
    pub kv_dim: usize,
    pub kv_mul: usize,
    pub head_size: usize,
    /// Q 投影输出维度 = head_num * head_size（当 head_size != dim/head_num 时不等于 dim）
    pub q_dim: usize,

    // === 标志位和元数据 ===
    pub is_shared_weight: bool,
    pub torch_dtype: String,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// tokenizer 实际的 vocab size（可能小于权重的 vocab_size）
    /// 采样时应只在 [0, tokenizer_vocab_size) 范围内进行 argmax
    pub tokenizer_vocab_size: usize,

    // === 模型特定参数 ===
    pub immediate_dim: Option<usize>,
}

impl RuntimeModelConfig {
    pub fn new(file_config: &ModelFileConfig) -> Result<Self> {
        // 进行参数有效性检查
        if !file_config.hidden_size.is_multiple_of(file_config.num_attention_heads) {
            return Err(Error::InvalidArgument(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                file_config.hidden_size, file_config.num_attention_heads
            )).into());
        }
        if !file_config.num_attention_heads.is_multiple_of(file_config.num_key_value_heads) {
            return Err(Error::InvalidArgument(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                file_config.num_attention_heads, file_config.num_key_value_heads
            )).into());
        }
        
        let dim = file_config.hidden_size;
        let intermediate_size = file_config.intermediate_size;
        let layer_num = file_config.num_hidden_layers;
        let head_num = file_config.num_attention_heads;
        let kv_head_num = file_config.num_key_value_heads;
        let seq_len = MAX_SEQ_LEN;

        
        let head_size = file_config.head_dim.unwrap_or(dim / head_num);
        let q_dim = head_num * head_size;
        let kv_dim = kv_head_num * head_size;
        let kv_mul = head_num / kv_head_num;

        let is_shared_weight = file_config.vocab_size > 0;
        let vocab_size = file_config.vocab_size.unsigned_abs() as usize;

        let immediate_dim = file_config.immediate_dim;


        Ok(Self {
            dim,
            intermediate_size,
            layer_num,
            head_num,
            kv_head_num,
            seq_len,
            vocab_size,
            kv_dim,
            kv_mul,
            head_size,
            q_dim,
            is_shared_weight,
            torch_dtype: file_config.torch_dtype.clone(),
            rope_theta: file_config.rope_theta as f32,
            rms_norm_eps: file_config.rms_norm_eps,
            tokenizer_vocab_size: vocab_size, // 初始值等于 config 的 vocab_size，后续被 tokenizer 覆盖
            immediate_dim,
        })
    }
}
