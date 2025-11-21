// src/model/config.rs

use std::io::{Cursor, Read};

use serde::Deserialize;
use crate::base::error::{Error, Result};

const MAX_SEQ_LEN: usize = 512; // 根据调整此值

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
    // 比如 Qwen3 的 immediate_dim
    #[serde(default)] // 如果 JSON 中没有这个字段，则使用默认值 (None)
    pub immediate_dim: Option<usize>,
}

/// 模型在运行时实际使用的配置，包含所有直接和派生参数。
#[derive(Debug, Clone)] // Clone is useful for passing config around
pub struct RuntimeModelConfig {
    // === 直接参数 ===
    pub dim: usize,
    pub hidden_dim: usize,
    pub layer_num: usize,
    pub head_num: usize,
    pub kv_head_num: usize,
    pub seq_len: usize,
    pub vocab_size: usize,

    // === 派生参数 ===
    pub kv_dim: usize,
    pub kv_mul: usize,
    pub head_size: usize,

    // === 标志位和元数据 ===
    pub is_shared_weight: bool,
    pub torch_dtype: String, // 保留原始数据类型信息

    // === 模型特定参数 ===
    pub immediate_dim: Option<usize>,
}

impl RuntimeModelConfig {
    pub fn new(file_config: &ModelFileConfig) -> Result<Self> {
        // 进行参数有效性检查
        if file_config.hidden_size % file_config.num_attention_heads != 0 {
            return Err(Error::InvalidArgument(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                file_config.hidden_size, file_config.num_attention_heads
            )).into());
        }
        if file_config.num_attention_heads % file_config.num_key_value_heads != 0 {
            return Err(Error::InvalidArgument(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                file_config.num_attention_heads, file_config.num_key_value_heads
            )).into());
        }
        
        let dim = file_config.hidden_size;
        let hidden_dim = file_config.intermediate_size;
        let layer_num = file_config.num_hidden_layers;
        let head_num = file_config.num_attention_heads;
        let kv_head_num = file_config.num_key_value_heads;
        let seq_len = MAX_SEQ_LEN;
        println!("模型最大序列长度 (max_position_embeddings) 设置为默认值 {}, 忽略 config.json 中的值 {}, 在model/config.rs中修改",MAX_SEQ_LEN, file_config.max_position_embeddings);
        
        let head_size = dim / head_num;
        let kv_dim = (dim * kv_head_num) / head_num;
        let kv_mul = head_num / kv_head_num;

        let is_shared_weight = file_config.vocab_size > 0;
        let vocab_size = file_config.vocab_size.abs() as usize;

        let immediate_dim = file_config.immediate_dim;


        Ok(Self {
            dim,
            hidden_dim,
            layer_num,
            head_num,
            kv_head_num,
            seq_len,
            vocab_size,
            kv_dim,
            kv_mul,
            head_size,
            is_shared_weight,
            torch_dtype: file_config.torch_dtype.clone(),
            immediate_dim,
        })
    }
}