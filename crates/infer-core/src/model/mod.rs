pub mod config;
pub mod safetensor_loader;
pub mod tokenizer;
pub mod llama3;
use tokenizer::{GenericHfTokenizer, Tokenizer};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use crate::model::config::{ModelFileConfig, RuntimeModelConfig};

// 引入我们自己的模块和统一的错误处理
use crate::base::error::{Error, Result};
use crate::model::safetensor_loader::SafetensorReader;
use safetensors::tensor::TensorView;
use crate::{base::DeviceType, tensor::Tensor};

pub trait Model {
    /// 初始化模型，包括创建层、加载权重等。
    fn init(&mut self, device_type: DeviceType) -> Result<()>;

    /// 执行一次完整的前向传播，并返回下一个 token 的 logits。
    fn forward(&mut self, input: &Tensor, pos: &Tensor) -> Result<Tensor>;
    /// 获取模型内部持有的 Tokenizer 的引用。
    fn tokenizer(&self) -> &dyn Tokenizer;

    /// 委托给内部的 Tokenizer 进行编码。
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        self.tokenizer().encode(text)
    }

    /// 委托给内部的 Tokenizer 进行解码。
    fn decode(&self, ids: &[i32]) -> Result<String> {
        self.tokenizer().decode(ids)
    }

    /// 判断给定的 token ID 是否为句末符 (end-of-sentence)。
    /// 这个逻辑与 Tokenizer 紧密相关。
    fn is_eos_token(&self, token_id: u32) -> bool;

    // === KV Cache 管理 ===
    // 含义：为了支持上下文切换或更复杂的采样策略，需要能从 KV Cache 中切片。
    // 在 Rust 中，我们返回一个元组 `(Tensor, Tensor)`，分别代表 K 和 V Cache。
    // `layer_idx`: 要操作的层索引。
    // `start_pos`, `end_pos`: 要切片的位置范围。
    fn slice_kv_cache(&self, layer_idx: usize, start_pos: usize, end_pos: usize) -> Result<(Tensor, Tensor)>;
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub file_path: PathBuf,
}

#[derive(Debug)]
pub struct ModelLoader {
    pub config: RuntimeModelConfig,
    tensor_names: Vec<String>,
    /// Mmap文件持有者。这些内存映射必须在readers之前被drop，因为readers依赖这些数据。
    /// 注意：字段声明顺序很重要 - Rust按声明顺序drop字段，所以_mmaps会在readers之后drop。
    _mmaps: HashMap<PathBuf, Mmap>,
    tensor_index: HashMap<String, TensorInfo>,
    /// SafetensorReader实例。这些reader的生命周期被标记为'static，
    /// 但实际上它们依赖于_mmaps中的数据。这是安全的，因为：
    /// 1. _mmaps和readers总是同时存在于同一个结构体中
    /// 2. Rust的drop顺序保证_mmaps会在readers之后drop（按字段声明相反顺序）
    /// 3. ModelLoader不提供任何方法来分别修改_mmaps或readers
    readers: HashMap<PathBuf, SafetensorReader<'static>>,
}

impl ModelLoader {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let model_dir = path.as_ref();

        // 1. 加载 config.json (不变)
        let config_path = model_dir.join("config.json");
        let config_file = File::open(&config_path)?;
        let file_config: ModelFileConfig = config::load_config_from_json(config_file)
            .map_err(|e| Error::InvalidArgument(format!("Failed to parse config.json: {}", e)))?;

        let config = RuntimeModelConfig::new(&file_config)?;
        println!("运行时配置生成成功: dim={}, heads={}, layers={}", config.dim, config.head_num, config.layer_num);
        // --- 核心改动：动态构建 weight_map ---
        let mut weight_map = HashMap::new();
        let index_path = model_dir.join("model.safetensors.index.json");

        if index_path.exists() {
            // **路径 A: 分片模型 (Sharded Model)**
            println!("检测到 index.json，按分片模型加载...");
            let index_file = File::open(&index_path)?;
            let index: serde_json::Value = serde_json::from_reader(index_file)
                .map_err(|e| Error::InvalidArgument(format!("Failed to parse index.json: {}", e)))?;
            let weight_map_json = index["weight_map"].as_object().ok_or_else(|| {
                Error::InvalidArgument("`weight_map` not found in index.json".to_string())
            })?;
            
            for (name, filename_val) in weight_map_json {
                if let Some(filename) = filename_val.as_str() {
                    weight_map.insert(name.clone(), filename.to_string());
                }
            }

        } else {
            // **路径 B: 单文件模型 (Single-file Model) - 备用逻辑**
            println!("未检测到 index.json，尝试按单文件模型加载...");
            let single_file_path = model_dir.join("model.safetensors");
            if !single_file_path.exists() {
                // 如果两种模式的文件都找不到，则报错
                return Err(Error::InvalidArgument(
                    "Model directory must contain either 'model.safetensors.index.json' or 'model.safetensors'".to_string()
                ).into());
            }

            // 临时映射文件以读取其内部的张量名称
            let mmap = safetensor_loader::load_and_mmap(&single_file_path)?;
            let reader = SafetensorReader::new(&mmap)?;
            // println!("[DEBUG] Available tensors in the model file:");
            // 为这个文件中的所有张量构建 weight_map
            for tensor_name in reader.get_tensor_names() {
                // println!("- {}", tensor_name);
                weight_map.insert(tensor_name, "model.safetensors".to_string());
            }
        }

        // --- 从这里开始，后续逻辑对于两种路径是统一的 ---

        // 3. 根据构建好的 weight_map 填充 tensor_index 和 files_to_mmap
        let mut tensor_index = HashMap::new();
        let mut files_to_mmap = std::collections::HashSet::new();
        let mut tensor_names = Vec::with_capacity(weight_map.len());
        for (tensor_name, file_name) in &weight_map {
            let file_path = model_dir.join(file_name);
            tensor_names.push(tensor_name.clone());
            tensor_index.insert(
                tensor_name.clone(),
                TensorInfo { name: tensor_name.clone(), file_path: file_path.clone() },
            );
            files_to_mmap.insert(file_path);
        }

        // 4. 内存映射所有需要的文件 (现在这个集合可能是1个或多个)
        let mut mmaps = HashMap::new();
        for file_path in files_to_mmap {
            let mmap = safetensor_loader::load_and_mmap(&file_path)?;
            mmaps.insert(file_path, mmap);
        }

        // 5. 为每个文件创建 SafetensorReader 实例
        let mut readers = HashMap::new();
        for (path, mmap) in &mmaps {
            // SAFETY: 这个transmute将mmap引用的生命周期从 &'a [u8] 扩展到 &'static [u8]。
            // 虽然这在一般情况下是不安全的，但在这个特定上下文中是安全的，原因如下：
            //
            // 1. **所有权保证**：mmaps HashMap和readers HashMap都存储在同一个ModelLoader结构体中。
            //    mmaps持有实际的Mmap对象，readers持有指向这些mmap数据的SafetensorReader。
            //
            // 2. **生命周期保证**：Rust的drop顺序是按字段声明的相反顺序。在ModelLoader的定义中，
            //    _mmaps字段在readers字段之前声明，因此_mmaps会在readers之后被drop。
            //    这保证了当readers被drop时，它们所引用的mmap数据仍然有效。
            //
            // 3. **不可变性保证**：ModelLoader不提供任何方法来分别修改_mmaps或readers。
            //    一旦创建，这两个HashMap就保持不变，确保引用关系不会被破坏。
            //
            // 4. **封装保证**：_mmaps字段是私有的，外部代码无法访问或修改它，
            //    因此无法在readers仍然存在时删除或替换mmap。
            //
            // 这个设计本质上实现了一个"自引用结构"，其中readers依赖于_mmaps的数据。
            // 理想情况下应该使用Pin或自引用crate（如ouroboros）来表达这种关系，
            // 但为了避免引入额外依赖和复杂性，我们选择了这种方式并明确文档化安全性保证。
            let mmap_static: &'static [u8] = unsafe { std::mem::transmute(mmap.as_ref()) };
            let reader = SafetensorReader::new(mmap_static)?;
            readers.insert(path.clone(), reader);
        }

        println!("所有权重文件均已成功映射，并为每个文件创建了 SafetensorReader。");

        Ok(Self { config, tensor_names, _mmaps: mmaps, tensor_index, readers })
    }

    /// 根据张量名称获取一个零拷贝的视图
    pub fn get_tensor<'a>(&'a self, name: &str) -> Result<TensorView<'a>> {
        let info = self.tensor_index.get(name).ok_or_else(|| {
            Error::InvalidArgument(format!("Tensor '{}' not found in model index", name))
        })?;

        let reader = self.readers.get(&info.file_path)
            .expect("Internal consistency error: reader not found for a known file");

        // 委托给 SafetensorReader 的 get_tensor 方法
        reader.get_tensor(name)
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.tensor_names.clone()
    }

    pub fn create_tokenizer(
        &mut self,
        model_dir: &Path,
    ) -> Result<Box<dyn Tokenizer>> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
             return Err(Error::InvalidArgument(format!(
                "Tokenizer file not found at: {:?}",
                tokenizer_path
            )).into());
        }

        // **核心改动**: 直接调用 GenericHfTokenizer::from_file
        let tokenizer = GenericHfTokenizer::from_file(&tokenizer_path)?;
        let boxed_tokenizer: Box<dyn Tokenizer> = Box::new(tokenizer);

        // 更新 vocab_size 的逻辑保持不变
        let tokenizer_vocab_size = boxed_tokenizer.vocab_size();
        if tokenizer_vocab_size == 0 {
            return Err(Error::InvalidArgument("Tokenizer vocabulary size must be positive".to_string()).into());
        }
        
        println!(
            "模型配置 vocab_size: {}, Tokenizer vocab_size: {}. 将使用 Tokenizer 的值。",
            self.config.vocab_size, tokenizer_vocab_size
        );
        self.config.vocab_size = tokenizer_vocab_size;

        Ok(boxed_tokenizer)
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
}

pub type Workspace = HashMap<BufferType, Tensor>;