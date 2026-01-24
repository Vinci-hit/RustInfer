use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use safetensors::Dtype;

use crate::model::config::{self, ModelFileConfig, RuntimeModelConfig};
use crate::base::error::Result;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;

/// 张量在 mmap 文件中的位置信息
#[derive(Debug, Clone)]
pub struct TensorLocation {
    /// 所在文件的索引（对应 mmaps 的下标）
    pub file_index: usize,
    /// 张量数据在 mmap 中的起始字节偏移量
    pub start_offset: usize,
    /// 张量数据的字节长度
    pub data_len: usize,
    /// 张量的形状
    pub shape: Vec<usize>,
    /// 张量的数据类型
    pub dtype: DType,
}

/// 数据类型枚举，用于简化类型处理
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    BOOL,
}

impl From<Dtype> for DType {
    fn from(dtype: Dtype) -> Self {
        match dtype {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            Dtype::I8 => DType::I8,
            Dtype::I16 => DType::I16,
            Dtype::I32 => DType::I32,
            Dtype::I64 => DType::I64,
            Dtype::U8 => DType::U8,
            Dtype::U16 => DType::U16,
            Dtype::U32 => DType::U32,
            Dtype::U64 => DType::U64,
            Dtype::BOOL => DType::BOOL,
            _ => panic!("Unsupported dtype: {:?}", dtype),
        }
    }
}

impl DType {
    /// 返回该类型的大小（字节数）
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 | DType::BOOL => 1,
            DType::I16 | DType::U16 => 2,
            DType::I32 | DType::U32 => 4,
            DType::I64 | DType::U64 => 8,
        }
    }

    /// 将 safetensors 的 Dtype 转换为我们的 DType
    pub fn from_safetensors(dtype: Dtype) -> Self {
        dtype.into()
    }
}

#[derive(Debug)]
pub struct ModelLoader {
    pub config: RuntimeModelConfig,
    /// 模型架构名称（从 config.json 的 architectures 字段读取）
    pub architecture: String,
    tensor_names: Vec<String>,
    /// mmap 文件持有者，使用 Arc 允许共享引用
    /// 这些 mmap 必须在 ModelLoader 生命周期内保持有效
    mmaps: Vec<Arc<Mmap>>,
    /// 张量索引：张量名称 -> 位置信息（包含文件索引、偏移量等）
    tensor_index: HashMap<String, TensorLocation>,
    /// 可选：保留 SafeTensors 用于兼容旧接口（仅在需要时使用）
    _readers: Vec<Arc<SafeTensors<'static>>>,
}

impl ModelLoader {
    /// 扫描模型目录，查找所有 .safetensors 文件
    fn find_safetensor_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        // 优先检查是否存在 model.safetensors.index.json（分片模型）
        let index_path = model_dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index_file = File::open(&index_path)?;
            let index: serde_json::Value = serde_json::from_reader(index_file)
                .map_err(|e| anyhow::anyhow!("Failed to parse index.json: {}", e))?;
            let weight_map = index["weight_map"].as_object().ok_or_else(|| {
                anyhow::anyhow!("`weight_map` not found in index.json")
            })?;

            // 收集所有被引用的文件名
            let mut file_names = std::collections::HashSet::new();
            for filename_val in weight_map.values() {
                if let Some(filename) = filename_val.as_str() {
                    file_names.insert(filename.to_string());
                }
            }

            // 返回所有文件路径
            for filename in file_names {
                files.push(model_dir.join(filename));
            }
        } else {
            // 单文件模型
            let single_file_path = model_dir.join("model.safetensors");
            if !single_file_path.exists() {
                return Err(anyhow::anyhow!(
                    "Model directory must contain either 'model.safetensors.index.json' or 'model.safetensors'"
                ));
            }
            files.push(single_file_path);
        }

        Ok(files)
    }

    /// 加载配置文件，同时检测模型架构
    fn load_config(model_dir: &Path) -> Result<(RuntimeModelConfig, String)> {
        let config_path = model_dir.join("config.json");

        // 读取原始 JSON 用于提取 architecture
        let config_bytes = std::fs::read(&config_path)?;
        let raw_json: serde_json::Value = serde_json::from_slice(&config_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to parse config.json: {}", e))?;

        // 提取 architecture
        let architecture = if let Some(archs) = raw_json.get("architectures").and_then(|v| v.as_array()) {
            archs.first()
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| anyhow::anyhow!("Empty 'architectures' array in config.json"))?
        } else if let Some(model_type) = raw_json.get("model_type").and_then(|v| v.as_str()) {
            // Fallback: 根据 model_type 推断
            match model_type {
                "llama" => "LlamaForCausalLM".to_string(),
                "mistral" => "MistralForCausalLM".to_string(),
                "qwen2" => "Qwen2ForCausalLM".to_string(),
                _ => return Err(anyhow::anyhow!("Unknown model_type: {}", model_type)),
            }
        } else {
            return Err(anyhow::anyhow!(
                "Cannot detect architecture: missing 'architectures' or 'model_type' in config.json"
            ));
        };

        // 解析模型配置
        let file_config: ModelFileConfig = config::load_config_from_json(std::io::Cursor::new(&config_bytes))
            .map_err(|e| anyhow::anyhow!("Failed to parse config.json: {}", e))?;

        let config = RuntimeModelConfig::new(&file_config)?;
        println!("运行时配置生成成功: dim={}, heads={}, layers={}, arch={}",
            config.dim, config.head_num, config.layer_num, architecture);
        Ok((config, architecture))
    }

    /// 加载模型
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let model_dir = path.as_ref();

        // 1. 加载 Config 和 Architecture
        let (config, architecture) = Self::load_config(model_dir)?;

        // 2. 扫描所有 .safetensors 文件
        let files = Self::find_safetensor_files(model_dir)?;
        println!("找到 {} 个 safetensors 文件", files.len());

        let mut mmaps = Vec::new();
        let mut readers = Vec::new();
        let mut tensor_index = HashMap::new();
        let mut tensor_names = Vec::new();

        for (file_idx, path) in files.iter().enumerate() {
            // 打开文件并映射
            let file = File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let mmap_arc = Arc::new(mmap);

            // --- 临时作用域开始：创建临时的 SafeTensors 视图来解析 Metadata ---
            {
                let st = SafeTensors::deserialize(&mmap_arc)?;
                let tensor_count = st.tensors().len();
                println!("  文件 {:?} 包含 {} 个张量", path.file_name(), tensor_count);

                for (name, view) in st.tensors() {
                    // 关键点：直接获取该 Tensor 在文件中的绝对字节位置
                    // view.data() 返回的是 slice，我们需要算出它相对于 mmap 起始位置的 offset
                    let slice_ptr = view.data().as_ptr() as usize;
                    let mmap_ptr = mmap_arc.as_ptr() as usize;
                    let start_offset = slice_ptr - mmap_ptr;

                    let info = TensorLocation {
                        file_index: file_idx,
                        start_offset,
                        data_len: view.data().len(),
                        shape: view.shape().to_vec(),
                        dtype: DType::from_safetensors(view.dtype()),
                    };

                    tensor_index.insert(name.to_string(), info);
                    tensor_names.push(name.to_string());
                }
            }
            // --- 临时作用域结束，SafeTensors 被 Drop ---

            // 保留 Arc<Mmap> 用于后续访问
            mmaps.push(mmap_arc);

            // 可选：为兼容性保留 SafeTensors 实例
            let mmap_static: &'static [u8] = unsafe { std::mem::transmute(&mmaps[file_idx] as &[u8]) };
            let st = SafeTensors::deserialize(mmap_static)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize safetensors: {}", e))?;
            readers.push(Arc::new(st));
        }

        println!("所有权重文件均已成功映射，张量索引构建完成。");

        Ok(Self {
            config,
            architecture,
            tensor_names,
            mmaps,
            tensor_index,
            _readers: readers,
        })
    }

    /// 根据张量名称获取一个零拷贝的视图
    pub fn get_tensor<'a>(&'a self, name: &str) -> Result<TensorView<'a>> {
        let info = self.tensor_index.get(name).ok_or_else(|| {
            anyhow::anyhow!("Tensor '{}' not found in model index", name)
        })?;

        // 获取对应的 SafeTensors 实例
        let st = self._readers.get(info.file_index)
            .ok_or_else(|| anyhow::anyhow!("Invalid file index: {}", info.file_index))?;

        // 获取张量视图
        Ok(st.tensor(name).map_err(|e| anyhow::anyhow!("Failed to get tensor '{}': {}", name, e))?)
    }

    /// 获取张量的原始数据（零拷贝）
    pub fn get_tensor_data<'a>(&'a self, name: &str) -> Result<&'a [u8]> {
        let info = self.tensor_index.get(name).ok_or_else(|| {
            anyhow::anyhow!("Tensor '{}' not found in model index", name)
        })?;

        // 直接通过偏移量获取数据，无需解析 SafeTensors
        let mmap = &self.mmaps[info.file_index];
        Ok(&mmap[info.start_offset..info.start_offset + info.data_len])
    }

    /// 获取张量的位置信息
    pub fn get_tensor_location(&self, name: &str) -> Result<TensorLocation> {
        self.tensor_index.get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Tensor '{}' not found in model index", name))
    }

    /// 获取张量信息（形状、数据类型等）
    pub fn get_tensor_info(&self, name: &str) -> Result<(Vec<usize>, DType)> {
        let loc = self.get_tensor_location(name)?;
        Ok((loc.shape, loc.dtype))
    }

    /// 检查张量是否存在
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }

    /// 获取所有张量名称
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensor_names.clone()
    }

    /// 获取张量总数
    pub fn tensor_count(&self) -> usize {
        self.tensor_index.len()
    }

    /// 获取已加载的文件数量
    pub fn file_count(&self) -> usize {
        self.mmaps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::Dtype as SafetensorsDtype;
    use std::io::Write;

    /// 创建一个最小的测试 safetensors 文件
    fn create_test_safetensors(dir: &Path, filename: &str) -> PathBuf {
        use safetensors::tensor::serialize;
        use safetensors::tensor::TensorView;

        let file_path = dir.join(filename);

        // 创建测试数据：两个 f32 值 [0.5, 1.0]
        let data: Vec<u8> = vec![
            0x00, 0x00, 0x00, 0x3F, // 0.5 in f32 (little-endian)
            0x00, 0x00, 0x80, 0x3F, // 1.0 in f32
        ];

        // 使用 safetensors 库创建一个张量视图
        let tensor_view = TensorView::new(
            safetensors::Dtype::F32,
            vec![2],
            &data,
        ).unwrap();

        // 使用 safetensors 库的序列化 API
        let tensors = [("test_tensor", tensor_view)];

        // 序列化为字节数组，metadata 为 None
        let serialized = serialize(tensors, None).unwrap();

        // 写入文件
        let mut file = File::create(&file_path).unwrap();
        file.write_all(&serialized).unwrap();

        file_path
    }

    /// 创建测试用的 config.json
    fn create_test_config(dir: &Path) {
        let config = serde_json::json!({
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "num_hidden_layers": 6,
            "num_key_value_heads": 4,
            "max_position_embeddings": 2048,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-5,
            "torch_dtype": "float16"
        });

        let config_path = dir.join("config.json");
        let mut file = File::create(config_path).unwrap();
        file.write_all(serde_json::to_vec_pretty(&config).unwrap().as_slice()).unwrap();
    }

    /// 创建测试用的 model.safetensors.index.json
    fn create_test_index(dir: &Path) {
        let index = serde_json::json!({
            "metadata": {},
            "weight_map": {
                "test_tensor_1": "model-00001-of-00002.safetensors",
                "test_tensor_2": "model-00002-of-00002.safetensors"
            }
        });

        let index_path = dir.join("model.safetensors.index.json");
        let mut file = File::create(index_path).unwrap();
        file.write_all(serde_json::to_vec_pretty(&index).unwrap().as_slice()).unwrap();
    }

    fn setup_test_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        dir
    }

    #[test]
    fn test_dtype_conversion() {
        // 测试 safetensors Dtype 到 DType 的转换
        assert_eq!(DType::from(SafetensorsDtype::F32), DType::F32);
        assert_eq!(DType::from(SafetensorsDtype::F16), DType::F16);
        assert_eq!(DType::from(SafetensorsDtype::BF16), DType::BF16);
        assert_eq!(DType::from(SafetensorsDtype::I32), DType::I32);
        assert_eq!(DType::from(SafetensorsDtype::U8), DType::U8);
    }

    #[test]
    fn test_dtype_size_of() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::F16.size_of(), 2);
        assert_eq!(DType::BF16.size_of(), 2);
        assert_eq!(DType::I8.size_of(), 1);
        assert_eq!(DType::I32.size_of(), 4);
        assert_eq!(DType::I64.size_of(), 8);
    }

    #[test]
    fn test_dtype_from_safetensors() {
        assert_eq!(DType::from_safetensors(SafetensorsDtype::F32), DType::F32);
        assert_eq!(DType::from_safetensors(SafetensorsDtype::BF16), DType::BF16);
    }

    #[test]
    fn test_find_safetensor_files_single_file() {
        let dir = setup_test_dir();
        create_test_config(dir.path());

        // 创建单文件模型
        create_test_safetensors(dir.path(), "model.safetensors");

        let files = ModelLoader::find_safetensor_files(dir.path()).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("model.safetensors"));
    }

    #[test]
    fn test_find_safetensor_files_sharded() {
        let dir = setup_test_dir();
        create_test_config(dir.path());
        create_test_index(dir.path());

        // 创建分片文件
        create_test_safetensors(dir.path(), "model-00001-of-00002.safetensors");
        create_test_safetensors(dir.path(), "model-00002-of-00002.safetensors");

        let files = ModelLoader::find_safetensor_files(dir.path()).unwrap();
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_find_safetensor_files_missing() {
        let dir = setup_test_dir();
        create_test_config(dir.path());
        // 不创建任何 safetensors 文件

        let result = ModelLoader::find_safetensor_files(dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must contain"));
    }

    #[test]
    fn test_load_config() {
        let dir = setup_test_dir();
        create_test_config(dir.path());

        let (config, _arch) = ModelLoader::load_config(dir.path()).unwrap();
        assert_eq!(config.dim, 512);
        assert_eq!(config.head_num, 8);
        assert_eq!(config.layer_num, 6);
        assert_eq!(config.vocab_size, 32000);
    }

    #[test]
    fn test_load_model_single_file() {
        let dir = setup_test_dir();
        create_test_config(dir.path());
        create_test_safetensors(dir.path(), "model.safetensors");

        // 这个测试只是验证加载过程不会失败
        let result = ModelLoader::load(dir.path());
        // 打印具体错误信息用于调试
        if let Err(e) = &result {
            eprintln!("ModelLoader::load failed with error: {}", e);
        }
        assert!(result.is_ok(), "ModelLoader::load failed: {:?}", result.unwrap_err());
    }

    #[test]
    fn test_tensor_location() {
        let loc = TensorLocation {
            file_index: 0,
            start_offset: 100,
            data_len: 200,
            shape: vec![10, 20],
            dtype: DType::F32,
        };

        assert_eq!(loc.file_index, 0);
        assert_eq!(loc.start_offset, 100);
        assert_eq!(loc.data_len, 200);
        assert_eq!(loc.shape, vec![10, 20]);
        assert_eq!(loc.dtype, DType::F32);
    }

    #[test]
    fn test_tensor_location_clone() {
        let loc1 = TensorLocation {
            file_index: 0,
            start_offset: 100,
            data_len: 200,
            shape: vec![10, 20],
            dtype: DType::F32,
        };

        let loc2 = loc1.clone();
        assert_eq!(loc1.file_index, loc2.file_index);
        assert_eq!(loc1.start_offset, loc2.start_offset);
        assert_eq!(loc1.data_len, loc2.data_len);
        assert_eq!(loc1.shape, loc2.shape);
        assert_eq!(loc1.dtype, loc2.dtype);
    }
}