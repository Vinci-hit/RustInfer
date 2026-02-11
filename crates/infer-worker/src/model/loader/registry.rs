// src/model/loader/registry.rs

use std::sync::Arc;
use std::sync::OnceLock;
use dashmap::DashMap;  

use crate::base::DeviceType;
use crate::base::error::{Error, Result};
use super::model_loader::ModelLoader;
use crate::model::Model;

/// 模型实例的线程安全指针
pub type ModelInstancePtr = Arc<dyn Model + Send + Sync>;

/// 全局模型注册表实例
static GLOBAL_REGISTRY: OnceLock<ModelRegistry> = OnceLock::new();

/// 模型注册表：管理 Worker 进程中已加载的模型实例
pub struct ModelRegistry {
    /// 已加载模型的存储：Key 为模型名称（如 "llama-3-8b"）
    /// 使用 DashMap 确保在多线程环境下（如 RPC 和推理线程同时访问）的无锁/细粒度锁读取
    models: DashMap<String, ModelInstancePtr>,
}

impl ModelRegistry {
    /// 获取全局单例
    pub fn global() -> &'static ModelRegistry {
        GLOBAL_REGISTRY.get_or_init(|| ModelRegistry {
            models: DashMap::new(),
        })
    }

    /// 根据名称获取模型用于推理
    pub fn get_model(&self, name: &str) -> Option<ModelInstancePtr> {
        self.models.get(name).map(|entry| entry.value().clone())
    }

    /// 核心工厂方法：将架构名称映射到具体的结构体实现
    /// 对于开源项目，新模型在此 match 中注册
    fn model_factory(
        &self,
        architecture: &str,
        loader: &ModelLoader,
        device_type: DeviceType,
        tp_rank: usize,
        tp_size: usize,
        pp_rank: usize,
        pp_size: usize,
    ) -> Result<Box<dyn Model + Send + Sync>> {
        match architecture {
            // "LlamaForCausalLM" | "LlamaModel" => {
            //     Ok(Box::new(crate::model::llama::LlamaModel::new(loader, device_type, tp_rank, tp_size)?))
            // },
            _ => Err(Error::ArchNotFound(format!("Architecture {} not supported", architecture)).into()),
        }
    }

    /// 加载模型并注册
    /// 
    /// # 参数
    /// * `name` - 给该模型实例起的别名（用于推理请求路由）
    /// * `loader` - 已初始化权重元数据的加载器
    pub async fn load_model(
        &self,
        name: String,
        loader: &ModelLoader,
        device_type: DeviceType,
        tp_rank: usize,
        tp_size: usize,
        pp_rank: usize,
        pp_size: usize,
    ) -> Result<()> {
        let architecture = &loader.architecture;
        println!("Worker: Creating model instance '{}' with arch '{}'...", name, architecture);

        // 1. 通过工厂创建具体模型实例（耗时操作，不持有 Map 锁）
        let model_impl = self.model_factory(
            architecture,
            loader,
            device_type,
            tp_rank,
            tp_size,
            pp_rank,
            pp_size
        )?;

        // 2. 将模型包装在 Arc 中
        let model_ptr: ModelInstancePtr = Arc::from(model_impl);

        // 3. 注册到 Map 中
        // 如果已存在同名模型，DashMap 的 insert 会自动替换旧的 Arc，
        // 旧模型会在所有存量请求结束后自动 Drop，释放显存。
        self.models.insert(name.clone(), model_ptr);

        println!("Worker: Model '{}' is ready for inference.", name);
        Ok(())
    }

    /// 卸载模型
    pub fn unload_model(&self, name: &str) {
        self.models.remove(name);
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }
}