// src/model/loader/registry.rs
//
// Model registry for managing model architectures and instances
// Based on architecture-based dispatch pattern inspired by sglang and vllm

use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use once_cell::sync::Lazy;

use crate::base::DeviceType;
use crate::base::error::{Error, Result};
use super::model_loader::ModelLoader;
use crate::model::Model;

// ======================= 全局 Registry 实例 =======================

/// 全局模型注册表实例
///
/// 程序启动时，各个模型架构会将自己的名字和构建函数注册到这里。
/// 在使用时，接收 config.json 里的架构名，找到对应的构建函数，
/// 传入 ModelLoader，返回构建好的模型对象。
pub static GLOBAL_REGISTRY: Lazy<ModelRegistry> = Lazy::new(ModelRegistry::new);

/// Model identifier for registry lookup
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ModelId(pub String);

impl ModelId {
    pub fn new(name: impl Into<String>) -> Self {
        ModelId(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for ModelId {
    fn from(s: String) -> Self {
        ModelId(s)
    }
}

impl From<&str> for ModelId {
    fn from(s: &str) -> Self {
        ModelId(s.to_string())
    }
}

/// Model builder function signature
///
/// This is the core function type that registry stores as "Value".
///
/// The builder receives a fully-initialized ModelLoader (containing weights metadata)
/// and distributed inference parameters, then constructs the model instance.
///
/// # Arguments
/// * `loader` - ModelLoader with weights already mapped and indexed
/// * `device_type` - Target device type for model execution
/// * `tp_rank` - Tensor Parallel rank (for distributed inference)
/// * `tp_size` - Tensor Parallel world size
/// * `pp_rank` - Pipeline Parallel rank (reserved for future use)
/// * `pp_size` - Pipeline Parallel world size
pub type ModelBuilderFn = fn(
    loader: &ModelLoader,
    device_type: DeviceType,
    tp_rank: usize,
    tp_size: usize,
    pp_rank: usize,
    pp_size: usize,
) -> Result<Box<dyn Model>>;

/// Enum to hold different model types
/// 
/// This allows the registry to store models of different concrete types
/// while maintaining type safety.
pub enum ModelInstance {
    /// Boxed model implementing the Model trait
    Model(Box<dyn Model>),
}

/// Thread-safe model registry for managing model architectures and instances
///
/// The ModelRegistry serves as:
/// 1. **Dispatch Center**: Maps architecture names to model builders
/// 2. **Lifecycle Manager**: Handles loading/unloading of models
/// 3. **Thread-safe Storage**: Provides concurrent access to models
pub struct ModelRegistry {
    /// Architecture to model builder mapping
    architectures: Arc<RwLock<HashMap<String, ModelBuilderFn>>>,
    /// Internal storage for loaded models
    models: Arc<RwLock<HashMap<ModelId, ModelInstance>>>,
}

impl ModelRegistry {
    /// Create a new empty model registry
    pub fn new() -> Self {
        Self {
            architectures: Arc::new(RwLock::new(HashMap::new())),
            models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a model architecture with its builder function
    /// 
    /// # Arguments
    /// * `architecture` - The architecture name from HuggingFace config.json (e.g., "LlamaForCausalLM")
    /// * `builder` - The function that builds the model instance
    pub fn register_architecture(
        &self,
        architecture: &str,
        builder: ModelBuilderFn,
    ) -> Result<()> {
        let mut architectures = self.architectures.write()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        architectures.insert(architecture.to_string(), builder);
        println!("Registered architecture: {}", architecture);
        Ok(())
    }

    /// Get the builder function for a specific architecture
    pub fn get_builder(&self, architecture: &str) -> Result<Option<ModelBuilderFn>> {
        let architectures = self.architectures.read()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        Ok(architectures.get(architecture).copied())
    }

    /// Load and register a model with pre-initialized ModelLoader
    ///
    /// Architecture is automatically detected from loader.architecture
    /// (which was read from config.json during ModelLoader::load)
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier for this model instance
    /// * `loader` - ModelLoader with weights and architecture already loaded
    /// * `device_type` - Target device type for model execution
    /// * `tp_rank` - Tensor Parallel rank
    /// * `tp_size` - Tensor Parallel world size
    /// * `pp_rank` - Pipeline Parallel rank
    /// * `pp_size` - Pipeline Parallel world size
    pub fn load_model(
        &self,
        model_id: ModelId,
        loader: &ModelLoader,
        device_type: DeviceType,
        tp_rank: usize,
        tp_size: usize,
        pp_rank: usize,
        pp_size: usize,
    ) -> Result<()> {
        let architecture = &loader.architecture;

        println!("Loading model '{}' with architecture '{}'...",
            model_id.as_str(), architecture);

        // Get the builder for this architecture
        let builder = self.get_builder(architecture)?
            .ok_or_else(|| Error::InvalidArgument(
                format!("Architecture '{}' is not registered", architecture)
            ))?;

        // Build the model
        let model = builder(loader, device_type, tp_rank, tp_size, pp_rank, pp_size)?;

        // Insert into registry
        let mut models = self.models.write()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        if models.contains_key(&model_id) {
            return Err(Error::InvalidArgument(
                format!("Model ID '{}' already exists in registry", model_id.as_str())
            ).into());
        }

        models.insert(model_id.clone(), ModelInstance::Model(model));
        println!("Model '{}' successfully loaded and registered.", model_id.as_str());

        Ok(())
    }

    /// Get read access to a model using a callback closure
    ///
    /// This method provides safe access to a model by holding the read lock
    /// only for the duration of the callback execution.
    ///
    /// # Example
    /// ```ignore
    /// registry.with_model(&model_id, |model| {
    ///     println!("Config: {}", model.config().dim);
    ///     Ok(())
    /// })?;
    /// ```
    pub fn with_model<F, R>(&self, model_id: &ModelId, f: F) -> Result<R>
    where
        F: FnOnce(&dyn Model) -> Result<R>,
    {
        let models = self.models.read()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        match models.get(model_id) {
            Some(ModelInstance::Model(model)) => f(model.as_ref()),
            None => Err(Error::InvalidArgument(
                format!("Model '{}' not found in registry", model_id.as_str())
            ).into()),
        }
    }

    /// Get mutable access to a model using a callback closure
    ///
    /// This method provides safe mutable access to a model by holding the write lock
    /// only for the duration of the callback execution.
    ///
    /// # Example
    /// ```ignore
    /// registry.with_model_mut(&model_id, |model| {
    ///     model.reset_kv_cache()?;
    ///     Ok(())
    /// })?;
    /// ```
    pub fn with_model_mut<F, R>(&self, model_id: &ModelId, f: F) -> Result<R>
    where
        F: FnOnce(&mut dyn Model) -> Result<R>,
    {
        let mut models = self.models.write()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        match models.get_mut(model_id) {
            Some(ModelInstance::Model(model)) => f(model.as_mut()),
            None => Err(Error::InvalidArgument(
                format!("Model '{}' not found in registry", model_id.as_str())
            ).into()),
        }
    }

    /// Get read access to a model
    ///
    /// # Deprecated
    /// Use `with_model` instead for simpler, safer API.
    ///
    /// This method returns a complex type with manual lifetime management.
    /// The callback-based `with_model` is recommended for new code.
    pub fn get_model(&self, model_id: &ModelId) -> Result<Option<impl std::ops::Deref<Target = dyn Model> + '_>> {
        let models = self.models.read()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        Ok(RwLockReadGuardModelMapper::new(models, model_id.clone()))
    }

    /// Get mutable access to a model
    ///
    /// # Deprecated
    /// Use `with_model_mut` instead for simpler, safer API.
    ///
    /// This method returns a complex type with manual lifetime management.
    /// The callback-based `with_model_mut` is recommended for new code.
    pub fn get_model_mut(&self, model_id: &ModelId) -> Result<Option<impl std::ops::DerefMut<Target = dyn Model> + '_>> {
        let models = self.models.write()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        Ok(RwLockWriteGuardModelMapper::new(models, model_id.clone()))
    }

    /// Unload a model from the registry
    pub fn unload_model(&self, model_id: &ModelId) -> Result<()> {
        let mut models = self.models.write()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        models.remove(model_id)
            .ok_or_else(|| Error::InvalidArgument(
                format!("Model '{}' not found in registry", model_id.as_str())
            ))?;

        println!("Model '{}' unloaded from registry.", model_id.as_str());
        Ok(())
    }

    /// List all loaded model IDs
    pub fn list_models(&self) -> Result<Vec<ModelId>> {
        let models = self.models.read()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        Ok(models.keys().cloned().collect())
    }

    /// List all registered architectures
    pub fn list_architectures(&self) -> Result<Vec<String>> {
        let architectures = self.architectures.read()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        Ok(architectures.keys().cloned().collect())
    }

    /// Check if a model is loaded
    pub fn contains(&self, model_id: &ModelId) -> bool {
        self.models.read()
            .map(|models| models.contains_key(model_id))
            .unwrap_or(false)
    }

    /// Check if an architecture is registered
    pub fn is_architecture_registered(&self, architecture: &str) -> bool {
        self.architectures.read()
            .map(|archs| archs.contains_key(architecture))
            .unwrap_or(false)
    }

    /// Get the number of loaded models
    pub fn len(&self) -> usize {
        self.models.read()
            .map(|models| models.len())
            .unwrap_or(0)
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Clone for ModelRegistry {
    fn clone(&self) -> Self {
        Self {
            architectures: Arc::clone(&self.architectures),
            models: Arc::clone(&self.models),
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Helper struct for mapping RwLockReadGuard to Model trait
struct RwLockReadGuardModelMapper<'a> {
    _guard: RwLockReadGuard<'a, HashMap<ModelId, ModelInstance>>,
    model_ptr: Option<*const dyn Model>,
}

impl<'a> RwLockReadGuardModelMapper<'a> {
    fn new(guard: RwLockReadGuard<'a, HashMap<ModelId, ModelInstance>>, model_id: ModelId) -> Option<Self> {
        // Get model pointer with correct lifetime
        let model_ptr = match guard.get(&model_id) {
            Some(ModelInstance::Model(model)) => {
                // Convert to raw pointer to manage lifetime manually
                Some(model.as_ref() as *const dyn Model)
            },
            _ => None,
        };

        model_ptr.map(|model_ptr| Self {
            _guard: guard,
            model_ptr: Some(model_ptr),
        })
    }
}

impl<'a> std::ops::Deref for RwLockReadGuardModelMapper<'a> {
    type Target = dyn Model;

    fn deref(&self) -> &Self::Target {
        // SAFETY: The _guard ensures the model instance is valid
        // The raw pointer is valid for the lifetime of the guard
        unsafe {
            &*self.model_ptr.unwrap()
        }
    }
}

// Helper struct for mapping RwLockWriteGuard to Model trait
struct RwLockWriteGuardModelMapper<'a> {
    _guard: RwLockWriteGuard<'a, HashMap<ModelId, ModelInstance>>,
    model_ptr: Option<*mut dyn Model>,
}

impl<'a> RwLockWriteGuardModelMapper<'a> {
    fn new(mut guard: RwLockWriteGuard<'a, HashMap<ModelId, ModelInstance>>, model_id: ModelId) -> Option<Self> {
        // Get mutable model pointer with correct lifetime
        let model_ptr = match guard.get_mut(&model_id) {
            Some(ModelInstance::Model(model)) => {
                // Convert to raw pointer to manage lifetime manually
                Some(model.as_mut() as *mut dyn Model)
            },
            _ => None,
        };

        model_ptr.map(|model_ptr| Self {
            _guard: guard,
            model_ptr: Some(model_ptr),
        })
    }
}

impl<'a> std::ops::Deref for RwLockWriteGuardModelMapper<'a> {
    type Target = dyn Model;

    fn deref(&self) -> &Self::Target {
        // SAFETY: The _guard ensures the model instance is valid
        // The raw pointer is valid for the lifetime of the guard
        unsafe {
            &*self.model_ptr.unwrap()
        }
    }
}

impl<'a> std::ops::DerefMut for RwLockWriteGuardModelMapper<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: The _guard ensures the model instance is valid
        // The raw pointer is valid for the lifetime of the guard
        unsafe {
            &mut *self.model_ptr.unwrap()
        }
    }
}
