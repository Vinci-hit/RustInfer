// src/model/registry.rs
//
// Model registry for managing multiple loaded models
// Provides thread-safe access to multiple models in memory

use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::path::Path;

use crate::base::DeviceType;
use crate::base::error::{Error, Result};
use super::factory::{ModelFactory, ModelType};
use super::llama3::Llama3;

/// Model identifier for registry lookup
///
/// Uniquely identifies a loaded model in the registry.
/// Typically uses a short name like "llama3-8b" or "qwen2-1.5b".
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ModelId(pub String);

impl ModelId {
    /// Create a new ModelId from a string
    pub fn new(name: impl Into<String>) -> Self {
        ModelId(name.into())
    }

    /// Get the underlying string representation
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

/// Enum to hold different model types
///
/// This allows the registry to store models of different concrete types
/// while maintaining type safety.
pub enum ModelInstance {
    Llama3(Llama3),
    // Qwen2(Qwen2),
    // Future: Qwen3VL(Qwen3VL), etc.
}

/// Thread-safe model registry for managing multiple loaded models
///
/// The ModelRegistry provides:
/// - Thread-safe concurrent access to models using RwLock
/// - Multiple models loaded simultaneously in memory
/// - Dynamic loading/unloading of models
/// - Model lifecycle management
///
/// # Thread Safety
/// - Multiple readers can access different models concurrently
/// - Writes (load/unload) are exclusive and block all access
/// - Safe to use across threads with Arc
///
/// # Example
/// ```
/// let registry = ModelRegistry::new(DeviceType::Cuda(0));
///
/// // Load multiple models
/// registry.load_model(
///     ModelId::new("llama3-8b"),
///     Path::new("/models/llama3-8b"),
///     ModelType::Llama3,
///     false,
/// )?;
///
/// registry.load_model(
///     ModelId::new("qwen2-1.5b"),
///     Path::new("/models/qwen2-1.5b"),
///     ModelType::Qwen2,
///     false,
/// )?;
///
/// // Access models
/// if let Some(model) = registry.get_llama3(&ModelId::new("llama3-8b"))? {
///     // Use model...
/// }
/// ```
pub struct ModelRegistry {
    /// Internal storage for loaded models
    models: Arc<RwLock<HashMap<ModelId, ModelInstance>>>,
    /// Default device type for loading models
    device_type: DeviceType,
}

impl ModelRegistry {
    /// Create a new empty model registry
    ///
    /// # Arguments
    /// * `device_type` - Default device type for models (CPU or CUDA)
    ///
    /// # Returns
    /// New ModelRegistry instance
    pub fn new(device_type: DeviceType) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            device_type,
        }
    }

    /// Load and register a model
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier for this model
    /// * `model_dir` - Path to model directory containing weights and config
    /// * `model_type` - Type of model to load
    /// * `is_quant_model` - Whether this is a quantized model
    ///
    /// # Returns
    /// Ok(()) if successful
    ///
    /// # Errors
    /// - Model loading fails
    /// - Model ID already exists in registry
    /// - Lock poisoning
    ///
    /// # Example
    /// ```
    /// registry.load_model(
    ///     ModelId::new("my-qwen2"),
    ///     Path::new("/models/qwen2-1.5b"),
    ///     ModelType::Qwen2,
    ///     false,
    /// )?;
    /// ```
    pub fn load_model(
        &self,
        model_id: ModelId,
        model_dir: &Path,
        model_type: ModelType,
        is_quant_model: bool,
    ) -> Result<()> {
        println!("Loading model '{}' of type {:?} from {:?}...",
            model_id.as_str(), model_type, model_dir);

        // Create the model instance
        let model_instance = match model_type {
            ModelType::Llama3 | ModelType::Mistral => {
                let model = ModelFactory::create_llama3(
                    model_dir,
                    self.device_type,
                    is_quant_model,
                )?;
                ModelInstance::Llama3(model)
            }
        };

        // Insert into registry
        let mut models = self.models.write()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        if models.contains_key(&model_id) {
            return Err(Error::InvalidArgument(
                format!("Model ID '{}' already exists in registry", model_id.as_str())
            ).into());
        }

        models.insert(model_id.clone(), model_instance);
        println!("Model '{}' successfully loaded and registered.", model_id.as_str());

        Ok(())
    }

    /// Load model with auto-detection of model type
    ///
    /// # Arguments
    /// * `model_id` - Unique identifier for this model
    /// * `model_dir` - Path to model directory
    /// * `is_quant_model` - Whether this is a quantized model
    ///
    /// # Returns
    /// Ok(ModelType) - the detected model type
    pub fn load_model_auto(
        &self,
        model_id: ModelId,
        model_dir: &Path,
        is_quant_model: bool,
    ) -> Result<ModelType> {
        let model_type = ModelFactory::detect_model_type(model_dir)?;
        self.load_model(model_id, model_dir, model_type, is_quant_model)?;
        Ok(model_type)
    }

    /// Get read access to a Llama3 model
    ///
    /// # Arguments
    /// * `model_id` - Model identifier
    ///
    /// # Returns
    /// Option<ReadGuard> containing the model if it exists and is Llama3
    pub fn get_llama3(&self, model_id: &ModelId) -> Result<Option<impl std::ops::Deref<Target = Llama3> + '_>> {
        let models = self.models.read()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        Ok(RwLockReadGuardMapper::new(models, model_id.clone()))
    }

    /// Get mutable access to a Llama3 model
    ///
    /// # Arguments
    /// * `model_id` - Model identifier
    ///
    /// # Returns
    /// Option<WriteGuard> containing mutable reference to model
    pub fn get_llama3_mut(&self, model_id: &ModelId) -> Result<Option<impl std::ops::DerefMut<Target = Llama3> + '_>> {
        let models = self.models.write()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        Ok(RwLockWriteGuardMapper::new(models, model_id.clone()))
    }

    /// Unload a model from the registry
    ///
    /// # Arguments
    /// * `model_id` - Model identifier to unload
    ///
    /// # Returns
    /// Ok(()) if successful
    ///
    /// # Errors
    /// - Model not found
    /// - Lock poisoning
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
    ///
    /// # Returns
    /// Vector of ModelIds currently in the registry
    pub fn list_models(&self) -> Result<Vec<ModelId>> {
        let models = self.models.read()
            .map_err(|_| Error::InternalError("Lock poisoned".to_string()))?;

        Ok(models.keys().cloned().collect())
    }

    /// Check if a model is loaded
    ///
    /// # Arguments
    /// * `model_id` - Model identifier to check
    ///
    /// # Returns
    /// true if model is loaded, false otherwise
    pub fn contains(&self, model_id: &ModelId) -> bool {
        self.models.read()
            .map(|models| models.contains_key(model_id))
            .unwrap_or(false)
    }

    /// Get the number of loaded models
    ///
    /// # Returns
    /// Number of models in registry
    pub fn len(&self) -> usize {
        self.models.read()
            .map(|models| models.len())
            .unwrap_or(0)
    }

    /// Check if registry is empty
    ///
    /// # Returns
    /// true if no models are loaded
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// Helper struct for mapping RwLockReadGuard to specific model type
struct RwLockReadGuardMapper<'a> {
    _guard: RwLockReadGuard<'a, HashMap<ModelId, ModelInstance>>,
    model_ref: Option<&'a Llama3>,
}

impl<'a> RwLockReadGuardMapper<'a> {
    fn new(guard: RwLockReadGuard<'a, HashMap<ModelId, ModelInstance>>, model_id: ModelId) -> Option<Self> {
        // SAFETY: We extend the lifetime of the reference to match the guard
        // This is safe because the guard keeps the lock held
        let model_ref = unsafe {
            let guard_ref: &HashMap<ModelId, ModelInstance> = &*(&*guard as *const _);
            match guard_ref.get(&model_id) {
                Some(ModelInstance::Llama3(model)) => Some(&*(model as *const Llama3)),
                _ => None,
            }
        };

        model_ref.map(|model_ref| Self {
            _guard: guard,
            model_ref: Some(model_ref),
        })
    }

}

impl<'a> std::ops::Deref for RwLockReadGuardMapper<'a> {
    type Target = Llama3;

    fn deref(&self) -> &Self::Target {
        self.model_ref.unwrap()
    }
}

// Helper struct for mapping RwLockWriteGuard to specific model type
struct RwLockWriteGuardMapper<'a> {
    _guard: RwLockWriteGuard<'a, HashMap<ModelId, ModelInstance>>,
    model_ref: Option<&'a mut Llama3>,
}

impl<'a> RwLockWriteGuardMapper<'a> {
    fn new(mut guard: RwLockWriteGuard<'a, HashMap<ModelId, ModelInstance>>, model_id: ModelId) -> Option<Self> {
        let model_ref = unsafe {
            let guard_ref: &mut HashMap<ModelId, ModelInstance> = &mut *(&mut *guard as *mut _);
            match guard_ref.get_mut(&model_id) {
                Some(ModelInstance::Llama3(model)) => Some(&mut *(model as *mut Llama3)),
                _ => None,
            }
        };

        model_ref.map(|model_ref| Self {
            _guard: guard,
            model_ref: Some(model_ref),
        })
    }

}

impl<'a> std::ops::Deref for RwLockWriteGuardMapper<'a> {
    type Target = Llama3;

    fn deref(&self) -> &Self::Target {
        self.model_ref.as_ref().unwrap()
    }
}

impl<'a> std::ops::DerefMut for RwLockWriteGuardMapper<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.model_ref.as_mut().unwrap()
    }
}

impl Clone for ModelRegistry {
    fn clone(&self) -> Self {
        Self {
            models: Arc::clone(&self.models),
            device_type: self.device_type,
        }
    }
}
