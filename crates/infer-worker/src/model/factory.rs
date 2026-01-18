// src/model/factory.rs
//
// Model factory for creating different model types with auto-detection

use std::path::Path;
use serde_json::Value;

use crate::base::DeviceType;
use crate::base::error::{Error, Result};
use super::llama3::Llama3;

/// Supported model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Llama3 and compatible models (Llama2, etc.)
    Llama3,
    /// Qwen2 models
    // Qwen2,
    /// Mistral models (uses Llama architecture)
    Mistral,
    // Future model types:
    // Qwen3VL,
    // MixtralMoE,
    // SD3,
}

/// Model factory for creating model instances
pub struct ModelFactory;

impl ModelFactory {
    /// Create a model instance from explicit model type
    ///
    /// # Arguments
    /// * `model_type` - The type of model to create
    /// * `model_dir` - Path to model directory
    /// * `device_type` - Target device (CPU or CUDA)
    /// * `is_quant_model` - Whether this is a quantized model
    ///
    /// # Returns
    /// Boxed model instance implementing the Model trait (future)
    ///
    /// # Example
    /// ```
    /// let model = ModelFactory::create(
    ///     ModelType::Qwen2,
    ///     Path::new("/models/qwen2-1.5b"),
    ///     DeviceType::Cuda(0),
    ///     false
    /// )?;
    /// ```
    pub fn create_llama3(
        model_dir: &Path,
        device_type: DeviceType,
        is_quant_model: bool,
    ) -> Result<Llama3> {
        Llama3::new(model_dir, device_type, is_quant_model)
    }

    /// Auto-detect model type from config.json
    ///
    /// Reads the model's config.json and determines the model type from:
    /// 1. "model_type" field (if present)
    /// 2. "architectures" field (if present)
    ///
    /// # Arguments
    /// * `model_dir` - Path to model directory containing config.json
    ///
    /// # Returns
    /// Detected ModelType
    ///
    /// # Errors
    /// Returns error if:
    /// - config.json cannot be read
    /// - Model type cannot be determined
    /// - Unknown/unsupported model type
    ///
    /// # Example
    /// ```
    /// let model_type = ModelFactory::detect_model_type(Path::new("/models/qwen2-1.5b"))?;
    /// println!("Detected model type: {:?}", model_type);
    /// ```
    pub fn detect_model_type(model_dir: &Path) -> Result<ModelType> {
        let config_path = model_dir.join("config.json");

        let config_file = std::fs::File::open(&config_path)
            .map_err(|e| Error::InvalidArgument(
                format!("Failed to open config.json at {:?}: {}", config_path, e)
            ))?;

        let config: Value = serde_json::from_reader(config_file)
            .map_err(|e| Error::InvalidArgument(
                format!("Failed to parse config.json: {}", e)
            ))?;

        // Strategy 1: Check "model_type" field
        if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
            return match model_type {
                "llama" => Ok(ModelType::Llama3),
                "mistral" => Ok(ModelType::Mistral),
                _ => Err(Error::InvalidArgument(
                    format!("Unknown model_type in config.json: {}", model_type)
                ).into())
            };
        }

        // Strategy 2: Check "architectures" field
        if let Some(architectures) = config.get("architectures").and_then(|v| v.as_array()) {
            if let Some(arch) = architectures.first().and_then(|v| v.as_str()) {
                return match arch {
                    "LlamaForCausalLM" => Ok(ModelType::Llama3),
                    "MistralForCausalLM" => Ok(ModelType::Mistral),
                    _ => Err(Error::InvalidArgument(
                        format!("Unknown architecture in config.json: {}", arch)
                    ).into())
                };
            }
        }

        // If neither strategy works, return error
        Err(Error::InvalidArgument(
            "Cannot determine model type from config.json. \
             Expected 'model_type' or 'architectures' field.".to_string()
        ).into())
    }

    /// Create model with automatic type detection
    ///
    /// Convenience method that auto-detects model type and creates the appropriate model.
    ///
    /// # Arguments
    /// * `model_dir` - Path to model directory
    /// * `device_type` - Target device (CPU or CUDA)
    /// * `is_quant_model` - Whether this is a quantized model
    ///
    /// # Returns
    /// Created model type enum
    ///
    /// # Example
    /// ```
    /// let (model_type, _) = ModelFactory::create_auto(
    ///     Path::new("/models/unknown-model"),
    ///     DeviceType::Cuda(0),
    ///     false
    /// )?;
    /// println!("Created model of type: {:?}", model_type);
    /// ```
    pub fn create_auto(
        model_dir: &Path,
        _device_type: DeviceType,
        _is_quant_model: bool,
    ) -> Result<ModelType> {
        let model_type = Self::detect_model_type(model_dir)?;
        println!("Auto-detected model type: {:?}", model_type);
        Ok(model_type)
    }
}
