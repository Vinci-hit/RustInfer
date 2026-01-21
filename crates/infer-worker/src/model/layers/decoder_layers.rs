// src/model/layers/decoder_layers.rs
//
// Generic decoder-only transformer layers
// Used by Llama3, Qwen2, Mistral, and other decoder-only models

use crate::base::{DeviceType, DataType};
use crate::base::error::{Error, Result};
use crate::op::add_inplace::AddInplace;
use crate::op::embedding::Embedding;
use crate::op::flash_gqa::FlashAttnGQA;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::rope::RoPEOp;
use crate::op::swiglu::SwiGLU;
use crate::op::scatter::Scatter;
use crate::tensor::Tensor;
use super::super::{ModelLoader, config::RuntimeModelConfig};
use super::weight_mapping::WeightMapping;

/// Generic decoder layers for decoder-only transformer models
///
/// This structure holds all operators for a decoder-only transformer model.
/// It's designed to be reusable across different model architectures like
/// Llama3, Qwen2, Mistral, etc.
pub struct DecoderLayers {
    // ---- Token Embedding and Output ----
    pub embedding_layer: Embedding,
    pub rmsnorm_final_layer: RMSNorm,
    pub cls_layer: Matmul,

    // ---- Transformer Blocks (organized by operator type) ----
    pub rmsnorm_attn_layers: Vec<RMSNorm>,
    pub rmsnorm_ffn_layers: Vec<RMSNorm>,

    // Attention operators
    pub wq_layers: Vec<Matmul>,
    pub wk_layers: Vec<Matmul>,
    pub wv_layers: Vec<Matmul>,
    pub wo_layers: Vec<Matmul>,
    pub mha_layers: Vec<FlashAttnGQA>,
    pub rope_layers: Vec<RoPEOp>,

    // FFN operators
    pub w1_layers: Vec<Matmul>, // gate_proj
    pub w2_layers: Vec<Matmul>, // down_proj
    pub w3_layers: Vec<Matmul>, // up_proj
    pub swiglu_layers: Vec<SwiGLU>,

    // Common operators
    pub add_layers: AddInplace,
    pub scatter_layer: Scatter,
}

impl DecoderLayers {
    /// Create DecoderLayers from ModelLoader using weight mapping
    ///
    /// # Arguments
    /// * `loader` - ModelLoader containing the safetensor weights
    /// * `config` - Runtime model configuration
    /// * `weight_mapping` - Weight naming mapping for this model type
    /// * `device_type` - Target device (CPU or CUDA)
    /// * `is_quant_model` - Whether this is a quantized model
    ///
    /// # Returns
    /// Initialized DecoderLayers with all operators loaded
    pub fn from_loader(
        loader: &ModelLoader,
        config: &RuntimeModelConfig,
        weight_mapping: &WeightMapping,
        device_type: DeviceType,
        is_quant_model: bool,
        block_size: usize,
        num_total_blocks: usize,
    ) -> Result<Self> {
        if is_quant_model {
            return Err(Error::InvalidArgument(
                "Quantized models are not yet supported.".to_string()
            ).into());
        }

        println!("Creating decoder layers using weight mapping...");

        let layer_num = config.layer_num;
        let tensor_names: std::collections::HashSet<String> =
            loader.tensor_names().into_iter().collect();

        // ========== OPTIMIZATION: Async GPU transfer ==========
        // Create CUDA stream for async transfers (only for CUDA devices)
        #[cfg(feature = "cuda")]
        let stream = if device_type.is_cuda() {
            let mut stream_ptr: crate::cuda::ffi::cudaStream_t = std::ptr::null_mut();
            unsafe {
                crate::cuda_check!(crate::cuda::ffi::cudaStreamCreate(&mut stream_ptr))?;
            }
            println!("  -> Created CUDA stream for async weight transfer");
            Some(stream_ptr)
        } else {
            None
        };

        // Pre-allocate vectors for all layers
        let mut rmsnorm_attn_layers = Vec::with_capacity(layer_num);
        let mut rmsnorm_ffn_layers = Vec::with_capacity(layer_num);
        let mut wq_layers = Vec::with_capacity(layer_num);
        let mut wk_layers = Vec::with_capacity(layer_num);
        let mut wv_layers = Vec::with_capacity(layer_num);
        let mut wo_layers = Vec::with_capacity(layer_num);
        let mut w1_layers = Vec::with_capacity(layer_num);
        let mut w2_layers = Vec::with_capacity(layer_num);
        let mut w3_layers = Vec::with_capacity(layer_num);

        println!("  -> Loading {} layers ...", layer_num);

        // Load per-layer weights
        for i in 0..layer_num {
            // Attention weights
            #[cfg(feature = "cuda")]
            {
                wq_layers.push(Self::load_matmul_async(
                    &weight_mapping.format_layer_weight(i, weight_mapping.attn_q),
                    loader,
                    device_type,
                    stream
                )?);
                wk_layers.push(Self::load_matmul_async(
                    &weight_mapping.format_layer_weight(i, weight_mapping.attn_k),
                    loader,
                    device_type,
                    stream
                )?);
                wv_layers.push(Self::load_matmul_async(
                    &weight_mapping.format_layer_weight(i, weight_mapping.attn_v),
                    loader,
                    device_type,
                    stream
                )?);
                wo_layers.push(Self::load_matmul_async(
                    &weight_mapping.format_layer_weight(i, weight_mapping.attn_o),
                    loader,
                    device_type,
                    stream
                )?);

                // FFN weights
                w1_layers.push(Self::load_matmul_async(
                    &weight_mapping.format_layer_weight(i, weight_mapping.ffn_gate),
                    loader,
                    device_type,
                    stream
                )?);
                w3_layers.push(Self::load_matmul_async(
                    &weight_mapping.format_layer_weight(i, weight_mapping.ffn_up),
                    loader,
                    device_type,
                    stream
                )?);
                w2_layers.push(Self::load_matmul_async(
                    &weight_mapping.format_layer_weight(i, weight_mapping.ffn_down),
                    loader,
                    device_type,
                    stream
                )?);

                // Normalization layers
                rmsnorm_attn_layers.push(Self::load_rmsnorm_async(
                    &weight_mapping.format_layer_weight(i, weight_mapping.rmsnorm_attn),
                    loader,
                    device_type,
                    stream,
                    config.rms_norm_eps,
                )?);
                rmsnorm_ffn_layers.push(Self::load_rmsnorm_async(
                    &weight_mapping.format_layer_weight(i, weight_mapping.rmsnorm_ffn),
                    loader,
                    device_type,
                    stream,
                    config.rms_norm_eps,
                )?);
            }
            #[cfg(not(feature = "cuda"))]
            {
                // CPU fallback - use synchronous loading
                wq_layers.push(Self::load_matmul(
                    &weight_mapping.format_layer_weight(i, weight_mapping.attn_q),
                    loader,
                    device_type
                )?);
                wk_layers.push(Self::load_matmul(
                    &weight_mapping.format_layer_weight(i, weight_mapping.attn_k),
                    loader,
                    device_type
                )?);
                wv_layers.push(Self::load_matmul(
                    &weight_mapping.format_layer_weight(i, weight_mapping.attn_v),
                    loader,
                    device_type
                )?);
                wo_layers.push(Self::load_matmul(
                    &weight_mapping.format_layer_weight(i, weight_mapping.attn_o),
                    loader,
                    device_type
                )?);

                w1_layers.push(Self::load_matmul(
                    &weight_mapping.format_layer_weight(i, weight_mapping.ffn_gate),
                    loader,
                    device_type
                )?);
                w3_layers.push(Self::load_matmul(
                    &weight_mapping.format_layer_weight(i, weight_mapping.ffn_up),
                    loader,
                    device_type
                )?);
                w2_layers.push(Self::load_matmul(
                    &weight_mapping.format_layer_weight(i, weight_mapping.ffn_down),
                    loader,
                    device_type
                )?);

                rmsnorm_attn_layers.push(Self::load_rmsnorm(
                    &weight_mapping.format_layer_weight(i, weight_mapping.rmsnorm_attn),
                    loader,
                    device_type
                )?);
                rmsnorm_ffn_layers.push(Self::load_rmsnorm(
                    &weight_mapping.format_layer_weight(i, weight_mapping.rmsnorm_ffn),
                    loader,
                    device_type
                )?);
            }
        }

        // Load global layers
        #[cfg(feature = "cuda")]
        let (embedding_layer, rmsnorm_final_layer, cls_layer) = {
            let embedding_layer = Self::load_embedding_async(
                weight_mapping.embedding,
                loader,
                device_type,
                stream
            )?;
            let rmsnorm_final_layer = Self::load_rmsnorm_async(
                weight_mapping.rmsnorm_final,
                loader,
                device_type,
                stream,
                config.rms_norm_eps,
            )?;

            // Load or tie classification layer
            let cls_layer = if tensor_names.contains(weight_mapping.cls) {
                println!("Found independent '{}'. Loading it.", weight_mapping.cls);
                Self::load_matmul_async(weight_mapping.cls, loader, device_type, stream)?
            } else {
                println!("'{}' not found. Reusing token embedding weights (tied weights).",
                    weight_mapping.cls);
                Matmul::from(embedding_layer.weight.clone(), None)
            };

            // ========== SYNCHRONIZE STREAM ==========
            // All async transfers are queued, now synchronize
            if let Some(stream_ptr) = stream {
                println!("  -> Synchronizing CUDA stream (waiting for all {} weight transfers)...", layer_num * 9 + 2);
                unsafe {
                    crate::cuda_check!(crate::cuda::ffi::cudaStreamSynchronize(stream_ptr))?;
                    crate::cuda_check!(crate::cuda::ffi::cudaStreamDestroy(stream_ptr))?;
                }
                println!("  -> All async transfers completed!");
            }

            (embedding_layer, rmsnorm_final_layer, cls_layer)
        };

        #[cfg(not(feature = "cuda"))]
        let (embedding_layer, rmsnorm_final_layer, cls_layer) = {
            let embedding_layer = Self::load_embedding(
                weight_mapping.embedding,
                loader,
                device_type
            )?;
            let rmsnorm_final_layer = Self::load_rmsnorm(
                weight_mapping.rmsnorm_final,
                loader,
                device_type
            )?;

            let cls_layer = if tensor_names.contains(weight_mapping.cls) {
                println!("Found independent '{}'. Loading it.", weight_mapping.cls);
                Self::load_matmul(weight_mapping.cls, loader, device_type)?
            } else {
                println!("'{}' not found. Reusing token embedding weights (tied weights).",
                    weight_mapping.cls);
                Matmul::from(embedding_layer.weight.clone(), None)
            };

            (embedding_layer, rmsnorm_final_layer, cls_layer)
        };

        // Create parameterless operators
        let mha_layers: Result<Vec<FlashAttnGQA>> = (0..layer_num)
            .map(|_| {
                FlashAttnGQA::new_paged(config.head_num, config.kv_head_num, config.head_size, block_size, num_total_blocks)
            })
            .collect();
        let mha_layers = mha_layers?;

        let rope_layers: Result<Vec<RoPEOp>> = (0..layer_num)
            .map(|_| RoPEOp::new(config.dim, config.kv_dim, config.head_size))
            .collect();
        let rope_layers = rope_layers?;

        let add_layers = AddInplace::new();
        let swiglu_layers: Vec<SwiGLU> = (0..layer_num)
            .map(|_| SwiGLU::new())
            .collect();
        let scatter_layer = Scatter::new();

        // Validation checks
        if rmsnorm_attn_layers.len() != layer_num || rmsnorm_ffn_layers.len() != layer_num {
            return Err(Error::InternalError(
                "Incorrect number of RMSNorm layers created.".to_string()
            ).into());
        }

        if wq_layers.len() != layer_num || wk_layers.len() != layer_num ||
           wv_layers.len() != layer_num || wo_layers.len() != layer_num {
            return Err(Error::InternalError(
                "Incorrect number of attention Matmul layers created.".to_string()
            ).into());
        }

        if w1_layers.len() != layer_num || w2_layers.len() != layer_num ||
           w3_layers.len() != layer_num {
            return Err(Error::InternalError(
                "Incorrect number of FFN Matmul layers created.".to_string()
            ).into());
        }

        if mha_layers.len() != layer_num || rope_layers.len() != layer_num ||
           swiglu_layers.len() != layer_num {
            return Err(Error::InternalError(
                "Incorrect number of non-parameterized layers created.".to_string()
            ).into());
        }

        println!("All decoder layers created and checked successfully.");

        Ok(Self {
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
            scatter_layer,
            w1_layers,
            w2_layers,
            w3_layers,
            swiglu_layers,
        })
    }

    /// Move all layers to CUDA device
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        // Move global layers
        self.embedding_layer.to_cuda(device_id)?;
        self.rmsnorm_final_layer.to_cuda(device_id)?;
        self.cls_layer.to_cuda(device_id)?;

        // Move per-layer operators
        self.rmsnorm_attn_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.rmsnorm_ffn_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;

        self.wq_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.wk_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.wv_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.wo_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;

        self.w1_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.w2_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.w3_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;

        self.mha_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.rope_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;
        self.swiglu_layers.iter_mut().try_for_each(|layer| layer.to_cuda(device_id))?;

        self.add_layers.to_cuda(device_id)?;

        println!("All decoder layers successfully moved to CUDA device {}.", device_id);
        Ok(())
    }

    fn load_matmul_async(
        name: &str,
        loader: &ModelLoader,
        device: DeviceType,
        stream: Option<crate::cuda::ffi::cudaStream_t>
    ) -> Result<Matmul> {
        let tensor_view = loader.get_tensor(name)?;
        let weight = Tensor::from_view_on_cpu(&tensor_view)?;

        // Convert to F32 for CPU device
        let weight_converted = if device.is_cpu() && weight.dtype() != DataType::F32 {
            weight.to_dtype(DataType::F32)?
        } else {
            weight
        };

        let weight_on_device = weight_converted.to_device_async(device, stream)?;
        Ok(Matmul::from(weight_on_device, None))
    }

    fn load_rmsnorm_async(
        name: &str,
        loader: &ModelLoader,
        device: DeviceType,
        stream: Option<crate::cuda::ffi::cudaStream_t>,
        rms_norm_eps: f32,
    ) -> Result<RMSNorm> {
        let tensor_view = loader.get_tensor(name)?;
        let weight = Tensor::from_view_on_cpu(&tensor_view)?;

        // Convert to F32 for CPU device
        let weight_converted = if device.is_cpu() && weight.dtype() != DataType::F32 {
            weight.to_dtype(DataType::F32)?
        } else {
            weight
        };

        let weight_on_device = weight_converted.to_device_async(device, stream)?;
        Ok(RMSNorm::from(weight_on_device,rms_norm_eps))
    }

    fn load_embedding_async(
        name: &str,
        loader: &ModelLoader,
        device: DeviceType,
        stream: Option<crate::cuda::ffi::cudaStream_t>
    ) -> Result<Embedding> {
        let tensor_view = loader.get_tensor(name)?;
        let weight = Tensor::from_view_on_cpu(&tensor_view)?;

        // Convert to F32 for CPU device
        let weight_converted = if device.is_cpu() && weight.dtype() != DataType::F32 {
            weight.to_dtype(DataType::F32)?
        } else {
            weight
        };

        let weight_on_device = weight_converted.to_device_async(device, stream)?;
        Ok(Embedding::from(weight_on_device))
    }
}
