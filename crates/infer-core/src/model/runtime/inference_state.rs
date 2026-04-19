use std::collections::HashMap;

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
#[cfg(feature = "cuda")]
use crate::cuda::CudaConfig;
use crate::model::common::config::RuntimeModelConfig;
use crate::model::{BufferType, Workspace};
use crate::op::sampler::{ArgmaxSampler, Sampler};
use crate::tensor::Tensor;

use super::KvCache;

pub struct InferenceState {
    pub kv_cache: KvCache,
    pub workspace: Workspace,
    pub output_token: Tensor,
    pub input_pos: Tensor,
    pub sampler: Box<dyn Sampler>,
    #[cfg(feature = "cuda")]
    pub cuda_config: Option<CudaConfig>,
}

impl InferenceState {
    pub fn new(config: &RuntimeModelConfig, device_type: DeviceType) -> Result<Self> {
        let kv_cache = KvCache::new(config, &device_type)?;
        let mut workspace = Self::init_workspace(config, &device_type)?;
        let sampler = Box::new(ArgmaxSampler::new(device_type));
        let output_token = Tensor::new(&[1], DataType::I32, device_type)?;
        let input_pos = Tensor::new(&[1], DataType::I32, device_type)?;

        Self::calculate_rope_cache(config, &mut workspace)?;

        #[cfg(feature = "cuda")]
        let cuda_config = {
            let cfg = CudaConfig::new()?;
            Some(cfg)
        };

        Ok(Self {
            kv_cache,
            workspace,
            output_token,
            input_pos,
            sampler,
            #[cfg(feature = "cuda")]
            cuda_config,
        })
    }

    fn calculate_rope_cache(
        config: &RuntimeModelConfig,
        workspace: &mut Workspace,
    ) -> Result<()> {
        let caches = workspace.get_disjoint_mut([
            &BufferType::SinCache,
            &BufferType::CosCache,
        ]);

        if let [Some(sin_cache), Some(cos_cache)] = caches {
            let target_device = sin_cache.device();
            let target_dtype = sin_cache.dtype();
            let head_size = config.head_size;
            let max_seq_len = config.seq_len;

            // 1. Compute inv_freq with optional llama3 scaling (always on CPU in f32)
            let half_head = head_size / 2;
            let rope_theta = config.rope_theta as f64;
            let mut inv_freq = vec![0.0f64; half_head];
            for k in 0..half_head {
                let dim = 2 * k;
                let exponent = dim as f64 / head_size as f64;
                inv_freq[k] = 1.0 / rope_theta.powf(exponent);
            }

            // Apply llama3 rope scaling if configured
            if let Some(ref scaling) = config.rope_scaling {
                if scaling.rope_type == "llama3" {
                    let factor = scaling.factor;
                    let low_freq_factor = scaling.low_freq_factor.unwrap_or(1.0);
                    let high_freq_factor = scaling.high_freq_factor.unwrap_or(4.0);
                    let old_context_len = scaling.original_max_position_embeddings.unwrap_or(8192) as f64;

                    let low_freq_wavelen = old_context_len / low_freq_factor;
                    let high_freq_wavelen = old_context_len / high_freq_factor;

                    for k in 0..half_head {
                        let freq = inv_freq[k];
                        let wavelen = 2.0 * std::f64::consts::PI / freq;

                        if wavelen > low_freq_wavelen {
                            // Low frequency: divide by factor
                            inv_freq[k] = freq / factor;
                        } else if wavelen >= high_freq_wavelen {
                            // Medium frequency: smooth interpolation
                            let smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                            let scaled = freq / factor;
                            inv_freq[k] = (1.0 - smooth) * scaled / factor + smooth * scaled;
                            // Actually the HF implementation is:
                            // inv_freq_llama = where(wavelen > low_freq_wavelen, freq / factor, freq)
                            // smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                            // smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
                            // But inv_freq_llama at this point (medium freq) equals freq (original, not divided)
                            // because wavelen is NOT > low_freq_wavelen for medium freq
                            inv_freq[k] = (1.0 - smooth) * freq / factor + smooth * freq;
                        }
                        // else: high frequency (wavelen < high_freq_wavelen): keep original
                    }
                }
            }

            // 2. Compute sin/cos cache on CPU in F32
            //    The BF16/FP16 CUDA kernels use [pos * half_head + k] layout
            //    But workspace allocates [max_seq_len, head_size] — we fill half_head per row
            //    We create matching-sized tensors and fill the first half_head columns per row
            let cache_cols = half_head;  // The BF16/FP16 kernels only use half_head columns
            let total = max_seq_len * cache_cols;
            let mut sin_f32 = vec![0.0f32; total];
            let mut cos_f32 = vec![0.0f32; total];

            for pos in 0..max_seq_len {
                for k in 0..half_head {
                    let val = pos as f64 * inv_freq[k];
                    let idx = pos * cache_cols + k;
                    sin_f32[idx] = val.sin() as f32;
                    cos_f32[idx] = val.cos() as f32;
                }
            }

            // 3. Write to target cache (may be CPU or CUDA, F32/BF16/FP16)
            //    The target tensor has shape [max_seq_len, head_size] but we only
            //    fill the first half_head elements per row. For BF16/FP16 on CUDA,
            //    we write directly into the GPU buffer which the kernel reads as
            //    [pos * half_head + k] — the allocation is larger but contiguous
            //    memory works because the kernel indexes as pos * half_head + k
            //    and the total data fits in max_seq_len * half_head elements.
            match (target_device, target_dtype) {
                (DeviceType::Cpu, DataType::F32) => {
                    // CPU F32 kernel uses [pos * head_size + dim] layout (full head_size)
                    // Recalculate with the correct layout
                    // Actually let's just fill it properly for F32 too with scaling
                    let sin_slice = sin_cache.as_f32_mut()?.as_slice_mut()?;
                    let cos_slice = cos_cache.as_f32_mut()?.as_slice_mut()?;
                    for pos in 0..max_seq_len {
                        for k in 0..half_head {
                            let val = pos as f64 * inv_freq[k];
                            let idx = pos * head_size + k;  // F32 uses head_size stride
                            sin_slice[idx] = val.sin() as f32;
                            cos_slice[idx] = val.cos() as f32;
                        }
                    }
                }
                #[cfg(feature = "cuda")]
                (DeviceType::Cuda(_), _) => {
                    // Write directly into GPU memory. The GPU tensor has
                    // max_seq_len * head_size elements but the BF16/FP16 kernel
                    // only uses the first max_seq_len * half_head elements (packed).
                    match target_dtype {
                        DataType::F16 => {
                            let data: Vec<half::f16> = sin_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
                            sin_cache.as_f16_mut()?.buffer_mut().copy_from_host(&data)?;
                            let data: Vec<half::f16> = cos_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
                            cos_cache.as_f16_mut()?.buffer_mut().copy_from_host(&data)?;
                        }
                        DataType::BF16 => {
                            let data: Vec<half::bf16> = sin_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
                            sin_cache.as_bf16_mut()?.buffer_mut().copy_from_host(&data)?;
                            let data: Vec<half::bf16> = cos_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
                            cos_cache.as_bf16_mut()?.buffer_mut().copy_from_host(&data)?;
                        }
                        DataType::F32 => {
                            sin_cache.as_f32_mut()?.buffer_mut().copy_from_host(&sin_f32)?;
                            cos_cache.as_f32_mut()?.buffer_mut().copy_from_host(&cos_f32)?;
                        }
                        _ => return Err(Error::InvalidArgument(format!("Unsupported dtype for RoPE cache: {:?}", target_dtype)).into()),
                    }
                }
                _ => {
                    // Fallback: use original CPU calculation (no scaling)
                    crate::op::kernels::cpu::rope_sin_cos_cache_calc(
                        config.head_size,
                        config.seq_len,
                        config.rope_theta,
                        sin_cache,
                        cos_cache,
                    )?;
                }
            }
            Ok(())
        } else {
            Err(Error::InternalError("SinCache or CosCache not found in workspace".to_string()).into())
        }
    }

    fn init_workspace(
        config: &RuntimeModelConfig,
        device: &DeviceType,
    ) -> Result<Workspace> {
        let mut buffers = HashMap::new();

        let float_dtype = config.runtime_float_dtype(*device)?;
        let int_dtype = DataType::I32;
        let max_seq_len = config.seq_len;

        buffers.insert(BufferType::InputTokens, Tensor::new(&[max_seq_len], int_dtype, *device)?);
        buffers.insert(BufferType::InputPos, Tensor::new(&[1], int_dtype, DeviceType::Cpu)?);
        buffers.insert(BufferType::InputEmbeddings, Tensor::new(&[max_seq_len, config.dim], float_dtype, *device)?);
        buffers.insert(BufferType::SinCache, Tensor::new(&[max_seq_len, config.head_size], float_dtype, *device)?);
        buffers.insert(BufferType::CosCache, Tensor::new(&[max_seq_len, config.head_size], float_dtype, *device)?);
        buffers.insert(BufferType::RmsOutput, Tensor::new(&[max_seq_len, config.dim], float_dtype, *device)?);
        buffers.insert(BufferType::Query, Tensor::new(&[max_seq_len, config.dim], float_dtype, *device)?);
        buffers.insert(BufferType::W1Output, Tensor::new(&[max_seq_len, config.intermediate_size], float_dtype, *device)?);
        buffers.insert(BufferType::W3Output, Tensor::new(&[max_seq_len, config.intermediate_size], float_dtype, *device)?);
        buffers.insert(BufferType::AttnScores, Tensor::new(&[config.head_num, max_seq_len, max_seq_len], float_dtype, *device)?);
        buffers.insert(BufferType::KeyCache, Tensor::new(&[max_seq_len, config.kv_dim], float_dtype, *device)?);
        buffers.insert(BufferType::ValueCache, Tensor::new(&[max_seq_len, config.kv_dim], float_dtype, *device)?);
        buffers.insert(BufferType::IntermediateBuffer1, Tensor::new(&[max_seq_len, config.dim], float_dtype, *device)?);
        buffers.insert(BufferType::QkvOutput, Tensor::new(&[max_seq_len, config.q_dim + 2 * config.kv_dim], float_dtype, *device)?);
        buffers.insert(BufferType::GateUpOutput, Tensor::new(&[max_seq_len, 2 * config.intermediate_size], float_dtype, *device)?);
        buffers.insert(BufferType::ForwardOutput, Tensor::new(&[config.vocab_size], float_dtype, *device)?);

        Ok(buffers)
    }
}
