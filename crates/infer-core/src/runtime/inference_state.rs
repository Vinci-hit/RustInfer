use std::collections::HashMap;

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
#[cfg(feature = "cuda")]
use crate::cuda::CudaConfig;
use crate::model::config::RuntimeModelConfig;
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
            match sin_cache.device() {
                DeviceType::Cpu => {
                    crate::op::kernels::cpu::rope_sin_cos_cache_calc(
                        config.head_size,
                        config.seq_len,
                        config.rope_theta,
                        sin_cache,
                        cos_cache,
                    )?;
                }
                #[cfg(feature = "cuda")]
                DeviceType::Cuda(_) => {
                    let stream = CudaConfig::new()?;
                    crate::op::kernels::cuda::rope_sin_cos_cache_calc_cuda(
                        config.head_size,
                        config.seq_len,
                        config.rope_theta,
                        sin_cache,
                        cos_cache,
                        Some(&stream),
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

        let float_dtype = if device.is_cpu() {
            DataType::F32
        } else {
            match config.torch_dtype.as_str() {
                "float32" => DataType::F32,
                "bfloat16" => DataType::BF16,
                _ => {
                    return Err(Error::InvalidArgument(format!(
                        "Unsupported torch_dtype: {}", config.torch_dtype
                    )).into());
                }
            }
        };
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
