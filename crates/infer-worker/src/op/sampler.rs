// src/op/sampler.rs
//
// Batch sampling operator for token generation
//
// Design:
// - Input: logits tensor [batch_size, vocab_size] (on GPU, BF16)
// - Output: token IDs tensor [batch_size] (on GPU, I32, supports CUDA Graph)
// - Server copies output to CPU for return value
// - GPU tensor can be reused for next batch (fixed memory address)

use crate::tensor::Tensor;
use crate::base::error::{Result, Error};
use crate::base::DeviceType;
use crate::op::{kernels, Op, OpContext};
use crate::cuda::CudaConfig;

/// Batch sampler trait for different sampling strategies
pub trait Sampler: Send + Sync {
    /// Sample next tokens from logits
    ///
    /// # Arguments
    /// * `logits` - Input logits [batch_size, vocab_size] (BF16 on GPU)
    /// * `output_tokens` - Output token IDs [batch_size], must be on GPU
    /// * `cuda_config` - CUDA configuration
    fn sample(
        &self,
        logits: &Tensor,
        output_tokens: &mut Tensor,
        cuda_config: Option<&CudaConfig>,
    ) -> Result<()>;
}

// ================ Argmax Sampler (Greedy Decoding) ================

pub struct ArgmaxSampler {
    device_type: DeviceType,
}

impl ArgmaxSampler {
    pub fn new(device_type: DeviceType) -> Self {
        Self { device_type }
    }
}

impl Sampler for ArgmaxSampler {
    fn sample(
        &self,
        logits: &Tensor,
        output_tokens: &mut Tensor,
        cuda_config: Option<&CudaConfig>,
    ) -> Result<()> {
        // ---- Validation ----
        if logits.shape().len() != 2 {
            return Err(Error::InvalidArgument(format!(
                "Input logits must be 2D [batch_size, vocab_size], but got shape {:?}",
                logits.shape()
            )).into());
        }

        let batch_size = logits.shape()[0];
        let output_size = output_tokens.shape()[0];

        // Output tensor must be at least batch_size (supports pre-allocated larger buffers)
        if output_size < batch_size {
            return Err(Error::InvalidArgument(format!(
                "Output tensor size {} must be >= batch_size {}, got shape {:?}",
                output_size, batch_size,
                output_tokens.shape()
            )).into());
        }

        if output_tokens.dtype() != crate::base::DataType::I32 {
            return Err(Error::InvalidArgument(format!(
                "Output must be I32, got {:?}",
                output_tokens.dtype()
            )).into());
        }

        if logits.device() != self.device_type {
            return Err(Error::DeviceMismatch {
                expected: self.device_type,
                actual: logits.device(),
                in_method: "ArgmaxSampler::sample".to_string(),
            }.into());
        }

        // ---- Dispatch to CUDA only (BF16) ----
        match self.device_type {
            DeviceType::Cpu => {
                unimplemented!("CPU sampling not supported, use CUDA")
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                kernels::cuda::argmax_batch(logits, output_tokens, cuda_config)?;
            }
        }

        Ok(())
    }
}

// ================ SamplerOp (Operator Wrapper) ================

/// Batch sampling operator
///
/// Wraps a Sampler implementation for use in computation graphs.
/// Input: logits [batch_size, vocab_size] (BF16 on GPU)
/// Output: token IDs [batch_size] (I32 on GPU, supports CUDA Graph fusion)
///
/// Note: Output tensor is pre-allocated on GPU and reused across batches to avoid
/// repeated allocations. Server copies result to CPU after each batch.
pub struct SamplerOp {
    sampler: Box<dyn Sampler>,
}

impl SamplerOp {
    pub fn new(sampler: Box<dyn Sampler>) -> Self {
        Self { sampler }
    }
}

impl Op for SamplerOp {
    fn name(&self) -> &'static str {
        "SamplerOp"
    }

    fn forward(&self, ctx: &mut OpContext) -> Result<()> {
        if ctx.inputs.len() != 1 || ctx.outputs.len() != 1 {
            return Err(Error::InvalidArgument(
                "SamplerOp expects 1 input (logits [batch_size, vocab_size]) and 1 output (token_ids [batch_size])".into(),
            ).into());
        }

        let logits = &ctx.inputs[0];
        let output_tokens = &mut ctx.outputs[0];

        self.sampler.sample(logits, output_tokens, ctx.cuda_config)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_sampler_creation() {
        #[cfg(feature = "cuda")]
        {
            let sampler = ArgmaxSampler::new(DeviceType::Cuda(0));
            assert_eq!(sampler.device_type, DeviceType::Cuda(0));
        }
    }
}

