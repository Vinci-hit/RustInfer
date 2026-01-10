use crate::base::error::Result;
use crate::base::DeviceType;
use crate::op::{kernels, Op, OpContext};

/// Scatter operator: copies src[0, :] to dst[pos, :]
///
/// This is used for KV cache updates in the decoding phase where seq_len == 1.
/// The source tensor has shape [1, kvdim] and needs to be written to a specific
/// position in the destination tensor which has shape [max_seq_len, kvdim].
///
/// # Context
/// * `ctx.inputs[0]` (src): Source tensor with shape [1, kvdim]
/// * `ctx.inputs[1]` (pos): Position tensor (I32 scalar) indicating where to write in dst
/// * `ctx.outputs[0]` (dst): Destination tensor with shape [max_seq_len, kvdim]
#[derive(Debug, Clone, Copy)]
pub struct Scatter;

impl Scatter {
    /// Creates a new Scatter operator instance.
    pub fn new() -> Self {
        Scatter
    }
}

impl Default for Scatter {
    fn default() -> Self {
        Self::new()
    }
}

impl Op for Scatter {
    fn name(&self) -> &'static str {
        "Scatter"
    }

    fn forward(&self, ctx: &mut OpContext) -> Result<()> {
        if ctx.inputs.len() != 2 {
            return Err(anyhow::anyhow!(
                "Scatter requires exactly 2 inputs (src, pos), got {}",
                ctx.inputs.len()
            ));
        }

        if ctx.outputs.len() != 1 {
            return Err(anyhow::anyhow!(
                "Scatter requires exactly 1 output (dst), got {}",
                ctx.outputs.len()
            ));
        }

        let src = ctx.inputs[0];
        let pos = ctx.inputs[1];
        let dst = &mut ctx.outputs[0];

        // Check device compatibility
        if src.device() != dst.device() {
            return Err(anyhow::anyhow!(
                "Device mismatch for Scatter: src={:?}, dst={:?}",
                src.device(), dst.device()
            ));
        }

        // Dispatch to kernel based on device
        match src.device() {
            DeviceType::Cpu => {
                return Err(anyhow::anyhow!(
                    "Scatter operator is only supported on CUDA devices"
                ));
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                kernels::cuda::scatter(dst, src, pos, ctx.cuda_config)?;
            }
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Scatter {
    /// Scatter is stateless; nothing to move to CUDA.
    pub fn to_cuda(&mut self, _device_id: i32) -> Result<()> {
        Ok(())
    }
}
