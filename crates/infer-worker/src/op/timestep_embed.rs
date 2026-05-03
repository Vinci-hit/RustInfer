//! Sinusoidal timestep embedding, CUDA-only.
//!
//! Used by diffusion pipelines (Z-Image DiT) during CUDA Graph capture: the
//! timestep `t` is pre-uploaded to a `[1]` F32 device tensor and the
//! embedding kernel reads it via pointer dereference. This lets the
//! captured graph be replayed across different `t` values without
//! re-capture.

use crate::base::DeviceType;
use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

use super::kernels;

/// Write a sinusoidal embedding into `out: [1, dim]`.
///
/// The scalar `t` comes from a `[1]` F32 **device** tensor (already scaled
/// by whatever `t_scale` factor the caller uses). `out.dtype()` selects
/// the precision at which the embedding is computed.
#[cfg(feature = "cuda")]
pub fn sinusoid_embedding_from_dev(out: &mut Tensor, d_t: &Tensor) -> Result<()> {
    match out.device() {
        DeviceType::Cuda(_) => kernels::cuda::sinusoid_embedding_from_dev(
            out, d_t, crate::cuda::get_current_cuda_stream(),
        ),
        other => Err(Error::InvalidArgument(format!(
            "sinusoid_embedding_from_dev: only CUDA is supported, got {:?}", other
        )).into()),
    }
}
