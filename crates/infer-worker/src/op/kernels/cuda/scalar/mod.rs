use crate::base::error::Result;
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::ffi::cudaStream_t;

/// dst[i] = src[i] * val  (CUDA)
#[cfg(feature = "cuda")]
pub fn scalar_mul(src: &Tensor, dst: &mut Tensor, val: f32, _stream: cudaStream_t) -> Result<()> {
    todo!("CUDA scalar_mul kernel — implement in P2")
}

/// dst[i] = src[i] + val  (CUDA)
#[cfg(feature = "cuda")]
pub fn scalar_add(src: &Tensor, dst: &mut Tensor, val: f32, _stream: cudaStream_t) -> Result<()> {
    todo!("CUDA scalar_add kernel — implement in P2")
}
