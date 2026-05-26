//! CUDA kernel wrappers — adapted to new generic Tensor<T, Cuda>.

pub mod rmsnorm;
pub mod add;
pub mod matmul;
pub mod softmax;
pub mod activation;
pub mod scalar;
pub mod embedding;
pub mod rope;
pub mod attention;
pub mod fused_add_rmsnorm;
pub mod sampler;
pub mod split_cols;
pub mod scatter_kv;
pub mod groupnorm;
pub mod upsample;
pub mod broadcast_mul;
pub mod ewise_mul;
pub mod layernorm;
pub mod conv2d;
