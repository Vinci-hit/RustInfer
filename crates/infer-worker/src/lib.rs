#![allow(
    clippy::too_many_arguments,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::mut_from_ref,
)]

pub mod base;
pub mod op;
pub mod tensor;

// Higher-level model and worker code is currently paused while we rebuild
// the tensor/stride foundation. Gate it behind `models` so the base
// infrastructure (tensor + kernels + cuda) can build and test alone.
#[cfg(feature = "models")]
pub mod model;
#[cfg(feature = "models")]
pub mod worker;
#[cfg(feature = "models")]
pub use model::runtime;

#[cfg(feature = "cuda")]
pub mod cuda;

/// 统一算子配置类型，所有算子接口使用 `Option<&OpConfig>` 作为参数。
/// - cuda 编译: `OpConfig = cuda::CudaConfig`（cublas/cudnn handle, stream 等）
/// - 纯 CPU 编译: `OpConfig = ()`（零开销）
/// CPU 路径传 `None` 即可。
#[cfg(feature = "cuda")]
pub type OpConfig = cuda::CudaConfig;
#[cfg(not(feature = "cuda"))]
pub type OpConfig = ();
