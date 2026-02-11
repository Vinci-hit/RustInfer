use crate::{base::error::Result, cuda::config::CudaConfig};
use super::tensor::Tensor;
pub mod rmsnorm;
pub mod kernels;
pub mod matmul;
pub mod add;
pub mod add_inplace;
pub mod swiglu;
pub mod embedding;
pub mod flash_gqa;
pub mod rope;
pub mod encode;
pub mod sampler;
pub mod scatter;
pub mod attn_backend;

/// OpContext 用于向 forward 方法传递可变数量的输入和输出。
pub struct OpContext<'a> {
    pub inputs: &'a [&'a Tensor],
    pub outputs: &'a mut [&'a mut Tensor],
    #[cfg(feature = "cuda")]
    pub cuda_config: Option<&'a CudaConfig>,
    #[cfg(feature = "cuda")]
    pub attn_backend: Option<&'a attn_backend::AttentionBackend>,
}

impl<'a> OpContext<'a> {
    pub fn new(
        inputs: &'a [&'a Tensor],
        outputs: &'a mut [&'a mut Tensor],
        cuda_config: Option<&'a CudaConfig>
    ) -> Self {
        Self {
            inputs,
            outputs,
            cuda_config,
            #[cfg(feature = "cuda")]
            attn_backend: None,
        }
    }
    // 链式调用，用于添加 CudaConfig
    pub fn with_cuda_config(mut self, config: Option<&'a CudaConfig>) -> Self {
        self.cuda_config = config;
        self
    }
}

pub trait Op {
    /// 返回算子的名字，用于调试和日志。
    fn name(&self) -> &'static str;

    /// 执行前向计算。
    /// 这是所有算子最核心的逻辑。
    fn forward(&self, ctx: &mut OpContext) -> Result<()>;

    #[cfg(feature = "cuda")]
    fn to_cuda_(&self, _cuda_config: i32) -> Result<()> {
        unimplemented!("未实现！");
    }
}
