//! 面向 worker runner 的最小 LLM 抽象。

use crate::base::error::Result;
use crate::model::common::config::RuntimeModelConfig;
use crate::model::common::tokenizer::Tokenizer;
use crate::model::runtime::InferenceState;
use crate::tensor::Tensor;
use crate::worker::batch_workspace::BatchWorkspace;
use crate::worker::runner::WorkerBatchMeta;

pub trait LlmModel: Send + Sync {
    fn config(&self) -> &RuntimeModelConfig;

    fn tokenizer(&self) -> &dyn Tokenizer;

    fn create_state(&self) -> Result<InferenceState>;

    /// Worker 唯一 forward 入口。
    ///
    /// 输入 token/position 已由 Server/Runner 写到 device workspace；模型只负责计算，
    /// 并把每个 seq 的采样结果写入唯一的 `output_tokens[0..num_seqs]`。
    fn forward(
        &self,
        states: &mut [&mut InferenceState],
        workspace: &mut BatchWorkspace,
        batch: &WorkerBatchMeta<'_>,
        output_tokens: &mut Tensor,
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<()>;

    fn fill_rope_cache(&self, dst_sin: &mut Tensor, dst_cos: &mut Tensor) -> Result<()>;
}
