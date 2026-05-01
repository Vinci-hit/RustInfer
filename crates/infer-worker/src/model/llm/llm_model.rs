//! 面向 scheduler / worker runner 的最小 LLM 抽象。
//!
//! 新增模型时实现这个 trait，`ModelRunner` 就可以直接驱动它。

use crate::base::error::Result;
use crate::model::common::config::RuntimeModelConfig;
use crate::model::common::tokenizer::Tokenizer;
use crate::model::runtime::InferenceState;
use crate::worker::batch_workspace::BatchWorkspace;

/// 一个 LLM 需要提供的三件事：prefill、单步 decode、batched decode。
///
/// prefill / decoding 的采样结果（一个 i32 token）直接通过返回值给出；
/// batch decode 额外通过 `BatchWorkspace` 做跨 seq 的 scratch buffer。
pub trait LlmModel: Send + Sync {
    /// 模型运行期配置（包含 dim / head_size / seq_len 等）。
    fn config(&self) -> &RuntimeModelConfig;

    /// Tokenizer 引用（scheduler 做 encode/decode 用）。
    fn tokenizer(&self) -> &dyn Tokenizer;

    /// 为一个新会话创建一份可变的推理状态（KV cache + per-layer workspace + sampler）。
    fn create_state(&self) -> Result<InferenceState>;

    /// 把 host 侧 `tokens[0..seq_len]` 投喂模型，返回采样出的下一个 token。
    ///
    /// `start_pos` 是 `tokens[0]` 对应 KV cache 的绝对位置（支持 continuation）。
    fn forward_prefill(
        &self,
        state: &mut InferenceState,
        tokens: &[i32],
        start_pos: i32,
        seq_len: usize,
    ) -> Result<i32>;

    /// 单步 decode（B=1 快路径）。input token 隐含从 `state.output_token` 读取。
    fn forward_decoding(&self, state: &mut InferenceState, pos: i32) -> Result<i32>;

    /// Batched decode（B≥1）。`positions[i]` 是 `states[i]` 本步的绝对 pos。
    fn forward_batch_decode(
        &self,
        states: &mut [&mut InferenceState],
        workspace: &mut BatchWorkspace,
        positions: &[i32],
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<Vec<i32>>;

    /// Fill `sin_cache` / `cos_cache` tensors with this model's RoPE basis.
    /// `dst_sin`/`dst_cos` shape = `[max_seq_len, head_size]`.
    ///
    /// Default: 要求模型方自己实现；无合理默认。
    fn fill_rope_cache(&self, dst_sin: &mut crate::tensor::Tensor, dst_cos: &mut crate::tensor::Tensor) -> Result<()>;
}
