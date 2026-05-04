//! 面向 worker runner 的最小 LLM 抽象。
//!
//! 职责纯粹：只暴露 token 级的 `forward(ctx)`。prompt / 文本 / 采样循环等属于
//! 调度器和网关层的事，本 crate 不参与。手动测试所需的"prompt → forward 循环
//! → 文本" helper 仅存在于各模型文件的 `#[cfg(test)]` 模块内，不会漏到生产
//! 代码。

pub mod llama3;
// pub mod qwen3;

use crate::base::DeviceType;
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

    fn device_type(&self) -> DeviceType;

    /// 默认实现：直接用 `config` + `device_type` 构造一个标准 `InferenceState`。
    /// 如果某个模型需要额外的 workspace buffer（例如 Qwen3 的 QK-norm），
    /// 请 override 本方法。
    fn create_state(&self) -> Result<InferenceState> {
        InferenceState::new(self.config(), self.device_type())
    }

    /// Worker 唯一 forward 入口 —— 只吃 `ForwardCtx`。
    ///
    /// 契约：
    /// - `ctx` 由 runner / scheduler 在 step 入口构造，里面的 buffer views、
    ///   `attn_plan`、`output_tokens` 等一切必需信息都**已就绪**；
    /// - 模型只负责按 ctx 计算，把采样得到的 token 写到 `ctx.output_tokens`；
    /// - 模型**不做**任何参数校验、KV 容量扩容、Graph 调度或 stream 管理。
    fn forward(&self, ctx: &mut ForwardCtx<'_, '_>) -> Result<()>;

    /// 默认实现：绝大多数 LLM 使用标准 RoPE cache（基于 `config`）。
    fn fill_rope_cache(&self, dst_sin: &mut Tensor, dst_cos: &mut Tensor) -> Result<()> {
        crate::model::runtime::compute_rope_cache(self.config(), dst_sin, dst_cos)
    }
}

/// Blanket impl: `Box<dyn LlmModel>` itself implements `LlmModel`.
impl<T: LlmModel + ?Sized> LlmModel for Box<T> {
    fn config(&self) -> &RuntimeModelConfig {
        (**self).config()
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        (**self).tokenizer()
    }

    fn device_type(&self) -> DeviceType {
        (**self).device_type()
    }

    fn create_state(&self) -> Result<InferenceState> {
        (**self).create_state()
    }

    fn forward(&self, ctx: &mut ForwardCtx<'_, '_>) -> Result<()> {
        (**self).forward(ctx)
    }

    fn fill_rope_cache(&self, dst_sin: &mut Tensor, dst_cos: &mut Tensor) -> Result<()> {
        (**self).fill_rope_cache(dst_sin, dst_cos)
    }
}

// ============================================================================
//  ForwardCtx & buffers —— 所有 decoder-only LLM 共用的 forward 上下文
// ============================================================================
//
// 设计：
// - `LlmBuffer` 聚合"每个 LLM 都有"的 batch-sliced views：输入 ids/pos、
//   residual stream、RMSNorm scratch、attn/mlp 共用输出、采样期 S 行 views。
// - `AttentionBuffer` 聚合 GQA/MHA 通用中间量：QKV fused + split + merged。
// - `SwigluBuffer` 聚合 SwiGLU MLP 通用中间量：gate_up fused + split。
// - `ForwardCtx` 组合（组合 > 继承）：上述三组 buffer + meta + states +
//   attn_plan + RoPE cache + OpConfig + output_tokens。新架构只需替换对应
//   buffer 字段，无需改 ctx 类型本体。
//
// 所有 buffer 字段都是零拷贝 Tensor views，第 0 维等于本次 forward 实际用到
// 的行数（token-level = T，seq-level = S），调用方无需再做 narrow 样板。

/// 每个 decoder-only LLM 都需要的通用 batch-sliced views。
pub struct LlmBuffer {
    // token-level (rows = total_tokens)
    /// 本 batch 的 token ids，`[T]`
    pub input_tokens: Tensor,
    /// 本 batch 每个 token 的位置，`[T]`
    pub input_pos: Tensor,
    /// Residual stream，`[T, dim]`
    pub hidden: Tensor,
    /// 任意 RMSNorm 的输出位，`[T, dim]`
    pub norm_out: Tensor,
    /// Attention / MLP 子块交替复用的输出位，`[T, dim]`
    pub block_out: Tensor,

    // seq-level (rows = num_seqs)
    /// 每 seq 最后一个 token 的 hidden state，`[S, dim]`
    pub sample_hidden: Tensor,
    /// lm_head 输出的 logits，`[S, vocab_size]`
    pub logits: Tensor,
    /// 设备对应的 KV 长度数组（CPU → kv_lens_cpu，CUDA → kv_lens_dev），`[S]`
    pub kv_lens: Tensor,
}

/// GQA / MHA 通用的中间 buffer。
///
/// 只保留**物理需要分配**的两块：fused QKV 的完整输出，以及跨 seq 写回的 attn
/// 合并位。Q/K/V 是 `qkv` 的 strided 列视图，在调用点用 `qkv.narrow(1, ..., ...)`
/// 零拷贝切出，不再占独立 buffer。
pub struct AttentionBuffer {
    /// Fused QKV 投影输出，`[T, q_dim + 2*kv_dim]`
    pub qkv: Tensor,
    /// flash attention 输出合并的位置，`[T, q_dim]`
    pub attn_merged: Tensor,
}

/// SwiGLU MLP 通用的中间 buffer。Fused gate+up projection → split → SwiGLU。
pub struct SwigluBuffer {
    /// Fused gate+up 投影输出，`[T, 2*intermediate_size]`
    pub gate_up: Tensor,
    /// Split 后的 gate 分支，`[T, intermediate_size]`
    pub gate: Tensor,
    /// Split 后的 up 分支，`[T, intermediate_size]`
    pub up: Tensor,
}

/// 单次 forward 的全体上下文。
///
/// 子模块只接受 `&mut ForwardCtx`，不再透传 workspace / device_type /
/// cuda_cfg / next_norm_weight 等细节。
pub struct ForwardCtx<'a, 's> {
    pub llm: LlmBuffer,
    pub attn: AttentionBuffer,
    pub mlp: SwigluBuffer,

    pub meta: &'a WorkerBatchMeta<'a>,
    pub states: &'a mut [&'s mut InferenceState],
    pub cuda_cfg: Option<&'a crate::OpConfig>,

    pub total_tokens: usize,
    pub num_seqs: usize,

    /// 本 step 的 attention 调度计划。由 runner / scheduler 在 step 入口准备，
    /// 所有层共用。模型层只读不改。
    pub attn_plan: crate::op::attention::AttentionPlan,

    /// RoPE sin cache，`[max_seq_len, head_size]`（不随 batch 变化）
    pub sin_cache: Tensor,
    /// RoPE cos cache，`[max_seq_len, head_size]`（不随 batch 变化）
    pub cos_cache: Tensor,

    /// 本 step 的采样输出位，`[S]` I32。模型把每 seq 采样得到的 token id
    /// 写到 `output_tokens[0..num_seqs]`。
    pub output_tokens: &'a mut Tensor,

    /// 直接引用 runner 持有的 `BatchWorkspace`，供少数**需要 device 常驻
    /// scratch 的 op**（目前是 `op::kv_cache::scatter`）访问其指针表字段。
    /// 模型层不直接读写它，只作为 op 的载体。
    pub workspace: &'a BatchWorkspace,
}

impl<'a, 's> ForwardCtx<'a, 's> {
    /// 从 `workspace` + `meta` + ... 构造一次 forward 的上下文，一次性把所有
    /// buffer 按当前 batch 大小 narrow 好。
    ///
    /// 假定 runner 已把 `input_tokens / input_pos / kv_lens_*` 写入 workspace，
    /// 并构造好了 `attn_plan`；本函数只做零拷贝 view 切分，不触碰数据。
    pub fn new(
        workspace: &'a BatchWorkspace,
        meta: &'a WorkerBatchMeta<'a>,
        states: &'a mut [&'s mut InferenceState],
        cuda_cfg: Option<&'a crate::OpConfig>,
        device: DeviceType,
        config: &RuntimeModelConfig,
        attn_plan: crate::op::attention::AttentionPlan,
        output_tokens: &'a mut Tensor,
    ) -> Result<Self> {
        let num_seqs = meta.num_seqs();
        let total_tokens = meta.seq_end(num_seqs - 1);
        let q_dim = config.q_dim;

        let kv_lens_src = match device {
            DeviceType::Cpu => &workspace.kv_lens_cpu,
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => &workspace.kv_lens_dev,
        };

        let llm = LlmBuffer {
            input_tokens: workspace.input_tokens.view_prefix(total_tokens)?,
            input_pos: workspace.input_pos.view_prefix(total_tokens)?,
            hidden: workspace.x.narrow(0, 0, total_tokens)?,
            norm_out: workspace.rms_out.narrow(0, 0, total_tokens)?,
            block_out: workspace.ffn_out.narrow(0, 0, total_tokens)?,
            sample_hidden: workspace.intermediate.narrow(0, 0, num_seqs)?,
            logits: workspace.logits.narrow(0, 0, num_seqs)?,
            kv_lens: kv_lens_src.narrow(0, 0, num_seqs)?,
        };

        let attn = AttentionBuffer {
            qkv: workspace.qkv_out.narrow(0, 0, total_tokens)?,
            attn_merged: workspace
                .intermediate
                .slice_ranges(&[0..total_tokens, 0..q_dim])?,
        };

        let mlp = SwigluBuffer {
            gate_up: workspace.gate_up_out.narrow(0, 0, total_tokens)?,
            gate: workspace.w1_out.narrow(0, 0, total_tokens)?,
            up: workspace.w3_out.narrow(0, 0, total_tokens)?,
        };

        Ok(Self {
            llm,
            attn,
            mlp,
            meta,
            states,
            cuda_cfg,
            total_tokens,
            num_seqs,
            attn_plan,
            sin_cache: workspace.sin_cache.clone(),
            cos_cache: workspace.cos_cache.clone(),
            output_tokens,
            workspace,
        })
    }
}
