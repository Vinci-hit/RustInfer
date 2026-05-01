use crate::base::error::Result;
use crate::base::{DataType, DeviceType};
use crate::model::common::config::RuntimeModelConfig;
use crate::tensor::Tensor;

/// Batch forward 用的共享 workspace buffer。
///
/// 所有 buffer 按 `max_batch_tokens × dim` 预分配一次，
/// 每步通过 `Tensor::slice` 零拷贝取前 N 行使用。
/// 与单 seq 的 `InferenceState.workspace` 不同，这里是跨 seq 共享的。
pub struct BatchWorkspace {
    // ═══ 主数据流 ═══
    /// embedding 输出 / residual stream, [max_batch_tokens, dim]
    pub x: Tensor,
    /// RMSNorm 输出 (复用做 attn_out, ffn_norm_out), [max_batch_tokens, dim]
    pub rms_out: Tensor,
    /// Fused QKV 输出, [max_batch_tokens, q_dim + 2 * kv_dim]
    pub qkv_out: Tensor,
    /// Q slice (不额外分配, 从 qkv_out slice)
    // pub q: slice of qkv_out
    /// Gate+Up fused 输出, [max_batch_tokens, 2 * intermediate_size]
    pub gate_up_out: Tensor,
    /// W2 (down proj) 输出 / FFN 中间 buffer, [max_batch_tokens, dim]
    pub ffn_out: Tensor,
    /// 额外的 [max_batch_tokens, dim] buffer (用于 residual 等)
    pub intermediate: Tensor,

    // ═══ Token 级 buffer ═══
    /// 输入 token ids, [max_batch_tokens], I32
    pub input_tokens: Tensor,
    /// Input positions, [max_batch_tokens], I32 (CPU, 因为 RoPE 需要 CPU pos)
    pub input_pos: Tensor,

    // ═══ Sin/Cos cache (从 InferenceState 复制或共享) ═══
    /// [max_seq_len, head_size]
    pub sin_cache: Tensor,
    pub cos_cache: Tensor,

    // ═══ 输出 ═══
    /// Logits, [max_batch_seqs, vocab_size]
    pub logits: Tensor,
    /// 采样输出 token ids, [max_batch_seqs], I32
    pub output_tokens: Tensor,

    // ═══ 容量 ═══
    pub max_batch_tokens: usize,
    pub max_batch_seqs: usize,
}

impl BatchWorkspace {
    pub fn new(
        config: &RuntimeModelConfig,
        max_batch_tokens: usize,
        max_batch_seqs: usize,
        device: DeviceType,
    ) -> Result<Self> {
        let dim = config.dim;
        let q_dim = config.q_dim;
        let kv_dim = config.kv_dim;
        let inter = config.intermediate_size;
        let head_size = config.head_size;
        let vocab_size = config.vocab_size;
        let max_seq_len = config.seq_len;

        let float_dtype = config.runtime_float_dtype(device)?;
        let int_dtype = DataType::I32;

        Ok(Self {
            x: Tensor::new(&[max_batch_tokens, dim], float_dtype, device)?,
            rms_out: Tensor::new(&[max_batch_tokens, dim], float_dtype, device)?,
            qkv_out: Tensor::new(&[max_batch_tokens, q_dim + 2 * kv_dim], float_dtype, device)?,
            gate_up_out: Tensor::new(&[max_batch_tokens, 2 * inter], float_dtype, device)?,
            ffn_out: Tensor::new(&[max_batch_tokens, dim], float_dtype, device)?,
            intermediate: Tensor::new(&[max_batch_tokens, dim], float_dtype, device)?,

            input_tokens: Tensor::new(&[max_batch_tokens], int_dtype, device)?,
            input_pos: Tensor::new(&[max_batch_tokens], int_dtype, DeviceType::Cpu)?,

            sin_cache: Tensor::new(&[max_seq_len, head_size], float_dtype, device)?,
            cos_cache: Tensor::new(&[max_seq_len, head_size], float_dtype, device)?,

            logits: Tensor::new(&[max_batch_seqs, vocab_size], float_dtype, device)?,
            output_tokens: Tensor::new(&[max_batch_seqs], int_dtype, device)?,

            max_batch_tokens,
            max_batch_seqs,
        })
    }
}
