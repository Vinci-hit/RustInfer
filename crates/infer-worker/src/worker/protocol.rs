use serde::{Deserialize, Serialize};

use crate::base::error::{Error, Result};

// ═══════════════════════════════════════════
// 调度器 → Worker (ZMQ_IN, MessagePack)
// ═══════════════════════════════════════════

/// 调度器发来的 prefill 请求 batch。
///
/// `input_ids` 是所有请求的 token 拼成的一维数组，
/// `q_start_loc[i]` 标记第 i 个请求在 `input_ids` 中的起始偏移。
/// 请求数量 = `q_start_loc.len()`（Vec 自带长度，不需要额外字段）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefillBatchCmd {
    /// 所有请求的 token 拼成一维
    pub input_ids: Vec<i32>,
    /// 每个请求在 input_ids 中的起始位置
    pub q_start_loc: Vec<u32>,
    /// 每个请求已经处理过的 token 数 (chunked prefill 时 > 0)
    pub num_computed_tokens: Vec<u32>,
    /// 调度器分配的 KV cache slot
    pub kv_slots: Vec<u32>,
    /// 采样参数
    pub sampling_params: Vec<SamplingParams>,
    /// 元信息
    pub request_metas: Vec<RequestMeta>,
}

impl PrefillBatchCmd {
    /// 请求数量
    pub fn num_requests(&self) -> usize {
        self.q_start_loc.len()
    }

    /// 校验外部调度器发来的 batch，避免 malformed 消息进入 unsafe 共享缓冲区。
    pub fn validate(&self, max_batch_tokens: usize, max_seqs: usize, max_kv_slots: usize) -> Result<()> {
        let n = self.num_requests();
        if n == 0 {
            return Err(Error::InvalidArgument("PrefillBatchCmd must contain at least one request".into()).into());
        }
        if n > max_seqs {
            return Err(Error::InvalidArgument(format!(
                "PrefillBatchCmd has {} requests, exceeds max_seqs {}",
                n, max_seqs
            )).into());
        }
        if self.input_ids.len() > max_batch_tokens {
            return Err(Error::InvalidArgument(format!(
                "PrefillBatchCmd has {} tokens, exceeds max_batch_tokens {}",
                self.input_ids.len(), max_batch_tokens
            )).into());
        }

        let lens = [
            ("num_computed_tokens", self.num_computed_tokens.len()),
            ("kv_slots", self.kv_slots.len()),
            ("sampling_params", self.sampling_params.len()),
            ("request_metas", self.request_metas.len()),
        ];
        for (name, len) in lens {
            if len != n {
                return Err(Error::InvalidArgument(format!(
                    "PrefillBatchCmd field {} length {} != q_start_loc length {}",
                    name, len, n
                )).into());
            }
        }

        for i in 0..n {
            let start = self.q_start_loc[i] as usize;
            let end = if i + 1 < n {
                self.q_start_loc[i + 1] as usize
            } else {
                self.input_ids.len()
            };
            if start > end || end > self.input_ids.len() {
                return Err(Error::InvalidArgument(format!(
                    "PrefillBatchCmd request {} has invalid token range [{}..{}) for input len {}",
                    i, start, end, self.input_ids.len()
                )).into());
            }
            if start == end {
                return Err(Error::InvalidArgument(format!(
                    "PrefillBatchCmd request {} has empty token range",
                    i
                )).into());
            }
            let kv_slot = self.kv_slots[i] as usize;
            if kv_slot >= max_kv_slots {
                return Err(Error::InvalidArgument(format!(
                    "PrefillBatchCmd request {} kv_slot {} out of range {}",
                    i, kv_slot, max_kv_slots
                )).into());
            }
            if self.request_metas[i].request_id.is_empty() {
                return Err(Error::InvalidArgument(format!(
                    "PrefillBatchCmd request {} has empty request_id",
                    i
                )).into());
            }
            if self.request_metas[i].max_tokens == 0 {
                return Err(Error::InvalidArgument(format!(
                    "PrefillBatchCmd request {} has max_tokens=0",
                    i
                )).into());
            }
            self.sampling_params[i].validate().map_err(|e| {
                Error::InvalidArgument(format!("PrefillBatchCmd request {} invalid sampling params: {}", i, e))
            })?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMeta {
    pub request_id: String,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
}

impl SamplingParams {
    pub fn validate(&self) -> std::result::Result<(), &'static str> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err("temperature must be finite and >= 0");
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err("top_p must be finite and in [0, 1]");
        }
        if self.top_k < -1 {
            return Err("top_k must be -1 or non-negative");
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════
// Worker → 调度器 (ZMQ_OUT, MessagePack)
// ═══════════════════════════════════════════

/// 一步的全量输出，每个活跃序列都包含在内。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutput {
    pub tokens: Vec<SeqToken>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqToken {
    pub request_id: String,
    pub token_id: i32,
    /// Worker 侧判断: true = 该请求已结束 (EOS 或 max_tokens)
    pub finished: bool,
}
