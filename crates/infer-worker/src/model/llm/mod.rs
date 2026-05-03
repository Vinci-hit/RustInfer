//! 面向 worker runner 的最小 LLM 抽象。

pub mod llama3;
pub mod qwen3;

use std::io::{self, Write};
use std::time::Instant;

use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
use crate::model::common::config::RuntimeModelConfig;
use crate::model::common::tokenizer::Tokenizer;
use crate::model::runtime::InferenceState;
use crate::tensor::Tensor;
use crate::worker::batch_workspace::BatchWorkspace;
use crate::worker::runner::WorkerBatchMeta;

/// `LlmModel::generate` 的返回值。
///
/// 用结构体而非元组（原则 1、3）：每个字段都有明确语义，
/// 新增字段不会破坏调用方模式匹配。
#[derive(Debug, Clone)]
pub struct GenerateStats {
    /// 解码后的文本（不含 prompt）。
    pub text: String,
    /// 实际生成的 token 数量（含第一个 prefill token）。
    pub num_tokens: u32,
    /// Prefill 阶段耗时（毫秒）。
    pub prefill_ms: u64,
    /// Decode 循环耗时（毫秒）。
    pub decode_ms: u64,
    /// Decode 迭代次数（= 有效调用 forward 的 decode 步数）。
    pub decode_iterations: usize,
}

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

    /// 默认实现：绝大多数 LLM 使用标准 RoPE cache（基于 `config`）。
    fn fill_rope_cache(&self, dst_sin: &mut Tensor, dst_cos: &mut Tensor) -> Result<()> {
        crate::model::runtime::compute_rope_cache(self.config(), dst_sin, dst_cos)
    }

    /// 钩子：在调用 `forward` 前把本轮的 token/position/kv_len 写到 `workspace`。
    ///
    /// - Llama3 风格的 batched forward 需要这些输入预写入到 device workspace；
    /// - Qwen3 目前在 `forward` 内部直接处理输入拷贝，因此使用默认 no-op。
    ///
    /// 本钩子是重构过渡期的产物（见 `doc/REFACTOR_GUIDE.md` 阶段 5）——
    /// 长期目标是让 `forward` 自己吃掉所有输入 staging 逻辑。
    fn stage_generate_inputs(
        &self,
        _workspace: &mut BatchWorkspace,
        _token_ids: &[i32],
        _positions: &[i32],
        _kv_len: i32,
    ) -> Result<()> {
        Ok(())
    }

    /// 单 prompt 文本生成（自包含测试/基准用）。
    ///
    /// 适用所有 `LlmModel` 实现：
    /// - 使用 `create_state` 提供的 per-request `InferenceState`；
    /// - 通过 `forward` 做 batched 推理（batch_size = 1）；
    /// - 通过 `tokenizer` 做编解码；
    /// - 通过 `stage_generate_inputs` 钩子处理模型特有的输入 staging。
    ///
    /// 不用于 Worker 热路径（Worker 走 `ModelRunner::run_step`）。
    fn generate(
        &self,
        state: &mut InferenceState,
        prompt: &str,
        max_tokens: usize,
        print_output: bool,
    ) -> Result<GenerateStats> {
        let mut stdout = io::stdout();
        if print_output {
            println!("----------------------------------------");
            println!("Prompt: {}", prompt);
            stdout.flush()?;
        }

        let tokenizer = self.tokenizer();
        let config = self.config();
        let device = self.device_type();

        let prompt_tokens = tokenizer.encode(prompt)?;
        if prompt_tokens.is_empty() {
            return Err(Error::InvalidArgument("Prompt cannot be empty.".to_string()).into());
        }

        // Batch workspace + RoPE cache（batch_size = 1）。
        let mut workspace = BatchWorkspace::new(
            config,
            prompt_tokens.len().max(1),
            1,
            device,
        )?;
        self.fill_rope_cache(&mut workspace.sin_cache, &mut workspace.cos_cache)?;

        // CUDA config（仅 CUDA feature + CUDA device 下启用）。
        #[cfg(feature = "cuda")]
        let cuda_cfg = if device.is_cuda() {
            Some(crate::cuda::CudaConfig::new()?.with_flash_decode(
                config.head_num,
                config.head_size,
                1,
            )?)
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        let cuda_ref = cuda_cfg.as_ref().map(|c| c as &crate::OpConfig);
        #[cfg(not(feature = "cuda"))]
        let cuda_ref = None;

        let mut gen_output = Tensor::new(&[1], DataType::I32, device)?;
        let slot_indices = [0i32];

        // ---------- Prefill ----------
        let prefill_start = Instant::now();
        let prefill_positions: Vec<i32> = (0..prompt_tokens.len()).map(|i| i as i32).collect();
        let prefill_q_start = [0i32, prompt_tokens.len() as i32];
        let prefill_meta = WorkerBatchMeta {
            q_start_loc: &prefill_q_start,
            slot_indices: &slot_indices,
            token_ids: &prompt_tokens,
            positions: &prefill_positions,
            num_decode: 0,
            num_prefill: 1,
        };
        let first_token = {
            self.stage_generate_inputs(&mut workspace, &prompt_tokens, &prefill_positions, 0)?;
            let mut refs = vec![&mut *state];
            self.forward(refs.as_mut_slice(), &mut workspace, &prefill_meta, &mut gen_output, cuda_ref)?;
            gen_output.to_cpu()?.as_i32()?.as_slice()?[0]
        };
        let prefill_ms = prefill_start.elapsed().as_millis() as u64;

        let mut generated_tokens = vec![first_token];
        let mut printed_len = 0usize;
        if print_output {
            let decoded = tokenizer.decode(&generated_tokens)?;
            let _ = write!(stdout, "{}", &decoded[printed_len..]);
            printed_len = decoded.len();
            stdout.flush()?;
        }

        // ---------- Decode ----------
        let decode_start = Instant::now();
        let mut decode_iterations = 0;
        let max_decode_end = (prompt_tokens.len() - 1 + max_tokens).min(config.seq_len);
        for pos in prompt_tokens.len()..max_decode_end {
            let last_token = [*generated_tokens.last().unwrap()];
            let positions = [pos as i32];
            let q_start = [0i32, 1];
            let meta = WorkerBatchMeta {
                q_start_loc: &q_start,
                slot_indices: &slot_indices,
                token_ids: &last_token,
                positions: &positions,
                num_decode: 1,
                num_prefill: 0,
            };
            let next_token = {
                self.stage_generate_inputs(&mut workspace, &last_token, &positions, positions[0])?;
                let mut refs = vec![&mut *state];
                self.forward(refs.as_mut_slice(), &mut workspace, &meta, &mut gen_output, cuda_ref)?;
                gen_output.to_cpu()?.as_i32()?.as_slice()?[0]
            };

            if tokenizer.is_eos(next_token) {
                break;
            }

            generated_tokens.push(next_token);
            decode_iterations += 1;

            if print_output {
                let decoded = tokenizer.decode(&generated_tokens)?;
                if decoded.len() > printed_len {
                    let new_text = &decoded[printed_len..];
                    if !new_text.contains('\u{FFFD}') {
                        let _ = write!(stdout, "{}", new_text);
                        printed_len = decoded.len();
                        stdout.flush()?;
                    }
                }
            }
        }
        let decode_ms = decode_start.elapsed().as_millis() as u64;
        if print_output {
            println!();
        }

        let text = tokenizer.decode(&generated_tokens)?;
        Ok(GenerateStats {
            text,
            num_tokens: generated_tokens.len() as u32,
            prefill_ms,
            decode_ms,
            decode_iterations,
        })
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

    fn forward(
        &self,
        states: &mut [&mut InferenceState],
        workspace: &mut BatchWorkspace,
        batch: &WorkerBatchMeta<'_>,
        output_tokens: &mut Tensor,
        cuda_config: Option<&crate::OpConfig>,
    ) -> Result<()> {
        (**self).forward(states, workspace, batch, output_tokens, cuda_config)
    }

    fn fill_rope_cache(&self, dst_sin: &mut Tensor, dst_cos: &mut Tensor) -> Result<()> {
        (**self).fill_rope_cache(dst_sin, dst_cos)
    }

    fn stage_generate_inputs(
        &self,
        workspace: &mut BatchWorkspace,
        token_ids: &[i32],
        positions: &[i32],
        kv_len: i32,
    ) -> Result<()> {
        (**self).stage_generate_inputs(workspace, token_ids, positions, kv_len)
    }
}
