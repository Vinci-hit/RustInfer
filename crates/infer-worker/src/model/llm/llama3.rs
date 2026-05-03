//! Llama3 model implementation.

use std::boxed::Box;
use std::path::Path;

use crate::OpConfig;
use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::model::common::config::RuntimeModelConfig;
use crate::model::common::tokenizer::Tokenizer;
use crate::model::common::{GateUpDims, QkvDims};
use crate::model::llm::LlmModel;
use crate::model::runtime::InferenceState;
use crate::model::ModelLoader;
use crate::op::add_inplace::AddInplace;
use crate::op::embedding::Embedding;
use crate::op::flash_gqa::FlashAttnGQA;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::rope::RoPEOp;
use crate::op::swiglu::SwiGLU;
use crate::tensor::Tensor;
use crate::worker::batch_workspace::BatchWorkspace;
use crate::worker::runner::WorkerBatchMeta;

// ============================================================================
//  Attention —— fused QKV projection + RoPE + flash GQA + o_proj
// ============================================================================

/// 对应 Python `LlamaAttention`。
///
/// 结构（硬编码 Llama3 的所有假设）：
/// - `wqkv`：`[q_proj; k_proj; v_proj]` 融合为一次 Matmul
/// - `wo`：output projection（`o_proj`）
/// - `rope`：RoPE，作用在 Q/K
/// - `mha`：GQA flash attention
struct Attention {
    wqkv: Matmul,
    wo: Matmul,
    rope: RoPEOp,
    mha: FlashAttnGQA,
}

impl Attention {
    fn load(
        layer_idx: usize,
        loader: &ModelLoader,
        config: &RuntimeModelConfig,
        device: DeviceType,
    ) -> Result<Self> {
        let qkv_dims = QkvDims {
            q_dim: config.q_dim,
            kv_dim: config.kv_dim,
            dim: config.dim,
        };
        // AWQ 下 attention 不量化，统一走原始精度。
        let wqkv = loader.load_fused_qkv(layer_idx, qkv_dims, device)?;
        let wo = loader.load_matmul(
            &format!("model.layers.{}.self_attn.o_proj.weight", layer_idx),
            device,
        )?;
        let rope = RoPEOp::new(config.dim, config.kv_dim, config.head_size)?;
        let mha = FlashAttnGQA::new(config.head_num, config.kv_head_num, config.head_size, true)?;
        Ok(Self { wqkv, wo, rope, mha })
    }

    /// Batched attention：`wqkv → split QKV → RoPE → scatter KV → flash attn per seq → wo`。
    ///
    /// 输入 `x_norm` 是 input_layernorm 之后的 hidden states；输出写到 `attn_out`
    /// （= `wo_out`）。`residual` 本函数不碰 —— 由 [`DecoderLayer`] 统一处理。
    #[allow(clippy::too_many_arguments)]
    fn forward_batch(
        &self,
        x_norm: &Tensor,
        attn_out: &mut Tensor,
        layer_idx: usize,
        config: &RuntimeModelConfig,
        states: &mut [&mut InferenceState],
        workspace: &mut BatchWorkspace,
        batch: &WorkerBatchMeta<'_>,
        device_type: DeviceType,
        cuda_cfg: Option<&OpConfig>,
    ) -> Result<()> {
        let dim = config.dim;
        let q_dim = config.q_dim;
        let kv_dim = config.kv_dim;
        let qkv_cols = q_dim + 2 * kv_dim;
        let total_tokens = x_norm.shape()[0];
        let num_seqs = batch.num_seqs();
        let is_cuda = device_type.is_cuda();

        #[cfg(feature = "cuda")]
        let split_stream = crate::cuda::CudaConfig::resolve_stream(cuda_cfg);

        // qkv = wqkv(x_norm)
        let mut qkv = workspace.qkv_out.slice(&[0, 0], &[total_tokens, qkv_cols])?;
        self.wqkv.forward(x_norm, &mut qkv, cuda_cfg)?;

        // split → Q/K/V
        let mut q = workspace.q_out.slice(&[0, 0], &[total_tokens, q_dim])?;
        let mut k = workspace.k_out.slice(&[0, 0], &[total_tokens, kv_dim])?;
        let mut v = workspace.v_out.slice(&[0, 0], &[total_tokens, kv_dim])?;
        crate::op::split_cols::split_cols_tensor(
            &qkv, &mut q, total_tokens, qkv_cols, 0, q_dim,
            #[cfg(feature = "cuda")] split_stream,
        )?;
        crate::op::split_cols::split_cols_tensor(
            &qkv, &mut k, total_tokens, qkv_cols, q_dim, kv_dim,
            #[cfg(feature = "cuda")] split_stream,
        )?;
        crate::op::split_cols::split_cols_tensor(
            &qkv, &mut v, total_tokens, qkv_cols, q_dim + kv_dim, kv_dim,
            #[cfg(feature = "cuda")] split_stream,
        )?;

        // RoPE on Q/K
        let input_pos_view = workspace.input_pos.slice(&[0], &[total_tokens])?;
        self.rope.forward(
            &input_pos_view,
            &workspace.sin_cache,
            &workspace.cos_cache,
            &mut q,
            &mut k,
            cuda_cfg,
        )?;

        // scatter K/V → per-seq KV cache
        for seq_idx in 0..num_seqs {
            let start = batch.seq_start(seq_idx);
            let len = batch.seq_len(seq_idx);
            let pos = batch.seq_pos(seq_idx);
            let (mut k_dst, mut v_dst) = states[seq_idx]
                .kv_cache
                .slice_kv_cache(layer_idx, pos, len, kv_dim)?;
            let k_src = k.slice(&[start, 0], &[len, kv_dim])?;
            let v_src = v.slice(&[start, 0], &[len, kv_dim])?;
            k_dst.copy_from_on_current_stream(&k_src)?;
            v_dst.copy_from_on_current_stream(&v_src)?;
        }

        // flash attention per seq（读取历史 KV）
        let attn_all = workspace.intermediate.slice(&[0, 0], &[total_tokens, q_dim])?;
        for seq_idx in 0..num_seqs {
            let start = batch.seq_start(seq_idx);
            let len = batch.seq_len(seq_idx);
            let q_seq = q.slice(&[start, 0], &[len, q_dim])?;
            let (k_hist, v_hist) = states[seq_idx].kv_cache.get(layer_idx)?;
            let mut out_seq = attn_all.slice(&[start, 0], &[len, q_dim])?;
            let kv_len = if is_cuda {
                workspace.kv_lens_dev.slice(&[seq_idx], &[1])?
            } else {
                workspace.kv_lens_cpu.slice(&[seq_idx], &[1])?
            };
            self.mha.forward(&q_seq, k_hist, v_hist, &kv_len, &mut out_seq, cuda_cfg)?;
        }

        // o_proj
        let mut attn_out_view = attn_out.slice(&[0, 0], &[total_tokens, dim])?;
        self.wo.forward(&attn_all, &mut attn_out_view, cuda_cfg)?;
        Ok(())
    }
}

// ============================================================================
//  Mlp —— fused gate_up projection + SwiGLU + down_proj
// ============================================================================

/// 对应 Python `LlamaMLP`。
///
/// 结构：
/// - `w_gate_up`：`[gate_proj; up_proj]` 融合为一次 Matmul
/// - `w2`：down_proj
/// - `swiglu`：`silu(gate) * up`
struct Mlp {
    w_gate_up: Matmul,
    w2: Matmul,
    swiglu: SwiGLU,
}

impl Mlp {
    fn load(
        layer_idx: usize,
        loader: &ModelLoader,
        config: &RuntimeModelConfig,
        is_awq: bool,
        device: DeviceType,
    ) -> Result<Self> {
        let gate_up_dims = GateUpDims {
            intermediate_size: config.intermediate_size,
            dim: config.dim,
        };
        let group_size = config
            .quant_config
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(128);

        let (w_gate_up, w2) = if is_awq {
            (
                loader.load_fused_gate_up_awq(
                    layer_idx,
                    config.intermediate_size,
                    device,
                    group_size,
                )?,
                loader.load_awq_matmul(
                    &format!("model.layers.{}.mlp.down_proj", layer_idx),
                    device,
                    group_size,
                )?,
            )
        } else {
            (
                loader.load_fused_gate_up(layer_idx, gate_up_dims, device)?,
                loader.load_matmul(
                    &format!("model.layers.{}.mlp.down_proj.weight", layer_idx),
                    device,
                )?,
            )
        };

        Ok(Self {
            w_gate_up,
            w2,
            swiglu: SwiGLU::new(),
        })
    }

    /// Batched SwiGLU FFN：`gate_up → split → swiglu → w2`。
    ///
    /// 输入 `x_norm` 是 post_attention_layernorm 之后的 hidden states；输出写到
    /// `ffn_out`。residual 同样由 [`DecoderLayer`] 处理。
    fn forward_batch(
        &self,
        x_norm: &Tensor,
        ffn_out: &mut Tensor,
        config: &RuntimeModelConfig,
        workspace: &mut BatchWorkspace,
        cuda_cfg: Option<&OpConfig>,
    ) -> Result<()> {
        let inter = config.intermediate_size;
        let total_tokens = x_norm.shape()[0];

        #[cfg(feature = "cuda")]
        let split_stream = crate::cuda::CudaConfig::resolve_stream(cuda_cfg);

        // gate_up = w_gate_up(x_norm)
        let mut gate_up = workspace
            .gate_up_out
            .slice(&[0, 0], &[total_tokens, 2 * inter])?;
        self.w_gate_up.forward(x_norm, &mut gate_up, cuda_cfg)?;

        // split → w1 (gate), w3 (up)
        let mut w1_out = workspace.w1_out.slice(&[0, 0], &[total_tokens, inter])?;
        let mut w3_out = workspace.w3_out.slice(&[0, 0], &[total_tokens, inter])?;
        crate::op::split_cols::split_cols_tensor(
            &gate_up, &mut w1_out, total_tokens, 2 * inter, 0, inter,
            #[cfg(feature = "cuda")] split_stream,
        )?;
        crate::op::split_cols::split_cols_tensor(
            &gate_up, &mut w3_out, total_tokens, 2 * inter, inter, inter,
            #[cfg(feature = "cuda")] split_stream,
        )?;

        // w1 = swiglu(w1, w3)  （就地写到 w1_out）
        self.swiglu.forward(&w3_out, &mut w1_out, cuda_cfg)?;

        // w2
        let mut w2_view = ffn_out.slice(&[0, 0], &[total_tokens, config.dim])?;
        self.w2.forward(&w1_out, &mut w2_view, cuda_cfg)?;
        Ok(())
    }
}

// ============================================================================
//  DecoderLayer —— RMSNorm + Attention + RMSNorm + MLP + residuals
// ============================================================================

/// 对应 Python `LlamaDecoderLayer`。
///
/// 结构：
/// - `input_layernorm` / `post_attention_layernorm`：两个 RMSNorm
/// - `self_attn`：[`Attention`]
/// - `mlp`：[`Mlp`]
///
/// CUDA fast path 里，层末的 `x += ffn_out` 会与**下一层** input_layernorm
/// fuse 成一个 kernel，所以 `forward_batch` 需要 caller 传入 `next_norm_weight`
/// 和 `skip_input_layernorm`。
struct DecoderLayer {
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    self_attn: Attention,
    mlp: Mlp,
}

impl DecoderLayer {
    fn load(
        layer_idx: usize,
        loader: &ModelLoader,
        config: &RuntimeModelConfig,
        is_awq: bool,
        device: DeviceType,
    ) -> Result<Self> {
        let input_layernorm = loader.load_rmsnorm(
            &format!("model.layers.{}.input_layernorm.weight", layer_idx),
            device,
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = loader.load_rmsnorm(
            &format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
            device,
            config.rms_norm_eps,
        )?;
        let self_attn = Attention::load(layer_idx, loader, config, device)?;
        let mlp = Mlp::load(layer_idx, loader, config, is_awq, device)?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            self_attn,
            mlp,
        })
    }

    /// 执行单层 batched forward。
    ///
    /// 语义（对照 Python `LlamaDecoderLayer.forward`）：
    /// 1. `x1 = input_layernorm(x)` —— CUDA 下除第 0 层外由上一层末尾 fused add+rmsnorm 代劳
    /// 2. `attn_out = self_attn(x1)`
    /// 3. `x += attn_out`；`x2 = post_attention_layernorm(x)`（CUDA 融合）
    /// 4. `ffn_out = mlp(x2)`
    /// 5. `x += ffn_out` —— CUDA 下与**下一层** input_layernorm 融合
    #[allow(clippy::too_many_arguments)]
    fn forward_batch(
        &self,
        x: &mut Tensor,
        layer_idx: usize,
        config: &RuntimeModelConfig,
        states: &mut [&mut InferenceState],
        workspace: &mut BatchWorkspace,
        batch: &WorkerBatchMeta<'_>,
        add_op: &AddInplace,
        device_type: DeviceType,
        cuda_cfg: Option<&OpConfig>,
        next_norm_weight: &Tensor,
        skip_input_layernorm: bool,
    ) -> Result<()> {
        let dim = config.dim;
        let total_tokens = x.shape()[0];
        let is_cuda = device_type.is_cuda();

        // ── 1. input_layernorm ──
        let mut attn_norm_out = workspace.rms_out.slice(&[0, 0], &[total_tokens, dim])?;
        if !skip_input_layernorm {
            self.input_layernorm
                .forward(x, &mut attn_norm_out, cuda_cfg)?;
        }

        // ── 2. self_attn ──（输出写到 ffn_out 这块 scratch）
        let mut attn_wo_out = workspace.ffn_out.slice(&[0, 0], &[total_tokens, dim])?;
        self.self_attn.forward_batch(
            &attn_norm_out,
            &mut attn_wo_out,
            layer_idx,
            config,
            states,
            workspace,
            batch,
            device_type,
            cuda_cfg,
        )?;

        // ── 3. residual + post_attention_layernorm（CUDA 融合）──
        //   重新取一次 view：forward_batch 内部借走过 workspace，这里恢复。
        let attn_wo_out = workspace.ffn_out.slice(&[0, 0], &[total_tokens, dim])?;
        let mut ffn_norm_out = workspace.rms_out.slice(&[0, 0], &[total_tokens, dim])?;
        if is_cuda {
            crate::op::fused_add_rmsnorm::fused_add_rmsnorm(
                &mut ffn_norm_out,
                x,
                &attn_wo_out,
                &self.post_attention_layernorm.weight,
                config.rms_norm_eps,
                cuda_cfg,
            )?;
        } else {
            add_op.forward(&attn_wo_out, x, cuda_cfg)?;
            self.post_attention_layernorm
                .forward(x, &mut ffn_norm_out, cuda_cfg)?;
        }

        // ── 4. mlp ──
        let mut w2_out = workspace.ffn_out.slice(&[0, 0], &[total_tokens, dim])?;
        self.mlp
            .forward_batch(&ffn_norm_out, &mut w2_out, config, workspace, cuda_cfg)?;

        // ── 5. residual + 下一层 input_layernorm（CUDA 融合）──
        let w2_out = workspace.ffn_out.slice(&[0, 0], &[total_tokens, dim])?;
        if is_cuda {
            let mut next_out = workspace.rms_out.slice(&[0, 0], &[total_tokens, dim])?;
            crate::op::fused_add_rmsnorm::fused_add_rmsnorm(
                &mut next_out,
                x,
                &w2_out,
                next_norm_weight,
                config.rms_norm_eps,
                cuda_cfg,
            )?;
        } else {
            add_op.forward(&w2_out, x, cuda_cfg)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        self.input_layernorm.to_cuda(device_id)?;
        self.post_attention_layernorm.to_cuda(device_id)?;
        self.self_attn.wqkv.to_cuda(device_id)?;
        self.self_attn.wo.to_cuda(device_id)?;
        self.self_attn.rope.to_cuda(device_id)?;
        self.self_attn.mha.to_cuda(device_id)?;
        self.mlp.w_gate_up.to_cuda(device_id)?;
        self.mlp.w2.to_cuda(device_id)?;
        self.mlp.swiglu.to_cuda(device_id)?;
        Ok(())
    }
}

// ============================================================================
//  Llama3 —— 对外门面（= vLLM 的 LlamaModel + LlamaForCausalLM 合体）
// ============================================================================

/// Llama3 model.
///
/// Top-level components mirror HuggingFace `LlamaForCausalLM`:
/// - `embed_tokens` — `Embedding`
/// - `layers` — `Vec<DecoderLayer>`
/// - `norm` — 最终 RMSNorm
/// - `lm_head` — 分类头 Matmul（tied embedding 时复用 `embed_tokens` 权重）
///
/// Request-level mutable state lives in `InferenceState`（per-seq KV cache + sampler）。
pub struct Llama3 {
    pub(crate) config: RuntimeModelConfig,
    pub(crate) device_type: DeviceType,
    pub(crate) tokenizer: Box<dyn Tokenizer>,

    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: Matmul,

    /// 无参数的就地加法算子（CPU fallback 路径用，CUDA 下走 fused_add_rmsnorm）。
    add_op: AddInplace,
}

impl Llama3 {
    pub fn new<P: AsRef<Path>>(model_dir: P, device_type: DeviceType) -> Result<Self> {
        let mut loader = ModelLoader::load(model_dir.as_ref())?;
        let tensor_names: std::collections::HashSet<String> =
            loader.tensor_names().into_iter().collect();
        let tokenizer = loader.create_tokenizer(model_dir.as_ref())?;
        let config = loader.config.clone();

        let is_awq = config
            .quant_config
            .as_ref()
            .is_some_and(|q| q.quant_method == "compressed-tensors");

        let layers: Vec<DecoderLayer> = (0..config.layer_num)
            .map(|i| DecoderLayer::load(i, &loader, &config, is_awq, device_type))
            .collect::<Result<_>>()?;

        let embed_tokens = loader.load_embedding("model.embed_tokens.weight", device_type)?;
        let norm = loader.load_rmsnorm("model.norm.weight", device_type, config.rms_norm_eps)?;
        let lm_head = if tensor_names.contains("lm_head.weight") {
            loader.load_matmul("lm_head.weight", device_type)?
        } else {
            // tied embedding：lm_head 复用 embed_tokens 权重
            Matmul::from(embed_tokens.weight.clone(), None)
        };

        Ok(Self {
            config,
            device_type,
            tokenizer,
            embed_tokens,
            layers,
            norm,
            lm_head,
            add_op: AddInplace::new(),
        })
    }

    fn compute_worker_batch_on_stream(
        &self,
        states: &mut [&mut InferenceState],
        workspace: &mut BatchWorkspace,
        batch: &WorkerBatchMeta<'_>,
        output_tokens: &mut Tensor,
        cuda_config_ref: Option<&OpConfig>,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Some(cfg) = cuda_config_ref {
            return crate::cuda::with_cuda_stream(cfg.stream, || {
                self.compute_worker_batch(states, workspace, batch, output_tokens, cuda_config_ref)
            });
        }
        self.compute_worker_batch(states, workspace, batch, output_tokens, cuda_config_ref)
    }

    fn compute_worker_batch(
        &self,
        states: &mut [&mut InferenceState],
        workspace: &mut BatchWorkspace,
        batch: &WorkerBatchMeta<'_>,
        output_tokens: &mut Tensor,
        cuda_config_ref: Option<&OpConfig>,
    ) -> Result<()> {
        let total_tokens = batch.seq_end(batch.num_seqs() - 1);
        let num_seqs = batch.num_seqs();
        let dim = self.config.dim;
        let is_cuda = self.device_type.is_cuda();

        // ── Embedding ──
        let input_tokens_view = workspace.input_tokens.slice(&[0], &[total_tokens])?;
        let mut x = workspace.x.slice(&[0, 0], &[total_tokens, dim])?;
        self.embed_tokens.forward(&input_tokens_view, &mut x, cuda_config_ref)?;

        // ── Decoder layers ──
        //
        // 对照 Python: `for layer in self.layers: x = layer(x, ...)`
        //
        // CUDA fast path 的跨层 fused add+rmsnorm：第 i 层末尾的 `x += ffn_out`
        // 会被融合到"写入下一层 input_layernorm 结果"的 kernel；所以：
        // - 传给本层的 `next_norm_weight` 必须是**下一层** input_layernorm.weight
        //   （最后一层指向 `self.norm.weight`，用于 final norm 的 fused 入口）；
        // - 本层开头的 input_layernorm 在 CUDA 下仅在 layer_idx == 0 时显式执行，
        //   其它层依赖上一层末尾 fusion 的产出。
        let layer_num = self.config.layer_num;
        for layer_idx in 0..layer_num {
            let next_norm_weight = if layer_idx + 1 < layer_num {
                &self.layers[layer_idx + 1].input_layernorm.weight
            } else {
                &self.norm.weight
            };
            let skip_input_layernorm = is_cuda && layer_idx != 0;

            self.layers[layer_idx].forward_batch(
                &mut x,
                layer_idx,
                &self.config,
                states,
                workspace,
                batch,
                &self.add_op,
                self.device_type,
                cuda_config_ref,
                next_norm_weight,
                skip_input_layernorm,
            )?;
        }

        // ── Final RMSNorm ──（CUDA 下已由最后一层末尾 fused 产生到 workspace.rms_out）
        let mut final_norm_all = workspace.rms_out.slice(&[0, 0], &[total_tokens, dim])?;
        if !is_cuda {
            self.norm.forward(&x, &mut final_norm_all, cuda_config_ref)?;
        }

        // ── 收集每个 seq 的最后 hidden state 用于采样 ──
        let sample_hidden = workspace.intermediate.slice(&[0, 0], &[num_seqs, dim])?;
        for seq_idx in 0..num_seqs {
            let last = batch.seq_end(seq_idx) - 1;
            let src = final_norm_all.slice(&[last, 0], &[1, dim])?;
            let mut dst = sample_hidden.slice(&[seq_idx, 0], &[1, dim])?;
            dst.copy_from_on_current_stream(&src)?;
        }

        // ── lm_head → logits → sampling ──
        let mut logits = workspace.logits.slice(&[0, 0], &[num_seqs, self.config.vocab_size])?;
        self.lm_head.forward(&sample_hidden, &mut logits, cuda_config_ref)?;

        let tok_vocab = self.config.tokenizer_vocab_size;
        #[cfg(feature = "cuda")]
        let use_batched_argmax = is_cuda && logits.dtype() == DataType::BF16;
        #[cfg(not(feature = "cuda"))]
        let use_batched_argmax = false;

        if use_batched_argmax {
            #[cfg(feature = "cuda")]
            {
                let mut out_view = output_tokens.slice(&[0], &[num_seqs])?;
                crate::op::kernels::cuda::argmax_batch_strided(
                    &logits, tok_vocab, self.config.vocab_size, 0, num_seqs,
                    &mut out_view, cuda_config_ref,
                )?;
            }
        } else {
            for i in 0..num_seqs {
                let logits_row = logits.slice(&[i, 0], &[1, self.config.vocab_size])?;
                let logits_trimmed = logits_row.slice(&[0, 0], &[1, tok_vocab])?;
                let logits_1d = logits_trimmed.reshape(&[tok_vocab])?;
                let mut dst = output_tokens.slice(&[i], &[1])?;
                states[i].sampler.sample(&logits_1d, &mut dst, cuda_config_ref)?;
            }
        }
        Ok(())
    }
}

// ============================================================================
//  LlmModel trait 实现
// ============================================================================
impl LlmModel for Llama3 {
    fn config(&self) -> &RuntimeModelConfig {
        &self.config
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Llama3 的 batched forward 要求 workspace 里预先写好 token_ids / input_pos /
    /// kv_lens。Worker 热路径由 `ModelRunner::stage_device_inputs` 做；`generate`
    /// 的单 seq 测试路径则通过本钩子提供等价写入。
    fn stage_generate_inputs(
        &self,
        workspace: &mut BatchWorkspace,
        token_ids: &[i32],
        positions: &[i32],
        kv_len: i32,
    ) -> Result<()> {
        workspace.input_tokens.write_from_i32_host(token_ids, token_ids.len())?;
        workspace.input_pos.write_from_i32_host(positions, positions.len())?;
        workspace.kv_lens_cpu.as_i32_mut()?.as_slice_mut()?[0] = kv_len;
        #[cfg(feature = "cuda")]
        {
            let src = workspace.kv_lens_cpu.slice(&[0], &[1])?;
            let mut dst = workspace.kv_lens_dev.slice(&[0], &[1])?;
            dst.copy_from(&src)?;
        }
        Ok(())
    }

    fn forward(
        &self,
        states: &mut [&mut InferenceState],
        workspace: &mut BatchWorkspace,
        batch: &WorkerBatchMeta<'_>,
        output_tokens: &mut Tensor,
        cuda_config: Option<&OpConfig>,
    ) -> Result<()> {
        let num_seqs = batch.num_seqs();
        if states.len() != num_seqs {
            return Err(Error::InvalidArgument(format!(
                "Llama3::forward states len {} != batch seqs {}",
                states.len(), num_seqs
            )).into());
        }
        if num_seqs == 0 {
            return Ok(());
        }
        let total_tokens = batch.seq_end(num_seqs - 1);
        if total_tokens > workspace.max_batch_tokens {
            return Err(Error::InvalidArgument(format!(
                "Llama3::forward total tokens {} exceeds workspace capacity {}",
                total_tokens, workspace.max_batch_tokens
            )).into());
        }
        if num_seqs > workspace.max_batch_seqs {
            return Err(Error::InvalidArgument(format!(
                "Llama3::forward seqs {} exceeds workspace capacity {}",
                num_seqs, workspace.max_batch_seqs
            )).into());
        }

        let mut kv_grew = false;
        for i in 0..num_seqs {
            if states[i].kv_cache.ensure_capacity(batch.seq_end_pos(i)?)? {
                states[i].invalidate_decode_graphs();
                kv_grew = true;
            }
        }
        if kv_grew {
            workspace.invalidate_batch_member_cache();
            #[cfg(feature = "cuda")]
            if let Some(cfg) = cuda_config {
                let cfg_ptr = cfg as *const crate::cuda::CudaConfig as *mut crate::cuda::CudaConfig;
                unsafe { (*cfg_ptr).graphs.clear(); }
            }
        }

        let can_full_graph = false
            && batch.is_decode_only()
            && self.device_type.is_cuda()
            && workspace.x.dtype() == DataType::BF16
            && self.config.head_size == 64;

        #[cfg(feature = "cuda")]
        if can_full_graph {
            let cfg = cuda_config.ok_or_else(|| Error::InvalidArgument(
                "DecodeOnly FullGraph path requires CudaConfig".into()
            ))?;
            let output_ptr = output_tokens.as_i32()?.buffer().as_ptr() as usize;
            let slot = crate::cuda::GraphSlot::LlmDecodeWithOutput { batch: num_seqs, output_ptr };
            if !cfg.graph_ready(slot) {
                cfg.sync_stream()?;
                let cfg_ptr = cfg as *const crate::cuda::CudaConfig as *mut crate::cuda::CudaConfig;
                cfg.capture_begin()?;
                self.compute_worker_batch_on_stream(states, workspace, batch, output_tokens, cuda_config)?;
                unsafe { (*cfg_ptr).capture_end(slot)?; }
            }
            cfg.launch(slot)?;
            cfg.sync_stream()?;
        } else {
            self.compute_worker_batch_on_stream(states, workspace, batch, output_tokens, cuda_config)?;
        }
        Ok(())
    }
}

// ============================================================================
//  Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::error::Result;
    use crate::model::llm::{GenerateStats, LlmModel};
    use std::time::Instant;

    fn generate_and_measure(
        model: &Llama3, state: &mut InferenceState,
        prompt: &str, max_tokens: usize, verbose: bool,
    ) -> Result<(u64, GenerateStats)> {
        let start = Instant::now();
        let stats = model.generate(state, prompt, max_tokens, verbose)?;
        Ok((start.elapsed().as_millis() as u64, stats))
    }

    /// Pre-run a tiny `generate()` on the same state the benchmark will
    /// use, so the real call sees a hot CUDA context:
    ///
    /// - kernel module load / PTX→SASS JIT for every prefill + decode
    ///   kernel (embedding, rmsnorm, qkv, rope, scatter_kv, flash-attn,
    ///   wo, silu, sampler, …).
    /// - cuBLASLt algorithm heuristics for every `(M,N,K)` shape hit.
    /// - the decode-path CUDA Graph: captured here, replayed for free
    ///   from the real benchmark's first decode step onward.
    ///
    /// After warmup returns, the real `generate(prompt, N)` call
    /// unconditionally overwrites `kv_cache[..prompt_len]` from pos=0
    /// (see [`Llama3::generate`]), so warmup has no correctness impact.
    ///
    /// ## Why the filler prompt
    ///
    /// The flash-attention prefill kernel processes the sequence in
    /// fixed-size tiles (~64 tokens). Feeding it a prompt of 1–2 tokens
    /// causes out-of-range reads inside the tile, putting the CUDA
    /// context into a sticky-error state that surfaces later as
    /// `CUBLAS_STATUS_EXECUTION_FAILED (13)` on the next cuBLASLt call.
    /// We pick a ~10-token filler string to stay above that floor while
    /// keeping the warmup cheap.
    fn warmup(model: &Llama3, state: &mut InferenceState) -> Result<()> {
        // ~10 tokens after BPE — safely above the flash-attn prefill
        // tile floor. A few decode steps also capture the CUDA Graph.
        let prompt = "The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog. \
            The quick brown fox jumps over the lazy dog.";
        let _ = model.generate(state, prompt, 4, false)?;
        Ok(())
    }

    #[test]
    #[ignore = "Long running test"]
    fn test_llama3_cpu_loading_and_generation() -> Result<()> {
        let model_path = get_dummy_model_path();
        assert!(model_path.exists(), "Model not found.");

        let model = Llama3::new(model_path, DeviceType::Cpu)?;
        let mut state = model.create_state()?;

        let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 14 Dec 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n你是算法糕手，写一段C++代码，实现一个简单的中序遍历函数。<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
        let (_dur_ms, stats) = generate_and_measure(&model, &mut state, prompt, 150, true)?;
        assert!(stats.num_tokens > 0, "No tokens generated.");

        let prompt_len = model.tokenizer.encode(prompt)?.len() as f64;
        let total_ms = (stats.prefill_ms + stats.decode_ms) as f64;
        println!("\n=== CPU: {} tok, {:.0}ms, {:.1} tok/s, decode {:.1} tok/s ===",
            stats.num_tokens, total_ms,
            (prompt_len + stats.num_tokens as f64) / (total_ms / 1000.0),
            if stats.decode_ms > 0 { stats.decode_iterations as f64 / (stats.decode_ms as f64 / 1000.0) } else { 0.0 });
        Ok(())
    }

    #[test]
    #[ignore = "Long running test"]
    #[cfg(feature = "cuda")]
    fn test_llama3_cuda_performance() -> Result<()> {
        let model_path = get_dummy_model_path();
        assert!(model_path.exists(), "Model not found.");

        let model = Llama3::new(model_path, DeviceType::Cuda(0))?;
        let mut state = model.create_state()?;
        warmup(&model, &mut state)?;

        let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 14 Dec 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n你是算法糕手，写一段C++代码，实现一个简单的中序遍历函数。<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
        let (_dur_ms, stats) = generate_and_measure(&model, &mut state, prompt, 2000, false)?;

        let prompt_len = model.tokenizer.encode(prompt)?.len() as f64;
        let total_ms = (stats.prefill_ms + stats.decode_ms) as f64;
        println!("\n=== BF16 CUDA: {} tok, {:.0}ms, {:.1} tok/s, decode {:.1} tok/s ===",
            stats.num_tokens, total_ms,
            (prompt_len + stats.num_tokens as f64) / (total_ms / 1000.0),
            if stats.decode_ms > 0 { stats.decode_iterations as f64 / (stats.decode_ms as f64 / 1000.0) } else { 0.0 });
        Ok(())
    }

    fn get_dummy_model_path() -> &'static Path {
        Path::new("/apdcephfs_qy2/share_303432435/vinciiliu/models/llama3.2-1b")
    }

    fn get_awq_model_path() -> &'static Path {
        Path::new("/apdcephfs_qy2/share_303432435/vinciiliu/vllm_test/llama3.2-1b-AWQ-mlp3")
    }

    #[test]
    #[ignore = "Long running test"]
    #[cfg(feature = "cuda")]
    fn test_llama3_awq_cuda() -> Result<()> {
        let model_path = get_awq_model_path();
        assert!(model_path.exists(), "AWQ model not found.");

        let model = Llama3::new(model_path, DeviceType::Cuda(0))?;
        let mut state = model.create_state()?;
        warmup(&model, &mut state)?;

        let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 14 Dec 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello, who are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
        let (_dur_ms, stats) = generate_and_measure(&model, &mut state, prompt, 2000, false)?;

        let prompt_len = model.tokenizer.encode(prompt)?.len() as f64;
        let total_ms = (stats.prefill_ms + stats.decode_ms) as f64;
        println!("\n=== K-packed INT4 CUDA: {} tok, {:.0}ms, {:.1} tok/s, decode {:.1} tok/s ===",
            stats.num_tokens, total_ms,
            (prompt_len + stats.num_tokens as f64) / (total_ms / 1000.0),
            if stats.decode_ms > 0 { stats.decode_iterations as f64 / (stats.decode_ms as f64 / 1000.0) } else { 0.0 });
        Ok(())
    }
}
