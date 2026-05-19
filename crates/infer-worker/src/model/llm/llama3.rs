//! Llama3 model —— 纯 forward 实现。
//!
//! 本文件只描述**模型数学结构**：embedding → N × (norm + attention + norm + mlp)
//! → final norm → lm_head → sampling。所有 CPU/CUDA 差异由底层 op 内部消化；
//! batch 维度默认存在，`runner` 负责写入所有 workspace 输入与扩容 KV，本模型
//! 一概假定这些信息 **已就绪**，不再做任何校验、staging、graph 或 stream 调度。

use std::boxed::Box;
use std::path::Path;

use crate::base::DeviceType;
use crate::base::error::Result;
use crate::model::common::config::RuntimeModelConfig;
use crate::model::common::tokenizer::Tokenizer;
use crate::model::common::{GateUpDims, QkvDims};
use crate::model::llm::{ForwardCtx, LlmModel};
use crate::model::ModelLoader;
use crate::op::attention::Attention as FlashAttn;
use crate::op::embedding::Embedding;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::rope::RoPEOp;
use crate::tensor::Tensor;

// ============================================================================
//  Attention —— fused QKV + RoPE + flash GQA + o_proj
// ============================================================================

struct Attention {
    wqkv: Matmul,
    wo: Matmul,
    rope: RoPEOp,
    mha: FlashAttn,
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
        let wqkv = loader.load_fused_qkv(layer_idx, qkv_dims, device)?;
        let wo = loader.load_matmul(
            &format!("model.layers.{}.self_attn.o_proj.weight", layer_idx),
            device,
        )?;
        let rope = RoPEOp::new(config.dim, config.kv_dim, config.head_size)?;
        let mha = FlashAttn::new(config.head_num, config.kv_head_num, config.head_size, true)?;
        Ok(Self { wqkv, wo, rope, mha })
    }

    /// 输入 `x_norm` = input_layernorm 的输出；输出写到 `attn_out`。residual 由
    /// [`DecoderLayer`] 统一处理。
    fn forward(
        &self,
        x_norm: &Tensor,
        attn_out: &mut Tensor,
        layer_idx: usize,
        config: &RuntimeModelConfig,
        ctx: &mut ForwardCtx<'_, '_>,
    ) -> Result<()> {
        let q_dim = config.q_dim;
        let kv_dim = config.kv_dim;
        let total_tokens = ctx.total_tokens;
        let cuda_cfg = ctx.cuda_cfg;

        // ── wqkv ──
        let mut qkv = ctx.attn.qkv.clone();
        self.wqkv.forward(x_norm, &mut qkv, cuda_cfg)?;

        // ── split → Q/K/V（零拷贝 strided view）──
        //
        // `qkv` 布局为 `[T, q_dim + 2*kv_dim]`，按列切出 Q/K/V 三段 view。stride
        // 保持与 `qkv` 一致（行步长 = qkv_cols），下游 RoPE / Attention 都接受
        // strided 输入，因此不再需要物理 copy。
        let mut q = qkv.narrow(1, 0, q_dim)?;
        let mut k = qkv.narrow(1, q_dim, kv_dim)?;
        let v = qkv.narrow(1, q_dim + kv_dim, kv_dim)?;

        // ── RoPE on Q/K ──
        self.rope.forward(
            &ctx.llm.input_pos,
            &ctx.sin_cache,
            &ctx.cos_cache,
            &mut q,
            &mut k,
            cuda_cfg,
        )?;

        // ── scatter K/V → per-seq KV cache ──（op 层负责循环 / kernel 化）
        crate::op::kv_cache::scatter(
            &k, &v, layer_idx, ctx.states, ctx.meta, ctx.workspace, cuda_cfg,
        )?;

        // ── flash attention (one launch for the whole batch) ──
        //
        // Q 在 qkv 前 q_dim 列内（RoPE 后已原地写回），qkv 本身是 contiguous
        // [T, qkv_cols]。对 qkv 整体 reshape 到 3D [T, total_heads, HD] 是零拷贝，
        // 然后 narrow head 维取前 head_num 个 head 即得 Q 的 3D view：
        //   shape  = [T, Hq, HD]
        //   stride = [qkv_cols, HD, 1]
        // attention kernel 通过 q_stride_b / q_stride_h 接受这种非 contiguous 布局，
        // 不再触发 permute_kernel。
        let head_num = config.head_num;
        let head_size = config.head_size;
        let num_kv_heads = config.kv_head_num;
        let total_heads = head_num + 2 * num_kv_heads;
        let qkv_3d = qkv.reshape(&[total_tokens, total_heads, head_size])?;
        let q3 = qkv_3d.narrow(1, 0, head_num)?;

        let mut attn_all_3d =
            ctx.attn.attn_merged.reshape(&[total_tokens, head_num, head_size])?;
        unsafe {
            self.mha.forward(&q3, &mut attn_all_3d, layer_idx, &ctx.attn_plan, cuda_cfg)?;
        }

        // ── o_proj ──（使用回 [T, q_dim] 的 view）
        let attn_all = ctx.attn.attn_merged.clone();
        self.wo.forward(&attn_all, attn_out, cuda_cfg)?;
        Ok(())
    }
}

// ============================================================================
//  Mlp —— fused gate_up + SwiGLU + down_proj
// ============================================================================

struct Mlp {
    w_gate_up: Matmul,
    w2: Matmul,
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
        })
    }

    /// `x_norm` = post_attention_layernorm 输出；结果写到 `ffn_out`。
    fn forward(
        &self,
        x_norm: &Tensor,
        ffn_out: &mut Tensor,
        config: &RuntimeModelConfig,
        ctx: &mut ForwardCtx<'_, '_>,
    ) -> Result<()> {
        let inter = config.intermediate_size;
        let total_tokens = ctx.total_tokens;
        let cuda_cfg = ctx.cuda_cfg;

        let mut gate_up = ctx.mlp.gate_up.clone();
        self.w_gate_up.forward(x_norm, &mut gate_up, cuda_cfg)?;

        // Fused packed SwiGLU: gate_up [T, 2*inter] → gate [T, inter]
        // 省掉 2 次 split_cols kernel launch。
        let mut gate = ctx.mlp.gate.clone();
        #[cfg(feature = "cuda")]
        {
            if matches!(gate_up.device(), crate::base::DeviceType::Cuda(_))
                && gate_up.dtype() == crate::base::DataType::BF16
            {
                crate::op::kernels::cuda::swiglu_packed_bf16(
                    &gate_up, &mut gate, total_tokens, inter, cuda_cfg,
                )?;
            } else {
                // fallback: split + swiglu
                let mut up = ctx.mlp.up.clone();
                crate::op::split_cols::split_cols_tensor(
                    &gate_up, &mut gate, total_tokens, 2 * inter, 0, inter, cuda_cfg,
                )?;
                crate::op::split_cols::split_cols_tensor(
                    &gate_up, &mut up, total_tokens, 2 * inter, inter, inter, cuda_cfg,
                )?;
                crate::op::swiglu::swiglu(&up, &mut gate, cuda_cfg)?;
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let mut up = ctx.mlp.up.clone();
            crate::op::split_cols::split_cols_tensor(
                &gate_up, &mut gate, total_tokens, 2 * inter, 0, inter, cuda_cfg,
            )?;
            crate::op::split_cols::split_cols_tensor(
                &gate_up, &mut up, total_tokens, 2 * inter, inter, inter, cuda_cfg,
            )?;
            crate::op::swiglu::swiglu(&up, &mut gate, cuda_cfg)?;
        }

        self.w2.forward(&gate, ffn_out, cuda_cfg)?;
        Ok(())
    }
}

// ============================================================================
//  DecoderLayer —— attn + rmsnorm + mlp + rmsnorm，两处 residual 走 fused op
// ============================================================================

/// 语义（对照 HuggingFace `LlamaDecoderLayer.forward`，使用跨层融合）：
///
/// 约定：进入本层的 `h_in` 已经是 `input_layernorm(x)` 的结果（由**上一层**的
/// 第 2 处 fused add+rmsnorm 顺手产出）。Layer 0 例外，由 [`Llama3::forward`]
/// 顶层在 embedding 之后**显式**跑一次 `layers[0].input_layernorm`。
/// 本层内因此**不再**显式调用 `self.input_layernorm`。
///
/// 1. `a      = self_attn(h_in)`
/// 2. `x     += a`；`h_mid = post_attention_rmsnorm(x)`         ← fused op
/// 3. `f      = mlp(h_mid)`
/// 4. `x     += f`；`h_out = next_input_rmsnorm(x)`             ← fused op
///
/// 第 4 步的 `next_input_rmsnorm` 是**下一层**的 `input_layernorm`，
/// 最后一层则传入模型级 `final norm`，让 `h_out` 直接等价于 `final_norm(x)`
/// —— 由此模型末尾不再需要一次独立的 final-norm 调用。
///
/// 跨层融合的效果：每层 **只做 2 次 rmsnorm**（都与前一个 add 融合），而不是
/// `input + post_attn + final` 的 3 次非融合版本。
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

    /// `x`                : residual stream，in-place 更新（本层结束时已加上 attn + ffn）。
    /// `h_in`              : 进入本层的 `input_layernorm(x)`，由上一层末尾的 fused op
    ///                       产出；layer 0 则由 [`Llama3::forward`] 顶层显式准备。
    /// `h_out`             : 本层结束时写入 —— 下一层的 `input_layernorm(x)`。
    ///                       最后一层则是 `final_norm(x)`。
    /// `next_input_rmsnorm`: 下一层的 `input_layernorm`；最后一层传入模型级 `norm`。
    fn forward(
        &self,
        x: &mut Tensor,
        h_in: &Tensor,
        h_out: &mut Tensor,
        layer_idx: usize,
        config: &RuntimeModelConfig,
        next_input_rmsnorm: &RMSNorm,
        ctx: &mut ForwardCtx<'_, '_>,
    ) -> Result<()> {
        // 1. a = self_attn(h_in) → 写到 block_out 复用 scratch
        let mut a = ctx.llm.block_out.clone();
        self.self_attn.forward(h_in, &mut a, layer_idx, config, ctx)?;

        // 2. x += a;  h_mid = post_attention_rmsnorm(x)
        let mut h_mid = ctx.llm.norm_out.clone();
        self.post_attention_layernorm
            .forward_with_residual(&mut h_mid, x, &a, ctx.cuda_cfg)?;

        // 3. f = mlp(h_mid) → 写到 block_out
        let mut f = ctx.llm.block_out.clone();
        self.mlp.forward(&h_mid, &mut f, config, ctx)?;

        // 4. x += f;  h_out = next_input_rmsnorm(x)
        //    —— 这步同时充当下一层 `input_layernorm` 的融合产物；模型入口的
        //       `input_layernorm` 仅在 layer 0 之前显式执行一次。
        next_input_rmsnorm.forward_with_residual(h_out, x, &f, ctx.cuda_cfg)
    }
}

// ============================================================================
//  Llama3 —— 对外门面
// ============================================================================

pub struct Llama3 {
    pub(crate) config: RuntimeModelConfig,
    pub(crate) device_type: DeviceType,
    pub(crate) tokenizer: Box<dyn Tokenizer>,

    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: Matmul,
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
        })
    }
}

// ============================================================================
//  LlmModel trait 实现 —— 纯 forward，假定 runner 已完成所有前置工作
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

    fn forward(&self, ctx: &mut ForwardCtx<'_, '_>) -> Result<()> {
        let cuda_cfg = ctx.cuda_cfg;

        // ── embedding ──
        let mut x = ctx.llm.hidden.clone();
        self.embed_tokens.forward(&ctx.llm.input_tokens, &mut x, cuda_cfg)?;

        // ── layer 0 的 input_rmsnorm：跨层 fuse 的"起点"──
        //
        // 之后每层不再显式跑 `input_layernorm`：由上一层末尾的 fused add+rmsnorm
        // 顺手产出，写到本 scratch。每层维护一个 `h` view 不断被覆盖、不断流转。
        let mut h = ctx.llm.norm_out.clone();
        self.layers[0].input_layernorm.forward(&x, &mut h, cuda_cfg)?;

        // ── N × DecoderLayer ──
        //
        // 最后一层的 `next_input_rmsnorm` 传入模型级 `self.norm`，使最终的 `h`
        // 直接等于 `final_norm(x)`，省掉一次独立的 final-norm 调用。
        let layer_num = self.layers.len();
        for i in 0..layer_num {
            let next_input_rmsnorm = if i + 1 < layer_num {
                &self.layers[i + 1].input_layernorm
            } else {
                &self.norm
            };
            // h_in / h_out 共用同一块 `norm_out` 底层存储：读完 h_in 走 attn+mlp
            // 后再写 h_out，二者时序不冲突。
            let h_in = h.clone();
            let mut h_out = ctx.llm.norm_out.clone();
            self.layers[i].forward(
                &mut x,
                &h_in,
                &mut h_out,
                i,
                &self.config,
                next_input_rmsnorm,
                ctx,
            )?;
            h = h_out;
        }
        // 循环结束后 `h` 即为 `final_norm(x)`。
        let final_norm_all = h;

        // ── 每个 seq 取最后一个 token 的 hidden state ──
        let sample_hidden = ctx.llm.sample_hidden.clone();
        for seq_idx in 0..ctx.num_seqs {
            let last = ctx.meta.seq_end(seq_idx) - 1;
            let src = final_norm_all.narrow(0, last, 1)?;
            let mut dst = sample_hidden.narrow(0, seq_idx, 1)?;
            dst.copy_from_on_current_stream(&src)?;
        }

        // ── lm_head → per-seq sampler ──
        let mut logits = ctx.llm.logits.clone();
        self.lm_head.forward(&sample_hidden, &mut logits, cuda_cfg)?;

        let tok_vocab = self.config.tokenizer_vocab_size;
        for i in 0..ctx.num_seqs {
            let logits_row = logits.narrow(0, i, 1)?;
            let logits_1d = logits_row.narrow(1, 0, tok_vocab)?.reshape(&[tok_vocab])?;
            let mut dst = ctx.output_tokens.narrow(0, i, 1)?;
            ctx.states[i].sampler.sample(&logits_1d, &mut dst, cuda_cfg)?;
        }
        Ok(())
    }
}
