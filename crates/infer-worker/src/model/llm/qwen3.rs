//! Qwen3 model —— 纯 forward 实现。
//!
//! 与 [`crate::model::llm::llama3`] 几乎同构（embedding → N × (norm + attention
//! + norm + mlp) → final norm → lm_head → sampling），跨层 fused add+rmsnorm
//! 的设计完全一样。唯一差异：**QK-norm**（Qwen3 特性）。
//!
//! QK-norm：QKV 投影 + split 后、RoPE 之前，分别对 Q 和 K 走一遍 per-head
//! RMSNorm。weight shape 是 `[head_size]`，对 strided 3-D 视图
//! `[T, head_num/kv_head_num, head_size]` 直接 in-place 做归一化（RMSNorm 算子
//! 原生支持 strided 输入）。weight 本身**可选** —— 取决于权重文件里是否含
//! `q_norm` / `k_norm`。
//!
//! 其它一切（fused gate_up MLP、跨层 fused norm、KV scatter、attention plan）
//! 都与 Llama3 的实现完全一致，详见 [`crate::model::llm::llama3`] 注释。

use std::boxed::Box;
use std::path::Path;

use crate::base::DeviceType;
use crate::base::error::Result;
use crate::model::ModelLoader;
use crate::model::common::config::RuntimeModelConfig;
use crate::model::common::tokenizer::Tokenizer;
use crate::model::common::{GateUpDims, QkvDims};
use crate::model::llm::{ForwardCtx, LlmModel};
use crate::op::attention::Attention as FlashAttn;
use crate::op::embedding::Embedding;
use crate::op::matmul::Matmul;
use crate::op::rmsnorm::RMSNorm;
use crate::op::rope::RoPEOp;
use crate::op::swiglu::swiglu;
use crate::tensor::Tensor;

// ============================================================================
//  Attention —— fused QKV + (optional) QK-norm + RoPE + flash GQA + o_proj
// ============================================================================

struct Attention {
    wqkv: Matmul,
    wo: Matmul,
    rope: RoPEOp,
    mha: FlashAttn,
    /// 可选 per-head Q-norm（weight shape `[head_size]`）。
    q_norm: Option<RMSNorm>,
    /// 可选 per-head K-norm（weight shape `[head_size]`）。
    k_norm: Option<RMSNorm>,
}

impl Attention {
    fn load(
        layer_idx: usize,
        loader: &ModelLoader,
        config: &RuntimeModelConfig,
        device: DeviceType,
        has_qnorm: bool,
        has_knorm: bool,
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

        let q_norm = if has_qnorm {
            Some(loader.load_rmsnorm(
                &format!("model.layers.{}.self_attn.q_norm.weight", layer_idx),
                device,
                config.rms_norm_eps,
            )?)
        } else {
            None
        };
        let k_norm = if has_knorm {
            Some(loader.load_rmsnorm(
                &format!("model.layers.{}.self_attn.k_norm.weight", layer_idx),
                device,
                config.rms_norm_eps,
            )?)
        } else {
            None
        };

        Ok(Self { wqkv, wo, rope, mha, q_norm, k_norm })
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
        let head_num = config.head_num;
        let kv_head_num = config.kv_head_num;
        let head_size = config.head_size;
        let total_tokens = ctx.total_tokens;
        let cuda_cfg = ctx.cuda_cfg;

        // ── wqkv ──
        let mut qkv = ctx.attn.qkv.clone();
        self.wqkv.forward(x_norm, &mut qkv, cuda_cfg)?;

        // ── split → Q/K/V（零拷贝 strided view）──
        //
        // 行 stride = q_dim + 2*kv_dim；最后一维 dense。下游 RoPE / scatter /
        // attention 都接受 strided 输入。
        let mut q = qkv.narrow(1, 0, q_dim)?;
        let mut k = qkv.narrow(1, q_dim, kv_dim)?;
        let v = qkv.narrow(1, q_dim + kv_dim, kv_dim)?;

        // ── Optional QK-norm（per-head RMSNorm，weight `[head_size]`）──
        //
        // RMSNorm 算子原生支持 strided 3-D `[T, n_heads, head_size]` 视图：
        // 通过 `unflatten` 零拷贝把 strided `[T, q_dim]` 视为 `[T, head_num,
        // head_size]`（strides = `[cols, head_size, 1]`），kernel 按
        // `T * head_num` 行做 norm。无需物化、无临时 buffer。
        if let Some(qn) = &self.q_norm {
            let mut q3 = q.unflatten(1, &[head_num, head_size])?;
            qn.forward_inplace(&mut q3, cuda_cfg)?;
        }
        if let Some(kn) = &self.k_norm {
            let mut k3 = k.unflatten(1, &[kv_head_num, head_size])?;
            kn.forward_inplace(&mut k3, cuda_cfg)?;
        }

        // ── RoPE on Q/K（in-place 写回 qkv 的 Q/K 列段）──
        self.rope.forward(
            &ctx.llm.input_pos,
            &ctx.sin_cache,
            &ctx.cos_cache,
            &mut q,
            &mut k,
            cuda_cfg,
        )?;

        // ── scatter K/V → per-seq KV cache ──
        crate::op::kv_cache::scatter(
            &k, &v, layer_idx, ctx.states, ctx.meta, ctx.workspace, cuda_cfg,
        )?;

        // ── flash attention（一次 launch 覆盖整个 batch）──
        // 对 qkv 整体做 3D reshape（零拷贝），然后 narrow head 维取 Q
        let total_heads = head_num + 2 * kv_head_num;
        let qkv_3d = qkv.reshape(&[total_tokens, total_heads, head_size])?;
        let q_for_attn = qkv_3d.narrow(1, 0, head_num)?;
        let mut attn_all_3d =
            ctx.attn.attn_merged.reshape(&[total_tokens, head_num, head_size])?;
        unsafe {
            self.mha.forward(&q_for_attn, &mut attn_all_3d, layer_idx, &ctx.attn_plan, cuda_cfg)?;
        }

        // ── o_proj ──（按 [T, q_dim] 视图喂回去）
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

        Ok(Self { w_gate_up, w2 })
    }

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
                let mut up = ctx.mlp.up.clone();
                crate::op::split_cols::split_cols_tensor(
                    &gate_up, &mut gate, total_tokens, 2 * inter, 0, inter, cuda_cfg,
                )?;
                crate::op::split_cols::split_cols_tensor(
                    &gate_up, &mut up, total_tokens, 2 * inter, inter, inter, cuda_cfg,
                )?;
                swiglu(&up, &mut gate, cuda_cfg)?;
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
            swiglu(&up, &mut gate, cuda_cfg)?;
        }

        self.w2.forward(&gate, ffn_out, cuda_cfg)?;
        Ok(())
    }
}

// ============================================================================
//  DecoderLayer —— attn + rmsnorm + mlp + rmsnorm，两处 residual 走 fused op
// ============================================================================

/// 跨层 fused 设计与 [`crate::model::llm::llama3::DecoderLayer`] 完全一致：
/// 进入本层的 `h_in` 已经是 `input_layernorm(x)` 的结果（由上一层末尾的 fused
/// add+rmsnorm 顺手产出），layer 0 由 [`Qwen3::forward`] 顶层显式跑一次。
///
/// 1. `a      = self_attn(h_in)`
/// 2. `x     += a`；`h_mid = post_attention_rmsnorm(x)`         ← fused op
/// 3. `f      = mlp(h_mid)`
/// 4. `x     += f`；`h_out = next_input_rmsnorm(x)`             ← fused op
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
        has_qnorm: bool,
        has_knorm: bool,
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
        let self_attn = Attention::load(layer_idx, loader, config, device, has_qnorm, has_knorm)?;
        let mlp = Mlp::load(layer_idx, loader, config, is_awq, device)?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            self_attn,
            mlp,
        })
    }

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
        next_input_rmsnorm.forward_with_residual(h_out, x, &f, ctx.cuda_cfg)
    }
}

// ============================================================================
//  Qwen3 —— 对外门面
// ============================================================================

pub struct Qwen3 {
    pub(crate) config: RuntimeModelConfig,
    pub(crate) device_type: DeviceType,
    pub(crate) tokenizer: Box<dyn Tokenizer>,

    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: Matmul,
}

impl Qwen3 {
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

        // Qwen3 与 Llama3 唯一结构性差异：可选的 per-head q_norm / k_norm。
        // 探测权重文件里是否包含相应键来决定是否加载。
        let has_qnorm = tensor_names.iter().any(|n| n.contains("q_norm"));
        let has_knorm = tensor_names.iter().any(|n| n.contains("k_norm"));

        let layers: Vec<DecoderLayer> = (0..config.layer_num)
            .map(|i| {
                DecoderLayer::load(i, &loader, &config, is_awq, device_type, has_qnorm, has_knorm)
            })
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

    /// Create an InferenceState for standalone (non-batch-runner) usage,
    /// e.g. text encoder in diffusion pipeline.
    pub fn create_state(&self) -> crate::base::error::Result<crate::model::runtime::InferenceState> {
        crate::model::runtime::InferenceState::new(&self.config, self.device_type)
    }

    /// Run prefill through `num_layers` layers and return hidden_states [seq_len, dim].
    /// Used by the diffusion text encoder which only needs intermediate hidden states,
    /// not the full LM head logits.
    ///
    /// This is a standalone path (no batch runner / ForwardCtx). It allocates temporary
    /// workspace internally — acceptable for text encoder use (seq_len ≤ 256, called once).
    pub fn forward_prefill_hidden_states(
        &self,
        state: &mut crate::model::runtime::InferenceState,
        input_tokens: &Tensor,
        seq_len: usize,
        num_layers: usize,
    ) -> crate::base::error::Result<Tensor> {
        use crate::OpConfig;
        #[cfg(feature = "cuda")]
        let cuda_cfg: Option<&OpConfig> = state.cuda_config.as_ref().map(|c| c as &OpConfig);
        #[cfg(not(feature = "cuda"))]
        let cuda_cfg: Option<&OpConfig> = None;

        let dtype = self.config.runtime_float_dtype(self.device_type)?;
        let dim = self.config.dim;
        let q_dim = self.config.q_dim;
        let kv_dim = self.config.kv_dim;
        let head_num = self.config.head_num;
        let kv_head_num = self.config.kv_head_num;
        let head_size = self.config.head_size;
        let inter = self.config.intermediate_size;

        // ── embedding ──
        let input_tokens_dev = if input_tokens.device() != self.device_type {
            input_tokens.to_device(self.device_type)?
        } else {
            input_tokens.clone()
        };
        let mut x = Tensor::new(&[seq_len, dim], dtype, self.device_type)?;
        self.embed_tokens.forward(&input_tokens_dev, &mut x, cuda_cfg)?;

        // ── sin/cos cache for RoPE ──
        let rope_cache_len = self.config.seq_len;
        let half_head = head_size / 2;
        let mut sin_cache = Tensor::new(&[rope_cache_len, half_head], dtype, self.device_type)?;
        let mut cos_cache = Tensor::new(&[rope_cache_len, half_head], dtype, self.device_type)?;
        crate::model::runtime::compute_rope_cache(&self.config, &mut sin_cache, &mut cos_cache)?;
        // positions [seq_len]: 0, 1, 2, ..., seq_len-1
        let mut input_pos = Tensor::new(&[seq_len], crate::base::DataType::I32, self.device_type)?;
        {
            let pos_data: Vec<i32> = (0..seq_len as i32).collect();
            input_pos.as_i32_mut()?.buffer_mut().copy_from_host(&pos_data)?;
        }

        // ── run layers ──
        let run_layers = num_layers.min(self.layers.len());
        let mut h = Tensor::new(&[seq_len, dim], dtype, self.device_type)?;
        self.layers[0].input_layernorm.forward(&x, &mut h, cuda_cfg)?;

        for i in 0..run_layers {
            // --- attention sublayer ---
            let qkv_cols = q_dim + 2 * kv_dim;
            let mut qkv = Tensor::new(&[seq_len, qkv_cols], dtype, self.device_type)?;
            self.layers[i].self_attn.wqkv.forward(&h, &mut qkv, cuda_cfg)?;

            // Q/K/V narrow views (strided)
            let mut q = qkv.narrow(1, 0, q_dim)?;
            let mut k = qkv.narrow(1, q_dim, kv_dim)?;
            let v = qkv.narrow(1, q_dim + kv_dim, kv_dim)?;

            // QK-norm (if present)
            if let Some(ref qn) = self.layers[i].self_attn.q_norm {
                let mut q3 = q.unflatten(1, &[head_num, head_size])?;
                qn.forward_inplace(&mut q3, cuda_cfg)?;
            }
            if let Some(ref kn) = self.layers[i].self_attn.k_norm {
                let mut k3 = k.unflatten(1, &[kv_head_num, head_size])?;
                kn.forward_inplace(&mut k3, cuda_cfg)?;
            }

            // RoPE
            self.layers[i].self_attn.rope.forward(
                &input_pos, &sin_cache, &cos_cache,
                &mut q, &mut k, cuda_cfg,
            )?;

            // Dense prefill attention
            let k_contig = k.contiguous()?;
            let v_contig = v.contiguous()?;
            let q_contig = q.contiguous()?;
            let mut attn_out = Tensor::new(&[seq_len, q_dim], dtype, self.device_type)?;
            #[cfg(feature = "cuda")]
            {
                let cuda_cfg_ref = state.cuda_config.as_ref();
                unsafe {
                    crate::op::kernels::cuda::flash_attn_gqa_prefill(
                        &q_contig, &k_contig, &v_contig, &mut attn_out,
                        seq_len, 0, // current_kv_len_host=0 (no prior cache)
                        head_num, kv_head_num, head_size,
                        true, // causal
                        cuda_cfg_ref,
                    )?;
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                // CPU fallback — not supported for now
                return Err(crate::base::error::Error::InvalidArgument(
                    "forward_prefill_hidden_states requires CUDA".into()
                ).into());
            }

            // o_proj
            let mut wo_out = Tensor::new(&[seq_len, dim], dtype, self.device_type)?;
            self.layers[i].self_attn.wo.forward(&attn_out, &mut wo_out, cuda_cfg)?;

            // residual + post_attention_layernorm
            let mut h_mid = Tensor::new(&[seq_len, dim], dtype, self.device_type)?;
            self.layers[i].post_attention_layernorm
                .forward_with_residual(&mut h_mid, &mut x, &wo_out, cuda_cfg)?;

            // --- MLP sublayer ---
            let mut gate_up = Tensor::new(&[seq_len, 2 * inter], dtype, self.device_type)?;
            self.layers[i].mlp.w_gate_up.forward(&h_mid, &mut gate_up, cuda_cfg)?;

            let mut gate = Tensor::new(&[seq_len, inter], dtype, self.device_type)?;
            #[cfg(feature = "cuda")]
            if matches!(self.device_type, crate::base::DeviceType::Cuda(_))
                && dtype == crate::base::DataType::BF16
            {
                crate::op::kernels::cuda::swiglu_packed_bf16(
                    &gate_up, &mut gate, seq_len, inter, cuda_cfg,
                )?;
            } else {
                let mut up = Tensor::new(&[seq_len, inter], dtype, self.device_type)?;
                crate::op::split_cols::split_cols_tensor(
                    &gate_up, &mut gate, seq_len, 2 * inter, 0, inter, cuda_cfg)?;
                crate::op::split_cols::split_cols_tensor(
                    &gate_up, &mut up, seq_len, 2 * inter, inter, inter, cuda_cfg)?;
                crate::op::swiglu::swiglu(&up, &mut gate, cuda_cfg)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let mut up = Tensor::new(&[seq_len, inter], dtype, self.device_type)?;
                crate::op::split_cols::split_cols_tensor(
                    &gate_up, &mut gate, seq_len, 2 * inter, 0, inter, cuda_cfg)?;
                crate::op::split_cols::split_cols_tensor(
                    &gate_up, &mut up, seq_len, 2 * inter, inter, inter, cuda_cfg)?;
                crate::op::swiglu::swiglu(&up, &mut gate, cuda_cfg)?;
            }

            let mut ffn_out = Tensor::new(&[seq_len, dim], dtype, self.device_type)?;
            self.layers[i].mlp.w2.forward(&gate, &mut ffn_out, cuda_cfg)?;

            // residual + next_input_layernorm
            let next_norm = if i + 1 < self.layers.len() {
                &self.layers[i + 1].input_layernorm
            } else {
                &self.norm
            };
            let mut h_out = Tensor::new(&[seq_len, dim], dtype, self.device_type)?;
            next_norm.forward_with_residual(&mut h_out, &mut x, &ffn_out, cuda_cfg)?;
            h = h_out;
        }

        // h = hidden_states after `run_layers` layers (including final norm)
        Ok(h)
    }
}

// ============================================================================
//  LlmModel trait 实现 —— 与 Llama3 同模板
// ============================================================================

impl LlmModel for Qwen3 {
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

        // ── layer 0 input_rmsnorm：跨层 fuse 的"起点"──
        let mut h = ctx.llm.norm_out.clone();
        self.layers[0].input_layernorm.forward(&x, &mut h, cuda_cfg)?;

        // ── N × DecoderLayer ──
        let layer_num = self.layers.len();
        for i in 0..layer_num {
            let next_input_rmsnorm = if i + 1 < layer_num {
                &self.layers[i + 1].input_layernorm
            } else {
                &self.norm
            };
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
