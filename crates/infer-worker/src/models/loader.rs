//! Weight loader — generic safetensors → tensor/`Linear`/`RMSNorm` primitives.
//!
//! This lives in models/ (not domain) because it's about constructing concrete
//! model tensors. It is deliberately *model-agnostic*: every method takes the
//! tensor name(s) from the caller and reads exactly what the file declares
//! (shape, dtype). Which tensors a given architecture needs — and how they wire
//! into components — lives in the model modules (`models/decoder.rs`, …), not
//! here. Filesystem access is delegated to `infra::io::SafetensorsReader`.

use safetensors::tensor::TensorView;

use super::layers::{Embedding, Linear, RMSNorm};
use crate::components::embed::{Embed as CompEmbed, EmbeddingParallelism};
use crate::components::linear::{Linear as CompLinear, LinearParallelism};
use crate::domain::dtype::Fp8E4m3;
use crate::domain::dtype::quant::QuantScheme;
use crate::domain::exec::RankPair;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{MemoryPort, OpBackend, OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype, Shape};
use crate::infrastructure::io::SafetensorsReader;

/// Llama-3 NTK-aware RoPE scaling parameters.
///
/// Mirrors the `rope_scaling` block in HuggingFace `config.json` when
/// `rope_type == "llama3"`. The frequency rescaling math lives in
/// `compute_rope_cache`.
#[derive(Debug, Clone, Copy)]
pub struct RopeScaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: u32,
}

/// Gated-DeltaNet (linear-attention) dimensions for the Qwen3.5 hybrid stack.
///
/// These describe the *recurrent* mixer that replaces full attention in 24 of
/// the 32 layers. They are attributes of the checkpoint, not a distinct config
/// type: a dense Llama/Qwen3 model simply has `LoadConfig::linear_attn = None`
/// and every layer is full attention. When `Some`, `layer_is_full[i]` selects
/// per layer which mixer (full vs Gated-DeltaNet) layer `i` uses.
#[derive(Debug, Clone)]
pub struct LinearAttnConfig {
    /// Number of key/query heads (query is L2-normed, shared GQA-style across
    /// value heads). Qwen3.5-4B: 16.
    pub num_key_heads: usize,
    /// Number of value heads (one recurrent state per value head). 32.
    pub num_value_heads: usize,
    /// Per-head dim for keys/queries. 128 → key_dim = 16*128 = 2048.
    pub key_head_dim: usize,
    /// Per-head dim for values. 128 → value_dim = 32*128 = 4096.
    pub value_head_dim: usize,
    /// Causal depthwise conv kernel width over the concatenated qkv channels. 4.
    pub conv_kernel_dim: usize,
    /// Per-layer mixer selector: `true` = full attention, `false` = Gated
    /// DeltaNet. Length == `LoadConfig::layer_num`.
    pub layer_is_full: Vec<bool>,
}

impl LinearAttnConfig {
    /// key_dim = num_key_heads * key_head_dim (also the q channel width).
    pub fn key_dim(&self) -> usize {
        self.num_key_heads * self.key_head_dim
    }
    /// value_dim = num_value_heads * value_head_dim.
    pub fn value_dim(&self) -> usize {
        self.num_value_heads * self.value_head_dim
    }
    /// conv operates over q(key_dim) | k(key_dim) | v(value_dim).
    pub fn conv_dim(&self) -> usize {
        self.key_dim() + self.key_dim() + self.value_dim()
    }
    /// Count of full-attention layers (the paged-KV pool is sized to these).
    pub fn num_full_layers(&self) -> usize {
        self.layer_is_full.iter().filter(|&&f| f).count()
    }
    /// Count of Gated-DeltaNet layers (the LinearStatePool is sized to these).
    pub fn num_linear_layers(&self) -> usize {
        self.layer_is_full.len() - self.num_full_layers()
    }
}

/// Configuration for model loading.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub dim: usize,
    pub intermediate_size: usize,
    pub layer_num: usize,
    pub head_num: usize,
    pub kv_head_num: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub rms_norm_eps: f32,
    /// RoPE theta (base frequency). Default 10000.0 for Llama3, 1000000.0 for Qwen3.
    pub rope_theta: f64,
    /// Optional NTK-aware frequency rescaling (Llama-3 family).
    pub rope_scaling: Option<RopeScaling>,
    /// When `Some`, the MLP `gate/up/down` projections are int4 group-quantized
    /// (compressed-tensors `pack-quantized`) with this scheme; attention and
    /// the head stay full-precision. `None` → a fully dense model.
    pub mlp_quant: Option<QuantScheme>,
    /// Block shape for a homogeneous FP8 checkpoint's `[N, K]` decoder linear
    /// weights. The raw E4M3 weights and their inverse-scale grid stay quantized
    /// on device; embeddings, norms, and the LM head remain in their stored
    /// dtype. A non-FP8 decoder projection is treated as a config mismatch.
    pub fp8_block: Option<[usize; 2]>,
    /// Number of head dims that receive RoPE. Equals `head_dim` for the usual
    /// full-rotary case; Qwen3.5 full-attn layers use partial rotary
    /// (`partial_rotary_factor = 0.25` → 64 of 256). Attribute, not a branch:
    /// the rope cache is simply built over `rotary_dim` columns.
    pub rotary_dim: usize,
    /// Qwen3.5 full-attn `attn_output_gate`: `q_proj` emits `[gate | query]` and
    /// the attention output is elementwise-gated by `sigmoid(gate)`. `false` for
    /// every model we ship today.
    pub attn_output_gate: bool,
    /// `Some` for the Qwen3.5 hybrid stack (Gated-DeltaNet + full attention).
    /// `None` → homogeneous full-attention decoder (Llama3 / Qwen3).
    pub linear_attn: Option<LinearAttnConfig>,
    /// MoE expert count. `0` for dense models.
    pub num_experts: usize,
    /// Top-k routed experts per token. `0` for dense models.
    pub experts_per_tok: usize,
    /// Per-expert SwiGLU intermediate width. `0` for dense models.
    pub moe_intermediate_size: usize,
    /// Whether router top-k probabilities are renormalized after top-k.
    pub norm_topk_prob: bool,
    /// HF sparse-layer interval. Qwen3-30B-A3B uses `1` (every layer is MoE).
    pub decoder_sparse_step: usize,
}

/// Weight loader — pulls tensors out of a `SafetensorsReader` and builds
/// typed model structs. The reader is borrowed (no copy until upload).
pub struct WeightLoader<'a> {
    reader: &'a SafetensorsReader,
    tp: RankPair,
}

impl<'a> WeightLoader<'a> {
    /// Wrap a reader for the default single-rank load. The reader owns the
    /// mmap; the loader only borrows.
    pub fn new(reader: &'a SafetensorsReader) -> Self {
        Self {
            reader,
            tp: RankPair { rank: 0, size: 1 },
        }
    }

    /// Wrap a reader and select the shard uploaded by this TP worker.
    pub fn with_tensor_parallel(
        reader: &'a SafetensorsReader,
        rank: usize,
        size: usize,
    ) -> OpResult<Self> {
        let tp = validate_tp(RankPair { rank, size })?;
        Ok(Self { reader, tp })
    }

    pub fn tensor_parallel(&self) -> RankPair {
        self.tp
    }

    /// Convert a global dimension into this rank's even local shard size.
    pub(crate) fn local_shard_size(&self, what: &str, global: usize) -> OpResult<usize> {
        even_shard_range(what, global, self.tp).map(|(_, len)| len)
    }

    fn require_tp1(&self, what: &str) -> OpResult<()> {
        if self.tp.size != 1 {
            return Err(OpError::Kernel(format!(
                "{} does not support TP{} yet; only dense BF16/F16/F32 weights can be sharded",
                what, self.tp.size
            )));
        }
        Ok(())
    }

    /// Whether a tensor with this name exists in the underlying file(s).
    pub fn has_tensor(&self, name: &str) -> bool {
        self.reader.contains(name)
    }

    /// Borrow a raw safetensors `TensorView` by name.
    pub fn read_view(&self, name: &str) -> Result<TensorView<'_>, String> {
        self.reader.read_view(name)
    }

    /// Load a tensor by name, cast to target dtype T, place on device D.
    pub fn load_tensor<T: Dtype, D: MemoryPort>(
        &self,
        name: &str,
        device: &D,
    ) -> OpResult<Tensor<T, D>> {
        let view = self
            .reader
            .read_view(name)
            .map_err(|e| OpError::Kernel(format!("tensor '{}' not found: {}", name, e)))?;
        tensor_from_safetensor_view::<T, D>(&view, device)
    }

    /// Load a Linear layer (weight + optional bias).
    pub fn load_linear<T: Dtype, D: OpBackend>(
        &self,
        weight_name: &str,
        bias_name: Option<&str>,
        device: &D,
    ) -> OpResult<Linear<T, D>> {
        let weight = self.load_tensor::<T, D>(weight_name, device)?;
        let bias = if let Some(bn) = bias_name {
            Some(self.load_tensor::<T, D>(bn, device)?)
        } else {
            None
        };
        Ok(Linear::new(weight, bias))
    }

    /// Load a row-parallel decoder Linear by sharding the input-feature axis.
    /// A configured block-FP8 weight stays quantized on device; FP8 TP>1 is
    /// rejected until its scale-grid sharding is implemented.
    pub fn load_row_parallel_linear_with_fp8<T: Dtype, D: OpBackend + LlmBackend>(
        &self,
        weight_name: &str,
        bias_name: Option<&str>,
        fp8_block: Option<[usize; 2]>,
        device: &D,
    ) -> OpResult<CompLinear<T, D>> {
        let Some(block) = fp8_block else {
            let view = self.reader.read_view(weight_name).map_err(|e| {
                OpError::Kernel(format!("tensor '{}' not found: {}", weight_name, e))
            })?;
            let host = prepare_matrix_shard::<T>(
                weight_name,
                &view,
                MatrixShardAxis::InputColumns,
                self.tp,
            )?;
            let weight = Tensor::<T, D>::from_host_bytes(
                &host.bytes,
                Shape::from_slice(&host.shape),
                device,
            )?;
            let bias = if let Some(bn) = bias_name {
                Some(self.load_tensor::<T, D>(bn, device)?)
            } else {
                None
            };
            return Ok(CompLinear::new(weight, bias)
                .with_parallelism(LinearParallelism::Row { tp: self.tp }));
        };

        self.require_tp1("block-FP8 row-parallel Linear")?;

        let view = self
            .reader
            .read_view(weight_name)
            .map_err(|e| OpError::Kernel(format!("tensor '{}' not found: {}", weight_name, e)))?;
        let scale_name = format!("{}_scale_inv", weight_name);
        let scale_view = self.reader.read_view(&scale_name).map_err(|e| {
            OpError::Kernel(format!(
                "FP8 tensor '{}' is missing scale tensor '{}': {}",
                weight_name, scale_name, e
            ))
        })?;
        let parts = [Fp8ViewPart {
            name: weight_name,
            weight: view,
            scale_inv: scale_view,
        }];
        let host = prepare_fp8_fused_host(&parts, block, weight_name)?;
        let weight = Tensor::<Fp8E4m3, D>::from_host_bytes(
            &host.weight,
            Shape::from_slice(&host.weight_shape),
            device,
        )?;
        let scales = Tensor::<f32, D>::from_host_bytes(
            &host.scales,
            Shape::from_slice(&host.scale_shape),
            device,
        )?;
        let bias = if let Some(bn) = bias_name {
            Some(self.load_tensor::<T, D>(bn, device)?)
        } else {
            None
        };
        Ok(CompLinear::from_fp8_block(weight, scales, block, bias)
            .with_parallelism(LinearParallelism::Row { tp: self.tp }))
    }

    /// Load an RMSNorm layer.
    pub fn load_rmsnorm<T: Dtype, D: OpBackend>(
        &self,
        name: &str,
        device: &D,
        eps: f32,
    ) -> OpResult<RMSNorm<T, D>> {
        let weight = self.load_tensor::<T, D>(name, device)?;
        Ok(RMSNorm::new(weight, eps))
    }

    /// Load an Embedding table.
    pub fn load_embedding<T: Dtype, D: OpBackend>(
        &self,
        name: &str,
        device: &D,
    ) -> OpResult<Embedding<T, D>> {
        let table = self.load_tensor::<T, D>(name, device)?;
        Ok(Embedding { table })
    }

    /// Load an LLM token embedding table sharded over vocabulary rows.
    ///
    /// The checkpoint tensor remains logically `[global_vocab_size, dim]`;
    /// only this rank's contiguous row range is uploaded. The returned
    /// component owns the corresponding global-id offset so its forward can
    /// mask non-local token ids before the TP all-reduce.
    pub fn load_vocab_parallel_embedding<T: Dtype, D: OpBackend + LlmBackend>(
        &self,
        name: &str,
        global_vocab_size: usize,
        dim: usize,
        device: &D,
    ) -> OpResult<CompEmbed<T, D>> {
        let view = self
            .reader
            .read_view(name)
            .map_err(|e| OpError::Kernel(format!("tensor '{}': {}", name, e)))?;
        validate_matrix_shape(name, &view, global_vocab_size, dim)?;
        let (vocab_start, _) = even_shard_range(name, global_vocab_size, self.tp)?;
        let host = prepare_matrix_shard::<T>(name, &view, MatrixShardAxis::OutputRows, self.tp)?;
        let table =
            Tensor::<T, D>::from_host_bytes(&host.bytes, Shape::from_slice(&host.shape), device)?;
        let parallelism = if self.tp.size == 1 {
            EmbeddingParallelism::Replicated { tp: self.tp }
        } else {
            EmbeddingParallelism::Vocab {
                tp: self.tp,
                vocab_start,
                global_vocab_size,
            }
        };
        Ok(CompEmbed::new(table).with_parallelism(parallelism))
    }

    /// Load an LM-head weight sharded over its output/vocabulary rows.
    /// An optional output bias is sliced over exactly the same row range.
    pub fn load_vocab_parallel_linear<T: Dtype, D: OpBackend + LlmBackend>(
        &self,
        weight_name: &str,
        bias_name: Option<&str>,
        global_vocab_size: usize,
        dim: usize,
        device: &D,
    ) -> OpResult<CompLinear<T, D>> {
        let view = self
            .reader
            .read_view(weight_name)
            .map_err(|e| OpError::Kernel(format!("tensor '{}': {}", weight_name, e)))?;
        validate_matrix_shape(weight_name, &view, global_vocab_size, dim)?;
        let host =
            prepare_matrix_shard::<T>(weight_name, &view, MatrixShardAxis::OutputRows, self.tp)?;
        let weight =
            Tensor::<T, D>::from_host_bytes(&host.bytes, Shape::from_slice(&host.shape), device)?;
        self.vocab_parallel_linear_from_weight(weight, bias_name, global_vocab_size, device)
    }

    /// Wrap an already-loaded local vocabulary shard as an LM head.
    ///
    /// This is the tied-weight path: callers clone the local embedding tensor,
    /// which shares its allocation, then this method only attaches the LM-head
    /// parallel semantics and (when present) loads the matching local bias.
    pub fn vocab_parallel_linear_from_weight<T: Dtype, D: OpBackend + LlmBackend>(
        &self,
        weight: Tensor<T, D>,
        bias_name: Option<&str>,
        global_vocab_size: usize,
        device: &D,
    ) -> OpResult<CompLinear<T, D>> {
        let bias = match bias_name {
            Some(name) => {
                Some(self.load_vocab_parallel_bias::<T, D>(name, global_vocab_size, device)?)
            }
            None => None,
        };
        make_vocab_parallel_linear(weight, bias, self.tp, global_vocab_size)
    }

    fn load_vocab_parallel_bias<T: Dtype, D: MemoryPort>(
        &self,
        name: &str,
        global_vocab_size: usize,
        device: &D,
    ) -> OpResult<Tensor<T, D>> {
        let view = self
            .reader
            .read_view(name)
            .map_err(|e| OpError::Kernel(format!("tensor '{}': {}", name, e)))?;
        let host = prepare_vector_shard::<T>(name, &view, global_vocab_size, self.tp)?;
        Tensor::<T, D>::from_host_bytes(&host.bytes, Shape::from_slice(&[host.len]), device)
    }

    /// Load fused QKV: concatenate q/k/v_proj along rows → [q_dim+2*kv_dim, dim].
    ///
    /// `prefix` is the caller-supplied tensor-name stem (e.g.
    /// `"model.layers.0"`); the three projections are read as
    /// `{prefix}.self_attn.{q,k,v}_proj.weight`. Keeping the naming in the
    /// caller is deliberate — the loader stays a generic primitive that knows
    /// nothing about any model's layout.
    ///
    /// To keep the tensor on `device` and avoid host↔device round-trips, we
    /// concatenate at the host-bytes level (in target dtype T) and upload
    /// the fused result in a single `Tensor::from_host_bytes`.
    pub fn load_fused_qkv<T: Dtype, D: OpBackend>(
        &self,
        prefix: &str,
        q_dim: usize,
        kv_dim: usize,
        dim: usize,
        device: &D,
    ) -> OpResult<Linear<T, D>> {
        let q_name = format!("{}.self_attn.q_proj.weight", prefix);
        let k_name = format!("{}.self_attn.k_proj.weight", prefix);
        let v_name = format!("{}.self_attn.v_proj.weight", prefix);
        let read = |name: &str| {
            self.reader
                .read_view(name)
                .map_err(|e| OpError::Kernel(format!("tensor '{}': {}", name, e)))
        };
        let q_view = read(&q_name)?;
        let k_view = read(&k_name)?;
        let v_view = read(&v_name)?;

        validate_matrix_shape(&q_name, &q_view, q_dim, dim)?;
        validate_matrix_shape(&k_name, &k_view, kv_dim, dim)?;
        validate_matrix_shape(&v_name, &v_view, kv_dim, dim)?;

        // Q/K/V must be sharded independently before fusion. Taking one
        // contiguous slice from the globally fused [Q; K; V] matrix would mix
        // projection boundaries for every rank after rank 0.
        let host = prepare_fused_output_shards::<T>(
            "fused QKV",
            &[(&q_name, &q_view), (&k_name, &k_view), (&v_name, &v_view)],
            self.tp,
        )?;
        let fused =
            Tensor::<T, D>::from_host_bytes(&host.bytes, Shape::from_slice(&host.shape), device)?;
        Ok(Linear::new(fused, None))
    }

    /// Decoder-component form of [`Self::load_fused_qkv`]. For block FP8, raw
    /// E4M3 rows and scale-grid rows are fused independently without expanding
    /// the weight to the activation dtype.
    pub fn load_fused_qkv_with_fp8<T: Dtype, D: OpBackend + LlmBackend>(
        &self,
        prefix: &str,
        q_dim: usize,
        kv_dim: usize,
        dim: usize,
        fp8_block: Option<[usize; 2]>,
        device: &D,
    ) -> OpResult<CompLinear<T, D>> {
        let Some(block) = fp8_block else {
            let dense = self.load_fused_qkv::<T, D>(prefix, q_dim, kv_dim, dim, device)?;
            return Ok(CompLinear::new(dense.weight, dense.bias).with_parallelism(
                LinearParallelism::Column {
                    tp: self.tp,
                    gather_output: false,
                },
            ));
        };

        self.require_tp1("block-FP8 column-parallel QKV")?;

        let q_name = format!("{}.self_attn.q_proj.weight", prefix);
        let k_name = format!("{}.self_attn.k_proj.weight", prefix);
        let v_name = format!("{}.self_attn.v_proj.weight", prefix);
        let read = |name: &str| {
            self.reader
                .read_view(name)
                .map_err(|e| OpError::Kernel(format!("tensor '{}': {}", name, e)))
        };
        let q_view = read(&q_name)?;
        let k_view = read(&k_name)?;
        let v_view = read(&v_name)?;

        validate_matrix_shape(&q_name, &q_view, q_dim, dim)?;
        validate_matrix_shape(&k_name, &k_view, kv_dim, dim)?;
        validate_matrix_shape(&v_name, &v_view, kv_dim, dim)?;

        let q_scale_name = format!("{}_scale_inv", q_name);
        let k_scale_name = format!("{}_scale_inv", k_name);
        let v_scale_name = format!("{}_scale_inv", v_name);
        let q_scale = read(&q_scale_name)?;
        let k_scale = read(&k_scale_name)?;
        let v_scale = read(&v_scale_name)?;
        let parts = [
            Fp8ViewPart {
                name: &q_name,
                weight: q_view,
                scale_inv: q_scale,
            },
            Fp8ViewPart {
                name: &k_name,
                weight: k_view,
                scale_inv: k_scale,
            },
            Fp8ViewPart {
                name: &v_name,
                weight: v_view,
                scale_inv: v_scale,
            },
        ];
        let host = prepare_fp8_fused_host(&parts, block, "fused_qkv")?;
        let weight = Tensor::<Fp8E4m3, D>::from_host_bytes(
            &host.weight,
            Shape::from_slice(&host.weight_shape),
            device,
        )?;
        let scales = Tensor::<f32, D>::from_host_bytes(
            &host.scales,
            Shape::from_slice(&host.scale_shape),
            device,
        )?;
        Ok(
            CompLinear::from_fp8_block(weight, scales, block, None).with_parallelism(
                LinearParallelism::Column {
                    tp: self.tp,
                    gather_output: false,
                },
            ),
        )
    }

    /// Load fused gate_up: concatenate gate_proj, up_proj along rows
    /// → `[2*intermediate_size, dim]`.
    ///
    /// One GEMV computes both gate and up in a single launch; downstream
    /// `swiglu_packed` consumes the fused output without splitting.
    pub fn load_fused_gate_up<T: Dtype, D: OpBackend>(
        &self,
        prefix: &str,
        intermediate_size: usize,
        dim: usize,
        device: &D,
    ) -> OpResult<Linear<T, D>> {
        let gate_name = format!("{}.mlp.gate_proj.weight", prefix);
        let up_name = format!("{}.mlp.up_proj.weight", prefix);
        let gate_view = self
            .reader
            .read_view(&gate_name)
            .map_err(|e| OpError::Kernel(format!("tensor '{}': {}", gate_name, e)))?;
        let up_view = self
            .reader
            .read_view(&up_name)
            .map_err(|e| OpError::Kernel(format!("tensor '{}': {}", up_name, e)))?;

        validate_matrix_shape(&gate_name, &gate_view, intermediate_size, dim)?;
        validate_matrix_shape(&up_name, &up_view, intermediate_size, dim)?;
        let host = prepare_fused_output_shards::<T>(
            "fused gate/up",
            &[(&gate_name, &gate_view), (&up_name, &up_view)],
            self.tp,
        )?;
        let fused =
            Tensor::<T, D>::from_host_bytes(&host.bytes, Shape::from_slice(&host.shape), device)?;
        Ok(Linear::new(fused, None))
    }

    /// Decoder-component form of [`Self::load_fused_gate_up`]. For block FP8,
    /// raw E4M3 rows and scale-grid rows are fused independently without host
    /// dequantization.
    pub fn load_fused_gate_up_with_fp8<T: Dtype, D: OpBackend + LlmBackend>(
        &self,
        prefix: &str,
        intermediate_size: usize,
        dim: usize,
        fp8_block: Option<[usize; 2]>,
        device: &D,
    ) -> OpResult<CompLinear<T, D>> {
        let Some(block) = fp8_block else {
            let dense = self.load_fused_gate_up::<T, D>(prefix, intermediate_size, dim, device)?;
            return Ok(CompLinear::new(dense.weight, dense.bias).with_parallelism(
                LinearParallelism::Column {
                    tp: self.tp,
                    gather_output: false,
                },
            ));
        };

        self.require_tp1("block-FP8 column-parallel gate/up")?;

        let gate_name = format!("{}.mlp.gate_proj.weight", prefix);
        let up_name = format!("{}.mlp.up_proj.weight", prefix);
        let gate_view = self
            .reader
            .read_view(&gate_name)
            .map_err(|e| OpError::Kernel(format!("tensor '{}': {}", gate_name, e)))?;
        let up_view = self
            .reader
            .read_view(&up_name)
            .map_err(|e| OpError::Kernel(format!("tensor '{}': {}", up_name, e)))?;

        validate_matrix_shape(&gate_name, &gate_view, intermediate_size, dim)?;
        validate_matrix_shape(&up_name, &up_view, intermediate_size, dim)?;

        let gate_scale_name = format!("{}_scale_inv", gate_name);
        let up_scale_name = format!("{}_scale_inv", up_name);
        let gate_scale = self
            .reader
            .read_view(&gate_scale_name)
            .map_err(|e| OpError::Kernel(format!("tensor '{}': {}", gate_scale_name, e)))?;
        let up_scale = self
            .reader
            .read_view(&up_scale_name)
            .map_err(|e| OpError::Kernel(format!("tensor '{}': {}", up_scale_name, e)))?;
        let parts = [
            Fp8ViewPart {
                name: &gate_name,
                weight: gate_view,
                scale_inv: gate_scale,
            },
            Fp8ViewPart {
                name: &up_name,
                weight: up_view,
                scale_inv: up_scale,
            },
        ];
        let host = prepare_fp8_fused_host(&parts, block, "fused_gate_up")?;
        let weight = Tensor::<Fp8E4m3, D>::from_host_bytes(
            &host.weight,
            Shape::from_slice(&host.weight_shape),
            device,
        )?;
        let scales = Tensor::<f32, D>::from_host_bytes(
            &host.scales,
            Shape::from_slice(&host.scale_shape),
            device,
        )?;
        Ok(
            CompLinear::from_fp8_block(weight, scales, block, None).with_parallelism(
                LinearParallelism::Column {
                    tp: self.tp,
                    gather_output: false,
                },
            ),
        )
    }

    // ─── AWQ / compressed-tensors `pack-quantized` MLP loading ───────────────
    //
    // Only the MLP `gate/up/down` projections are int4-quantized; the tensors
    // are stored verbatim (no cast) and fed straight to the `matmul_quant`
    // kernel. `gate` and `up` are fused by vertical (row / `N`-axis) concat,
    // mirroring the dense `load_fused_gate_up` so downstream SwiGLU is
    // unchanged. `N` is divisible by 8, so the zero-point rows (`[N/8, g]`,
    // packed 8-along-`N`) concatenate on a clean word boundary.

    /// Vertically concatenate two same-width row-major safetensors views of
    /// dtype `E` into a fresh device tensor `[rows_a + rows_b, cols]`. Bytes
    /// are copied verbatim (no cast), so the view dtype must already be `E`.
    fn fuse_rows_verbatim<E: Dtype, D: MemoryPort>(
        &self,
        a: &TensorView,
        b: &TensorView,
        what: &str,
        device: &D,
    ) -> OpResult<Tensor<E, D>> {
        let (sa, sb) = (a.shape(), b.shape());
        if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[1] {
            return Err(OpError::Shape(format!(
                "{}: cannot fuse views of shape {:?} and {:?}",
                what, sa, sb
            )));
        }
        if st_dtype(a)? != E::DATA_TYPE || st_dtype(b)? != E::DATA_TYPE {
            return Err(OpError::Kernel(format!(
                "{}: expected dtype {:?}, got {:?}/{:?}",
                what,
                E::DATA_TYPE,
                a.dtype(),
                b.dtype()
            )));
        }
        let mut host = Vec::with_capacity(a.data().len() + b.data().len());
        host.extend_from_slice(a.data());
        host.extend_from_slice(b.data());
        Tensor::<E, D>::from_host_bytes(&host, Shape::from_slice(&[sa[0] + sb[0], sa[1]]), device)
    }

    /// Load a single int4 (`pack-quantized`) projection — e.g. `down_proj` —
    /// into a quantized `Linear`. `prefix` is the tensor-name stem, e.g.
    /// `"model.layers.0.mlp.down_proj"`.
    pub(crate) fn load_awq_linear<T: Dtype, D: OpBackend + LlmBackend>(
        &self,
        prefix: &str,
        scheme: QuantScheme,
        device: &D,
    ) -> OpResult<CompLinear<T, D>> {
        self.require_tp1("AWQ row-parallel Linear")?;
        let packed = self.load_tensor::<i32, D>(&format!("{}.weight_packed", prefix), device)?;
        let zeros = self.load_tensor::<i32, D>(&format!("{}.weight_zero_point", prefix), device)?;
        let scales = self.load_tensor::<T, D>(&format!("{}.weight_scale", prefix), device)?;
        Ok(CompLinear::from_awq(packed, zeros, scales, scheme, None)
            .with_parallelism(LinearParallelism::Row { tp: self.tp }))
    }

    /// Load int4 `gate_proj` + `up_proj` fused along rows into one quantized
    /// `Linear` (`[2*inter, K/8]` packed), matching the dense fused layout.
    /// `mlp_prefix` is the caller-supplied MLP stem, e.g. `"model.layers.0.mlp"`.
    pub(crate) fn load_fused_gate_up_awq<T: Dtype, D: OpBackend + LlmBackend>(
        &self,
        mlp_prefix: &str,
        scheme: QuantScheme,
        device: &D,
    ) -> OpResult<CompLinear<T, D>> {
        self.require_tp1("AWQ column-parallel gate/up")?;
        let view = |proj: &str, part: &str| -> OpResult<TensorView<'_>> {
            let name = format!("{}.{}.{}", mlp_prefix, proj, part);
            self.reader
                .read_view(&name)
                .map_err(|e| OpError::Kernel(format!("{}: {}", name, e)))
        };
        let packed = self.fuse_rows_verbatim::<i32, D>(
            &view("gate_proj", "weight_packed")?,
            &view("up_proj", "weight_packed")?,
            "fused_gate_up_awq packed",
            device,
        )?;
        let zeros = self.fuse_rows_verbatim::<i32, D>(
            &view("gate_proj", "weight_zero_point")?,
            &view("up_proj", "weight_zero_point")?,
            "fused_gate_up_awq zeros",
            device,
        )?;
        let scales = self.fuse_rows_verbatim::<T, D>(
            &view("gate_proj", "weight_scale")?,
            &view("up_proj", "weight_scale")?,
            "fused_gate_up_awq scales",
            device,
        )?;
        Ok(
            CompLinear::from_awq(packed, zeros, scales, scheme, None).with_parallelism(
                LinearParallelism::Column {
                    tp: self.tp,
                    gather_output: false,
                },
            ),
        )
    }
}

// ─── Internal: convert safetensor view to Tensor<T, D> ───────────────────────

fn make_vocab_parallel_linear<T: Dtype, D: LlmBackend>(
    weight: Tensor<T, D>,
    bias: Option<Tensor<T, D>>,
    tp: RankPair,
    global_vocab_size: usize,
) -> OpResult<CompLinear<T, D>> {
    let (_, expected_local) = even_shard_range("LM head vocabulary", global_vocab_size, tp)?;
    let weight_shape = weight.shape().as_slice();
    if weight_shape.len() != 2 || weight_shape[0] != expected_local {
        return Err(OpError::Shape(format!(
            "LM head local weight must have {} rows for rank {}/{}, got {:?}",
            expected_local, tp.rank, tp.size, weight_shape
        )));
    }
    if let Some(local_bias) = &bias
        && local_bias.shape().as_slice() != [expected_local]
    {
        return Err(OpError::Shape(format!(
            "LM head local bias must have shape [{}] for rank {}/{}, got {:?}",
            expected_local,
            tp.rank,
            tp.size,
            local_bias.shape().as_slice()
        )));
    }
    let parallelism = if tp.size == 1 {
        LinearParallelism::Replicated { tp }
    } else {
        LinearParallelism::Column {
            tp,
            gather_output: true,
        }
    };
    Ok(CompLinear::new(weight, bias).with_parallelism(parallelism))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixShardAxis {
    /// Column-parallel Linear: split logical `[N, K]` on `N`.
    OutputRows,
    /// Row-parallel Linear: split logical `[N, K]` on `K`.
    InputColumns,
}

#[derive(Debug, PartialEq, Eq)]
struct MatrixShardHost {
    bytes: Vec<u8>,
    shape: [usize; 2],
}

#[derive(Debug, PartialEq, Eq)]
struct VectorShardHost {
    bytes: Vec<u8>,
    len: usize,
}

fn validate_tp(tp: RankPair) -> OpResult<RankPair> {
    if tp.size == 0 {
        return Err(OpError::Shape(
            "tensor parallel size must be nonzero".into(),
        ));
    }
    if tp.rank >= tp.size {
        return Err(OpError::Shape(format!(
            "tensor parallel rank {} is outside size {}",
            tp.rank, tp.size
        )));
    }
    Ok(tp)
}

fn even_shard_range(what: &str, global: usize, tp: RankPair) -> OpResult<(usize, usize)> {
    let tp = validate_tp(tp)?;
    if global == 0 {
        return Err(OpError::Shape(format!(
            "{}: cannot shard an empty dimension",
            what
        )));
    }
    if !global.is_multiple_of(tp.size) {
        return Err(OpError::Shape(format!(
            "{}: dimension {} is not divisible by TP size {}",
            what, global, tp.size
        )));
    }
    let local = global / tp.size;
    Ok((tp.rank * local, local))
}

/// Cast a rank-2 safetensor to `T`, then retain only this rank's matrix shard.
/// The returned host buffer is contiguous and is the only data uploaded to the
/// device. Row shards are contiguous; column shards are gathered row by row.
fn prepare_matrix_shard<T: Dtype>(
    name: &str,
    view: &TensorView<'_>,
    axis: MatrixShardAxis,
    tp: RankPair,
) -> OpResult<MatrixShardHost> {
    let shape = view.shape();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(OpError::Shape(format!(
            "tensor '{}': TP sharding requires a non-empty rank-2 matrix, got {:?}",
            name, shape
        )));
    }
    let (rows, cols) = (shape[0], shape[1]);
    let src_dtype = st_dtype(view)?;
    let src_elem_bytes = src_dtype.size_in_bytes();
    let expected_src_bytes = rows
        .checked_mul(cols)
        .and_then(|numel| numel.checked_mul(src_elem_bytes))
        .ok_or_else(|| OpError::Shape(format!("tensor '{}': byte size overflows", name)))?;
    if view.data().len() != expected_src_bytes {
        return Err(OpError::Shape(format!(
            "tensor '{}': byte length {} does not match shape {:?} and dtype {:?}",
            name,
            view.data().len(),
            shape,
            view.dtype()
        )));
    }

    match axis {
        MatrixShardAxis::OutputRows => {
            let (row_start, local_rows) = even_shard_range(name, rows, tp)?;
            let src_row_bytes = cols.checked_mul(src_elem_bytes).ok_or_else(|| {
                OpError::Shape(format!("tensor '{}': row byte size overflows", name))
            })?;
            let byte_start = row_start.checked_mul(src_row_bytes).ok_or_else(|| {
                OpError::Shape(format!("tensor '{}': shard offset overflows", name))
            })?;
            let byte_len = local_rows.checked_mul(src_row_bytes).ok_or_else(|| {
                OpError::Shape(format!("tensor '{}': shard byte size overflows", name))
            })?;
            Ok(MatrixShardHost {
                bytes: convert_host_bytes::<T>(
                    &view.data()[byte_start..byte_start + byte_len],
                    src_dtype,
                    local_rows * cols,
                    name,
                )?,
                shape: [local_rows, cols],
            })
        }
        MatrixShardAxis::InputColumns => {
            let (col_start, local_cols) = even_shard_range(name, cols, tp)?;
            let local_src_row_bytes = local_cols.checked_mul(src_elem_bytes).ok_or_else(|| {
                OpError::Shape(format!("tensor '{}': local row byte size overflows", name))
            })?;
            let capacity = rows.checked_mul(local_src_row_bytes).ok_or_else(|| {
                OpError::Shape(format!("tensor '{}': shard byte size overflows", name))
            })?;
            let mut src_shard = Vec::with_capacity(capacity);
            for row in 0..rows {
                let elem_start = row
                    .checked_mul(cols)
                    .and_then(|offset| offset.checked_add(col_start))
                    .ok_or_else(|| {
                        OpError::Shape(format!("tensor '{}': shard offset overflows", name))
                    })?;
                let byte_start = elem_start.checked_mul(src_elem_bytes).ok_or_else(|| {
                    OpError::Shape(format!("tensor '{}': shard offset overflows", name))
                })?;
                src_shard
                    .extend_from_slice(&view.data()[byte_start..byte_start + local_src_row_bytes]);
            }
            Ok(MatrixShardHost {
                bytes: convert_host_bytes::<T>(&src_shard, src_dtype, rows * local_cols, name)?,
                shape: [rows, local_cols],
            })
        }
    }
}

/// Cast a rank-1 output bias to `T`, then retain the same even vocabulary
/// range used by the corresponding row-sharded LM-head matrix.
fn prepare_vector_shard<T: Dtype>(
    name: &str,
    view: &TensorView<'_>,
    expected_len: usize,
    tp: RankPair,
) -> OpResult<VectorShardHost> {
    if view.shape() != [expected_len] {
        return Err(OpError::Shape(format!(
            "tensor '{}': expected bias shape [{}], got {:?}",
            name,
            expected_len,
            view.shape()
        )));
    }
    let (start, len) = even_shard_range(name, expected_len, tp)?;
    let src_dtype = st_dtype(view)?;
    let src_elem_bytes = src_dtype.size_in_bytes();
    let expected_bytes = expected_len
        .checked_mul(src_elem_bytes)
        .ok_or_else(|| OpError::Shape(format!("tensor '{}': byte size overflows", name)))?;
    if view.data().len() != expected_bytes {
        return Err(OpError::Shape(format!(
            "tensor '{}': byte length {} does not match shape {:?} and dtype {:?}",
            name,
            view.data().len(),
            view.shape(),
            view.dtype()
        )));
    }
    let byte_start = start
        .checked_mul(src_elem_bytes)
        .ok_or_else(|| OpError::Shape(format!("tensor '{}': shard offset overflows", name)))?;
    let byte_len = len
        .checked_mul(src_elem_bytes)
        .ok_or_else(|| OpError::Shape(format!("tensor '{}': shard byte size overflows", name)))?;
    Ok(VectorShardHost {
        bytes: convert_host_bytes::<T>(
            &view.data()[byte_start..byte_start + byte_len],
            src_dtype,
            len,
            name,
        )?,
        len,
    })
}

/// Shard each projection on output rows first, then concatenate the local
/// pieces. This preserves layouts such as `[Q_rank; K_rank; V_rank]` and
/// `[gate_rank; up_rank]`.
fn prepare_fused_output_shards<T: Dtype>(
    what: &str,
    parts: &[(&str, &TensorView<'_>)],
    tp: RankPair,
) -> OpResult<MatrixShardHost> {
    if parts.is_empty() {
        return Err(OpError::Shape(format!(
            "{}: cannot fuse an empty projection list",
            what
        )));
    }

    let mut bytes = Vec::new();
    let mut total_rows = 0usize;
    let mut cols = None;
    for &(name, view) in parts {
        let shard = prepare_matrix_shard::<T>(name, view, MatrixShardAxis::OutputRows, tp)?;
        if let Some(expected) = cols {
            if shard.shape[1] != expected {
                return Err(OpError::Shape(format!(
                    "{}: tensor '{}' has {} columns, expected {}",
                    what, name, shard.shape[1], expected
                )));
            }
        } else {
            cols = Some(shard.shape[1]);
        }
        total_rows = total_rows
            .checked_add(shard.shape[0])
            .ok_or_else(|| OpError::Shape(format!("{}: fused row count overflows", what)))?;
        bytes.extend_from_slice(&shard.bytes);
    }

    Ok(MatrixShardHost {
        bytes,
        shape: [total_rows, cols.expect("parts is non-empty")],
    })
}

/// Map a safetensors view's dtype to our `DataType`, erroring on unsupported.
fn st_dtype(view: &TensorView) -> OpResult<DataType> {
    Ok(match view.dtype() {
        safetensors::Dtype::F32 => DataType::F32,
        safetensors::Dtype::F16 => DataType::F16,
        safetensors::Dtype::BF16 => DataType::BF16,
        safetensors::Dtype::F8_E4M3 => DataType::F8E4M3,
        safetensors::Dtype::I32 => DataType::I32,
        safetensors::Dtype::I8 => DataType::I8,
        other => {
            return Err(OpError::Kernel(format!(
                "unsupported safetensor dtype: {:?}",
                other
            )));
        }
    })
}

fn validate_matrix_shape(
    name: &str,
    view: &TensorView<'_>,
    expected_rows: usize,
    expected_cols: usize,
) -> OpResult<()> {
    let shape = view.shape();
    if shape != [expected_rows, expected_cols] {
        return Err(OpError::Shape(format!(
            "tensor '{}': expected [{}, {}], got {:?}",
            name, expected_rows, expected_cols, shape
        )));
    }
    Ok(())
}

struct Fp8ViewPart<'name, 'data> {
    name: &'name str,
    weight: TensorView<'data>,
    scale_inv: TensorView<'data>,
}

#[derive(Debug)]
struct Fp8FusedHost {
    weight: Vec<u8>,
    scales: Vec<u8>,
    weight_shape: [usize; 2],
    scale_shape: [usize; 2],
}

/// Validate and row-fuse block-FP8 weight segments without dequantizing them.
///
/// A scale row covers `block_n` consecutive output rows. Therefore every
/// boundary *between* fused projections must land on a block-row boundary;
/// otherwise a simple row concatenation would assign the wrong scale tile to
/// the following projection. The final segment may contain a tail block.
/// Scale tensors are normalized to f32 for the native backend contract.
fn prepare_fp8_fused_host(
    parts: &[Fp8ViewPart<'_, '_>],
    [block_n, block_k]: [usize; 2],
    what: &str,
) -> OpResult<Fp8FusedHost> {
    if parts.is_empty() {
        return Err(OpError::Shape(format!(
            "{}: cannot fuse an empty FP8 projection list",
            what
        )));
    }
    if block_n == 0 || block_k == 0 {
        return Err(OpError::Shape(format!(
            "{}: FP8 block dimensions must be nonzero, got [{}, {}]",
            what, block_n, block_k
        )));
    }

    let mut cols = None;
    let mut total_rows = 0usize;
    let mut total_scale_rows = 0usize;
    let mut weight_host = Vec::new();
    let mut scale_host = Vec::new();

    for (part_idx, part) in parts.iter().enumerate() {
        if part.weight.dtype() != safetensors::Dtype::F8_E4M3 {
            return Err(OpError::Kernel(format!(
                "{}: tensor '{}' must be F8_E4M3, got {:?}",
                what,
                part.name,
                part.weight.dtype()
            )));
        }
        let shape = part.weight.shape();
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            return Err(OpError::Shape(format!(
                "{}: tensor '{}' must be a non-empty rank-2 matrix, got {:?}",
                what, part.name, shape
            )));
        }
        let (rows, part_cols) = (shape[0], shape[1]);
        if let Some(expected_cols) = cols {
            if part_cols != expected_cols {
                return Err(OpError::Shape(format!(
                    "{}: tensor '{}' has {} columns, expected {}",
                    what, part.name, part_cols, expected_cols
                )));
            }
        } else {
            cols = Some(part_cols);
        }

        if part_idx + 1 < parts.len() && rows % block_n != 0 {
            return Err(OpError::Shape(format!(
                "{}: fusion boundary after '{}' is not aligned: rows {} is not divisible by block_n {}",
                what, part.name, rows, block_n
            )));
        }

        let expected_weight_bytes = rows.checked_mul(part_cols).ok_or_else(|| {
            OpError::Shape(format!("{}: tensor '{}' shape overflows", what, part.name))
        })?;
        if part.weight.data().len() != expected_weight_bytes {
            return Err(OpError::Shape(format!(
                "{}: tensor '{}' has {} bytes, expected {} for shape {:?}",
                what,
                part.name,
                part.weight.data().len(),
                expected_weight_bytes,
                shape
            )));
        }

        let scale_rows = rows.div_ceil(block_n);
        let scale_cols = part_cols.div_ceil(block_k);
        if part.scale_inv.shape() != [scale_rows, scale_cols] {
            return Err(OpError::Shape(format!(
                "{}: tensor '{}_scale_inv' must have shape [{}, {}], got {:?}",
                what,
                part.name,
                scale_rows,
                scale_cols,
                part.scale_inv.shape()
            )));
        }
        if !matches!(
            part.scale_inv.dtype(),
            safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16
        ) {
            return Err(OpError::Kernel(format!(
                "{}: tensor '{}_scale_inv' must be F32/F16/BF16, got {:?}",
                what,
                part.name,
                part.scale_inv.dtype()
            )));
        }

        if let Some(index) = part
            .weight
            .data()
            .iter()
            .position(|bits| bits & 0x7f == 0x7f)
        {
            return Err(OpError::Kernel(format!(
                "{}: tensor '{}' contains an E4M3FN NaN encoding at flat index {}",
                what, part.name, index
            )));
        }
        weight_host.extend_from_slice(part.weight.data());

        let part_scale_host = safetensor_view_to_host_bytes::<f32>(&part.scale_inv)?;
        if let Some((index, value)) = part_scale_host
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32-sized chunk")))
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(OpError::Kernel(format!(
                "{}: tensor '{}_scale_inv' contains non-finite value {} at flat index {}",
                what, part.name, value, index
            )));
        }
        scale_host.extend_from_slice(&part_scale_host);
        total_rows = total_rows
            .checked_add(rows)
            .ok_or_else(|| OpError::Shape(format!("{}: fused weight row count overflows", what)))?;
        total_scale_rows = total_scale_rows
            .checked_add(scale_rows)
            .ok_or_else(|| OpError::Shape(format!("{}: fused scale row count overflows", what)))?;
    }

    let cols = cols.expect("parts is non-empty");
    let scale_cols = cols.div_ceil(block_k);
    let expected_scale_rows = total_rows.div_ceil(block_n);
    if total_scale_rows != expected_scale_rows {
        return Err(OpError::Shape(format!(
            "{}: fused scale rows {} do not match ceil({}/{})={}; projection boundaries must align to block_n",
            what, total_scale_rows, total_rows, block_n, expected_scale_rows
        )));
    }
    let expected_weight_len = total_rows
        .checked_mul(cols)
        .ok_or_else(|| OpError::Shape(format!("{}: fused weight byte count overflows", what)))?;
    let expected_scale_len = total_scale_rows
        .checked_mul(scale_cols)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| OpError::Shape(format!("{}: fused scale byte count overflows", what)))?;
    debug_assert_eq!(weight_host.len(), expected_weight_len);
    debug_assert_eq!(scale_host.len(), expected_scale_len);

    Ok(Fp8FusedHost {
        weight: weight_host,
        scales: scale_host,
        weight_shape: [total_rows, cols],
        scale_shape: [total_scale_rows, scale_cols],
    })
}

fn safetensor_view_to_host_bytes<T: Dtype>(view: &TensorView<'_>) -> OpResult<Vec<u8>> {
    let shape_vec: Vec<usize> = view.shape().to_vec();
    let numel: usize = shape_vec.iter().product();
    let src_dtype = st_dtype(view)?;
    convert_host_bytes::<T>(view.data(), src_dtype, numel, "safetensor view")
}

fn convert_host_bytes<T: Dtype>(
    src_bytes: &[u8],
    src_dtype: DataType,
    numel: usize,
    what: &str,
) -> OpResult<Vec<u8>> {
    let expected_src_bytes = numel
        .checked_mul(src_dtype.size_in_bytes())
        .ok_or_else(|| OpError::Shape(format!("{}: source byte size overflows", what)))?;
    if src_bytes.len() != expected_src_bytes {
        return Err(OpError::Shape(format!(
            "{}: source byte length {} != expected {} for {} {:?} elements",
            what,
            src_bytes.len(),
            expected_src_bytes,
            numel,
            src_dtype
        )));
    }
    let size_bytes = numel
        .checked_mul(T::SIZE_BYTES)
        .ok_or_else(|| OpError::Shape(format!("{}: target byte size overflows", what)))?;
    if src_dtype == T::DATA_TYPE {
        return Ok(src_bytes.to_vec());
    }
    if src_dtype == DataType::F8E4M3 || T::DATA_TYPE == DataType::F8E4M3 {
        return Err(OpError::Kernel(format!(
            "unsupported safetensor cast from {:?} to {:?}; FP8 weights must stay raw",
            src_dtype,
            T::DATA_TYPE
        )));
    }
    let mut host_buf = vec![0u8; size_bytes];
    cast_bytes(
        src_bytes,
        src_dtype,
        host_buf.as_mut_ptr(),
        T::DATA_TYPE,
        numel,
    );
    Ok(host_buf)
}

fn tensor_from_safetensor_view<T: Dtype, D: MemoryPort>(
    view: &TensorView,
    device: &D,
) -> OpResult<Tensor<T, D>> {
    let shape_vec: Vec<usize> = view.shape().to_vec();
    let shape = Shape::from_slice(&shape_vec);
    let host_buf = safetensor_view_to_host_bytes::<T>(view)?;
    Tensor::<T, D>::from_host_bytes(&host_buf, shape, device)
}

/// Element-wise dtype cast via f64 intermediate.
fn cast_bytes(src: &[u8], src_dt: DataType, dst: *mut u8, dst_dt: DataType, numel: usize) {
    use half::{bf16, f16};
    for i in 0..numel {
        let val: f64 = match src_dt {
            DataType::F32 => {
                let b = &src[i * 4..i * 4 + 4];
                f64::from(f32::from_le_bytes(b.try_into().unwrap()))
            }
            DataType::BF16 => {
                let b = &src[i * 2..i * 2 + 2];
                f64::from(bf16::from_le_bytes(b.try_into().unwrap()).to_f32())
            }
            DataType::F16 => {
                let b = &src[i * 2..i * 2 + 2];
                f64::from(f16::from_le_bytes(b.try_into().unwrap()).to_f32())
            }
            DataType::I32 => {
                let b = &src[i * 4..i * 4 + 4];
                i32::from_le_bytes(b.try_into().unwrap()) as f64
            }
            DataType::I8 => src[i] as i8 as f64,
            DataType::F8E4M3 => {
                <Fp8E4m3 as crate::domain::dtype::Dtype>::read_f64(&Fp8E4m3(src[i]))
            }
        };
        unsafe {
            match dst_dt {
                DataType::F32 => {
                    std::ptr::copy_nonoverlapping(
                        (val as f32).to_le_bytes().as_ptr(),
                        dst.add(i * 4),
                        4,
                    );
                }
                DataType::BF16 => {
                    std::ptr::copy_nonoverlapping(
                        bf16::from_f64(val).to_le_bytes().as_ptr(),
                        dst.add(i * 2),
                        2,
                    );
                }
                DataType::F16 => {
                    std::ptr::copy_nonoverlapping(
                        f16::from_f64(val).to_le_bytes().as_ptr(),
                        dst.add(i * 2),
                        2,
                    );
                }
                DataType::I32 => {
                    std::ptr::copy_nonoverlapping(
                        (val as i32).to_le_bytes().as_ptr(),
                        dst.add(i * 4),
                        4,
                    );
                }
                DataType::I8 => {
                    *dst.add(i) = val as i8 as u8;
                }
                DataType::F8E4M3 => {
                    *dst.add(i) = <Fp8E4m3 as crate::domain::dtype::Dtype>::write_f64(val).0;
                }
            }
        }
    }
}

// ─── RoPE cache computation ─────────────────────────────────────────────────

/// Precompute interleaved RoPE sin/cos cache.
///
/// For each position p and dimension pair (2i, 2i+1):
///   freq = 1 / (theta ^ (2i / head_dim))
///   sin_cache[p, i] = sin(p * freq)
///   cos_cache[p, i] = cos(p * freq)
///
/// Output shape: [max_seq_len, head_dim] where:
///   - sin_cache[p, 2i]   = sin(p * freq_i)  (stored interleaved for the RoPE kernel)
///   - cos_cache[p, 2i]   = cos(p * freq_i)
///
/// Actually stored as [max_seq_len, head_dim/2] for half-dim RoPE variant
/// (matching our rope_inplace kernel which processes pairs).
pub(crate) fn compute_rope_cache<T: Dtype, D: OpBackend>(
    max_seq_len: usize,
    head_dim: usize,
    theta: f64,
    rope_scaling: Option<&RopeScaling>,
    device: &D,
) -> OpResult<(Tensor<T, D>, Tensor<T, D>)> {
    let half_dim = head_dim / 2;

    // Compute base frequencies: freq_i = 1 / theta^(2i / head_dim)
    let mut freqs: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    // Llama-3 NTK-aware frequency rescaling. See `transformers`
    // `_compute_llama3_parameters`: positions ≤ orig_max_pos use the
    // unmodified base RoPE; longer positions get a piecewise-smooth
    // rescaling per-frequency.
    if let Some(s) = rope_scaling {
        let two_pi = std::f64::consts::TAU;
        let orig = s.original_max_position_embeddings as f64;
        let lambda_low = orig / s.low_freq_factor as f64;
        let lambda_high = orig / s.high_freq_factor as f64;
        let factor = s.factor as f64;
        for f in freqs.iter_mut() {
            let wavelength = two_pi / *f;
            if wavelength < lambda_high {
                // High frequency: keep unchanged.
            } else if wavelength > lambda_low {
                // Low frequency: stretch by `factor`.
                *f /= factor;
            } else {
                // Smooth transition (linear ramp on the inverse-factor side).
                let smooth = (orig / wavelength - s.low_freq_factor as f64)
                    / (s.high_freq_factor as f64 - s.low_freq_factor as f64);
                *f = (1.0 - smooth) * (*f / factor) + smooth * *f;
            }
        }
    }

    // Build typed host buffers through the canonical scalar conversion API.
    let n = max_seq_len * half_dim;
    let mut sin_host = vec![T::write_f64(0.0); n];
    let mut cos_host = vec![T::write_f64(0.0); n];
    for pos in 0..max_seq_len {
        for (i, &frequency) in freqs.iter().enumerate().take(half_dim) {
            let angle = pos as f64 * frequency;
            let offset = pos * half_dim + i;
            sin_host[offset] = T::write_f64(angle.sin());
            cos_host[offset] = T::write_f64(angle.cos());
        }
    }

    let shape = Shape::from_slice(&[max_seq_len, half_dim]);
    let sin_tensor = Tensor::<T, D>::from_host_slice(&sin_host, shape, device)?;
    let cos_tensor = Tensor::<T, D>::from_host_slice(&cos_host, shape, device)?;
    Ok((sin_tensor, cos_tensor))
}

#[cfg(test)]
mod tp_tests {
    use super::{
        MatrixShardAxis, even_shard_range, make_vocab_parallel_linear, prepare_fused_output_shards,
        prepare_matrix_shard, prepare_vector_shard, validate_tp,
    };
    use crate::components::linear::LinearParallelism;
    use crate::domain::exec::RankPair;
    use crate::domain::tensor::Tensor;
    use crate::domain::types::Shape;
    use crate::infrastructure::cpu::Cpu;
    use half::bf16;
    use safetensors::{Dtype, tensor::TensorView};

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_le_bytes())
            .collect()
    }

    fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_le_bytes(chunk.try_into().unwrap()).to_f32())
            .collect()
    }

    #[test]
    fn tp1_matrix_shards_preserve_the_complete_weight() {
        let bytes = bf16_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let view = TensorView::new(Dtype::BF16, vec![2, 3], &bytes).unwrap();
        let tp = RankPair { rank: 0, size: 1 };

        for axis in [MatrixShardAxis::OutputRows, MatrixShardAxis::InputColumns] {
            let shard = prepare_matrix_shard::<bf16>("weight", &view, axis, tp).unwrap();
            assert_eq!(shard.shape, [2, 3]);
            assert_eq!(shard.bytes, bytes);
        }
    }

    #[test]
    fn row_parallel_shards_each_matrix_row_on_input_columns() {
        let bytes = bf16_bytes(&[
            0.0, 1.0, 2.0, 3.0, // row 0
            10.0, 11.0, 12.0, 13.0, // row 1
            20.0, 21.0, 22.0, 23.0, // row 2
        ]);
        let view = TensorView::new(Dtype::BF16, vec![3, 4], &bytes).unwrap();

        let rank0 = prepare_matrix_shard::<bf16>(
            "o_proj.weight",
            &view,
            MatrixShardAxis::InputColumns,
            RankPair { rank: 0, size: 2 },
        )
        .unwrap();
        let rank1 = prepare_matrix_shard::<bf16>(
            "o_proj.weight",
            &view,
            MatrixShardAxis::InputColumns,
            RankPair { rank: 1, size: 2 },
        )
        .unwrap();

        assert_eq!(rank0.shape, [3, 2]);
        assert_eq!(
            decode_bf16(&rank0.bytes),
            vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0]
        );
        assert_eq!(rank1.shape, [3, 2]);
        assert_eq!(
            decode_bf16(&rank1.bytes),
            vec![2.0, 3.0, 12.0, 13.0, 22.0, 23.0]
        );
    }

    #[test]
    fn vocab_weight_and_bias_use_the_identical_row_range() {
        let weight_bytes = bf16_bytes(&[
            0.0, 1.0, // vocab row 0
            10.0, 11.0, // vocab row 1
            20.0, 21.0, // vocab row 2
            30.0, 31.0, // vocab row 3
            40.0, 41.0, // vocab row 4
            50.0, 51.0, // vocab row 5
        ]);
        let bias_bytes = bf16_bytes(&[100.0, 110.0, 120.0, 130.0, 140.0, 150.0]);
        let weight = TensorView::new(Dtype::BF16, vec![6, 2], &weight_bytes).unwrap();
        let bias = TensorView::new(Dtype::BF16, vec![6], &bias_bytes).unwrap();
        let tp = RankPair { rank: 1, size: 2 };

        let local_weight = prepare_matrix_shard::<bf16>(
            "lm_head.weight",
            &weight,
            MatrixShardAxis::OutputRows,
            tp,
        )
        .unwrap();
        let local_bias = prepare_vector_shard::<bf16>("lm_head.bias", &bias, 6, tp).unwrap();

        assert_eq!(local_weight.shape, [3, 2]);
        assert_eq!(
            decode_bf16(&local_weight.bytes),
            vec![30.0, 31.0, 40.0, 41.0, 50.0, 51.0]
        );
        assert_eq!(local_bias.len, 3);
        assert_eq!(decode_bf16(&local_bias.bytes), vec![130.0, 140.0, 150.0]);
    }

    #[test]
    fn tied_vocab_linear_keeps_the_local_embedding_allocation() {
        let tp = RankPair { rank: 1, size: 2 };
        let local_table = Tensor::from_host_slice(
            &[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
            Shape::from_slice(&[3, 2]),
            &Cpu,
        )
        .unwrap();

        let lm_head = make_vocab_parallel_linear(local_table.clone(), None, tp, 6).unwrap();
        let lm_weight = lm_head.weight.as_dense().unwrap();

        assert!(std::sync::Arc::ptr_eq(
            local_table.storage(),
            lm_weight.storage()
        ));
        assert_eq!(
            lm_head.parallelism(),
            LinearParallelism::Column {
                tp,
                gather_output: true,
            }
        );
    }

    #[test]
    fn tp1_vocab_linear_remains_replicated() {
        let tp = RankPair { rank: 0, size: 1 };
        let weight =
            Tensor::from_host_slice(&[0.0f32, 1.0, 2.0, 3.0], Shape::from_slice(&[2, 2]), &Cpu)
                .unwrap();

        let lm_head = make_vocab_parallel_linear(weight, None, tp, 2).unwrap();

        assert_eq!(lm_head.parallelism(), LinearParallelism::Replicated { tp });
    }

    #[test]
    fn qkv_column_parallel_shards_each_projection_before_fusion() {
        let q_bytes = bf16_bytes(&[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
        let k_bytes = bf16_bytes(&[100.0, 101.0, 110.0, 111.0]);
        let v_bytes = bf16_bytes(&[200.0, 201.0, 210.0, 211.0]);
        let q = TensorView::new(Dtype::BF16, vec![4, 2], &q_bytes).unwrap();
        let k = TensorView::new(Dtype::BF16, vec![2, 2], &k_bytes).unwrap();
        let v = TensorView::new(Dtype::BF16, vec![2, 2], &v_bytes).unwrap();
        let parts = [("q", &q), ("k", &k), ("v", &v)];

        let rank0 =
            prepare_fused_output_shards::<bf16>("qkv", &parts, RankPair { rank: 0, size: 2 })
                .unwrap();
        let rank1 =
            prepare_fused_output_shards::<bf16>("qkv", &parts, RankPair { rank: 1, size: 2 })
                .unwrap();

        assert_eq!(rank0.shape, [4, 2]);
        assert_eq!(
            decode_bf16(&rank0.bytes),
            vec![0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 200.0, 201.0]
        );
        assert_eq!(rank1.shape, [4, 2]);
        assert_eq!(
            decode_bf16(&rank1.bytes),
            vec![20.0, 21.0, 30.0, 31.0, 110.0, 111.0, 210.0, 211.0]
        );
    }

    #[test]
    fn gate_up_column_parallel_preserves_local_gate_then_up_layout() {
        let gate_bytes = bf16_bytes(&[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
        let up_bytes = bf16_bytes(&[100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 130.0, 131.0]);
        let gate = TensorView::new(Dtype::BF16, vec![4, 2], &gate_bytes).unwrap();
        let up = TensorView::new(Dtype::BF16, vec![4, 2], &up_bytes).unwrap();

        let rank1 = prepare_fused_output_shards::<bf16>(
            "gate_up",
            &[("gate", &gate), ("up", &up)],
            RankPair { rank: 1, size: 2 },
        )
        .unwrap();

        assert_eq!(rank1.shape, [4, 2]);
        assert_eq!(
            decode_bf16(&rank1.bytes),
            vec![20.0, 21.0, 30.0, 31.0, 120.0, 121.0, 130.0, 131.0]
        );
    }

    #[test]
    fn invalid_tp_rank_and_non_divisible_dimensions_fail_early() {
        let rank_err = validate_tp(RankPair { rank: 2, size: 2 }).unwrap_err();
        assert!(rank_err.to_string().contains("rank 2"));

        let size_err = validate_tp(RankPair { rank: 0, size: 0 }).unwrap_err();
        assert!(size_err.to_string().contains("nonzero"));

        let dim_err =
            even_shard_range("KV head count", 3, RankPair { rank: 0, size: 2 }).unwrap_err();
        assert!(dim_err.to_string().contains("KV head count"));
        assert!(dim_err.to_string().contains("not divisible"));
    }
}

#[cfg(test)]
mod fp8_tests {
    use super::{Fp8ViewPart, prepare_fp8_fused_host};
    use crate::domain::dtype::Fp8E4m3;
    use crate::domain::tensor::Tensor;
    use crate::domain::types::Shape;
    use crate::infrastructure::cpu::Cpu;
    use half::bf16;
    use safetensors::{Dtype, tensor::TensorView};

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_le_bytes())
            .collect()
    }

    fn decode_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn native_fp8_fusion_preserves_one_byte_weights_on_host_and_device() {
        // block [2,2]. The first segment ends on a block_n boundary; the final
        // segment deliberately has a tail block.
        let q_bytes: Vec<u8> = (0..8).collect();
        let k_bytes: Vec<u8> = (8..20).collect();
        let q_scale_bytes = bf16_bytes(&[1.0, 2.0]);
        let k_scale_bytes = bf16_bytes(&[3.0, 4.0, 5.0, 6.0]);
        let q = TensorView::new(Dtype::F8_E4M3, vec![2, 4], &q_bytes).unwrap();
        let k = TensorView::new(Dtype::F8_E4M3, vec![3, 4], &k_bytes).unwrap();
        let q_scales = TensorView::new(Dtype::BF16, vec![1, 2], &q_scale_bytes).unwrap();
        let k_scales = TensorView::new(Dtype::BF16, vec![2, 2], &k_scale_bytes).unwrap();
        let parts = [
            Fp8ViewPart {
                name: "q.weight",
                weight: q,
                scale_inv: q_scales,
            },
            Fp8ViewPart {
                name: "k.weight",
                weight: k,
                scale_inv: k_scales,
            },
        ];

        let host = prepare_fp8_fused_host(&parts, [2, 2], "test_qk").unwrap();
        let expected_weight: Vec<u8> = q_bytes.into_iter().chain(k_bytes).collect();
        assert_eq!(host.weight, expected_weight);
        assert_eq!(host.weight_shape, [5, 4]);
        assert_eq!(host.weight.len(), 5 * 4); // E4M3 stays one byte per element.
        assert_eq!(host.scale_shape, [3, 2]);
        assert_eq!(decode_f32(&host.scales), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let device_weight = Tensor::<Fp8E4m3, Cpu>::from_host_bytes(
            &host.weight,
            Shape::from_slice(&host.weight_shape),
            &Cpu,
        )
        .unwrap();
        assert_eq!(device_weight.storage().size(), 5 * 4);
        assert_eq!(
            device_weight
                .to_host_vec()
                .unwrap()
                .into_iter()
                .map(|value| value.0)
                .collect::<Vec<_>>(),
            host.weight
        );
    }

    #[test]
    fn native_fp8_fusion_rejects_unaligned_projection_boundary() {
        let first_bytes = vec![0x38; 3 * 4];
        let second_bytes = vec![0x38; 2 * 4];
        let first_scale_bytes = bf16_bytes(&[1.0; 4]);
        let second_scale_bytes = bf16_bytes(&[1.0; 2]);
        let first = TensorView::new(Dtype::F8_E4M3, vec![3, 4], &first_bytes).unwrap();
        let second = TensorView::new(Dtype::F8_E4M3, vec![2, 4], &second_bytes).unwrap();
        let first_scales = TensorView::new(Dtype::BF16, vec![2, 2], &first_scale_bytes).unwrap();
        let second_scales = TensorView::new(Dtype::BF16, vec![1, 2], &second_scale_bytes).unwrap();
        let parts = [
            Fp8ViewPart {
                name: "first.weight",
                weight: first,
                scale_inv: first_scales,
            },
            Fp8ViewPart {
                name: "second.weight",
                weight: second,
                scale_inv: second_scales,
            },
        ];

        let err = prepare_fp8_fused_host(&parts, [2, 2], "test_fusion")
            .expect_err("unaligned first projection must fail");
        assert!(format!("{err:?}").contains("fusion boundary"));
    }

    #[test]
    fn native_fp8_single_linear_allows_tail_block() {
        let weight_bytes = vec![0x38; 3 * 5];
        let scale_bytes = bf16_bytes(&[1.0; 6]);
        let weight = TensorView::new(Dtype::F8_E4M3, vec![3, 5], &weight_bytes).unwrap();
        let scales = TensorView::new(Dtype::BF16, vec![2, 3], &scale_bytes).unwrap();
        let parts = [Fp8ViewPart {
            name: "tail.weight",
            weight,
            scale_inv: scales,
        }];

        let host = prepare_fp8_fused_host(&parts, [2, 2], "tail").unwrap();
        assert_eq!(host.weight.len(), 15);
        assert_eq!(host.scale_shape, [2, 3]);
    }

    #[test]
    fn native_fp8_rejects_non_finite_metadata() {
        let nan_weight_bytes = vec![0x7f; 2 * 2];
        let good_scale_bytes = 1.0f32.to_le_bytes();
        let nan_weight = TensorView::new(Dtype::F8_E4M3, vec![2, 2], &nan_weight_bytes).unwrap();
        let good_scale = TensorView::new(Dtype::F32, vec![1, 1], &good_scale_bytes).unwrap();
        let err = prepare_fp8_fused_host(
            &[Fp8ViewPart {
                name: "nan.weight",
                weight: nan_weight,
                scale_inv: good_scale,
            }],
            [2, 2],
            "nan_weight",
        )
        .expect_err("E4M3FN NaN must be rejected");
        assert!(format!("{err:?}").contains("NaN encoding"));

        let good_weight_bytes = vec![0x38; 2 * 2];
        let infinite_scale_bytes = f32::INFINITY.to_le_bytes();
        let good_weight = TensorView::new(Dtype::F8_E4M3, vec![2, 2], &good_weight_bytes).unwrap();
        let infinite_scale =
            TensorView::new(Dtype::F32, vec![1, 1], &infinite_scale_bytes).unwrap();
        let err = prepare_fp8_fused_host(
            &[Fp8ViewPart {
                name: "infinite.weight",
                weight: good_weight,
                scale_inv: infinite_scale,
            }],
            [2, 2],
            "infinite_scale",
        )
        .expect_err("non-finite scale must be rejected");
        assert!(format!("{err:?}").contains("non-finite"));
    }
}
