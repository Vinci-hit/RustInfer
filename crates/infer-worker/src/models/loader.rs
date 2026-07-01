//! Weight loader — loads safetensors into model structures.
//!
//! This lives in models/ (not domain) because it's about constructing
//! concrete model instances. Filesystem access is delegated to
//! `infra::io::SafetensorsReader` to keep the I/O concern in the
//! infrastructure layer.

use safetensors::tensor::TensorView;

use super::decoder::Decoder;
use super::layers::{Embedding, Linear, RMSNorm};
use super::llama3::Llama3Model;
use super::qwen3::Qwen3Model;
use crate::components::{
    Attention, DecoderBlock, DenseFfn, Embed, Linear as CompLinear, LmHead, RmsNorm as CompRmsNorm,
};
use crate::domain::model::ModelDims;
use crate::domain::dtype::quant::QuantScheme;
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

/// Configuration for model loading.
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
}

/// Weight loader — pulls tensors out of a `SafetensorsReader` and builds
/// typed model structs. The reader is borrowed (no copy until upload).
pub struct WeightLoader<'a> {
    reader: &'a SafetensorsReader,
}

impl<'a> WeightLoader<'a> {
    /// Wrap a reader. The reader owns the mmap; the loader only borrows.
    pub fn new(reader: &'a SafetensorsReader) -> Self {
        Self { reader }
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

    /// Load fused QKV: concatenate q_proj, k_proj, v_proj along rows → [q_dim+2*kv_dim, dim].
    ///
    /// To keep the tensor on `device` and avoid host↔device round-trips, we
    /// concatenate at the host-bytes level (in target dtype T) and upload
    /// the fused result in a single `Tensor::from_host_bytes`.
    pub fn load_fused_qkv<T: Dtype, D: OpBackend>(
        &self,
        layer_idx: usize,
        q_dim: usize,
        kv_dim: usize,
        dim: usize,
        device: &D,
    ) -> OpResult<Linear<T, D>> {
        let q_view = self
            .reader
            .read_view(&format!(
                "model.layers.{}.self_attn.q_proj.weight",
                layer_idx
            ))
            .map_err(|e| OpError::Kernel(format!("q_proj layer {}: {}", layer_idx, e)))?;
        let k_view = self
            .reader
            .read_view(&format!(
                "model.layers.{}.self_attn.k_proj.weight",
                layer_idx
            ))
            .map_err(|e| OpError::Kernel(format!("k_proj layer {}: {}", layer_idx, e)))?;
        let v_view = self
            .reader
            .read_view(&format!(
                "model.layers.{}.self_attn.v_proj.weight",
                layer_idx
            ))
            .map_err(|e| OpError::Kernel(format!("v_proj layer {}: {}", layer_idx, e)))?;

        let total_rows = q_dim + 2 * kv_dim;
        let elem = T::SIZE_BYTES;
        let total_bytes = total_rows * dim * elem;
        let mut host = vec![0u8; total_bytes];

        // Helper: cast a single weight view into the host buffer at `row_offset`.
        let mut cast_into = |view: &TensorView,
                             row_offset: usize,
                             expected_rows: usize|
         -> OpResult<()> {
            let shape: Vec<usize> = view.shape().to_vec();
            if shape.len() != 2 || shape[0] != expected_rows || shape[1] != dim {
                return Err(OpError::Shape(format!(
                    "fused_qkv layer {}: expected [{}, {}], got {:?}",
                    layer_idx, expected_rows, dim, shape,
                )));
            }
            let src = view.data();
            let src_dt = match view.dtype() {
                safetensors::Dtype::F32 => DataType::F32,
                safetensors::Dtype::F16 => DataType::F16,
                safetensors::Dtype::BF16 => DataType::BF16,
                safetensors::Dtype::I32 => DataType::I32,
                safetensors::Dtype::I8 => DataType::I8,
                other => return Err(OpError::Kernel(format!("unsupported dtype: {:?}", other))),
            };
            let numel = expected_rows * dim;
            let dst = unsafe { host.as_mut_ptr().add(row_offset * dim * elem) };
            if src_dt == T::DATA_TYPE {
                let n = numel * elem;
                if src.len() != n {
                    return Err(OpError::Shape(format!(
                        "fused_qkv layer {}: view byte length {} != expected {} for shape {:?} \
                         (corrupt safetensors?)",
                        layer_idx,
                        src.len(),
                        n,
                        shape,
                    )));
                }
                // SAFETY: host buffer has at least row_offset*dim*elem + n bytes
                // by construction, and src has exactly n bytes (checked above).
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, n);
                }
            } else {
                cast_bytes(src, src_dt, dst, T::DATA_TYPE, numel);
            }
            Ok(())
        };

        cast_into(&q_view, 0, q_dim)?;
        cast_into(&k_view, q_dim, kv_dim)?;
        cast_into(&v_view, q_dim + kv_dim, kv_dim)?;

        let fused =
            Tensor::<T, D>::from_host_bytes(&host, Shape::from_slice(&[total_rows, dim]), device)?;
        Ok(Linear::new(fused, None))
    }

    /// Load fused gate_up: concatenate gate_proj, up_proj along rows
    /// → `[2*intermediate_size, dim]`.
    ///
    /// One GEMV computes both gate and up in a single launch; downstream
    /// `swiglu_packed` consumes the fused output without splitting.
    pub fn load_fused_gate_up<T: Dtype, D: OpBackend>(
        &self,
        layer_idx: usize,
        intermediate_size: usize,
        dim: usize,
        device: &D,
    ) -> OpResult<Linear<T, D>> {
        let g_view = self
            .reader
            .read_view(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx))
            .map_err(|e| OpError::Kernel(format!("gate_proj layer {}: {}", layer_idx, e)))?;
        let u_view = self
            .reader
            .read_view(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx))
            .map_err(|e| OpError::Kernel(format!("up_proj layer {}: {}", layer_idx, e)))?;

        let total_rows = 2 * intermediate_size;
        let elem = T::SIZE_BYTES;
        let total_bytes = total_rows * dim * elem;
        let mut host = vec![0u8; total_bytes];

        let mut cast_into = |view: &TensorView,
                             row_offset: usize,
                             expected_rows: usize|
         -> OpResult<()> {
            let shape: Vec<usize> = view.shape().to_vec();
            if shape.len() != 2 || shape[0] != expected_rows || shape[1] != dim {
                return Err(OpError::Shape(format!(
                    "fused_gate_up layer {}: expected [{}, {}], got {:?}",
                    layer_idx, expected_rows, dim, shape,
                )));
            }
            let src = view.data();
            let src_dt = match view.dtype() {
                safetensors::Dtype::F32 => DataType::F32,
                safetensors::Dtype::F16 => DataType::F16,
                safetensors::Dtype::BF16 => DataType::BF16,
                safetensors::Dtype::I32 => DataType::I32,
                safetensors::Dtype::I8 => DataType::I8,
                other => return Err(OpError::Kernel(format!("unsupported dtype: {:?}", other))),
            };
            let numel = expected_rows * dim;
            let dst = unsafe { host.as_mut_ptr().add(row_offset * dim * elem) };
            if src_dt == T::DATA_TYPE {
                let n = numel * elem;
                if src.len() != n {
                    return Err(OpError::Shape(format!(
                        "fused_gate_up layer {}: view byte length {} != expected {} for shape {:?} \
                         (corrupt safetensors?)",
                        layer_idx,
                        src.len(),
                        n,
                        shape,
                    )));
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, n);
                }
            } else {
                cast_bytes(src, src_dt, dst, T::DATA_TYPE, numel);
            }
            Ok(())
        };

        cast_into(&g_view, 0, intermediate_size)?;
        cast_into(&u_view, intermediate_size, intermediate_size)?;

        let fused =
            Tensor::<T, D>::from_host_bytes(&host, Shape::from_slice(&[total_rows, dim]), device)?;
        Ok(Linear::new(fused, None))
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
    fn load_awq_linear<T: Dtype + crate::domain::dtype::Dtype, D: OpBackend + LlmBackend>(
        &self,
        prefix: &str,
        scheme: QuantScheme,
        device: &D,
    ) -> OpResult<CompLinear<T, D>> {
        let packed = self.load_tensor::<i32, D>(&format!("{}.weight_packed", prefix), device)?;
        let zeros = self.load_tensor::<i32, D>(&format!("{}.weight_zero_point", prefix), device)?;
        let scales = self.load_tensor::<T, D>(&format!("{}.weight_scale", prefix), device)?;
        Ok(CompLinear::from_awq(packed, zeros, scales, scheme, None))
    }

    /// Load int4 `gate_proj` + `up_proj` fused along rows into one quantized
    /// `Linear` (`[2*inter, K/8]` packed), matching the dense fused layout.
    fn load_fused_gate_up_awq<T: Dtype + crate::domain::dtype::Dtype, D: OpBackend + LlmBackend>(
        &self,
        layer_idx: usize,
        scheme: QuantScheme,
        device: &D,
    ) -> OpResult<CompLinear<T, D>> {
        let view = |proj: &str, part: &str| -> OpResult<TensorView<'_>> {
            let name = format!("model.layers.{}.mlp.{}.{}", layer_idx, proj, part);
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
        Ok(CompLinear::from_awq(packed, zeros, scales, scheme, None))
    }

    /// Build the shared dense decoder. Per-block Q/K norms are populated when
    /// the weights contain them (Qwen3) and left absent otherwise (Llama3), so
    /// one builder backs both `load_llama3` and `load_qwen3`.
    fn build_decoder<T: Dtype + crate::domain::dtype::Dtype, D: OpBackend + LlmBackend>(
        &self,
        cfg: &LoadConfig,
        device: &D,
    ) -> OpResult<Decoder<T, D>> {
        let embed_table = self
            .load_embedding("model.embed_tokens.weight", device)?
            .table;
        let final_norm = self.load_rmsnorm("model.norm.weight", device, cfg.rms_norm_eps)?;
        let lm_head = if self.reader.contains("lm_head.weight") {
            self.load_linear("lm_head.weight", None, device)?
        } else {
            Linear::new(embed_table.clone(), None)
        };

        let (sin_cache, cos_cache) = compute_rope_cache::<T, D>(
            cfg.seq_len,
            cfg.head_dim,
            cfg.rope_theta,
            cfg.rope_scaling.as_ref(),
            device,
        )?;

        let q_dim = cfg.head_num * cfg.head_dim;
        let kv_dim = cfg.kv_head_num * cfg.head_dim;
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();

        let mut blocks = Vec::with_capacity(cfg.layer_num);
        for i in 0..cfg.layer_num {
            let input_layernorm = self.load_rmsnorm(
                &format!("model.layers.{}.input_layernorm.weight", i),
                device,
                cfg.rms_norm_eps,
            )?;
            let post_attention_layernorm = self.load_rmsnorm(
                &format!("model.layers.{}.post_attention_layernorm.weight", i),
                device,
                cfg.rms_norm_eps,
            )?;
            let qkv_proj = self.load_fused_qkv(i, q_dim, kv_dim, cfg.dim, device)?;
            let o_proj = self.load_linear(
                &format!("model.layers.{}.self_attn.o_proj.weight", i),
                None,
                device,
            )?;
            // MLP: int4 `pack-quantized` when `mlp_quant` is set, else dense.
            // Both arms yield a `components::Linear`; the quant arm carries the
            // packed weight + scales/zeros and drives `matmul_quant` internally.
            let (gate_up_proj, down_proj): (CompLinear<T, D>, CompLinear<T, D>) =
                if let Some(scheme) = cfg.mlp_quant {
                    (
                        self.load_fused_gate_up_awq(i, scheme, device)?,
                        self.load_awq_linear(
                            &format!("model.layers.{}.mlp.down_proj", i),
                            scheme,
                            device,
                        )?,
                    )
                } else {
                    (
                        comp_linear(self.load_fused_gate_up(
                            i,
                            cfg.intermediate_size,
                            cfg.dim,
                            device,
                        )?),
                        comp_linear(self.load_linear(
                            &format!("model.layers.{}.mlp.down_proj.weight", i),
                            None,
                            device,
                        )?),
                    )
                };

            let q_norm_name = format!("model.layers.{}.self_attn.q_norm.weight", i);
            let k_norm_name = format!("model.layers.{}.self_attn.k_norm.weight", i);
            let q_norm = if self.reader.contains(&q_norm_name) {
                Some(comp_rms(self.load_rmsnorm(
                    &q_norm_name,
                    device,
                    cfg.rms_norm_eps,
                )?))
            } else {
                None
            };
            let k_norm = if self.reader.contains(&k_norm_name) {
                Some(comp_rms(self.load_rmsnorm(
                    &k_norm_name,
                    device,
                    cfg.rms_norm_eps,
                )?))
            } else {
                None
            };
            if q_norm.is_some() != k_norm.is_some() {
                return Err(OpError::Shape(format!(
                    "load: layer {} has q_norm={} but k_norm={}; require both or neither",
                    i,
                    q_norm.is_some(),
                    k_norm.is_some()
                )));
            }

            blocks.push(DecoderBlock {
                attention: Attention {
                    input_layernorm: comp_rms(input_layernorm),
                    qkv_proj: comp_linear(qkv_proj),
                    o_proj: comp_linear(o_proj),
                    q_norm,
                    k_norm,
                    sin: sin_cache.clone(),
                    cos: cos_cache.clone(),
                    head_num: cfg.head_num,
                    kv_head_num: cfg.kv_head_num,
                    head_dim: cfg.head_dim,
                    scale,
                    scratch: None,
                },
                ffn: DenseFfn {
                    post_attention_layernorm: comp_rms(post_attention_layernorm),
                    gate_up_proj,
                    down_proj,
                    scratch: None,
                },
            });
        }

        Ok(Decoder {
            embed: Embed { table: embed_table },
            blocks,
            norm: comp_rms(final_norm),
            lm_head: LmHead {
                proj: comp_linear(lm_head),
            },
            dims: ModelDims {
                dim: cfg.dim,
                q_dim,
                kv_dim,
                qkv_dim: q_dim + 2 * kv_dim,
                intermediate_size: cfg.intermediate_size,
                vocab_size: cfg.vocab_size,
                head_num: cfg.head_num,
                head_dim: cfg.head_dim,
                kv_head_num: cfg.kv_head_num,
                num_layers: cfg.layer_num,
                num_experts: 0,
                experts_per_tok: 0,
                moe_intermediate_size: 0,
                num_shared_experts: 0,
            },
            scratch: None,
        })
    }

    /// Load a complete Llama3 model (no per-block Q/K norms).
    pub fn load_llama3<T: Dtype + crate::domain::dtype::Dtype, D: OpBackend + LlmBackend>(
        &self,
        cfg: &LoadConfig,
        device: &D,
    ) -> OpResult<Llama3Model<T, D>> {
        self.build_decoder(cfg, device)
    }

    /// Load a complete Qwen3 model. Per-block Q/K RMSNorms are populated when
    /// present in the weights.
    pub fn load_qwen3<T: Dtype + crate::domain::dtype::Dtype, D: OpBackend + LlmBackend>(
        &self,
        cfg: &LoadConfig,
        device: &D,
    ) -> OpResult<Qwen3Model<T, D>> {
        self.build_decoder(cfg, device)
    }
}

fn comp_linear<T: Dtype + crate::domain::dtype::Dtype, D: OpBackend + LlmBackend>(
    l: Linear<T, D>,
) -> CompLinear<T, D> {
    CompLinear::new(l.weight, l.bias)
}

fn comp_rms<T: Dtype + crate::domain::dtype::Dtype, D: OpBackend + LlmBackend>(
    r: RMSNorm<T, D>,
) -> CompRmsNorm<T, D> {
    CompRmsNorm {
        weight: r.weight,
        eps: r.eps,
    }
}

// ─── Internal: convert safetensor view to Tensor<T, D> ───────────────────────

/// Map a safetensors view's dtype to our `DataType`, erroring on unsupported.
fn st_dtype(view: &TensorView) -> OpResult<DataType> {
    Ok(match view.dtype() {
        safetensors::Dtype::F32 => DataType::F32,
        safetensors::Dtype::F16 => DataType::F16,
        safetensors::Dtype::BF16 => DataType::BF16,
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

fn tensor_from_safetensor_view<T: Dtype, D: MemoryPort>(
    view: &TensorView,
    device: &D,
) -> OpResult<Tensor<T, D>> {
    let shape_vec: Vec<usize> = view.shape().to_vec();
    let numel: usize = shape_vec.iter().product();
    let src_bytes = view.data();
    let src_dtype = match view.dtype() {
        safetensors::Dtype::F32 => DataType::F32,
        safetensors::Dtype::F16 => DataType::F16,
        safetensors::Dtype::BF16 => DataType::BF16,
        safetensors::Dtype::I32 => DataType::I32,
        safetensors::Dtype::I8 => DataType::I8,
        other => {
            return Err(OpError::Kernel(format!(
                "unsupported safetensor dtype: {:?}",
                other
            )));
        }
    };

    let shape = Shape::from_slice(&shape_vec);
    let size_bytes = numel * T::SIZE_BYTES;

    // Build a host buffer in target dtype, then upload to device in one shot.
    let mut host_buf: Vec<u8> = vec![0u8; size_bytes];
    if src_dtype == T::DATA_TYPE {
        // Direct byte copy — the view must carry exactly the bytes its shape
        // implies, otherwise the weight would be silently zero-padded.
        if src_bytes.len() != size_bytes {
            return Err(OpError::Shape(format!(
                "tensor_from_safetensor_view: view byte length {} != expected {} for shape {:?} \
                 (corrupt safetensors?)",
                src_bytes.len(),
                size_bytes,
                shape_vec,
            )));
        }
        host_buf.copy_from_slice(src_bytes);
    } else {
        // Element-wise cast through f64 intermediate.
        cast_bytes(
            src_bytes,
            src_dtype,
            host_buf.as_mut_ptr(),
            T::DATA_TYPE,
            numel,
        );
    }

    Tensor::<T, D>::from_host_bytes(&host_buf, shape, device)
}

/// Public re-export of the internal cast helper for diffusion loaders.
pub unsafe fn cast_bytes_pub(
    src: &[u8],
    src_dt: DataType,
    dst: *mut u8,
    dst_dt: DataType,
    numel: usize,
) {
    cast_bytes(src, src_dt, dst, dst_dt, numel);
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
fn compute_rope_cache<T: Dtype, D: OpBackend>(
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

    // Build host buffers in target dtype, then upload.
    let elem = T::SIZE_BYTES;
    let n = max_seq_len * half_dim;
    let mut sin_host = vec![0u8; n * elem];
    let mut cos_host = vec![0u8; n * elem];
    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let angle = pos as f64 * freqs[i];
            let offset = (pos * half_dim + i) * elem;
            // SAFETY: offset + elem <= n * elem by construction.
            unsafe {
                write_dtype_bytes(sin_host.as_mut_ptr().add(offset), angle.sin(), T::DATA_TYPE);
                write_dtype_bytes(cos_host.as_mut_ptr().add(offset), angle.cos(), T::DATA_TYPE);
            }
        }
    }

    let shape = Shape::from_slice(&[max_seq_len, half_dim]);
    let sin_tensor = Tensor::<T, D>::from_host_bytes(&sin_host, shape, device)?;
    let cos_tensor = Tensor::<T, D>::from_host_bytes(&cos_host, shape, device)?;
    Ok((sin_tensor, cos_tensor))
}

/// Write an f64 value as the target dtype bytes.
unsafe fn write_dtype_bytes(dst: *mut u8, val: f64, dt: DataType) {
    unsafe {
        match dt {
            DataType::F32 => {
                std::ptr::copy_nonoverlapping((val as f32).to_le_bytes().as_ptr(), dst, 4)
            }
            DataType::BF16 => std::ptr::copy_nonoverlapping(
                half::bf16::from_f64(val).to_le_bytes().as_ptr(),
                dst,
                2,
            ),
            DataType::F16 => std::ptr::copy_nonoverlapping(
                half::f16::from_f64(val).to_le_bytes().as_ptr(),
                dst,
                2,
            ),
            DataType::I32 => {
                std::ptr::copy_nonoverlapping((val as i32).to_le_bytes().as_ptr(), dst, 4)
            }
            DataType::I8 => {
                *dst = val as i8 as u8;
            }
        }
    }
}
