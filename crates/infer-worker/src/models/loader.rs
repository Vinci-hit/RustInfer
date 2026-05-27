//! Weight loader — loads safetensors into model structures.
//!
//! This lives in models/ (not domain) because it's about constructing
//! concrete model instances. Filesystem access is delegated to
//! `infra::io::SafetensorsReader` to keep the I/O concern in the
//! infrastructure layer.

use safetensors::tensor::TensorView;

use crate::domain::ports::{MemoryPort, OpBackend, OpResult, OpError};
use crate::domain::types::{Dtype, DataType, Shape};
use crate::domain::tensor::Tensor;
use crate::infra::io::SafetensorsReader;
use super::layers::{Linear, RMSNorm, Embedding};
use super::llama3::{Llama3Model, Llama3Layer};
use super::qwen3::{Qwen3Model, Qwen3Layer};

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

    /// Load a tensor by name, cast to target dtype T, place on device D.
    pub fn load_tensor<T: Dtype, D: MemoryPort>(&self, name: &str, device: &D) -> OpResult<Tensor<T, D>> {
        let view = self.reader.read_view(name)
            .map_err(|e| OpError::Kernel(format!("tensor '{}' not found: {}", name, e)))?;
        tensor_from_safetensor_view::<T, D>(&view, device)
    }

    /// Load a Linear layer (weight + optional bias).
    pub fn load_linear<T: Dtype, D: OpBackend>(&self, weight_name: &str, bias_name: Option<&str>, device: &D) -> OpResult<Linear<T, D>> {
        let weight = self.load_tensor::<T, D>(weight_name, device)?;
        let bias = if let Some(bn) = bias_name {
            Some(self.load_tensor::<T, D>(bn, device)?)
        } else {
            None
        };
        Ok(Linear::new(weight, bias))
    }

    /// Load an RMSNorm layer.
    pub fn load_rmsnorm<T: Dtype, D: OpBackend>(&self, name: &str, device: &D, eps: f32) -> OpResult<RMSNorm<T, D>> {
        let weight = self.load_tensor::<T, D>(name, device)?;
        Ok(RMSNorm::new(weight, eps))
    }

    /// Load an Embedding table.
    pub fn load_embedding<T: Dtype, D: OpBackend>(&self, name: &str, device: &D) -> OpResult<Embedding<T, D>> {
        let table = self.load_tensor::<T, D>(name, device)?;
        Ok(Embedding { table })
    }

    /// Load fused QKV: concatenate q_proj, k_proj, v_proj along rows → [q_dim+2*kv_dim, dim].
    ///
    /// To keep the tensor on `device` and avoid host↔device round-trips, we
    /// concatenate at the host-bytes level (in target dtype T) and upload
    /// the fused result in a single `Tensor::from_host_bytes`.
    pub fn load_fused_qkv<T: Dtype, D: OpBackend>(
        &self, layer_idx: usize, q_dim: usize, kv_dim: usize, dim: usize, device: &D,
    ) -> OpResult<Linear<T, D>> {
        let q_view = self.reader.read_view(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx))
            .map_err(|e| OpError::Kernel(format!("q_proj layer {}: {}", layer_idx, e)))?;
        let k_view = self.reader.read_view(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx))
            .map_err(|e| OpError::Kernel(format!("k_proj layer {}: {}", layer_idx, e)))?;
        let v_view = self.reader.read_view(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx))
            .map_err(|e| OpError::Kernel(format!("v_proj layer {}: {}", layer_idx, e)))?;

        let total_rows = q_dim + 2 * kv_dim;
        let elem = T::SIZE_BYTES;
        let total_bytes = total_rows * dim * elem;
        let mut host = vec![0u8; total_bytes];

        // Helper: cast a single weight view into the host buffer at `row_offset`.
        let mut cast_into = |view: &TensorView, row_offset: usize, expected_rows: usize| -> OpResult<()> {
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
                // SAFETY: host buffer has at least row_offset*dim*elem + n bytes by construction.
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, n.min(src.len())); }
            } else {
                cast_bytes(src, src_dt, dst, T::DATA_TYPE, numel);
            }
            Ok(())
        };

        cast_into(&q_view, 0, q_dim)?;
        cast_into(&k_view, q_dim, kv_dim)?;
        cast_into(&v_view, q_dim + kv_dim, kv_dim)?;

        let fused = Tensor::<T, D>::from_host_bytes(
            &host,
            Shape::from_slice(&[total_rows, dim]),
            device,
        )?;
        Ok(Linear::new(fused, None))
    }

    /// Load a complete Llama3 model.
    pub fn load_llama3<T: Dtype, D: OpBackend>(&self, cfg: &LoadConfig, device: &D) -> OpResult<Llama3Model<T, D>> {
        let embed_tokens = self.load_embedding("model.embed_tokens.weight", device)?;
        let norm = self.load_rmsnorm("model.norm.weight", device, cfg.rms_norm_eps)?;

        // lm_head — may share weights with embed_tokens
        let lm_head = if self.reader.contains("lm_head.weight") {
            self.load_linear("lm_head.weight", None, device)?
        } else {
            Linear::new(embed_tokens.table.clone(), None)
        };

        let mut layers = Vec::with_capacity(cfg.layer_num);
        for i in 0..cfg.layer_num {
            layers.push(Llama3Layer {
                input_layernorm: self.load_rmsnorm(&format!("model.layers.{}.input_layernorm.weight", i), device, cfg.rms_norm_eps)?,
                post_attention_layernorm: self.load_rmsnorm(&format!("model.layers.{}.post_attention_layernorm.weight", i), device, cfg.rms_norm_eps)?,
                qkv_proj: self.load_fused_qkv(i, cfg.head_num * cfg.head_dim, cfg.kv_head_num * cfg.head_dim, cfg.dim, device)?,
                o_proj: self.load_linear(&format!("model.layers.{}.self_attn.o_proj.weight", i), None, device)?,
                gate_proj: self.load_linear(&format!("model.layers.{}.mlp.gate_proj.weight", i), None, device)?,
                up_proj: self.load_linear(&format!("model.layers.{}.mlp.up_proj.weight", i), None, device)?,
                down_proj: self.load_linear(&format!("model.layers.{}.mlp.down_proj.weight", i), None, device)?,
            });
        }

        // RoPE sin/cos cache — precomputed from theta
        let (sin_cache, cos_cache) = compute_rope_cache::<T, D>(cfg.seq_len, cfg.head_dim, cfg.rope_theta, cfg.rope_scaling.as_ref(), device)?;

        Ok(Llama3Model {
            embed_tokens, layers, norm, lm_head, sin_cache, cos_cache,
            head_num: cfg.head_num, kv_head_num: cfg.kv_head_num, head_dim: cfg.head_dim,
            dim: cfg.dim, kv_dim: cfg.kv_head_num * cfg.head_dim,
            intermediate_size: cfg.intermediate_size, vocab_size: cfg.vocab_size,
        })
    }

    /// Load a complete Qwen3 model (adds q_norm / k_norm per layer).
    pub fn load_qwen3<T: Dtype, D: OpBackend>(&self, cfg: &LoadConfig, device: &D) -> OpResult<Qwen3Model<T, D>> {
        let embed_tokens = self.load_embedding("model.embed_tokens.weight", device)?;
        let norm = self.load_rmsnorm("model.norm.weight", device, cfg.rms_norm_eps)?;
        let lm_head = if self.reader.contains("lm_head.weight") {
            self.load_linear("lm_head.weight", None, device)?
        } else {
            Linear::new(embed_tokens.table.clone(), None)
        };

        let mut layers = Vec::with_capacity(cfg.layer_num);
        for i in 0..cfg.layer_num {
            let q_norm_name = format!("model.layers.{}.self_attn.q_norm.weight", i);
            let k_norm_name = format!("model.layers.{}.self_attn.k_norm.weight", i);
            let q_norm = if self.reader.contains(&q_norm_name) {
                Some(self.load_rmsnorm(&q_norm_name, device, cfg.rms_norm_eps)?)
            } else { None };
            let k_norm = if self.reader.contains(&k_norm_name) {
                Some(self.load_rmsnorm(&k_norm_name, device, cfg.rms_norm_eps)?)
            } else { None };

            layers.push(Qwen3Layer {
                input_layernorm: self.load_rmsnorm(&format!("model.layers.{}.input_layernorm.weight", i), device, cfg.rms_norm_eps)?,
                post_attention_layernorm: self.load_rmsnorm(&format!("model.layers.{}.post_attention_layernorm.weight", i), device, cfg.rms_norm_eps)?,
                qkv_proj: self.load_fused_qkv(i, cfg.head_num * cfg.head_dim, cfg.kv_head_num * cfg.head_dim, cfg.dim, device)?,
                o_proj: self.load_linear(&format!("model.layers.{}.self_attn.o_proj.weight", i), None, device)?,
                gate_proj: self.load_linear(&format!("model.layers.{}.mlp.gate_proj.weight", i), None, device)?,
                up_proj: self.load_linear(&format!("model.layers.{}.mlp.up_proj.weight", i), None, device)?,
                down_proj: self.load_linear(&format!("model.layers.{}.mlp.down_proj.weight", i), None, device)?,
                q_norm,
                k_norm,
            });
        }

        let (sin_cache, cos_cache) = compute_rope_cache::<T, D>(cfg.seq_len, cfg.head_dim, cfg.rope_theta, cfg.rope_scaling.as_ref(), device)?;

        Ok(Qwen3Model {
            embed_tokens, layers, norm, lm_head, sin_cache, cos_cache,
            head_num: cfg.head_num, kv_head_num: cfg.kv_head_num, head_dim: cfg.head_dim,
            dim: cfg.dim, kv_dim: cfg.kv_head_num * cfg.head_dim,
            intermediate_size: cfg.intermediate_size, vocab_size: cfg.vocab_size,
        })
    }
}

// ─── Internal: convert safetensor view to Tensor<T, D> ───────────────────────

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
        other => return Err(OpError::Kernel(format!("unsupported safetensor dtype: {:?}", other))),
    };

    let shape = Shape::from_slice(&shape_vec);
    let size_bytes = numel * T::SIZE_BYTES;

    // Build a host buffer in target dtype, then upload to device in one shot.
    let mut host_buf: Vec<u8> = vec![0u8; size_bytes];
    if src_dtype == T::DATA_TYPE {
        // Direct byte copy.
        let n = size_bytes.min(src_bytes.len());
        host_buf[..n].copy_from_slice(&src_bytes[..n]);
    } else {
        // Element-wise cast through f64 intermediate.
        cast_bytes(src_bytes, src_dtype, host_buf.as_mut_ptr(), T::DATA_TYPE, numel);
    }

    Tensor::<T, D>::from_host_bytes(&host_buf, shape, device)
}

/// Element-wise dtype cast via f64 intermediate.
fn cast_bytes(src: &[u8], src_dt: DataType, dst: *mut u8, dst_dt: DataType, numel: usize) {
    use half::{bf16, f16};
    for i in 0..numel {
        let val: f64 = match src_dt {
            DataType::F32 => { let b = &src[i*4..i*4+4]; f64::from(f32::from_le_bytes(b.try_into().unwrap())) }
            DataType::BF16 => { let b = &src[i*2..i*2+2]; f64::from(bf16::from_le_bytes(b.try_into().unwrap()).to_f32()) }
            DataType::F16 => { let b = &src[i*2..i*2+2]; f64::from(f16::from_le_bytes(b.try_into().unwrap()).to_f32()) }
            DataType::I32 => { let b = &src[i*4..i*4+4]; i32::from_le_bytes(b.try_into().unwrap()) as f64 }
            DataType::I8 => { src[i] as i8 as f64 }
        };
        unsafe {
            match dst_dt {
                DataType::F32 => { std::ptr::copy_nonoverlapping((val as f32).to_le_bytes().as_ptr(), dst.add(i*4), 4); }
                DataType::BF16 => { std::ptr::copy_nonoverlapping(bf16::from_f64(val).to_le_bytes().as_ptr(), dst.add(i*2), 2); }
                DataType::F16 => { std::ptr::copy_nonoverlapping(f16::from_f64(val).to_le_bytes().as_ptr(), dst.add(i*2), 2); }
                DataType::I32 => { std::ptr::copy_nonoverlapping((val as i32).to_le_bytes().as_ptr(), dst.add(i*4), 4); }
                DataType::I8 => { *dst.add(i) = val as i8 as u8; }
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
                let smooth =
                    (orig / wavelength - s.low_freq_factor as f64)
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
            DataType::F32 => std::ptr::copy_nonoverlapping((val as f32).to_le_bytes().as_ptr(), dst, 4),
            DataType::BF16 => std::ptr::copy_nonoverlapping(half::bf16::from_f64(val).to_le_bytes().as_ptr(), dst, 2),
            DataType::F16 => std::ptr::copy_nonoverlapping(half::f16::from_f64(val).to_le_bytes().as_ptr(), dst, 2),
            DataType::I32 => std::ptr::copy_nonoverlapping((val as i32).to_le_bytes().as_ptr(), dst, 4),
            DataType::I8 => { *dst = val as i8 as u8; }
        }
    }
}

