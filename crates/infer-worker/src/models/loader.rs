//! Weight loader — loads safetensors into model structures.
//!
//! This lives in models/ (not domain) because it's about
//! constructing concrete model instances from files — an application concern.

use std::marker::PhantomData;

use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

use crate::domain::ports::{Device, OpBackend, OpResult, OpError};
use crate::domain::types::{Dtype, DataType, Shape};
use crate::domain::tensor::Tensor;
use super::layers::{Linear, RMSNorm, Embedding};
use super::llama3::{alloc, Llama3Model, Llama3Layer};
use super::qwen3::{Qwen3Model, Qwen3Layer};

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
}

/// Weight loader — reads safetensors and constructs typed tensors.
pub struct WeightLoader<'a> {
    safetensors: SafeTensors<'a>,
}

impl<'a> WeightLoader<'a> {
    /// Create from raw safetensors bytes.
    pub fn new(data: &'a [u8]) -> Result<Self, String> {
        let safetensors = SafeTensors::deserialize(data)
            .map_err(|e| format!("safetensors parse: {}", e))?;
        Ok(Self { safetensors })
    }

    /// Load a tensor by name, cast to target dtype T, place on device D.
    pub fn load_tensor<T: Dtype, D: Device>(&self, name: &str, device: &D) -> OpResult<Tensor<T, D>> {
        let view = self.safetensors.tensor(name)
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

    /// Load fused QKV: concatenate q_proj, k_proj, v_proj along rows → [q_dim+2*kv_dim, dim]
    pub fn load_fused_qkv<T: Dtype, D: OpBackend>(
        &self, layer_idx: usize, q_dim: usize, kv_dim: usize, dim: usize, device: &D,
    ) -> OpResult<Linear<T, D>> {
        let q = self.load_tensor::<T, D>(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx), device)?;
        let k = self.load_tensor::<T, D>(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx), device)?;
        let v = self.load_tensor::<T, D>(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx), device)?;
        // Concatenate: [q_dim, dim] + [kv_dim, dim] + [kv_dim, dim] → [q_dim+2*kv_dim, dim]
        let total_rows = q_dim + 2 * kv_dim;
        let mut fused = alloc::<T, D>(total_rows, dim, device)?;
        let elem = T::SIZE_BYTES;
        unsafe {
            let dst = fused.data_ptr_mut() as *mut u8;
            std::ptr::copy_nonoverlapping(q.data_ptr() as *const u8, dst, q_dim * dim * elem);
            std::ptr::copy_nonoverlapping(k.data_ptr() as *const u8, dst.add(q_dim * dim * elem), kv_dim * dim * elem);
            std::ptr::copy_nonoverlapping(v.data_ptr() as *const u8, dst.add((q_dim + kv_dim) * dim * elem), kv_dim * dim * elem);
        }
        Ok(Linear::new(fused, None))
    }

    /// Load a complete Llama3 model.
    pub fn load_llama3<T: Dtype, D: OpBackend>(&self, cfg: &LoadConfig, device: &D) -> OpResult<Llama3Model<T, D>> {
        let embed_tokens = self.load_embedding("model.embed_tokens.weight", device)?;
        let norm = self.load_rmsnorm("model.norm.weight", device, cfg.rms_norm_eps)?;

        // lm_head — may share weights with embed_tokens
        let lm_head = if self.safetensors.tensor("lm_head.weight").is_ok() {
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
        let (sin_cache, cos_cache) = compute_rope_cache::<T, D>(cfg.seq_len, cfg.head_dim, cfg.rope_theta, device)?;

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
        let lm_head = if self.safetensors.tensor("lm_head.weight").is_ok() {
            self.load_linear("lm_head.weight", None, device)?
        } else {
            Linear::new(embed_tokens.table.clone(), None)
        };

        let mut layers = Vec::with_capacity(cfg.layer_num);
        for i in 0..cfg.layer_num {
            let q_norm_name = format!("model.layers.{}.self_attn.q_norm.weight", i);
            let k_norm_name = format!("model.layers.{}.self_attn.k_norm.weight", i);
            let q_norm = if self.safetensors.tensor(&q_norm_name).is_ok() {
                Some(self.load_rmsnorm(&q_norm_name, device, cfg.rms_norm_eps)?)
            } else { None };
            let k_norm = if self.safetensors.tensor(&k_norm_name).is_ok() {
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

        let (sin_cache, cos_cache) = compute_rope_cache::<T, D>(cfg.seq_len, cfg.head_dim, cfg.rope_theta, device)?;

        Ok(Qwen3Model {
            embed_tokens, layers, norm, lm_head, sin_cache, cos_cache,
            head_num: cfg.head_num, kv_head_num: cfg.kv_head_num, head_dim: cfg.head_dim,
            dim: cfg.dim, kv_dim: cfg.kv_head_num * cfg.head_dim,
            intermediate_size: cfg.intermediate_size, vocab_size: cfg.vocab_size,
        })
    }
}

// ─── Internal: convert safetensor view to Tensor<T, D> ───────────────────────

fn tensor_from_safetensor_view<T: Dtype, D: Device>(view: &TensorView, device: &D) -> OpResult<Tensor<T, D>> {
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
    let layout = std::alloc::Layout::from_size_align(size_bytes.max(1), 16)
        .map_err(|e| OpError::Kernel(format!("{}", e)))?;
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() { return Err(OpError::Kernel("alloc failed".into())); }

    // Cast src_dtype → T::DATA_TYPE if needed
    if src_dtype == T::DATA_TYPE {
        // Direct copy
        unsafe { std::ptr::copy_nonoverlapping(src_bytes.as_ptr(), ptr, size_bytes.min(src_bytes.len())); }
    } else {
        // Element-wise cast through f64 intermediate
        cast_bytes(src_bytes, src_dtype, ptr, T::DATA_TYPE, numel);
    }

    Ok(Tensor {
        shape, strides: shape.contiguous_strides(),
        offset_elems: 0, numel, is_contiguous: true,
        storage_ptr: ptr, storage_len: size_bytes,
        device: device.clone(), _marker: PhantomData,
    })
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
    device: &D,
) -> OpResult<(Tensor<T, D>, Tensor<T, D>)> {
    let half_dim = head_dim / 2;

    // Compute frequencies: freq_i = 1 / theta^(2i / head_dim)
    let freqs: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    // Allocate host buffers and fill
    let sin_shape = Shape::from_slice(&[max_seq_len, half_dim]);
    let mut sin_tensor = D::alloc_tensor::<T>(sin_shape, device)?;
    let mut cos_tensor = D::alloc_tensor::<T>(sin_shape, device)?;

    let sin_ptr = sin_tensor.data_ptr_mut() as *mut u8;
    let cos_ptr = cos_tensor.data_ptr_mut() as *mut u8;
    let elem = T::SIZE_BYTES;

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let angle = pos as f64 * freqs[i];
            let sin_val = angle.sin();
            let cos_val = angle.cos();

            let offset = (pos * half_dim + i) * elem;
            unsafe {
                write_dtype_bytes(sin_ptr.add(offset), sin_val, T::DATA_TYPE);
                write_dtype_bytes(cos_ptr.add(offset), cos_val, T::DATA_TYPE);
            }
        }
    }

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

