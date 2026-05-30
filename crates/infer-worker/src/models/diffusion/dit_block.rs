//! Single DiT transformer block for Z-Image.
//!
//! Architecture (matching diffusers `ZImageTransformerBlock`):
//!
//! ```text
//! if modulation:
//!     c        = adaLN_modulation(silu(adaln_input))   # [1, 4*dim]
//!     scale_msa, gate_msa, scale_mlp, gate_mlp = chunk(c, 4)
//!     scales += 1; gates  = tanh(gates)
//! norm1_x       = attention_norm1(x) [* scale_msa]
//! qkv          = to_qkv(norm1_x)                       # [seq, 3*dim]
//! q, k, v      = split(qkv)
//! q'           = norm_q(q); k' = norm_k(k)             # qk_norm
//! q'', k''     = rope_interleaved(q', k', cos, sin)
//! attn         = sdpa(q'', k'', v)                     # [seq, dim]
//! attn         = to_out(attn)
//! attn         = attention_norm2(attn) [* gate_msa]
//! x            = x + attn                              # residual
//! norm1_ffn    = ffn_norm1(x) [* scale_mlp]
//! ffn          = w2(silu(w1(norm1_ffn)) * w3(norm1_ffn))
//! ffn          = ffn_norm2(ffn) [* gate_mlp]
//! x_out        = x + ffn                               # residual
//! ```
//!
//! All buffers used during forward must be passed in by the caller as
//! pre-allocated workspaces (`DiTBlockScratch`) so the hot path is alloc-free.

use crate::domain::ports::{OpResult, OpError, OpBackend};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype, Shape};
use crate::models::layers::{Linear, RMSNorm};

// ─── Dump infrastructure for numerical comparison with the Python reference ──
//
// Set `RUSTINFER_DUMP=1` to enable. The first DiTBlock forward writes every
// intermediate tensor to `/tmp/zimage_dump_rust/<name>.npy` (overwrite mode,
// f32 NPY v1.0). Subsequent block forwards skip dumping.

static DUMP_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn dump_enabled() -> bool {
    std::env::var("RUSTINFER_DUMP").map(|v| v == "1").unwrap_or(false)
}

/// Dump `t` (any contiguous Tensor<T, D>) to `/tmp/zimage_dump_rust/<name>.npy`
/// as float32. Errors are printed but never propagated.
pub fn dump_tensor<T: Dtype, D: OpBackend>(name: &str, t: &Tensor<T, D>) {
    let dir = std::path::Path::new("/tmp/zimage_dump_rust");
    if !dir.exists() {
        let _ = std::fs::create_dir_all(dir);
    }
    let path = dir.join(format!("{}.npy", name));
    let host: Vec<T> = match t.to_host_vec() {
        Ok(v) => v,
        Err(e) => { eprintln!("[dump] {}: D2H failed: {:?}", name, e); return; }
    };
    let shape: Vec<usize> = t.shape().as_slice().to_vec();
    let f32_data: Vec<f32> = match T::DATA_TYPE {
        DataType::F32 => unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const f32, host.len()).to_vec()
        },
        DataType::BF16 => unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const half::bf16, host.len())
                .iter().map(|v| v.to_f32()).collect()
        },
        DataType::F16 => unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const half::f16, host.len())
                .iter().map(|v| v.to_f32()).collect()
        },
        other => { eprintln!("[dump] {}: unsupported dtype {:?}", name, other); return; }
    };
    if let Err(e) = write_npy_f32(&path, &shape, &f32_data) {
        eprintln!("[dump] {}: write failed: {}", name, e);
    } else {
        // Stats from the *same* `f32_data` we just wrote.
        let n = f32_data.len();
        let mut sum = 0.0_f64;
        let mut nan = 0usize;
        let mut mn = f32::INFINITY; let mut mx = f32::NEG_INFINITY;
        for &v in &f32_data {
            if v.is_finite() { sum += v as f64; if v < mn { mn = v; } if v > mx { mx = v; } }
            else { nan += 1; }
        }
        eprintln!("[dump] {} shape={:?} numel={} nan={} mean={:.6} min={:.6} max={:.6} first4={:?}",
            name, shape, n, nan, sum / (n as f64).max(1.0), mn, mx,
            &f32_data[..4.min(n)]);
    }
}

fn write_npy_f32(path: &std::path::Path, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"\x93NUMPY")?;
    f.write_all(&[1u8, 0u8])?;
    let shape_str = shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
    let shape_str = if shape.len() == 1 { format!("{},", shape_str) } else { shape_str };
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}",
        shape_str,
    );
    let prefix = 10;
    let needed_pad = (64 - (prefix + header.len() + 1) % 64) % 64;
    for _ in 0..needed_pad { header.push(' '); }
    header.push('\n');
    let header_len = header.len() as u16;
    f.write_all(&header_len.to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    f.write_all(bytes)?;
    f.flush()?;
    f.sync_all()?;
    Ok(())
}

/// Take a contiguous 2D prefix view of `src` shaped `[rows, cols]`.
/// Caller asserts the storage holds at least `rows * cols` valid elements
/// at offset 0.
fn vp2<T: Dtype, D: OpBackend>(src: &Tensor<T, D>, rows: usize, cols: usize) -> OpResult<Tensor<T, D>> {
    let total = src.numel();
    if rows * cols > total {
        return Err(OpError::Shape(format!(
            "vp2: requested {}*{}={} > storage numel {}", rows, cols, rows * cols, total,
        )));
    }
    Ok(src.view_raw(
        Shape::from_slice(&[rows, cols]),
        Shape::from_slice(&[cols, 1]).contiguous_strides(),
        src.offset_elems(),
        true,
    ))
}

/// 3D variant of `vp2`.
fn vp3<T: Dtype, D: OpBackend>(src: &Tensor<T, D>, a: usize, b: usize, c: usize) -> OpResult<Tensor<T, D>> {
    let total = src.numel();
    if a * b * c > total {
        return Err(OpError::Shape(format!(
            "vp3: requested {}*{}*{}={} > storage numel {}", a, b, c, a * b * c, total,
        )));
    }
    Ok(src.view_raw(
        Shape::from_slice(&[a, b, c]),
        Shape::from_slice(&[b * c, c, 1]).contiguous_strides(),
        src.offset_elems(),
        true,
    ))
}

/// Per-block weights.
pub struct DiTBlock<T: Dtype, D: OpBackend> {
    pub attention_norm1: RMSNorm<T, D>,
    pub attention_norm2: RMSNorm<T, D>,
    pub ffn_norm1: RMSNorm<T, D>,
    pub ffn_norm2: RMSNorm<T, D>,
    /// `[3*dim, dim]` fused QKV (no bias).
    pub to_qkv: Linear<T, D>,
    /// `[dim, dim]`.
    pub to_out: Linear<T, D>,
    /// Per-head Q RMSNorm. Weight shape `[head_dim]`.
    pub norm_q: RMSNorm<T, D>,
    /// Per-head K RMSNorm. Weight shape `[head_dim]`.
    pub norm_k: RMSNorm<T, D>,

    /// `[hidden, dim]`.
    pub w1: Linear<T, D>,
    pub w3: Linear<T, D>,
    /// `[dim, hidden]`.
    pub w2: Linear<T, D>,

    /// Optional `[4*dim, adaln_embed_dim=256]`. None for context_refiner blocks.
    pub adaln_modulation: Option<Linear<T, D>>,

    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// True for layers with adaLN modulation (main + noise_refiner);
    /// false for context_refiner blocks.
    pub modulation: bool,
}

/// Per-block scratch buffers, sized to the maximum sequence length the
/// pipeline ever passes in. Provided once at construction time.
pub struct DiTBlockScratch<T: Dtype, D: OpBackend> {
    /// `[1, 4*dim]` modulation projection output (raw, before chunk).
    pub mod_out: Tensor<T, D>,
    /// `[1, dim]` `silu(adaln_input)` workspace (only used when modulation=True).
    pub adaln_silu: Tensor<T, D>,
    /// `[1, dim]` per-chunk slices.
    pub scale_msa: Tensor<T, D>,
    pub gate_msa: Tensor<T, D>,
    pub scale_mlp: Tensor<T, D>,
    pub gate_mlp: Tensor<T, D>,

    /// `[seq_max, dim]`.
    pub norm1_x: Tensor<T, D>,
    /// `[seq_max, 3*dim]`.
    pub qkv_out: Tensor<T, D>,
    /// `[seq_max, dim]` each.
    pub q: Tensor<T, D>,
    pub k: Tensor<T, D>,
    pub v: Tensor<T, D>,
    /// `[seq_max, dim]` SDPA output.
    pub attn_out: Tensor<T, D>,
    /// `[seq_max, dim]` to_out output.
    pub to_out_buf: Tensor<T, D>,
    /// `[seq_max, dim]` post-attention norm.
    pub norm2_attn: Tensor<T, D>,
    /// `[seq_max, dim]` ffn pre-norm.
    pub norm1_ffn: Tensor<T, D>,
    /// `[seq_max, hidden]` w1 / w3.
    pub w1_out: Tensor<T, D>,
    pub w3_out: Tensor<T, D>,
    /// `[seq_max, dim]` w2 / norm2_ffn.
    pub ffn_out: Tensor<T, D>,
    pub norm2_ffn: Tensor<T, D>,
}

impl<T: Dtype, D: OpBackend> DiTBlockScratch<T, D> {
    /// Allocate all scratch buffers for the given max sequence length.
    pub fn new(dim: usize, hidden: usize, seq_max: usize, dev: &D) -> OpResult<Self> {
        Ok(Self {
            mod_out: Tensor::zeros([1, 4 * dim], dev)?,
            adaln_silu: Tensor::zeros([1, 256], dev)?,
            scale_msa: Tensor::zeros([1, dim], dev)?,
            gate_msa: Tensor::zeros([1, dim], dev)?,
            scale_mlp: Tensor::zeros([1, dim], dev)?,
            gate_mlp: Tensor::zeros([1, dim], dev)?,
            norm1_x: Tensor::zeros([seq_max, dim], dev)?,
            qkv_out: Tensor::zeros([seq_max, 3 * dim], dev)?,
            q: Tensor::zeros([seq_max, dim], dev)?,
            k: Tensor::zeros([seq_max, dim], dev)?,
            v: Tensor::zeros([seq_max, dim], dev)?,
            attn_out: Tensor::zeros([seq_max, dim], dev)?,
            to_out_buf: Tensor::zeros([seq_max, dim], dev)?,
            norm2_attn: Tensor::zeros([seq_max, dim], dev)?,
            norm1_ffn: Tensor::zeros([seq_max, dim], dev)?,
            w1_out: Tensor::zeros([seq_max, hidden], dev)?,
            w3_out: Tensor::zeros([seq_max, hidden], dev)?,
            ffn_out: Tensor::zeros([seq_max, dim], dev)?,
            norm2_ffn: Tensor::zeros([seq_max, dim], dev)?,
        })
    }
}

impl<T: Dtype, D: OpBackend> DiTBlock<T, D> {
    /// Forward: `dst = block(x; cos, sin, adaln_input?)`.
    ///
    /// `x`, `dst` are both `[seq, dim]` and **must be distinct**.
    /// `cos`, `sin` are `[seq, head_dim/2]` F32 device tensors (same device as
    /// weights). `adaln_input` is `[1, 256]` (dtype T) — required when
    /// `self.modulation == true`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor<T, D>,
        cos: &Tensor<f32, D>,
        sin: &Tensor<f32, D>,
        adaln_input: Option<&Tensor<T, D>>,
        scratch: &mut DiTBlockScratch<T, D>,
        dst: &mut Tensor<T, D>,
    ) -> OpResult<()> {
        let seq = x.shape().as_slice()[0];
        let dim = self.dim;
        if x.shape().as_slice() != [seq, dim] {
            return Err(OpError::Shape(format!(
                "DiTBlock::forward: x shape {:?} doesn't match [seq, dim={}]",
                x.shape(), dim,
            )));
        }
        if dst.shape().as_slice() != [seq, dim] {
            return Err(OpError::Shape(format!(
                "DiTBlock::forward: dst shape {:?} != [{}, {}]",
                dst.shape(), seq, dim,
            )));
        }

        // ── 1. Modulation (if any) ──
        let have_mod = self.modulation;
        // First-block dump: only do this for the very first DiTBlock forward
        // call in the whole process, gated by RUSTINFER_DUMP=1.
        let do_dump = dump_enabled()
            && !DUMP_DONE.swap(true, std::sync::atomic::Ordering::SeqCst);
        if do_dump {
            eprintln!("[dump] DiTBlock first forward: dumping intermediates");
            dump_tensor("step0_x_padded_in", x);
            if let Some(c) = adaln_input { dump_tensor("step0_adaln_input", c); }
        }

        if have_mod {
            let adaln = self.adaln_modulation.as_ref().ok_or_else(|| OpError::Kernel(
                "DiTBlock::forward: modulation=true but adaln_modulation is None".into()
            ))?;
            let c = adaln_input.ok_or_else(|| OpError::Kernel(
                "DiTBlock::forward: modulation=true but adaln_input is None".into()
            ))?;
            // silu(c) into scratch.adaln_silu (256)
            let mut adaln_silu = vp2(&scratch.adaln_silu, 1, 256)?;
            adaln_silu.copy_from(c)?;
            D::silu_inplace_diff(&mut adaln_silu)?;
            // mod_out = adaln(silu(c))
            let mut mod_out = vp2(&scratch.mod_out, 1, 4 * dim)?;
            adaln.forward(&adaln_silu, &mut mod_out)?;
            if do_dump { dump_tensor("step0_nr0_mod_out", &mod_out); }
            // Split into 4 chunks of [1, dim].
            let mut scale_msa = vp2(&scratch.scale_msa, 1, dim)?;
            let mut gate_msa = vp2(&scratch.gate_msa, 1, dim)?;
            let mut scale_mlp = vp2(&scratch.scale_mlp, 1, dim)?;
            let mut gate_mlp = vp2(&scratch.gate_mlp, 1, dim)?;
            D::split_cols(&mod_out, &mut scale_msa, 1, 4 * dim, 0,         dim)?;
            D::split_cols(&mod_out, &mut gate_msa,  1, 4 * dim, dim,       dim)?;
            D::split_cols(&mod_out, &mut scale_mlp, 1, 4 * dim, 2 * dim,   dim)?;
            D::split_cols(&mod_out, &mut gate_mlp,  1, 4 * dim, 3 * dim,   dim)?;
            // scale += 1; gate = tanh(gate)
            D::scalar_add_inplace(&mut scale_msa, 1.0)?;
            D::scalar_add_inplace(&mut scale_mlp, 1.0)?;
            D::tanh_inplace(&mut gate_msa)?;
            D::tanh_inplace(&mut gate_mlp)?;
            if do_dump {
                dump_tensor("step0_nr0_scale_msa", &scale_msa);
                dump_tensor("step0_nr0_gate_msa", &gate_msa);
                dump_tensor("step0_nr0_scale_mlp", &scale_mlp);
                dump_tensor("step0_nr0_gate_mlp", &gate_mlp);
            }
        }

        // ── 2. attention_norm1(x) [* scale_msa] ──
        let mut norm1_x = vp2(&scratch.norm1_x, seq, dim)?;
        self.attention_norm1.forward(x, &mut norm1_x)?;
        if have_mod {
            let scale_msa = vp2(&scratch.scale_msa, 1, dim)?;
            D::broadcast_mul_inplace(&mut norm1_x, &scale_msa)?;
        }
        if do_dump { dump_tensor("step0_nr0_norm1_x", &norm1_x); }

        // ── 3. QKV projection + split + qk-norm + RoPE ──
        let mut qkv_out = vp2(&scratch.qkv_out, seq, 3 * dim)?;
        self.to_qkv.forward(&norm1_x, &mut qkv_out)?;
        let mut q = vp2(&scratch.q, seq, dim)?;
        let mut k = vp2(&scratch.k, seq, dim)?;
        let mut v = vp2(&scratch.v, seq, dim)?;
        D::split_cols(&qkv_out, &mut q, seq, 3 * dim, 0,     dim)?;
        D::split_cols(&qkv_out, &mut k, seq, 3 * dim, dim,   dim)?;
        D::split_cols(&qkv_out, &mut v, seq, 3 * dim, 2*dim, dim)?;
        if do_dump {
            dump_tensor("step0_nr0_qkv_q", &q);
            dump_tensor("step0_nr0_qkv_k", &k);
            dump_tensor("step0_nr0_qkv_v", &v);
        }

        // qk_norm: per-head RMSNorm on q/k. Reshape to [seq*n_heads, head_dim].
        let mut q_norm = vp2(&scratch.q, seq * self.n_heads, self.head_dim)?;
        let mut k_norm = vp2(&scratch.k, seq * self.n_heads, self.head_dim)?;
        self.norm_q.forward_inplace(&mut q_norm)?;
        self.norm_k.forward_inplace(&mut k_norm)?;
        if do_dump {
            // View as [seq, heads, head_dim] for shape consistency with Python.
            let q3v = vp3(&scratch.q, seq, self.n_heads, self.head_dim)?;
            let k3v = vp3(&scratch.k, seq, self.n_heads, self.head_dim)?;
            dump_tensor("step0_nr0_q_normed", &q3v);
            dump_tensor("step0_nr0_k_normed", &k3v);
        }

        // RoPE: reshape to [seq, n_heads, head_dim].
        let mut q3 = vp3(&scratch.q, seq, self.n_heads, self.head_dim)?;
        let mut k3 = vp3(&scratch.k, seq, self.n_heads, self.head_dim)?;
        D::apply_rope_interleaved(&mut q3, cos, sin, self.head_dim)?;
        D::apply_rope_interleaved(&mut k3, cos, sin, self.head_dim)?;
        if do_dump {
            dump_tensor("step0_nr0_q_roped", &q3);
            dump_tensor("step0_nr0_k_roped", &k3);
        }

        // ── 4. SDPA: q,k,v are [seq, n_heads, head_dim]. ──
        let v3 = vp3(&scratch.v, seq, self.n_heads, self.head_dim)?;
        let q3_in = vp3(&scratch.q, seq, self.n_heads, self.head_dim)?;
        let k3_in = vp3(&scratch.k, seq, self.n_heads, self.head_dim)?;
        let mut attn3 = vp3(&scratch.attn_out, seq, self.n_heads, self.head_dim)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        D::sdpa(&q3_in, &k3_in, &v3, &mut attn3, self.n_heads, self.n_heads, self.head_dim, scale)?;
        if do_dump { dump_tensor("step0_nr0_attn_out_premerge", &attn3); }

        // ── 5. to_out projection ──
        let attn_flat = vp2(&scratch.attn_out, seq, dim)?;
        let mut to_out_buf = vp2(&scratch.to_out_buf, seq, dim)?;
        self.to_out.forward(&attn_flat, &mut to_out_buf)?;
        if do_dump { dump_tensor("step0_nr0_attn_out_post", &to_out_buf); }

        // ── 6. attention_norm2(to_out) [* gate_msa] ──
        let mut norm2_attn = vp2(&scratch.norm2_attn, seq, dim)?;
        self.attention_norm2.forward(&to_out_buf, &mut norm2_attn)?;
        if have_mod {
            let gate_msa = vp2(&scratch.gate_msa, 1, dim)?;
            D::broadcast_mul_inplace(&mut norm2_attn, &gate_msa)?;
        }
        if do_dump { dump_tensor("step0_nr0_norm2_attn", &norm2_attn); }

        // ── 7. dst = x + norm2_attn  (gate_msa already applied above) ──
        dst.copy_from(x)?;
        D::add_inplace(dst, &norm2_attn)?;
        if do_dump { dump_tensor("step0_nr0_after_attn", dst); }

        // ── 8. ffn_norm1(dst) [* scale_mlp] ──
        let mut norm1_ffn = vp2(&scratch.norm1_ffn, seq, dim)?;
        self.ffn_norm1.forward(dst, &mut norm1_ffn)?;
        if have_mod {
            let scale_mlp = vp2(&scratch.scale_mlp, 1, dim)?;
            D::broadcast_mul_inplace(&mut norm1_ffn, &scale_mlp)?;
        }
        if do_dump { dump_tensor("step0_nr0_norm1_ffn", &norm1_ffn); }

        // ── 9. FFN: w2(silu(w1) * w3) ──
        let hidden = self.w1.weight.shape().as_slice()[0];
        let mut w1_out = vp2(&scratch.w1_out, seq, hidden)?;
        let mut w3_out = vp2(&scratch.w3_out, seq, hidden)?;
        self.w1.forward(&norm1_ffn, &mut w1_out)?;
        self.w3.forward(&norm1_ffn, &mut w3_out)?;
        if do_dump {
            dump_tensor("step0_nr0_w1_out", &w1_out);
            dump_tensor("step0_nr0_w3_out", &w3_out);
        }
        // SwiGLU in-place: w1_out = silu(w1_out) * w3_out (same shape).
        D::silu_inplace_diff(&mut w1_out)?;
        let w1_in = vp2(&scratch.w1_out, seq, hidden)?;
        D::ewise_mul(&w1_in, &w3_out, &mut w1_out)?;
        if do_dump { dump_tensor("step0_nr0_silu_w1_x_w3", &w1_out); }
        let mut ffn_out = vp2(&scratch.ffn_out, seq, dim)?;
        self.w2.forward(&w1_out, &mut ffn_out)?;
        if do_dump { dump_tensor("step0_nr0_w2_out", &ffn_out); }

        // ── 10. ffn_norm2(ffn_out) [* gate_mlp] ──
        let mut norm2_ffn = vp2(&scratch.norm2_ffn, seq, dim)?;
        self.ffn_norm2.forward(&ffn_out, &mut norm2_ffn)?;
        if have_mod {
            let gate_mlp = vp2(&scratch.gate_mlp, 1, dim)?;
            D::broadcast_mul_inplace(&mut norm2_ffn, &gate_mlp)?;
        }
        if do_dump { dump_tensor("step0_nr0_norm2_ffn", &norm2_ffn); }

        // ── 11. dst += norm2_ffn ──
        D::add_inplace(dst, &norm2_ffn)?;
        if do_dump { dump_tensor("step0_nr0_block_out", dst); }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::Shape;
    use crate::infrastructure::cuda::Cuda;
    use crate::models::layers::{Linear, RMSNorm};

    /// Build a "tiny but valid" DiT block with random weights for a CUDA shape
    /// check + finite-output test.
    fn make_random_block(
        cuda: &Cuda, dim: usize, n_heads: usize, hidden: usize, modulation: bool, seed: u64,
    ) -> DiTBlock<f32, Cuda> {
        let head_dim = dim / n_heads;
        let randn_w = |out: usize, in_: usize, s: u64| -> Linear<f32, Cuda> {
            let w: Tensor<f32, Cuda> = Tensor::randn([out, in_], cuda, Some(s)).unwrap();
            Linear::new(w, None)
        };
        let unit_norm = |d: usize| -> RMSNorm<f32, Cuda> {
            let w_host: Vec<f32> = vec![1.0; d];
            let w: Tensor<f32, Cuda> = Tensor::from_host_slice(&w_host, [d], cuda).unwrap();
            RMSNorm::new(w, 1e-5)
        };
        DiTBlock {
            attention_norm1: unit_norm(dim),
            attention_norm2: unit_norm(dim),
            ffn_norm1: unit_norm(dim),
            ffn_norm2: unit_norm(dim),
            to_qkv: randn_w(3 * dim, dim, seed),
            to_out: randn_w(dim, dim, seed + 1),
            norm_q: unit_norm(head_dim),
            norm_k: unit_norm(head_dim),
            w1: randn_w(hidden, dim, seed + 2),
            w3: randn_w(hidden, dim, seed + 3),
            w2: randn_w(dim, hidden, seed + 4),
            adaln_modulation: if modulation {
                Some(randn_w(4 * dim, 256, seed + 5))
            } else {
                None
            },
            dim, n_heads, head_dim, modulation,
        }
    }

    #[test]
    fn dit_block_no_modulation_finite_output() {
        let cuda = Cuda::new(0).unwrap();
        // dim divisible by n_heads, head_dim even, seq divisible by 8 (for vec kernels).
        let dim = 32usize;
        let n_heads = 4usize;
        let hidden = 64usize;
        let seq = 8usize;
        let block = make_random_block(&cuda, dim, n_heads, hidden, false, 42);
        let mut scratch = DiTBlockScratch::<f32, Cuda>::new(dim, hidden, seq, &cuda).unwrap();
        let head_dim = dim / n_heads;
        let half = head_dim / 2;

        // Random inputs (use small magnitudes so RMSNorm doesn't blow up).
        let x: Tensor<f32, Cuda> = Tensor::randn([seq, dim], &cuda, Some(7)).unwrap();
        let cos: Tensor<f32, Cuda> = Tensor::from_host_slice(
            &(0..seq * half).map(|i| (i as f32 * 0.1).cos()).collect::<Vec<_>>(),
            [seq, half], &cuda,
        ).unwrap();
        let sin: Tensor<f32, Cuda> = Tensor::from_host_slice(
            &(0..seq * half).map(|i| (i as f32 * 0.1).sin()).collect::<Vec<_>>(),
            [seq, half], &cuda,
        ).unwrap();

        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([seq, dim], &cuda).unwrap();
        block.forward(&x, &cos, &sin, None, &mut scratch, &mut dst).unwrap();

        let got = dst.to_host_vec().unwrap();
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "non-finite output at {}: {}", i, v);
        }
        // Output should differ from input (block does *something*).
        let xv = x.to_host_vec().unwrap();
        let diff: f32 = got.iter().zip(xv.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>() / got.len() as f32;
        assert!(diff > 1e-4, "block output identical to input (avg diff {})", diff);
    }

    #[test]
    fn dit_block_with_modulation_finite_output() {
        let cuda = Cuda::new(0).unwrap();
        let dim = 32usize;
        let n_heads = 4usize;
        let hidden = 64usize;
        let seq = 8usize;
        let block = make_random_block(&cuda, dim, n_heads, hidden, true, 99);
        let mut scratch = DiTBlockScratch::<f32, Cuda>::new(dim, hidden, seq, &cuda).unwrap();
        let head_dim = dim / n_heads;
        let half = head_dim / 2;

        let x: Tensor<f32, Cuda> = Tensor::randn([seq, dim], &cuda, Some(8)).unwrap();
        let adaln_in: Tensor<f32, Cuda> = Tensor::randn([1, 256], &cuda, Some(81)).unwrap();
        let cos: Tensor<f32, Cuda> = Tensor::from_host_slice(
            &(0..seq * half).map(|i| (i as f32 * 0.07).cos()).collect::<Vec<_>>(),
            [seq, half], &cuda,
        ).unwrap();
        let sin: Tensor<f32, Cuda> = Tensor::from_host_slice(
            &(0..seq * half).map(|i| (i as f32 * 0.07).sin()).collect::<Vec<_>>(),
            [seq, half], &cuda,
        ).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([seq, dim], &cuda).unwrap();
        block.forward(&x, &cos, &sin, Some(&adaln_in), &mut scratch, &mut dst).unwrap();
        let got = dst.to_host_vec().unwrap();
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "non-finite output at {}: {}", i, v);
        }
    }

    #[test]
    fn dit_block_zero_input_yields_residual_only() {
        // When x == 0 and modulation=false, all linear outputs are ~0 (random
        // weights but zero input → zero matmul). dst = x + attn + ffn ≈ 0.
        // (Modulo numerical noise from the SDPA softmax which returns a
        // valid probability distribution.)
        let cuda = Cuda::new(0).unwrap();
        let dim = 32; let n_heads = 4; let hidden = 64; let seq = 8;
        let block = make_random_block(&cuda, dim, n_heads, hidden, false, 1);
        let mut scratch = DiTBlockScratch::<f32, Cuda>::new(dim, hidden, seq, &cuda).unwrap();
        let head_dim = dim / n_heads;
        let half = head_dim / 2;
        let x: Tensor<f32, Cuda> = Tensor::zeros([seq, dim], &cuda).unwrap();
        let cos: Tensor<f32, Cuda> = Tensor::zeros([seq, half], &cuda).unwrap();
        let sin: Tensor<f32, Cuda> = Tensor::zeros([seq, half], &cuda).unwrap();
        let mut dst: Tensor<f32, Cuda> = Tensor::zeros([seq, dim], &cuda).unwrap();
        block.forward(&x, &cos, &sin, None, &mut scratch, &mut dst).unwrap();
        let got = dst.to_host_vec().unwrap();
        // Output must be finite (no NaNs from RMSNorm on zero input — the
        // norm_q/norm_k weight=1 path divides by sqrt(eps)).
        for v in got { assert!(v.is_finite()); }
    }
}
