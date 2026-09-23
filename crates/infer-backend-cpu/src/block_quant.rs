//! Pure Rust reference decoding of little-endian GGUF block weights.
//! FP32 reconstruction and accumulation; no external inference runtime.

use infer_core::dtype::quant::codebooks as tables;

use crate::Cpu;
use half::f16;
use infer_core::dtype::quant::BlockQuantFormat as F;
use infer_core::error::{OpError, OpResult};
use infer_core::quantized::{BlockQuantView, BlockQuantWeight};
use infer_core::tensor::Tensor;
use infer_core::types::{DTypeId, Dtype};

fn u16le(b: &[u8], p: usize) -> u16 {
    u16::from_le_bytes([b[p], b[p + 1]])
}
fn u32le(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}
fn fp16(b: &[u8], p: usize) -> f32 {
    f16::from_bits(u16le(b, p)).to_f32()
}
fn sign(mask: u8, bit: usize) -> f32 {
    if mask & (1 << bit) == 0 { 1.0 } else { -1.0 }
}
// Seven stored sign bits; the eighth enforces even parity.
fn signs7(bits: u8) -> u8 {
    bits | (((bits.count_ones() & 1) as u8) << 7)
}
const IQ4: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn scale_min(b: &[u8], g: usize) -> (u8, u8) {
    if g < 4 {
        (b[4 + g] & 63, b[8 + g] & 63)
    } else {
        (
            (b[8 + g] & 15) | ((b[g] >> 6) << 4),
            (b[8 + g] >> 4) | ((b[4 + g] >> 6) << 4),
        )
    }
}

/// Decode exactly one block. Length errors leave `out` untouched.
/// FP16 NaNs/infinities are propagated, not silently sanitized.
pub fn decode_block(format: F, b: &[u8], out: &mut [f32]) -> OpResult<()> {
    let layout = format.layout();
    if b.len() != layout.bytes || out.len() != layout.elements {
        return Err(OpError::Shape(format!(
            "{format:?}: expected {} bytes and {} output values",
            layout.bytes, layout.elements
        )));
    }
    for (i, y) in out.iter_mut().enumerate() {
        *y = match format {
            F::Q8_0 => fp16(b, 0) * (b[2 + i] as i8) as f32,
            F::Q2_K => {
                let s = b[i / 16];
                let q = (b[16 + (i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3;
                (fp16(b, 80) * (s & 15) as f32) * q as f32 - fp16(b, 82) * (s >> 4) as f32
            }
            F::Q3_K => {
                let g = i / 16;
                let scale = ((b[96 + g % 8] >> (4 * (g / 8))) & 15)
                    | (((b[104 + g % 4] >> (2 * (g / 4))) & 3) << 4);
                let low = (b[32 + (i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3;
                let high = (b[i % 32] >> (i / 32)) & 1;
                (fp16(b, 108) * (scale as i32 - 32) as f32)
                    * (low as i32 - 4 * (1 - high) as i32) as f32
            }
            F::Q4_K | F::Q5_K => {
                let (s, m) = scale_min(b, i / 32);
                let base = if format == F::Q5_K { 48 } else { 16 };
                let mut q = (b[base + (i / 64) * 32 + i % 32] >> (4 * ((i % 64) / 32))) & 15;
                if format == F::Q5_K {
                    q |= ((b[16 + i % 32] >> (i / 32)) & 1) << 4;
                }
                (fp16(b, 0) * s as f32) * q as f32 - fp16(b, 2) * m as f32
            }
            F::Q6_K => {
                let lo = (b[(i / 128) * 64 + i % 64] >> (4 * ((i % 128) / 64))) & 15;
                let hi = (b[128 + (i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3;
                (fp16(b, 208) * (b[192 + i / 16] as i8) as f32)
                    * ((lo | (hi << 4)) as i32 - 32) as f32
            }
            F::IQ4_NL => fp16(b, 0) * IQ4[((b[2 + i % 16] >> (4 * (i / 16))) & 15) as usize] as f32,
            F::IQ4_XS => {
                let g = i / 32;
                let s = ((b[4 + g / 2] >> (4 * (g % 2))) & 15)
                    | (((u16le(b, 2) >> (2 * g)) as u8 & 3) << 4);
                let q = (b[8 + g * 16 + i % 16] >> (4 * ((i % 32) / 16))) & 15;
                (fp16(b, 0) * (s as i32 - 32) as f32) * IQ4[q as usize] as f32
            }
            F::IQ2_XXS => {
                let g = i / 32;
                let j = (i % 32) / 8;
                let aux = u32le(b, 2 + g * 8 + 4);
                let mask = signs7(((aux >> (7 * j)) & 127) as u8);
                let grid = b[2 + g * 8 + j] as usize;
                let d = fp16(b, 0) * (0.5 + (aux >> 28) as f32) * 0.25;
                (d * tables::IQ2_XXS[grid * 8 + i % 8] as f32) * sign(mask, i % 8)
            }
            F::IQ2_XS | F::IQ2_S => {
                let g = i / 8;
                let sg = i / 16;
                let (grid, mask, scale, table) = if format == F::IQ2_XS {
                    let q = u16le(b, 2 + 2 * g);
                    (
                        (q & 511) as usize,
                        signs7((q >> 9) as u8),
                        (b[66 + sg / 2] >> (4 * (sg % 2))) & 15,
                        &tables::IQ2_XS[..],
                    )
                } else {
                    let q = b[2 + g] as usize
                        | ((((b[66 + g / 4] >> (2 * (g % 4))) & 3) as usize) << 8);
                    (
                        q,
                        b[34 + g],
                        (b[74 + sg / 2] >> (4 * (sg % 2))) & 15,
                        &tables::IQ2_S[..],
                    )
                };
                let d = fp16(b, 0) * (0.5 + scale as f32) * 0.25;
                (d * table[grid * 8 + i % 8] as f32) * sign(mask, i % 8)
            }
            F::IQ3_XXS => {
                let g = i / 32;
                let aux = u32le(b, 66 + 4 * g);
                let mask = signs7(((aux >> (7 * ((i % 32) / 8))) & 127) as u8);
                let grid = b[2 + i / 4] as usize;
                let d = fp16(b, 0) * (0.5 + (aux >> 28) as f32) * 0.5;
                (d * tables::IQ3_XXS[grid * 4 + i % 4] as f32) * sign(mask, i % 8)
            }
            F::IQ3_S => {
                let g = i / 4;
                let sg = i / 32;
                let grid = b[2 + g] as usize | ((((b[66 + g / 8] >> (g % 8)) & 1) as usize) << 8);
                let s = (b[106 + sg / 2] >> (4 * (sg % 2))) & 15;
                let d = fp16(b, 0) * (1 + 2 * s) as f32;
                (d * tables::IQ3_S[grid * 4 + i % 4] as f32) * sign(b[74 + i / 8], i % 8)
            }
        };
    }
    Ok(())
}

/// Decode one logical row, including row-sliced borrowed views.
pub fn decode_row(view: BlockQuantView<'_>, row: usize, out: &mut [f32]) -> OpResult<()> {
    let l = view.layout();
    if row >= l.shape()[0] || out.len() != l.shape()[1] {
        return Err(OpError::Shape(
            "block quant decode row/width mismatch".into(),
        ));
    }
    let b = l.format().layout();
    let range = l.row_byte_range(row..row + 1)?;
    for (src, dst) in view.bytes()[range]
        .chunks_exact(b.bytes)
        .zip(out.chunks_exact_mut(b.elements))
    {
        decode_block(l.format(), src, dst)?;
    }
    Ok(())
}

fn float_dtype<T: Dtype>() -> OpResult<()> {
    if !matches!(T::ID, DTypeId::F32 | DTypeId::F16 | DTypeId::BF16) {
        return Err(OpError::unsupported(
            "cpu",
            "block quant activation dtype (requires f32/f16/bf16)",
        ));
    }
    Ok(())
}

fn disjoint<A: Dtype, B: Dtype>(src: &Tensor<A, Cpu>, dst: &Tensor<B, Cpu>) -> OpResult<()> {
    if std::sync::Arc::ptr_eq(src.storage(), dst.storage()) {
        return Err(OpError::Shape(
            "block quant output must not share input/weight/bias storage".into(),
        ));
    }
    Ok(())
}

fn host_view(w: &BlockQuantWeight<Cpu>) -> OpResult<BlockQuantView<'_>> {
    // SAFETY: validated contiguous byte weight on the host; borrow lasts only
    // through this synchronous operation. Output storage is checked disjoint.
    let bytes = unsafe { std::slice::from_raw_parts(w.bytes().data_ptr(), w.layout().byte_len()) };
    BlockQuantView::new(*w.layout(), bytes)
}

/// Reference X[M,K] @ W[N,K]^T + bias. Only one FP32 weight row is resident.
/// Arbitrary valid tensor strides are respected. Bias precedes output rounding.
pub(crate) fn matmul<T: Dtype>(
    input: &Tensor<T, Cpu>,
    weight: &BlockQuantWeight<Cpu>,
    bias: Option<&Tensor<T, Cpu>>,
    output: &mut Tensor<T, Cpu>,
) -> OpResult<()> {
    float_dtype::<T>()?;
    let [n, k] = weight.layout().shape();
    if input.ndim() != 2
        || input.shape()[1] != k
        || output.shape().as_slice() != [input.shape()[0], n]
    {
        return Err(OpError::Shape("block quant matmul shape mismatch".into()));
    }
    disjoint(input, output)?;
    disjoint(weight.bytes(), output)?;
    if let Some(b) = bias {
        if b.shape().as_slice() != [n] {
            return Err(OpError::Shape("block quant bias shape mismatch".into()));
        }
        disjoint(b, output)?;
    }
    let view = host_view(weight)?;
    let mut row = vec![0.0f32; k];
    let xs = input.strides().as_slice();
    let ys = output.strides().as_slice();
    for j in 0..n {
        decode_row(view, j, &mut row)?;
        for i in 0..input.shape()[0] {
            let mut sum = 0.0f32;
            for (p, &w) in row.iter().enumerate() {
                // SAFETY: Tensor validates storage bounds for its shape/strides;
                // indices are in bounds and output does not alias any source.
                let x =
                    unsafe { T::read_f64(&*input.data_ptr().add(i * xs[0] + p * xs[1])) as f32 };
                sum += x * w;
            }
            if let Some(b) = bias {
                sum += unsafe {
                    T::read_f64(&*b.data_ptr().add(j * b.strides().as_slice()[0])) as f32
                };
            }
            unsafe {
                output
                    .data_ptr_mut()
                    .add(i * ys[0] + j * ys[1])
                    .write(T::write_f64(sum as f64));
            }
        }
    }
    Ok(())
}

pub(crate) fn embedding<T: Dtype>(
    weight: &BlockQuantWeight<Cpu>,
    ids: &Tensor<i32, Cpu>,
    output: &mut Tensor<T, Cpu>,
) -> OpResult<()> {
    float_dtype::<T>()?;
    let [n, k] = weight.layout().shape();
    if ids.ndim() != 1 || output.shape().as_slice() != [ids.numel(), k] {
        return Err(OpError::Shape(
            "block quant embedding shape mismatch".into(),
        ));
    }
    disjoint(ids, output)?;
    disjoint(weight.bytes(), output)?;
    // Check ALL token IDs before any output writes.
    let tokens: Vec<i32> = (0..ids.numel())
        .map(|i| {
            // SAFETY: host Tensor validates this strided, in-bounds address.
            unsafe { *ids.data_ptr().add(i * ids.strides().as_slice()[0]) }
        })
        .collect();
    if tokens.iter().any(|&id| id < 0 || id as usize >= n) {
        return Err(OpError::Shape(
            "block quant token ID outside vocabulary".into(),
        ));
    }
    let view = host_view(weight)?;
    let mut row = vec![0.0f32; k];
    let ys = output.strides().as_slice();
    for (i, &token) in tokens.iter().enumerate() {
        decode_row(view, token as usize, &mut row)?;
        for (j, &w) in row.iter().enumerate() {
            // SAFETY: valid tensor layout, in-bounds indices, disjoint storage.
            unsafe {
                output
                    .data_ptr_mut()
                    .add(i * ys[0] + j * ys[1])
                    .write(T::write_f64(w as f64));
            }
        }
    }
    Ok(())
}
