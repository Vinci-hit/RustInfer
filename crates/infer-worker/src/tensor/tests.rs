//! Tensor-level tests. Kernel regression tests stay with their kernels in
//! `crate::op::*`; this module targets metadata, views, strides, copies
//! and dtype conversion.

#![cfg(test)]

use crate::base::error::Result;
use crate::base::{DataType, DeviceType};

use super::tensor::Tensor;

// ───────────────────────── helpers ─────────────────────────

fn make_f32_cpu(data: &[f32]) -> Result<Tensor> {
    let mut t = Tensor::empty(&[data.len()], DataType::F32, DeviceType::Cpu)?;
    t.as_f32_mut()?.as_slice_mut()?.copy_from_slice(data);
    Ok(t)
}

fn make_f32_cpu_2d(rows: usize, cols: usize, seed: usize) -> Result<Tensor> {
    let mut t = Tensor::empty(&[rows, cols], DataType::F32, DeviceType::Cpu)?;
    let s = t.as_f32_mut()?.as_slice_mut()?;
    for i in 0..rows * cols {
        s[i] = (i + seed) as f32;
    }
    Ok(t)
}

// ───────────────────────── shape / strides ─────────────────────────

#[test]
fn strides_default_contiguous() -> Result<()> {
    let t = Tensor::empty(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
    assert_eq!(t.shape(),   &[2, 3, 4]);
    assert_eq!(t.strides(), &[12, 4, 1]);
    assert!(t.is_contiguous());
    assert_eq!(t.storage_offset(), 0);
    Ok(())
}

#[test]
fn transpose_flips_strides() -> Result<()> {
    let t = Tensor::empty(&[2, 3], DataType::F32, DeviceType::Cpu)?;
    let tt = t.transpose(0, 1)?;
    assert_eq!(tt.shape(),   &[3, 2]);
    assert_eq!(tt.strides(), &[1, 3]);
    assert!(!tt.is_contiguous());
    Ok(())
}

#[test]
fn permute_view_zero_copy() -> Result<()> {
    let t = make_f32_cpu_2d(4, 5, 0)?;
    let p = t.permute(&[1, 0])?;
    assert_eq!(p.shape(), &[5, 4]);
    assert_eq!(p.strides(), &[1, 5]);
    assert!(!p.is_contiguous());
    Ok(())
}

// ───────────────────────── view vs reshape ─────────────────────────

#[test]
fn view_requires_contiguous() -> Result<()> {
    let t = make_f32_cpu_2d(4, 5, 0)?;
    let p = t.transpose(0, 1)?;
    assert!(p.view(&[20]).is_err(), "view on strided must fail");
    assert!(p.reshape(&[20]).is_ok(), "reshape should densify");
    Ok(())
}

#[test]
fn reshape_preserves_row_major_order() -> Result<()> {
    let t = make_f32_cpu_2d(2, 3, 0)?;
    let r = t.reshape(&[3, 2])?;
    assert_eq!(r.as_f32()?.as_slice()?, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    Ok(())
}

// ───────────────────────── narrow / select / slice ─────────────────────────

#[test]
fn narrow_is_zero_copy_and_offsets_storage() -> Result<()> {
    // t: 3×4 matrix, narrow rows 1..3 → 2×4
    let t = make_f32_cpu_2d(3, 4, 0)?;
    let n = t.narrow(0, 1, 2)?;
    assert_eq!(n.shape(), &[2, 4]);
    assert_eq!(n.strides(), &[4, 1]);
    assert_eq!(n.storage_offset(), 4);
    assert!(n.is_contiguous());
    // Materialise and verify values.
    let dense = n.contiguous()?;
    assert_eq!(dense.as_f32()?.as_slice()?, &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    Ok(())
}

#[test]
fn narrow_on_middle_dim_stays_strided() -> Result<()> {
    // [2, 3, 4] → narrow dim 1 from [1..2] → [2, 1, 4]. The dim-1 slot is
    // gone, but dim-0's stride (12) no longer aligns with the product of
    // the trailing dims (4), so the view is *not* storage-contiguous.
    let mut t = Tensor::empty(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
    let s = t.as_f32_mut()?.as_slice_mut()?;
    for i in 0..24 { s[i] = i as f32; }
    let n = t.narrow(1, 1, 1)?;
    assert_eq!(n.shape(),   &[2, 1, 4]);
    assert_eq!(n.strides(), &[12, 4, 1]);
    assert!(!n.is_contiguous(),
        "narrow on a middle dim with original stride should not be contiguous");
    // Materialise and verify the right four-wide chunks were picked.
    let dense = n.contiguous()?;
    assert_eq!(dense.as_f32()?.as_slice()?,
               &[4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0]);
    Ok(())
}

#[test]
fn slice_legacy_matches_narrow_chain() -> Result<()> {
    let t = make_f32_cpu_2d(3, 4, 0)?;
    let a = t.slice(&[1, 0], &[2, 4])?;
    let b = t.narrow(0, 1, 2)?.narrow(1, 0, 4)?;
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.storage_offset(), b.storage_offset());
    assert_eq!(a.contiguous()?.as_f32()?.as_slice()?,
               b.contiguous()?.as_f32()?.as_slice()?);
    Ok(())
}

#[test]
fn select_drops_dim() -> Result<()> {
    let t = make_f32_cpu_2d(3, 4, 0)?;
    let r = t.select(0, 2)?;
    assert_eq!(r.shape(), &[4]);
    assert_eq!(r.contiguous()?.as_f32()?.as_slice()?, &[8.0, 9.0, 10.0, 11.0]);
    Ok(())
}

// ───────────────────────── squeeze / unsqueeze / expand ─────────────────────

#[test]
fn unsqueeze_adds_axis() -> Result<()> {
    let t = Tensor::empty(&[3, 4], DataType::F32, DeviceType::Cpu)?;
    let u = t.unsqueeze(0)?;
    assert_eq!(u.shape(), &[1, 3, 4]);
    let u2 = t.unsqueeze(2)?;
    assert_eq!(u2.shape(), &[3, 4, 1]);
    Ok(())
}

#[test]
fn squeeze_drops_size_one_axis() -> Result<()> {
    let t = Tensor::empty(&[1, 5, 1], DataType::F32, DeviceType::Cpu)?;
    assert_eq!(t.squeeze(0)?.shape(), &[5, 1]);
    assert_eq!(t.squeeze_all()?.shape(), &[5]);
    assert!(t.squeeze(1).is_err());
    Ok(())
}

#[test]
fn expand_broadcasts_size_one() -> Result<()> {
    // [3, 1] → [3, 5] with stride 0 on the broadcast axis.
    let mut t = Tensor::empty(&[3, 1], DataType::F32, DeviceType::Cpu)?;
    let s = t.as_f32_mut()?.as_slice_mut()?;
    s.copy_from_slice(&[10.0, 20.0, 30.0]);
    let e = t.expand(&[3, 5])?;
    assert_eq!(e.shape(),   &[3, 5]);
    assert_eq!(e.strides(), &[1, 0]);
    // Materialise to compare.
    let dense = e.contiguous()?;
    let got = dense.as_f32()?.as_slice()?;
    let expected: Vec<f32> = [10.0f32, 20.0, 30.0]
        .iter().flat_map(|&v| std::iter::repeat(v).take(5)).collect();
    assert_eq!(got, expected.as_slice());
    Ok(())
}

// ───────────────────────── flatten / unflatten ─────────────────────────

#[test]
fn flatten_and_unflatten_roundtrip() -> Result<()> {
    let t = Tensor::empty(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
    let flat = t.flatten(0, 2)?;
    assert_eq!(flat.shape(), &[24]);
    let back = flat.unflatten(0, &[2, 3, 4])?;
    assert_eq!(back.shape(), &[2, 3, 4]);
    Ok(())
}

// ───────────────────────── split / chunk ─────────────────────────

#[test]
fn chunk_last_dim_evenly() -> Result<()> {
    let t = make_f32_cpu_2d(2, 8, 0)?;
    let cs = t.chunk(4, 1)?;
    assert_eq!(cs.len(), 4);
    for c in &cs { assert_eq!(c.shape(), &[2, 2]); }
    // first chunk, dense values: rows 0..2 × cols 0..2 → (0,1,8,9)
    let d0 = cs[0].contiguous()?;
    assert_eq!(d0.as_f32()?.as_slice()?, &[0.0, 1.0, 8.0, 9.0]);
    Ok(())
}

#[test]
fn split_sizes_must_sum_to_extent() -> Result<()> {
    let t = make_f32_cpu_2d(2, 6, 0)?;
    let cs = t.split(&[2, 4], 1)?;
    assert_eq!(cs[0].shape(), &[2, 2]);
    assert_eq!(cs[1].shape(), &[2, 4]);
    assert!(t.split(&[3, 4], 1).is_err());
    Ok(())
}

// ───────────────────────── contiguous / to_owned ─────────────────────────

#[test]
fn contiguous_materialises_transposed_view() -> Result<()> {
    let t = make_f32_cpu_2d(2, 3, 0)?;         // [0..6]
    let tt = t.transpose(0, 1)?;               // shape [3, 2], strides [1, 3]
    let c  = tt.contiguous()?;
    assert!(c.is_contiguous());
    assert_eq!(c.as_f32()?.as_slice()?, &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    Ok(())
}

#[test]
fn to_owned_always_allocates() -> Result<()> {
    let t = make_f32_cpu_2d(2, 3, 0)?;
    let o = t.to_owned()?;
    assert_eq!(o.as_f32()?.as_slice()?, t.as_f32()?.as_slice()?);
    // Distinct buffer: mutating `o` must not affect `t`.
    let mut o2 = o;
    o2 += 1.0;
    assert_ne!(o2.as_f32()?.as_slice()?, t.as_f32()?.as_slice()?);
    Ok(())
}

// ───────────────────────── as_slice / contiguity guards ─────────────────────

#[test]
fn as_slice_rejects_non_contiguous() -> Result<()> {
    let t = make_f32_cpu_2d(2, 3, 0)?;
    let tt = t.transpose(0, 1)?;
    assert!(tt.as_f32()?.as_slice().is_err());
    assert!(tt.contiguous()?.as_f32()?.as_slice().is_ok());
    Ok(())
}

// ───────────────────────── fill / zero / ones ─────────────────────────

#[test]
fn fill_and_zero_contiguous() -> Result<()> {
    let mut t = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
    t.fill_(3.0)?;
    assert_eq!(t.as_f32()?.as_slice()?, &[3.0, 3.0, 3.0, 3.0]);
    t.zero_()?;
    assert_eq!(t.as_f32()?.as_slice()?, &[0.0, 0.0, 0.0, 0.0]);

    let o = Tensor::ones(&[3], DataType::F32, DeviceType::Cpu)?;
    assert_eq!(o.as_f32()?.as_slice()?, &[1.0, 1.0, 1.0]);
    Ok(())
}

// ───────────────────────── operator overloads ─────────────────────────

#[test]
fn op_add_assign_tensor() -> Result<()> {
    let mut a = make_f32_cpu(&[1.0, 2.0, 3.0])?;
    let b     = make_f32_cpu(&[100.0, 200.0, 300.0])?;
    a += &b;
    assert_eq!(a.as_f32()?.as_slice()?, &[101.0, 202.0, 303.0]);
    Ok(())
}

#[test]
fn op_neg_and_mul_assign_scalar() -> Result<()> {
    let mut x = make_f32_cpu(&[1.0, -2.0, 3.0])?;
    x *= -1.0;
    assert_eq!(x.as_f32()?.as_slice()?, &[-1.0, 2.0, -3.0]);
    Ok(())
}

// ───────────────────────── randn statistics ─────────────────────────

#[test]
fn randn_shape_and_determinism() -> Result<()> {
    let a = Tensor::randn(&[1000], DataType::F32, DeviceType::Cpu, Some(1))?;
    let b = Tensor::randn(&[1000], DataType::F32, DeviceType::Cpu, Some(1))?;
    assert_eq!(a.as_f32()?.as_slice()?, b.as_f32()?.as_slice()?);
    Ok(())
}

#[test]
fn randn_mean_variance_close_to_01() -> Result<()> {
    let t = Tensor::randn(&[10000], DataType::F32, DeviceType::Cpu, Some(0))?;
    let d = t.as_f32()?.as_slice()?;
    let mean: f32 = d.iter().sum::<f32>() / d.len() as f32;
    let var : f32 = d.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / d.len() as f32;
    assert!(mean.abs() < 0.05);
    assert!((var - 1.0).abs() < 0.1);
    Ok(())
}

// ───────────────────────── permute_into (CPU) ─────────────────────────

#[test]
fn permute_into_cpu_matches_reference() -> Result<()> {
    let src = make_f32_cpu_2d(2, 3, 0)?;      // [0..6]
    let mut dst = Tensor::empty(&[3, 2], DataType::F32, DeviceType::Cpu)?;
    src.permute_into(&[1, 0], &mut dst)?;
    assert_eq!(dst.as_f32()?.as_slice()?, &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    Ok(())
}

// ───────────────────────── copy_from over strided src ─────────────────────

#[test]
fn copy_from_handles_strided_src() -> Result<()> {
    let a = make_f32_cpu_2d(2, 3, 0)?;        // shape [2, 3]
    let at = a.transpose(0, 1)?;              // shape [3, 2], strided
    let mut dst = Tensor::empty(&[3, 2], DataType::F32, DeviceType::Cpu)?;
    dst.copy_from(&at)?;
    assert_eq!(dst.as_f32()?.as_slice()?, &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    Ok(())
}

// ───────────────────────── CUDA smoke (feature-gated) ─────────────────────

#[cfg(feature = "cuda")]
#[test]
fn cuda_roundtrip_preserves_data() -> Result<()> {
    let cpu = make_f32_cpu(&[1.0, 2.0, 3.0, 4.0])?;
    let gpu = cpu.to_cuda(0)?;
    let back = gpu.to_cpu()?;
    assert_eq!(back.as_f32()?.as_slice()?, &[1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_contiguous_materialises_transpose() -> Result<()> {
    let cpu = make_f32_cpu_2d(4, 5, 0)?;
    let gpu = cpu.to_cuda(0)?;
    let gpu_t = gpu.transpose(0, 1)?;
    let gpu_c = gpu_t.contiguous()?;
    let cpu_c = gpu_c.to_cpu()?;
    let cpu_t = cpu.transpose(0, 1)?.contiguous()?;
    assert_eq!(cpu_c.as_f32()?.as_slice()?, cpu_t.as_f32()?.as_slice()?);
    Ok(())
}

// ───────── prefix-narrowed view materialisation regressions ──────────
//
// 一个 base 张量被 `narrow(0, 0, n)` 切成前缀视图后：
//   - `is_contiguous() == true`
//   - `storage_offset() == 0`
//   - **但** `buffer.len_bytes()` 仍是 base 整张的尺寸（> shape.numel * elem）
//
// 早期 `to_owned` / `to_cpu` / `to_cuda` 用 `is_contiguous && offset==0`
// 走快路径，会按 src buffer 长度整段拷贝 → `from_buffer` 的尺寸校验直接 panic。
// 这里固化精确判定 (`owns_storage_tightly`) 后的行为。

#[test]
fn owns_storage_tightly_distinguishes_prefix_view() -> Result<()> {
    let base = make_f32_cpu_2d(8, 4, 0)?;     // shape=[8,4]，buffer=8*4*4=128B
    assert!(base.owns_storage_tightly());

    let prefix = base.narrow(0, 0, 3)?;       // shape=[3,4]，buffer 仍 128B
    assert!(prefix.is_contiguous());
    assert_eq!(prefix.storage_offset(), 0);
    assert!(
        !prefix.owns_storage_tightly(),
        "prefix-narrowed view must NOT be considered tightly owned"
    );
    Ok(())
}

#[test]
fn to_owned_on_prefix_view_returns_tight_buffer() -> Result<()> {
    let base = make_f32_cpu_2d(8, 4, 0)?;
    let prefix = base.narrow(0, 0, 3)?;
    let owned = prefix.to_owned()?;
    // 内容前 12 个 f32 与 base 前 12 个一致
    let base_slice = base.as_f32()?.as_slice()?;
    let owned_slice = owned.as_f32()?.as_slice()?;
    assert_eq!(owned_slice, &base_slice[..12]);
    // 且 owned 的 buffer 大小 = numel * 4
    assert!(owned.owns_storage_tightly());
    assert_eq!(owned.buffer().len_bytes(), 12 * std::mem::size_of::<f32>());
    Ok(())
}

#[test]
fn contiguous_on_prefix_view_does_not_panic_and_is_tight() -> Result<()> {
    let base = make_f32_cpu_2d(8, 4, 0)?;
    let prefix = base.narrow(0, 0, 3)?;
    let c = prefix.contiguous()?;
    assert!(c.owns_storage_tightly());
    assert_eq!(c.shape(), &[3, 4]);
    let want = base.as_f32()?.as_slice()?[..12].to_vec();
    assert_eq!(c.as_f32()?.as_slice()?, want.as_slice());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn to_cpu_on_cuda_prefix_view_does_not_panic() -> Result<()> {
    let cpu = make_f32_cpu_2d(8, 4, 0)?;
    let gpu = cpu.to_cuda(0)?;
    let gpu_prefix = gpu.narrow(0, 0, 3)?;
    let back = gpu_prefix.to_cpu()?;
    let want = cpu.as_f32()?.as_slice()?[..12].to_vec();
    assert_eq!(back.as_f32()?.as_slice()?, want.as_slice());
    assert!(back.owns_storage_tightly());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn to_cuda_on_cpu_prefix_view_does_not_panic() -> Result<()> {
    let cpu = make_f32_cpu_2d(8, 4, 0)?;
    let cpu_prefix = cpu.narrow(0, 0, 3)?;
    let gpu = cpu_prefix.to_cuda(0)?;
    let back = gpu.to_cpu()?;
    let want = cpu.as_f32()?.as_slice()?[..12].to_vec();
    assert_eq!(back.as_f32()?.as_slice()?, want.as_slice());
    Ok(())
}
