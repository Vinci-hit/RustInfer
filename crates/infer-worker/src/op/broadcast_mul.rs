use crate::base::DeviceType;
use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

use super::kernels;

/// 广播逐元素乘法: dst[i, j] = a[i, j] * b[j]
/// a: [rows, D], b: [D], dst: [rows, D]
///
/// CUDA kernel 按 index 独立写，允许 `dst` 与 `a` 指向同一 Tensor；若需要
/// 就地（`a *= b`），请用 [`broadcast_mul_inplace`] 以绕过 Rust 借用检查。
pub fn broadcast_mul(a: &Tensor, b: &Tensor, dst: &mut Tensor) -> Result<()> {
    match a.device() {
        DeviceType::Cpu => kernels::cpu::broadcast_mul(a, b, dst),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let d = *a.shape().last().unwrap() as i32;
            let rows = (a.num_elements() / d as usize) as i32;
            kernels::cuda::broadcast_mul(a, b, dst, rows, d, crate::cuda::get_current_cuda_stream())
        }
    }
}

/// 原地广播逐元素乘法: a[i, j] *= b[j]
///
/// 等价于 `broadcast_mul(&a, &b, &mut a)`，但绕过 Rust 借用检查并省去一块
/// 中间 buffer。CUDA kernel 天然支持 `dst == src`。
pub fn broadcast_mul_inplace(a: &mut Tensor, b: &Tensor) -> Result<()> {
    match a.device() {
        DeviceType::Cpu => broadcast_mul_inplace_cpu(a, b),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => broadcast_mul_inplace_cuda(a, b),
    }
}

#[cfg(feature = "cuda")]
fn broadcast_mul_inplace_cuda(a: &mut Tensor, b: &Tensor) -> Result<()> {
    use crate::cuda::ffi::cudaStream_t;
    let d = *a.shape().last().unwrap() as i32;
    let rows = (a.num_elements() / d as usize) as i32;
    let stream: cudaStream_t = crate::cuda::get_current_cuda_stream();

    unsafe extern "C" {
        fn broadcast_mul_f32_forward(dst: *mut f32, a: *const f32, b: *const f32,
            rows: i32, d: i32, stream: cudaStream_t);
        fn broadcast_mul_bf16_forward(dst: *mut half::bf16, a: *const half::bf16, b: *const half::bf16,
            rows: i32, d: i32, stream: cudaStream_t);
        fn broadcast_mul_f16_forward(dst: *mut half::f16, a: *const half::f16, b: *const half::f16,
            rows: i32, d: i32, stream: cudaStream_t);
    }

    match a.dtype() {
        crate::base::DataType::F32 => {
            let ap = a.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            let bp = b.as_f32()?.buffer().as_ptr() as *const f32;
            unsafe { broadcast_mul_f32_forward(ap, ap as *const f32, bp, rows, d, stream); }
        }
        crate::base::DataType::BF16 => {
            let ap = a.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let bp = b.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            unsafe { broadcast_mul_bf16_forward(ap, ap as *const half::bf16, bp, rows, d, stream); }
        }
        crate::base::DataType::F16 => {
            let ap = a.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            let bp = b.as_f16()?.buffer().as_ptr() as *const half::f16;
            unsafe { broadcast_mul_f16_forward(ap, ap as *const half::f16, bp, rows, d, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "broadcast_mul_inplace CUDA: unsupported dtype {:?}", other)).into()),
    }
    Ok(())
}

fn broadcast_mul_inplace_cpu(a: &mut Tensor, b: &Tensor) -> Result<()> {
    let last = *a.shape().last().unwrap();
    let rows = a.num_elements() / last;
    match (a, b) {
        (Tensor::F32(a_t), Tensor::F32(b_t)) => {
            let bv: Vec<f32> = b_t.as_slice()?.to_vec();
            let as_slice = a_t.as_slice_mut()?;
            for r in 0..rows {
                let base = r * last;
                for j in 0..last { as_slice[base + j] *= bv[j]; }
            }
        }
        (Tensor::BF16(a_t), Tensor::BF16(b_t)) => {
            let bv: Vec<half::bf16> = b_t.as_slice()?.to_vec();
            let as_slice = a_t.as_slice_mut()?;
            for r in 0..rows {
                let base = r * last;
                for j in 0..last {
                    let v = as_slice[base + j].to_f32() * bv[j].to_f32();
                    as_slice[base + j] = half::bf16::from_f32(v);
                }
            }
        }
        (Tensor::F16(a_t), Tensor::F16(b_t)) => {
            let bv: Vec<half::f16> = b_t.as_slice()?.to_vec();
            let as_slice = a_t.as_slice_mut()?;
            for r in 0..rows {
                let base = r * last;
                for j in 0..last {
                    let v = as_slice[base + j].to_f32() * bv[j].to_f32();
                    as_slice[base + j] = half::f16::from_f32(v);
                }
            }
        }
        (a, _) => return Err(Error::InvalidArgument(format!(
            "broadcast_mul_inplace CPU: unsupported dtype {:?}", a.dtype())).into()),
    }
    Ok(())
}

// ───────────────────────────── Tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::DataType;
    use half::bf16;

    fn rows_cols() -> (usize, usize) { (7, 16) }

    fn fill_f32(t: &mut Tensor, f: impl Fn(usize) -> f32) {
        let s = t.as_f32_mut().unwrap().as_slice_mut().unwrap();
        for i in 0..s.len() { s[i] = f(i); }
    }

    // `broadcast_mul_inplace(a, b)` must produce the same bytes as
    // `broadcast_mul(a_copy, b, a)` modulo the absence of a separate
    // destination — verified against a from-scratch clone of `a`.

    #[test]
    fn broadcast_mul_inplace_matches_nonalias_cpu_f32() -> Result<()> {
        let (r, d) = rows_cols();
        let mut a_src = Tensor::new(&[r, d], DataType::F32, DeviceType::Cpu)?;
        fill_f32(&mut a_src, |i| (i as f32) * 0.1 - 0.5);
        let mut b = Tensor::new(&[d], DataType::F32, DeviceType::Cpu)?;
        fill_f32(&mut b, |i| (i as f32 + 1.0) * 0.25);

        // baseline via non-inplace kernel
        let mut baseline = Tensor::new(&[r, d], DataType::F32, DeviceType::Cpu)?;
        broadcast_mul(&a_src, &b, &mut baseline)?;

        // under test: same input, inplace
        let mut a = Tensor::new(&[r, d], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(a_src.as_f32()?.as_slice()?);
        broadcast_mul_inplace(&mut a, &b)?;

        assert_eq!(baseline.as_f32()?.as_slice()?, a.as_f32()?.as_slice()?);
        Ok(())
    }

    #[test]
    fn broadcast_mul_inplace_matches_nonalias_cpu_bf16() -> Result<()> {
        let (r, d) = rows_cols();
        let mut a_src = Tensor::new(&[r, d], DataType::BF16, DeviceType::Cpu)?;
        for (i, v) in a_src.as_bf16_mut()?.as_slice_mut()?.iter_mut().enumerate() {
            *v = bf16::from_f32((i as f32) * 0.1 - 0.5);
        }
        let mut b = Tensor::new(&[d], DataType::BF16, DeviceType::Cpu)?;
        for (i, v) in b.as_bf16_mut()?.as_slice_mut()?.iter_mut().enumerate() {
            *v = bf16::from_f32((i as f32 + 1.0) * 0.25);
        }

        let mut baseline = Tensor::new(&[r, d], DataType::BF16, DeviceType::Cpu)?;
        broadcast_mul(&a_src, &b, &mut baseline)?;

        let mut a = Tensor::new(&[r, d], DataType::BF16, DeviceType::Cpu)?;
        a.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(a_src.as_bf16()?.as_slice()?);
        broadcast_mul_inplace(&mut a, &b)?;

        let bl: Vec<f32> = baseline.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect();
        let ac: Vec<f32> = a.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect();
        assert_eq!(bl, ac);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn broadcast_mul_inplace_matches_nonalias_cuda_f32() -> Result<()> {
        let (r, d) = rows_cols();
        let mut a_cpu = Tensor::new(&[r, d], DataType::F32, DeviceType::Cpu)?;
        fill_f32(&mut a_cpu, |i| (i as f32) * 0.01 + 0.2);
        let mut b_cpu = Tensor::new(&[d], DataType::F32, DeviceType::Cpu)?;
        fill_f32(&mut b_cpu, |i| (i as f32 + 1.0) * 0.5);

        // baseline on CUDA
        let a_src_gpu = a_cpu.to_cuda(0)?;
        let b_gpu = b_cpu.to_cuda(0)?;
        let mut baseline = Tensor::new(&[r, d], DataType::F32, DeviceType::Cuda(0))?;
        broadcast_mul(&a_src_gpu, &b_gpu, &mut baseline)?;
        let b_cpu_res = baseline.to_cpu()?;

        let mut a_gpu = a_cpu.to_cuda(0)?;
        broadcast_mul_inplace(&mut a_gpu, &b_gpu)?;
        let a_cpu_res = a_gpu.to_cpu()?;

        assert_eq!(b_cpu_res.as_f32()?.as_slice()?, a_cpu_res.as_f32()?.as_slice()?);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn broadcast_mul_inplace_matches_nonalias_cuda_bf16() -> Result<()> {
        let (r, d) = (4, 32);
        let mut a_cpu = Tensor::new(&[r, d], DataType::BF16, DeviceType::Cpu)?;
        for (i, v) in a_cpu.as_bf16_mut()?.as_slice_mut()?.iter_mut().enumerate() {
            *v = bf16::from_f32((i as f32) * 0.01 - 0.3);
        }
        let mut b_cpu = Tensor::new(&[d], DataType::BF16, DeviceType::Cpu)?;
        for (i, v) in b_cpu.as_bf16_mut()?.as_slice_mut()?.iter_mut().enumerate() {
            *v = bf16::from_f32((i as f32 + 1.0) * 0.1);
        }

        let a_src_gpu = a_cpu.to_cuda(0)?;
        let b_gpu = b_cpu.to_cuda(0)?;
        let mut baseline = Tensor::new(&[r, d], DataType::BF16, DeviceType::Cuda(0))?;
        broadcast_mul(&a_src_gpu, &b_gpu, &mut baseline)?;
        let b_cpu_res = baseline.to_cpu()?;

        let mut a_gpu = a_cpu.to_cuda(0)?;
        broadcast_mul_inplace(&mut a_gpu, &b_gpu)?;
        let a_cpu_res = a_gpu.to_cpu()?;

        let bv: Vec<f32> = b_cpu_res.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect();
        let av: Vec<f32> = a_cpu_res.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect();
        assert_eq!(bv, av);
        Ok(())
    }
}
