use crate::base::DeviceType;
use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use half::{bf16, f16};

use super::kernels;

/// dst[i] = src[i] * val
///
/// `src` 与 `dst` 可以是同一张 `Tensor`（同指针），kernel 按 index 独立写，
/// 无数据竞争；但 Rust 借用检查要求二者通过不同引用路径进入，所以这个
/// 签名依然是 `&Tensor + &mut Tensor`。若需就地执行（`dst == src`）请直接
/// 调用 [`scalar_mul_inplace`]。
pub fn scalar_mul(
    src: &Tensor,
    dst: &mut Tensor,
    val: f32,
) -> Result<()> {
    match src.device() {
        DeviceType::Cpu => kernels::cpu::scalar_mul(src, dst, val),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::scalar_mul(
            src, dst, val, crate::cuda::get_current_cuda_stream(),
        ),
    }
}

/// dst[i] = src[i] + val
pub fn scalar_add(
    src: &Tensor,
    dst: &mut Tensor,
    val: f32,
) -> Result<()> {
    match src.device() {
        DeviceType::Cpu => kernels::cpu::scalar_add(src, dst, val),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::scalar_add(
            src, dst, val, crate::cuda::get_current_cuda_stream(),
        ),
    }
}

// ─────────────────────── In-place variants ─────────────────────────────
//
// Motivation: the CUDA kernels above write one output per thread with no
// inter-thread dependency, so `dst == src` (same buffer, same pointer) is
// mathematically legal. We expose dedicated `*_inplace` entry points both
// to sidestep Rust's `&Tensor + &mut Tensor` aliasing rules and to give
// hot diffusion paths a single-tensor API (`x += 1.0` etc.).

/// 原地标量乘：x[i] *= val
///
/// 底层直接复用 `scalar_mul` 的 kernel（该 kernel 允许 `dst == src`）。
pub fn scalar_mul_inplace(x: &mut Tensor, val: f32) -> Result<()> {
    match x.device() {
        DeviceType::Cpu => scalar_mul_inplace_cpu(x, val),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => scalar_mul_inplace_cuda(x, val),
    }
}

/// 原地标量加：x[i] += val
pub fn scalar_add_inplace(x: &mut Tensor, val: f32) -> Result<()> {
    match x.device() {
        DeviceType::Cpu => scalar_add_inplace_cpu(x, val),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => scalar_add_inplace_cuda(x, val),
    }
}

#[cfg(feature = "cuda")]
fn scalar_mul_inplace_cuda(x: &mut Tensor, val: f32) -> Result<()> {
    use crate::cuda::ffi::cudaStream_t;
    let stream: cudaStream_t = crate::cuda::get_current_cuda_stream();
    let n = x.num_elements() as i32;
    // Grab a raw mut pointer; then run the kernel with src_ptr == dst_ptr.
    // Safety: the CUDA kernel writes dst[i] purely from src[i], with each
    // thread owning a distinct i — aliasing src and dst is race-free.
    unsafe extern "C" {
        fn scalar_mul_f32_forward(dst: *mut f32, src: *const f32, val: f32, n: i32, stream: cudaStream_t);
        fn scalar_mul_bf16_forward(dst: *mut half::bf16, src: *const half::bf16, val: f32, n: i32, stream: cudaStream_t);
        fn scalar_mul_f16_forward(dst: *mut half::f16, src: *const half::f16, val: f32, n: i32, stream: cudaStream_t);
    }
    match x.dtype() {
        crate::base::DataType::F32 => {
            let p = x.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe { scalar_mul_f32_forward(p, p as *const f32, val, n, stream); }
        }
        crate::base::DataType::BF16 => {
            let p = x.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut bf16;
            unsafe { scalar_mul_bf16_forward(p, p as *const bf16, val, n, stream); }
        }
        crate::base::DataType::F16 => {
            let p = x.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut f16;
            unsafe { scalar_mul_f16_forward(p, p as *const f16, val, n, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "scalar_mul_inplace CUDA: unsupported dtype {:?}", other)).into()),
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn scalar_add_inplace_cuda(x: &mut Tensor, val: f32) -> Result<()> {
    use crate::cuda::ffi::cudaStream_t;
    let stream: cudaStream_t = crate::cuda::get_current_cuda_stream();
    let n = x.num_elements() as i32;
    unsafe extern "C" {
        fn scalar_add_f32_forward(dst: *mut f32, src: *const f32, val: f32, n: i32, stream: cudaStream_t);
        fn scalar_add_bf16_forward(dst: *mut half::bf16, src: *const half::bf16, val: f32, n: i32, stream: cudaStream_t);
        fn scalar_add_f16_forward(dst: *mut half::f16, src: *const half::f16, val: f32, n: i32, stream: cudaStream_t);
    }
    match x.dtype() {
        crate::base::DataType::F32 => {
            let p = x.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe { scalar_add_f32_forward(p, p as *const f32, val, n, stream); }
        }
        crate::base::DataType::BF16 => {
            let p = x.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut bf16;
            unsafe { scalar_add_bf16_forward(p, p as *const bf16, val, n, stream); }
        }
        crate::base::DataType::F16 => {
            let p = x.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut f16;
            unsafe { scalar_add_f16_forward(p, p as *const f16, val, n, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "scalar_add_inplace CUDA: unsupported dtype {:?}", other)).into()),
    }
    Ok(())
}

fn scalar_mul_inplace_cpu(x: &mut Tensor, val: f32) -> Result<()> {
    match x {
        Tensor::F32(t) => t.as_slice_mut()?.iter_mut().for_each(|v| *v *= val),
        Tensor::BF16(t) => t.as_slice_mut()?.iter_mut()
            .for_each(|v| *v = bf16::from_f32(v.to_f32() * val)),
        Tensor::F16(t) => t.as_slice_mut()?.iter_mut()
            .for_each(|v| *v = f16::from_f32(v.to_f32() * val)),
        _ => return Err(Error::InvalidArgument(format!(
            "scalar_mul_inplace CPU: unsupported dtype {:?}", x.dtype())).into()),
    }
    Ok(())
}

fn scalar_add_inplace_cpu(x: &mut Tensor, val: f32) -> Result<()> {
    match x {
        Tensor::F32(t) => t.as_slice_mut()?.iter_mut().for_each(|v| *v += val),
        Tensor::BF16(t) => t.as_slice_mut()?.iter_mut()
            .for_each(|v| *v = bf16::from_f32(v.to_f32() + val)),
        Tensor::F16(t) => t.as_slice_mut()?.iter_mut()
            .for_each(|v| *v = f16::from_f32(v.to_f32() + val)),
        _ => return Err(Error::InvalidArgument(format!(
            "scalar_add_inplace CPU: unsupported dtype {:?}", x.dtype())).into()),
    }
    Ok(())
}

/// 原地 SiLU: x[i] = x[i] * sigmoid(x[i])
pub fn silu_inplace(x: &mut Tensor) -> Result<()> {
    match x.device() {
        DeviceType::Cpu => kernels::cpu::silu_inplace(x),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::silu_inplace(
            x, crate::cuda::get_current_cuda_stream(),
        ),
    }
}

/// 原地 tanh: x[i] = tanh(x[i])
pub fn tanh_inplace(x: &mut Tensor) -> Result<()> {
    match x.device() {
        DeviceType::Cpu => kernels::cpu::tanh_inplace(x),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => kernels::cuda::tanh_inplace(
            x, crate::cuda::get_current_cuda_stream(),
        ),
    }
}

// ───────────────────────────── Tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::DataType;
    use half::bf16;

    fn vals(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.25 - 3.0).collect()
    }

    fn cpu_f32(data: &[f32]) -> Tensor {
        let mut t = Tensor::new(&[data.len()], DataType::F32, DeviceType::Cpu).unwrap();
        t.as_f32_mut().unwrap().as_slice_mut().unwrap().copy_from_slice(data);
        t
    }

    // `*_inplace` must produce byte-identical results to `scalar_*(src, dst, c)`
    // when `src == dst`'s initial copy. We verify that invariant on CPU (exact
    // equality) and on CUDA (BF16 allows ≤1 LSB difference in principle but in
    // practice the same kernel produces the same bits).

    #[test]
    fn scalar_mul_inplace_matches_nonalias_cpu_f32() -> Result<()> {
        let data = vals(257);
        // non-inplace baseline
        let src = cpu_f32(&data);
        let mut baseline = Tensor::new(&[data.len()], DataType::F32, DeviceType::Cpu)?;
        scalar_mul(&src, &mut baseline, -2.5)?;
        // inplace under test
        let mut x = cpu_f32(&data);
        scalar_mul_inplace(&mut x, -2.5)?;
        assert_eq!(baseline.as_f32()?.as_slice()?, x.as_f32()?.as_slice()?);
        Ok(())
    }

    #[test]
    fn scalar_add_inplace_matches_nonalias_cpu_f32() -> Result<()> {
        let data = vals(257);
        let src = cpu_f32(&data);
        let mut baseline = Tensor::new(&[data.len()], DataType::F32, DeviceType::Cpu)?;
        scalar_add(&src, &mut baseline, 1.75)?;
        let mut x = cpu_f32(&data);
        scalar_add_inplace(&mut x, 1.75)?;
        assert_eq!(baseline.as_f32()?.as_slice()?, x.as_f32()?.as_slice()?);
        Ok(())
    }

    #[test]
    fn scalar_mul_inplace_matches_nonalias_cpu_bf16() -> Result<()> {
        let data = vals(128);
        let mut src = Tensor::new(&[data.len()], DataType::BF16, DeviceType::Cpu)?;
        let s_sl = src.as_bf16_mut()?.as_slice_mut()?;
        for (i, v) in data.iter().enumerate() { s_sl[i] = bf16::from_f32(*v); }

        let mut baseline = Tensor::new(&[data.len()], DataType::BF16, DeviceType::Cpu)?;
        scalar_mul(&src, &mut baseline, 0.75)?;

        // inplace on a clone of the original data
        let mut x = Tensor::new(&[data.len()], DataType::BF16, DeviceType::Cpu)?;
        let x_sl = x.as_bf16_mut()?.as_slice_mut()?;
        for (i, v) in data.iter().enumerate() { x_sl[i] = bf16::from_f32(*v); }
        scalar_mul_inplace(&mut x, 0.75)?;

        let a: Vec<f32> = baseline.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect();
        let b: Vec<f32> = x.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect();
        assert_eq!(a, b);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn scalar_mul_inplace_matches_nonalias_cuda_f32() -> Result<()> {
        let data = vals(1024);
        let src_cpu = cpu_f32(&data);

        // Baseline on CUDA: non-inplace
        let src_gpu = src_cpu.to_cuda(0)?;
        let mut baseline = Tensor::new(&[data.len()], DataType::F32, DeviceType::Cuda(0))?;
        scalar_mul(&src_gpu, &mut baseline, 3.125)?;
        let b_cpu = baseline.to_cpu()?;

        // Inplace on CUDA
        let mut x = src_cpu.to_cuda(0)?;
        scalar_mul_inplace(&mut x, 3.125)?;
        let x_cpu = x.to_cpu()?;

        assert_eq!(b_cpu.as_f32()?.as_slice()?, x_cpu.as_f32()?.as_slice()?);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn scalar_add_inplace_matches_nonalias_cuda_bf16() -> Result<()> {
        let data = vals(1024);
        let mut src_cpu = Tensor::new(&[data.len()], DataType::BF16, DeviceType::Cpu)?;
        let s_sl = src_cpu.as_bf16_mut()?.as_slice_mut()?;
        for (i, v) in data.iter().enumerate() { s_sl[i] = bf16::from_f32(*v); }

        let src_gpu = src_cpu.to_cuda(0)?;
        let mut baseline = Tensor::new(&[data.len()], DataType::BF16, DeviceType::Cuda(0))?;
        scalar_add(&src_gpu, &mut baseline, -1.25)?;
        let b_cpu = baseline.to_cpu()?;

        let mut x = src_cpu.to_cuda(0)?;
        scalar_add_inplace(&mut x, -1.25)?;
        let x_cpu = x.to_cpu()?;

        let a: Vec<f32> = b_cpu.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect();
        let b: Vec<f32> = x_cpu.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect();
        assert_eq!(a, b);
        Ok(())
    }
}
