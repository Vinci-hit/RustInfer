//! Materialisation: turn a possibly-strided view into contiguous storage.
//!
//! [`Tensor::contiguous`] is the universal escape hatch that every kernel
//! can rely on. For CPU tensors we run a compact N-D stride walker; for
//! CUDA we reuse the existing `permute_*_forward` kernels with the
//! identity permutation — that kernel already accepts arbitrary
//! `old_strides`, which is precisely what we need.
//!
//! [`Tensor::permute_into`] is preserved as the write-into-preallocated-dst
//! entry point used by KV-cache movement and patchify. It now accepts any
//! source (contiguous or strided) because it, too, routes through the
//! same strided-copy kernels.

use half::{bf16, f16};

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};

use super::dims::Dims;
use super::tensor::Tensor;

impl Tensor {
    /// Return a contiguous copy. If `self` already owns its storage tightly
    /// (contiguous, zero offset, and the backing buffer is exactly
    /// `numel * sizeof(dtype)`) this is a cheap metadata clone (the
    /// underlying buffer `Arc` is shared).
    ///
    /// Note that a *prefix-narrowed* view (e.g. `base.narrow(0, 0, n)` with
    /// `n < base.shape[0]`) is contiguous with `storage_offset == 0` yet
    /// shares a buffer larger than `numel`; for that case we deliberately
    /// fall through to the strided-copy path so the returned tensor's
    /// `buffer.len_bytes()` matches its logical size.
    pub fn contiguous(&self) -> Result<Self> {
        if self.owns_storage_tightly() {
            return Ok(self.clone());
        }
        // Either non-contiguous, nonzero-offset, or sharing an oversized
        // buffer with a parent. Materialise via the identity-permute
        // strided-copy path, which allocates a fresh buffer sized to
        // `shape` and walks `self.strides()` to gather the right elements.
        let ndim = self.ndim();
        let identity: Vec<usize> = (0..ndim).collect();
        let mut dst = Self::empty(self.shape(), self.dtype(), self.device())?;
        self.permute_into(&identity, &mut dst)?;
        Ok(dst)
    }

    /// Deep copy into a freshly allocated, contiguous, exclusive buffer.
    /// Unlike [`Tensor::contiguous`] this always allocates, even when the
    /// source already owns its storage tightly — callers asking for
    /// `to_owned` want independent storage.
    pub fn to_owned(&self) -> Result<Self> {
        if self.owns_storage_tightly() {
            let mut dst = Self::empty(self.shape(), self.dtype(), self.device())?;
            dst.buffer_mut().copy_from(self.buffer())?;
            return Ok(dst);
        }
        // Prefix-narrowed / strided / offset views: `contiguous()` already
        // allocates a fresh buffer of the right size and gathers via
        // strides, which gives us independent storage too.
        self.contiguous()
    }

    /// `dst = self.permute(perm)` materialised into a caller-provided
    /// destination. `dst` must already have the correct shape, dtype and
    /// device. `self` may be contiguous *or* strided.
    pub fn permute_into(&self, perm: &[usize], dst: &mut Tensor) -> Result<()> {
        let ndim = self.ndim();
        if perm.len() != ndim {
            return Err(Error::InvalidArgument(format!(
                "permute_into: perm length {} != ndim {}", perm.len(), ndim
            )).into());
        }
        let mut seen = [false; super::dims::MAX_RANK];
        for &p in perm {
            if p >= ndim || seen[p] {
                return Err(Error::InvalidArgument(format!(
                    "permute_into: invalid permutation {:?}", perm
                )).into());
            }
            seen[p] = true;
        }
        let expected_shape: Dims = {
            let mut d = Dims::new();
            for &p in perm { d.push(self.shape()[p]); }
            d
        };
        if dst.shape() != expected_shape.as_slice() {
            return Err(Error::InvalidArgument(format!(
                "permute_into: dst shape {:?} does not match permuted shape {:?}",
                dst.shape(), expected_shape.as_slice()
            )).into());
        }
        if dst.dtype() != self.dtype() {
            return Err(Error::InvalidArgument(format!(
                "permute_into: dtype mismatch src={:?} dst={:?}",
                self.dtype(), dst.dtype()
            )).into());
        }
        if dst.device() != self.device() {
            return Err(Error::InvalidArgument(format!(
                "permute_into: device mismatch src={:?} dst={:?}",
                self.device(), dst.device()
            )).into());
        }
        if !dst.is_contiguous() || dst.storage_offset() != 0 {
            return Err(Error::InvalidArgument(
                "permute_into: dst must be a contiguous tensor with zero offset".into()
            ).into());
        }

        #[cfg(feature = "cuda")]
        if self.device() != DeviceType::Cpu {
            return permute_into_cuda(self, perm, dst);
        }
        permute_into_cpu(self, perm, dst)
    }
}

// ─────────────────────────── CPU implementation ─────────────────────────

fn permute_into_cpu(src: &Tensor, perm: &[usize], dst: &mut Tensor) -> Result<()> {
    let ndim = src.ndim();
    let src_shape   = src.shape();
    let src_strides = src.strides();
    let new_shape: Vec<usize>   = perm.iter().map(|&i| src_shape[i]).collect();
    let new_strides: Vec<usize> = {
        let mut s = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * new_shape[i + 1];
        }
        s
    };
    let n = src.numel();

    macro_rules! gather {
        ($ty:ty, $src_t:expr, $dst_t:expr) => {{
            // Iterate the dst in contiguous order and compute the matching
            // src offset via perm-mapped multi-index.
            let src_ptr = $src_t.data_ptr();
            let dst_ptr = $dst_t.data_ptr_mut();
            for flat_new in 0..n {
                let mut rem = flat_new;
                let mut old_flat = 0usize;
                for j in 0..ndim {
                    let coord = rem / new_strides[j];
                    rem %= new_strides[j];
                    old_flat += coord * src_strides[perm[j]];
                }
                unsafe { *dst_ptr.add(flat_new) = *src_ptr.add(old_flat); }
            }
        }};
    }

    match (src, dst) {
        (Tensor::F32(s),  Tensor::F32(d))  => gather!(f32 , s, d),
        (Tensor::BF16(s), Tensor::BF16(d)) => gather!(bf16, s, d),
        (Tensor::F16(s),  Tensor::F16(d))  => gather!(f16 , s, d),
        (Tensor::I32(s),  Tensor::I32(d))  => gather!(i32 , s, d),
        (Tensor::I8(s),   Tensor::I8(d))   => gather!(i8  , s, d),
        _ => return Err(Error::InvalidArgument(
            "permute_into_cpu: dtype pair mismatch".into()
        ).into()),
    }
    Ok(())
}

// ────────────────────────── CUDA implementation ─────────────────────────

#[cfg(feature = "cuda")]
fn permute_into_cuda(src: &Tensor, perm: &[usize], dst: &mut Tensor) -> Result<()> {
    use crate::cuda::ffi::cudaStream_t;

    unsafe extern "C" {
        fn permute_f32_forward(dst: *mut f32, src: *const f32,
            ndim: i32, new_shape: *const i64, new_strides: *const i64,
            old_strides: *const i64, perm: *const i32,
            num_elements: i64, stream: cudaStream_t);
        fn permute_bf16_forward(dst: *mut bf16, src: *const bf16,
            ndim: i32, new_shape: *const i64, new_strides: *const i64,
            old_strides: *const i64, perm: *const i32,
            num_elements: i64, stream: cudaStream_t);
        fn permute_f16_forward(dst: *mut f16, src: *const f16,
            ndim: i32, new_shape: *const i64, new_strides: *const i64,
            old_strides: *const i64, perm: *const i32,
            num_elements: i64, stream: cudaStream_t);
        fn permute_i32_forward(dst: *mut i32, src: *const i32,
            ndim: i32, new_shape: *const i64, new_strides: *const i64,
            old_strides: *const i64, perm: *const i32,
            num_elements: i64, stream: cudaStream_t);
    }

    if !matches!(
        src.dtype(),
        DataType::F32 | DataType::BF16 | DataType::F16 | DataType::I32
    ) {
        // I8: fall back via CPU round-trip. Not hot.
        let cpu = src.to_device(DeviceType::Cpu)?;
        let permuted = {
            let mut tmp = Tensor::empty(
                &perm.iter().map(|&i| src.shape()[i]).collect::<Vec<_>>(),
                src.dtype(),
                DeviceType::Cpu,
            )?;
            permute_into_cpu(&cpu, perm, &mut tmp)?;
            tmp
        };
        let uploaded = permuted.to_device(src.device())?;
        dst.buffer_mut().copy_from(uploaded.buffer())?;
        return Ok(());
    }

    let ndim = src.ndim();
    let src_shape = src.shape();
    let old_strides: Vec<i64> = src.strides().iter().map(|&s| s as i64).collect();
    let new_shape  : Vec<i64> = perm.iter().map(|&i| src_shape[i] as i64).collect();
    let mut new_strides: Vec<i64> = vec![1; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }
    let perm_i32: Vec<i32> = perm.iter().map(|&p| p as i32).collect();
    let num_elements = src.numel() as i64;
    let stream = crate::cuda::get_current_cuda_stream();

    // The kernels expect base pointers, so pass through data_ptr() which
    // already accounts for storage_offset.
    unsafe {
        match src.dtype() {
            DataType::F32 => permute_f32_forward(
                dst.as_f32_mut()?.data_ptr_mut(),
                src.as_f32()?.data_ptr(),
                ndim as i32, new_shape.as_ptr(), new_strides.as_ptr(),
                old_strides.as_ptr(), perm_i32.as_ptr(), num_elements, stream,
            ),
            DataType::BF16 => permute_bf16_forward(
                dst.as_bf16_mut()?.data_ptr_mut(),
                src.as_bf16()?.data_ptr(),
                ndim as i32, new_shape.as_ptr(), new_strides.as_ptr(),
                old_strides.as_ptr(), perm_i32.as_ptr(), num_elements, stream,
            ),
            DataType::F16 => permute_f16_forward(
                dst.as_f16_mut()?.data_ptr_mut(),
                src.as_f16()?.data_ptr(),
                ndim as i32, new_shape.as_ptr(), new_strides.as_ptr(),
                old_strides.as_ptr(), perm_i32.as_ptr(), num_elements, stream,
            ),
            DataType::I32 => permute_i32_forward(
                dst.as_i32_mut()?.data_ptr_mut(),
                src.as_i32()?.data_ptr(),
                ndim as i32, new_shape.as_ptr(), new_strides.as_ptr(),
                old_strides.as_ptr(), perm_i32.as_ptr(), num_elements, stream,
            ),
            _ => unreachable!(),
        }
    }
    Ok(())
}
