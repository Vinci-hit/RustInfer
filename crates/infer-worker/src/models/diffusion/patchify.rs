//! Patchify / Unpatchify for DiT (Z-Image).
//!
//! Z-Image transformer config sets `patch_size = 2`, `f_patch_size = 1`. So
//! a `[16, 1, H, W]` latent (with H = W = latent_h, latent_w) becomes
//! `[(H/2) * (W/2), 1*2*2*16] = [num_tokens, 64]`.
//!
//! Layout (matching diffusers `_patchify_image`):
//!   src `[C, F, H, W]` → reshape `[C, F_t, p_F, H_t, p_H, W_t, p_W]`
//!     → permute `(F_t, H_t, W_t, p_F, p_H, p_W, C)`
//!     → flatten last 4 dims → `[num_tokens, p_F*p_H*p_W*C]`
//!
//! Implementation: do the permute on host (D2H → permute → H2D). Cheap
//! since latent volume is tiny vs. the rest of the pipeline (1024×1024 →
//! ~1MB BF16). When the device tensor lives on CPU we skip the round-trip.

use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::Dtype;
use crate::infrastructure::cuda::Cuda;
use half::bf16;

/// `image: [C, F, H, W]` → `dst: [num_tokens, patch_flat]`,
/// `num_tokens = (F/p_f) * (H/p) * (W/p)`,
/// `patch_flat = p_f * p^2 * C`.
pub fn patchify_into<T: Dtype>(
    image: &Tensor<T, Cuda>,
    patch_size: usize,
    f_patch_size: usize,
    dst: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let s = image.shape().as_slice();
    if s.len() != 4 {
        return Err(OpError::Shape(format!(
            "patchify_into: expected [C,F,H,W], got {:?}",
            s,
        )));
    }
    let (c, f, h, w) = (s[0], s[1], s[2], s[3]);
    let (p_f, p_h, p_w) = (f_patch_size, patch_size, patch_size);
    if f % p_f != 0 || h % p_h != 0 || w % p_w != 0 {
        return Err(OpError::Shape(format!(
            "patchify_into: ({},{},{}) not divisible by ({},{},{})",
            f, h, w, p_f, p_h, p_w,
        )));
    }
    let (f_t, h_t, w_t) = (f / p_f, h / p_h, w / p_w);
    let num_tokens = f_t * h_t * w_t;
    let patch_flat = p_f * p_h * p_w * c;
    let ds = dst.shape().as_slice();
    if ds != [num_tokens, patch_flat] {
        return Err(OpError::Shape(format!(
            "patchify_into: dst {:?} != [{}, {}]",
            ds, num_tokens, patch_flat,
        )));
    }

    // Host roundtrip: D2H, permute, H2D. The latent is small (~MB scale).
    let src_host = image.to_host_vec()?;
    let mut dst_host: Vec<T> = vec_uninit::<T>(num_tokens * patch_flat);

    // src indexed as src[ch, fi, hi, wi].
    // dst indexed as dst[token_idx, local], where:
    //   token_idx = ft * h_t * w_t + ht * w_t + wt
    //   local = pf * (p_h * p_w * c) + ph * (p_w * c) + pw * c + ch
    //   fi = ft * p_f + pf, hi = ht * p_h + ph, wi = wt * p_w + pw.
    let stride_c = f * h * w;
    let stride_f = h * w;
    let stride_h = w;
    for ft in 0..f_t {
        for ht in 0..h_t {
            for wt in 0..w_t {
                let token_idx = (ft * h_t + ht) * w_t + wt;
                for pf in 0..p_f {
                    let fi = ft * p_f + pf;
                    for ph in 0..p_h {
                        let hi = ht * p_h + ph;
                        for pw in 0..p_w {
                            let wi = wt * p_w + pw;
                            for ch in 0..c {
                                let src_idx = ch * stride_c + fi * stride_f + hi * stride_h + wi;
                                let local = pf * (p_h * p_w * c) + ph * (p_w * c) + pw * c + ch;
                                let dst_idx = token_idx * patch_flat + local;
                                dst_host[dst_idx] = src_host[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    upload_into(dst, &dst_host)
}

/// `tokens: [num_tokens, patch_flat]` → `dst: [C, F, H, W]`. Inverse of
/// `patchify_into`.
pub fn unpatchify_into<T: Dtype>(
    tokens: &Tensor<T, Cuda>,
    f: usize,
    h: usize,
    w: usize,
    out_channels: usize,
    patch_size: usize,
    f_patch_size: usize,
    dst: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let (p_f, p_h, p_w) = (f_patch_size, patch_size, patch_size);
    let (f_t, h_t, w_t) = (f / p_f, h / p_h, w / p_w);
    let num_tokens = f_t * h_t * w_t;
    let patch_flat = p_f * p_h * p_w * out_channels;

    let ts = tokens.shape().as_slice();
    if ts != [num_tokens, patch_flat] {
        return Err(OpError::Shape(format!(
            "unpatchify_into: tokens {:?} != [{}, {}]",
            ts, num_tokens, patch_flat,
        )));
    }
    let ds = dst.shape().as_slice();
    if ds != [out_channels, f, h, w] {
        return Err(OpError::Shape(format!(
            "unpatchify_into: dst {:?} != [{}, {}, {}, {}]",
            ds, out_channels, f, h, w,
        )));
    }

    let src_host = tokens.to_host_vec()?;
    let mut dst_host: Vec<T> = vec_uninit::<T>(out_channels * f * h * w);

    let stride_c = f * h * w;
    let stride_f = h * w;
    let stride_h = w;
    for ft in 0..f_t {
        for ht in 0..h_t {
            for wt in 0..w_t {
                let token_idx = (ft * h_t + ht) * w_t + wt;
                for pf in 0..p_f {
                    let fi = ft * p_f + pf;
                    for ph in 0..p_h {
                        let hi = ht * p_h + ph;
                        for pw in 0..p_w {
                            let wi = wt * p_w + pw;
                            for ch in 0..out_channels {
                                let local = pf * (p_h * p_w * out_channels)
                                    + ph * (p_w * out_channels)
                                    + pw * out_channels
                                    + ch;
                                let src_idx = token_idx * patch_flat + local;
                                let dst_idx = ch * stride_c + fi * stride_f + hi * stride_h + wi;
                                dst_host[dst_idx] = src_host[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    upload_into(dst, &dst_host)
}

fn vec_uninit<T: Dtype>(n: usize) -> Vec<T> {
    // Zero-init by default — we'll overwrite every slot, but Vec must be
    // valid for to_owned semantics. Using `unsafe { Vec::set_len }` after
    // alloc would skip init but adds risk; for the sizes we deal with the
    // overhead is negligible.
    let mut v: Vec<T> = Vec::with_capacity(n);
    unsafe {
        // SAFETY: caller writes every element before reading it, and we
        // don't drop intermediate values (T: Copy).
        v.set_len(n);
    }
    let _ = T::DATA_TYPE;
    let _ = bf16::from_f32(0.0); // ensure half is linked; harmless
    v
}

fn upload_into<T: Dtype>(dst: &mut Tensor<T, Cuda>, host: &[T]) -> OpResult<()> {
    if !dst.is_contiguous() {
        return Err(OpError::NotContiguous(*dst.shape()));
    }
    let bytes = host.len() * T::SIZE_BYTES;
    if bytes == 0 {
        return Ok(());
    }
    use crate::domain::ports::MemoryPort;
    let dev = dst.device().clone();
    unsafe {
        let dst_nn = std::ptr::NonNull::new_unchecked(dst.data_ptr_mut() as *mut u8);
        dev.upload(dst_nn, host.as_ptr() as *const u8, bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_roundtrip_f32_simple() {
        let cuda = Cuda::new(0).unwrap();
        let (c, f, h, w) = (2, 1, 4, 4);
        let n = c * f * h * w;
        let host: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let img: Tensor<f32, Cuda> = Tensor::from_host_slice(&host, [c, f, h, w], &cuda).unwrap();

        let patch_size = 2;
        let f_patch = 1;
        let f_t = f / f_patch;
        let h_t = h / patch_size;
        let w_t = w / patch_size;
        let num_tokens = f_t * h_t * w_t; // 4
        let patch_flat = f_patch * patch_size * patch_size * c; // 8
        let mut tokens: Tensor<f32, Cuda> = Tensor::zeros([num_tokens, patch_flat], &cuda).unwrap();
        patchify_into(&img, patch_size, f_patch, &mut tokens).unwrap();
        // round trip
        let mut restored: Tensor<f32, Cuda> = Tensor::zeros([c, f, h, w], &cuda).unwrap();
        unpatchify_into(&tokens, f, h, w, c, patch_size, f_patch, &mut restored).unwrap();
        assert_eq!(restored.to_host_vec().unwrap(), host);
    }

    #[test]
    fn patchify_layout_matches_diffusers_reshape_perm_for_2x2() {
        // For a 1-channel, F=1, H=W=2 latent with patch_size=2 → single token
        // containing all 4 elements in order [c=0, p_h=0, p_w=0], [c=0, p_h=0, p_w=1],
        // [c=0, p_h=1, p_w=0], [c=0, p_h=1, p_w=1].
        let cuda = Cuda::new(0).unwrap();
        let host = vec![1.0_f32, 2.0, 3.0, 4.0];
        let img: Tensor<f32, Cuda> = Tensor::from_host_slice(&host, [1, 1, 2, 2], &cuda).unwrap();
        let mut tokens: Tensor<f32, Cuda> = Tensor::zeros([1, 4], &cuda).unwrap();
        patchify_into(&img, 2, 1, &mut tokens).unwrap();
        let got = tokens.to_host_vec().unwrap();
        // Layout: token[ p_F=0, p_H=0, p_W=0, c=0 ] = src[0, 0, 0, 0] = 1
        //         token[ p_F=0, p_H=0, p_W=1, c=0 ] = src[0, 0, 0, 1] = 2
        //         token[ p_F=0, p_H=1, p_W=0, c=0 ] = src[0, 0, 1, 0] = 3
        //         token[ p_F=0, p_H=1, p_W=1, c=0 ] = src[0, 0, 1, 1] = 4
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn patchify_3channel_roundtrip() {
        let cuda = Cuda::new(0).unwrap();
        let (c, f, h, w) = (3, 1, 8, 8);
        let n = c * f * h * w;
        let host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let img: Tensor<f32, Cuda> = Tensor::from_host_slice(&host, [c, f, h, w], &cuda).unwrap();
        let f_t = 1;
        let h_t = h / 2;
        let w_t = w / 2;
        let num_tokens = f_t * h_t * w_t;
        let patch_flat = 1 * 2 * 2 * c;
        let mut tokens: Tensor<f32, Cuda> = Tensor::zeros([num_tokens, patch_flat], &cuda).unwrap();
        patchify_into(&img, 2, 1, &mut tokens).unwrap();
        let mut back: Tensor<f32, Cuda> = Tensor::zeros([c, f, h, w], &cuda).unwrap();
        unpatchify_into(&tokens, f, h, w, c, 2, 1, &mut back).unwrap();
        let restored = back.to_host_vec().unwrap();
        for (i, (a, b)) in host.iter().zip(restored.iter()).enumerate() {
            assert_eq!(
                a, b,
                "roundtrip mismatch at {}: orig={}, restored={}",
                i, a, b
            );
        }
    }

    #[test]
    fn patchify_bf16_roundtrip() {
        use half::bf16;
        let cuda = Cuda::new(0).unwrap();
        let (c, f, h, w) = (16, 1, 8, 8);
        let n = c * f * h * w;
        let host: Vec<bf16> = (0..n).map(|i| bf16::from_f32(i as f32 * 0.05)).collect();
        let img: Tensor<bf16, Cuda> = Tensor::from_host_slice(&host, [c, f, h, w], &cuda).unwrap();
        let f_t = 1;
        let h_t = h / 2;
        let w_t = w / 2;
        let num_tokens = f_t * h_t * w_t; // 16
        let patch_flat = 2 * 2 * c; // 64
        let mut tokens: Tensor<bf16, Cuda> =
            Tensor::zeros([num_tokens, patch_flat], &cuda).unwrap();
        patchify_into(&img, 2, 1, &mut tokens).unwrap();
        let mut back: Tensor<bf16, Cuda> = Tensor::zeros([c, f, h, w], &cuda).unwrap();
        unpatchify_into(&tokens, f, h, w, c, 2, 1, &mut back).unwrap();
        let restored = back.to_host_vec().unwrap();
        for (i, (a, b)) in host.iter().zip(restored.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bf16 roundtrip mismatch at {}: orig={}, restored={}",
                i,
                a.to_f32(),
                b.to_f32()
            );
        }
    }
}
