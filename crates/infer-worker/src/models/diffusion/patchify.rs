//! Patchify / Unpatchify for DiT (Z-Image).
//!
//! Converts between image-space [C, F, H, W] and token-space [num_tokens, patch_flat].
//! These are CPU-side reshape+permute operations for the patch embedding.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{Dtype, Shape};
use crate::domain::tensor::Tensor;
use crate::infra::cpu::Cpu;

/// Patchify: [C, F, H, W] → [num_tokens, patch_flat] via reshape + permute.
///
/// - patch_size: spatial patch size (H and W dimensions)
/// - f_patch_size: temporal patch size (F dimension, usually 1 for images)
///
/// Layout: tokens ordered as (f_t, h_t, w_t), each token contains (p_f, p_h, p_w, C) elements.
pub fn patchify_cpu<T: Dtype>(
    image: &Tensor<T, Cpu>,
    patch_size: usize,
    f_patch_size: usize,
) -> OpResult<Tensor<T, Cpu>> {
    let shape = image.shape().as_slice();
    if shape.len() != 4 {
        return Err(OpError::Shape(format!("patchify: expected [C, F, H, W], got {:?}", shape)));
    }
    let (c, f, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let (p_f, p_h, p_w) = (f_patch_size, patch_size, patch_size);

    if f % p_f != 0 || h % p_h != 0 || w % p_w != 0 {
        return Err(OpError::Shape(format!(
            "patchify: ({f},{h},{w}) not divisible by ({p_f},{p_h},{p_w})"
        )));
    }
    let (f_t, h_t, w_t) = (f / p_f, h / p_h, w / p_w);
    let num_tokens = f_t * h_t * w_t;
    let patch_flat = p_f * p_h * p_w * c;

    // Perform the permute: src[c, f_t, p_f, h_t, p_h, w_t, p_w] → dst[f_t, h_t, w_t, p_f, p_h, p_w, c]
    let src = image.as_slice();
    let mut dst_data = vec![0u8; num_tokens * patch_flat * T::SIZE_BYTES];
    let elem = T::SIZE_BYTES;

    for ft in 0..f_t {
        for ht in 0..h_t {
            for wt in 0..w_t {
                let token_idx = ft * h_t * w_t + ht * w_t + wt;
                for pf in 0..p_f {
                    for ph in 0..p_h {
                        for pw in 0..p_w {
                            for ch in 0..c {
                                // Source index in [C, F, H, W]
                                let fi = ft * p_f + pf;
                                let hi = ht * p_h + ph;
                                let wi = wt * p_w + pw;
                                let src_idx = ch * f * h * w + fi * h * w + hi * w + wi;

                                // Dest index in [num_tokens, patch_flat]
                                // patch_flat layout: [p_f, p_h, p_w, c]
                                let dst_local = pf * p_h * p_w * c + ph * p_w * c + pw * c + ch;
                                let dst_idx = token_idx * patch_flat + dst_local;

                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        (src.as_ptr() as *const u8).add(src_idx * elem),
                                        dst_data.as_mut_ptr().add(dst_idx * elem),
                                        elem,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Build output tensor (CPU host allocation via Tensor::from_host_bytes).
    let out_shape = Shape::from_slice(&[num_tokens, patch_flat]);
    Tensor::<T, Cpu>::from_host_bytes(&dst_data, out_shape, &Cpu)
}

/// Unpatchify: [num_tokens, patch_flat] → [C, F, H, W] (inverse of patchify).
pub fn unpatchify_cpu<T: Dtype>(
    tokens: &Tensor<T, Cpu>,
    c: usize, f: usize, h: usize, w: usize,
    patch_size: usize,
    f_patch_size: usize,
) -> OpResult<Tensor<T, Cpu>> {
    let (p_f, p_h, p_w) = (f_patch_size, patch_size, patch_size);
    let (f_t, h_t, w_t) = (f / p_f, h / p_h, w / p_w);
    let patch_flat = p_f * p_h * p_w * c;
    let numel = c * f * h * w;
    let elem = T::SIZE_BYTES;

    let src = tokens.as_slice();
    let mut dst_data = vec![0u8; numel * elem];

    for ft in 0..f_t {
        for ht in 0..h_t {
            for wt in 0..w_t {
                let token_idx = ft * h_t * w_t + ht * w_t + wt;
                for pf in 0..p_f {
                    for ph in 0..p_h {
                        for pw in 0..p_w {
                            for ch in 0..c {
                                let fi = ft * p_f + pf;
                                let hi = ht * p_h + ph;
                                let wi = wt * p_w + pw;
                                let dst_idx = ch * f * h * w + fi * h * w + hi * w + wi;
                                let src_local = pf * p_h * p_w * c + ph * p_w * c + pw * c + ch;
                                let src_idx = token_idx * patch_flat + src_local;

                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        (src.as_ptr() as *const u8).add(src_idx * elem),
                                        dst_data.as_mut_ptr().add(dst_idx * elem),
                                        elem,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let out_shape = Shape::from_slice(&[c, f, h, w]);
    Tensor::<T, Cpu>::from_host_bytes(&dst_data, out_shape, &Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_patchify_unpatchify() {
        let (c, f, h, w) = (2, 1, 4, 4);
        let numel = c * f * h * w;
        let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
        let image = Tensor::<f32, Cpu>::from_slice(&data, Shape::from_slice(&[c, f, h, w]));

        let tokens = patchify_cpu(&image, 2, 1).unwrap();
        assert_eq!(tokens.shape().as_slice(), &[4, 8]); // 4 tokens, each 1*2*2*2=8

        let restored = unpatchify_cpu(&tokens, c, f, h, w, 2, 1).unwrap();
        assert_eq!(restored.shape().as_slice(), &[c, f, h, w]);

        let orig = image.as_slice();
        let rest = restored.as_slice();
        for i in 0..numel {
            assert_eq!(orig[i], rest[i], "mismatch at {i}");
        }
    }
}
