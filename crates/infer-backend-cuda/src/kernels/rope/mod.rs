//! Standard half-split Rotary Position Embedding CUDA wrapper.
//!
//! `rotary_dim` may be smaller than `head_dim`: only the prefix
//! `[0, rotary_dim)` is rotated, with pairs `(i, i + rotary_dim / 2)`. The
//! remaining per-head dimensions are left byte-for-byte unchanged.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn rope_kernel_cu(
        q: *mut f32,
        k: *mut f32,
        sin_cache: *const f32,
        cos_cache: *const f32,
        positions: *const i32,
        num_tokens: i32,
        head_num: i32,
        kv_head_num: i32,
        head_dim: i32,
        rotary_dim: i32,
        q_row_stride: i64,
        k_row_stride: i64,
        stream: cudaStream_t,
    );
    fn rope_kernel_cu_bf16(
        q: *mut half::bf16,
        k: *mut half::bf16,
        sin_cache: *const half::bf16,
        cos_cache: *const half::bf16,
        positions: *const i32,
        num_tokens: i32,
        head_num: i32,
        kv_head_num: i32,
        head_dim: i32,
        rotary_dim: i32,
        q_row_stride: i64,
        k_row_stride: i64,
        stream: cudaStream_t,
    );
    fn rope_kernel_cu_fp16(
        q: *mut half::f16,
        k: *mut half::f16,
        sin_cache: *const half::f16,
        cos_cache: *const half::f16,
        positions: *const i32,
        num_tokens: i32,
        head_num: i32,
        kv_head_num: i32,
        head_dim: i32,
        rotary_dim: i32,
        q_row_stride: i64,
        k_row_stride: i64,
        stream: cudaStream_t,
    );
}

/// Element types with a partial/full RoPE CUDA kernel.
///
/// # Safety
/// All pointers and strides must describe the validated device tensors on
/// `stream`; every active position must index a valid sin/cos cache row.
pub trait RopeKernel: CudaFloat {
    #[allow(clippy::too_many_arguments)]
    unsafe fn rope(
        q: *mut Self,
        k: *mut Self,
        sin_cache: *const Self,
        cos_cache: *const Self,
        positions: *const i32,
        num_tokens: i32,
        head_num: i32,
        kv_head_num: i32,
        head_dim: i32,
        rotary_dim: i32,
        q_row_stride: i64,
        k_row_stride: i64,
        stream: cudaStream_t,
    );
}

macro_rules! impl_rope_kernel {
    ($ty:ty, $entry:ident) => {
        impl RopeKernel for $ty {
            #[inline]
            unsafe fn rope(
                q: *mut Self,
                k: *mut Self,
                sin_cache: *const Self,
                cos_cache: *const Self,
                positions: *const i32,
                num_tokens: i32,
                head_num: i32,
                kv_head_num: i32,
                head_dim: i32,
                rotary_dim: i32,
                q_row_stride: i64,
                k_row_stride: i64,
                stream: cudaStream_t,
            ) {
                unsafe {
                    $entry(
                        q,
                        k,
                        sin_cache,
                        cos_cache,
                        positions,
                        num_tokens,
                        head_num,
                        kv_head_num,
                        head_dim,
                        rotary_dim,
                        q_row_stride,
                        k_row_stride,
                        stream,
                    )
                }
            }
        }
    };
}

impl_rope_kernel!(f32, rope_kernel_cu);
impl_rope_kernel!(half::bf16, rope_kernel_cu_bf16);
impl_rope_kernel!(half::f16, rope_kernel_cu_fp16);

fn checked_i32(value: usize, name: &str) -> OpResult<i32> {
    i32::try_from(value).map_err(|_| OpError::Shape(format!("rope_inplace: {name} exceeds i32")))
}

fn checked_i64(value: usize, name: &str) -> OpResult<i64> {
    i64::try_from(value).map_err(|_| OpError::Shape(format!("rope_inplace: {name} exceeds i64")))
}

/// Apply standard half-split RoPE in-place to Q and K.
///
/// - `q`: `[num_tokens, head_num * head_dim]`
/// - `k`: `[num_tokens, kv_head_num * head_dim]`
/// - `sin/cos`: `[max_position, rotary_dim / 2]`
/// - `positions`: at least `num_tokens` active i32 entries
///
/// Q/K may be zero-copy column views with arbitrary row strides, but their
/// columns and all cache/control tensors must be packed.
#[allow(clippy::too_many_arguments)]
pub fn rope_inplace<T: RopeKernel>(
    stream: cudaStream_t,
    q: &mut Tensor<T, Cuda>,
    k: &mut Tensor<T, Cuda>,
    sin: &Tensor<T, Cuda>,
    cos: &Tensor<T, Cuda>,
    positions: &Tensor<i32, Cuda>,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> OpResult<()> {
    if head_num == 0
        || kv_head_num == 0
        || head_dim == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || rotary_dim / 2 > 1024
    {
        return Err(OpError::Shape(format!(
            "rope_inplace: invalid head_num={head_num} kv_head_num={kv_head_num} head_dim={head_dim} rotary_dim={rotary_dim}"
        )));
    }
    let q_dim = head_num
        .checked_mul(head_dim)
        .ok_or_else(|| OpError::Shape("rope_inplace: q width overflows".into()))?;
    let kv_dim = kv_head_num
        .checked_mul(head_dim)
        .ok_or_else(|| OpError::Shape("rope_inplace: k width overflows".into()))?;
    let q_shape = q.shape().as_slice();
    if q_shape.len() != 2 || q_shape[0] == 0 || q_shape[1] != q_dim {
        return Err(OpError::Shape(format!(
            "rope_inplace: q must be non-empty [num_tokens, {q_dim}], got {q_shape:?}"
        )));
    }
    let num_tokens = q_shape[0];
    if k.shape().as_slice() != [num_tokens, kv_dim] {
        return Err(OpError::Shape(format!(
            "rope_inplace: k shape {:?} != [{num_tokens}, {kv_dim}]",
            k.shape().as_slice()
        )));
    }
    if q.strides().as_slice()[1] != 1 || k.strides().as_slice()[1] != 1 {
        return Err(OpError::Shape(format!(
            "rope_inplace: q/k columns must be packed, got strides {:?}/{:?}",
            q.strides().as_slice(),
            k.strides().as_slice()
        )));
    }
    let sin_shape = sin.shape().as_slice();
    let cos_shape = cos.shape().as_slice();
    if sin_shape.len() != 2
        || sin_shape != cos_shape
        || sin_shape[0] == 0
        || sin_shape[1] != rotary_dim / 2
        || !sin.is_contiguous()
        || !cos.is_contiguous()
    {
        return Err(OpError::Shape(format!(
            "rope_inplace: sin/cos must be matching contiguous [max_position, {}], got {:?}/{:?}",
            rotary_dim / 2,
            sin_shape,
            cos_shape
        )));
    }
    if !positions.is_contiguous() || positions.numel() < num_tokens {
        return Err(OpError::Shape(format!(
            "rope_inplace: positions must be contiguous with at least {num_tokens} entries, got shape {:?}",
            positions.shape().as_slice()
        )));
    }
    let launch_blocks = num_tokens
        .checked_mul(head_num.max(kv_head_num))
        .ok_or_else(|| OpError::Shape("rope_inplace: launch grid overflows".into()))?;
    checked_i32(launch_blocks, "launch block count")?;

    let num_tokens = checked_i32(num_tokens, "num_tokens")?;
    let head_num = checked_i32(head_num, "head_num")?;
    let kv_head_num = checked_i32(kv_head_num, "kv_head_num")?;
    let head_dim = checked_i32(head_dim, "head_dim")?;
    let rotary_dim = checked_i32(rotary_dim, "rotary_dim")?;
    let q_row_stride = checked_i64(q.strides().as_slice()[0], "q row stride")?;
    let k_row_stride = checked_i64(k.strides().as_slice()[0], "k row stride")?;
    unsafe {
        T::rope(
            q.data_ptr_mut(),
            k.data_ptr_mut(),
            sin.data_ptr(),
            cos.data_ptr(),
            positions.data_ptr(),
            num_tokens,
            head_num,
            kv_head_num,
            head_dim,
            rotary_dim,
            q_row_stride,
            k_row_stride,
            stream,
        )
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaScope;
    use infer_core::ports::MathOps;

    fn rotate_reference(
        values: &mut [f32],
        sin: &[f32],
        cos: &[f32],
        positions: &[i32],
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) {
        let row_width = heads * head_dim;
        let half_rotary = rotary_dim / 2;
        for (token, &position) in positions.iter().enumerate() {
            let cache_base = position as usize * half_rotary;
            for head in 0..heads {
                let head_base = token * row_width + head * head_dim;
                for pair in 0..half_rotary {
                    let left_index = head_base + pair;
                    let right_index = left_index + half_rotary;
                    let left = values[left_index];
                    let right = values[right_index];
                    let sin_value = sin[cache_base + pair];
                    let cos_value = cos[cache_base + pair];
                    values[left_index] = left * cos_value - right * sin_value;
                    values[right_index] = left * sin_value + right * cos_value;
                }
            }
        }
    }

    #[test]
    fn f32_partial_rope_supports_strided_qk_and_preserves_tail_and_padding() {
        let cuda = Cuda::new(0).expect("cuda init");
        let scope = CudaScope::new(cuda.clone());
        let (tokens, q_heads, kv_heads, head_dim, rotary_dim) =
            (3usize, 2usize, 1usize, 8usize, 4usize);
        let (q_width, k_width) = (q_heads * head_dim, kv_heads * head_dim);
        let (q_pitch, k_pitch) = (q_width + 5, k_width + 3);
        let sentinel = -77.0f32;
        let mut q_storage = vec![sentinel; tokens * q_pitch];
        let mut k_storage = vec![sentinel; tokens * k_pitch];
        let mut q_logical = vec![0.0; tokens * q_width];
        let mut k_logical = vec![0.0; tokens * k_width];
        for token in 0..tokens {
            for col in 0..q_width {
                let value = ((token * q_width + col) as f32 * 0.17).sin() * 1.3;
                q_logical[token * q_width + col] = value;
                q_storage[token * q_pitch + 2 + col] = value;
            }
            for col in 0..k_width {
                let value = ((token * k_width + col) as f32 * 0.23).cos() * 0.9;
                k_logical[token * k_width + col] = value;
                k_storage[token * k_pitch + 1 + col] = value;
            }
        }
        let positions = vec![2, 0, 1];
        let sin = vec![0.0, 0.0, 0.25, -0.5, -0.75, 0.4];
        let cos = sin
            .iter()
            .map(|&value| (1.0f32 - value * value).sqrt())
            .collect::<Vec<_>>();
        let mut expected_q = q_logical.clone();
        let mut expected_k = k_logical.clone();
        rotate_reference(
            &mut expected_q,
            &sin,
            &cos,
            &positions,
            q_heads,
            head_dim,
            rotary_dim,
        );
        rotate_reference(
            &mut expected_k,
            &sin,
            &cos,
            &positions,
            kv_heads,
            head_dim,
            rotary_dim,
        );

        let q_backing = Tensor::from_host_slice(&q_storage, [tokens, q_pitch], &cuda).unwrap();
        let k_backing = Tensor::from_host_slice(&k_storage, [tokens, k_pitch], &cuda).unwrap();
        let mut q = q_backing.narrow(1, 2, q_width).unwrap();
        let mut k = k_backing.narrow(1, 1, k_width).unwrap();
        let sin = Tensor::from_host_slice(&sin, [3, rotary_dim / 2], &cuda).unwrap();
        let cos = Tensor::from_host_slice(&cos, [3, rotary_dim / 2], &cuda).unwrap();
        let positions = Tensor::from_host_slice(&positions, [3], &cuda).unwrap();

        <Cuda as MathOps>::rope_inplace(
            &scope, &mut q, &mut k, &sin, &cos, &positions, q_heads, kv_heads, head_dim, rotary_dim,
        )
        .unwrap();

        let q_got = q_backing.to_host_vec().unwrap();
        let k_got = k_backing.to_host_vec().unwrap();
        for token in 0..tokens {
            for col in 0..q_pitch {
                let got = q_got[token * q_pitch + col];
                if (2..2 + q_width).contains(&col) {
                    let logical = token * q_width + col - 2;
                    assert!(
                        (got - expected_q[logical]).abs() < 2e-6,
                        "q mismatch token={token} col={col}: got={got}, expected={}",
                        expected_q[logical]
                    );
                    if (col - 2) % head_dim >= rotary_dim {
                        assert_eq!(got.to_bits(), q_logical[logical].to_bits());
                    }
                } else {
                    assert_eq!(got, sentinel);
                }
            }
            for col in 0..k_pitch {
                let got = k_got[token * k_pitch + col];
                if (1..1 + k_width).contains(&col) {
                    let logical = token * k_width + col - 1;
                    assert!((got - expected_k[logical]).abs() < 2e-6);
                    if (col - 1) % head_dim >= rotary_dim {
                        assert_eq!(got.to_bits(), k_logical[logical].to_bits());
                    }
                } else {
                    assert_eq!(got, sentinel);
                }
            }
        }
    }
}
