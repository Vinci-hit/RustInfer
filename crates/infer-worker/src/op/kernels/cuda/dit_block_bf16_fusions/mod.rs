use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
use crate::cuda::config::CudaConfig;
use crate::tensor::Tensor;

unsafe extern "C" {
    fn zimage_fused_adaln_modulation_split_scale_add_tanh_bf16_forward(
        mod_out: *mut half::bf16,
        dim: i32,
        stream: crate::cuda::ffi::cudaStream_t,
    );

    fn zimage_fused_qkv_split_head_rmsnorm_rope_interleaved_bf16_forward(
        q: *mut half::bf16,
        k: *mut half::bf16,
        v: *mut half::bf16,
        qkv_out: *const half::bf16,
        norm_q_weight: *const half::bf16,
        norm_k_weight: *const half::bf16,
        cos: *const f32,
        sin: *const f32,
        seq: i32,
        n_heads: i32,
        head_dim: i32,
        eps: f32,
        stream: crate::cuda::ffi::cudaStream_t,
    );

    fn zimage_fused_post_attention_rmsnorm_gate_residual_and_ffn_prenorm_scale_bf16_forward(
        residual_mid: *mut half::bf16,
        ffn_in: *mut half::bf16,
        x: *const half::bf16,
        to_out_result: *const half::bf16,
        gate_msa: *const half::bf16,
        scale_mlp: *const half::bf16,
        attention_norm2_weight: *const half::bf16,
        ffn_norm1_weight: *const half::bf16,
        rows: i32,
        dim: i32,
        attention_norm2_eps: f32,
        ffn_norm1_eps: f32,
        stream: crate::cuda::ffi::cudaStream_t,
    );

    fn zimage_fused_ffn_down_rmsnorm_gate_residual_bf16_forward(
        dst: *mut half::bf16,
        residual_mid: *const half::bf16,
        ffn_out: *const half::bf16,
        gate_mlp: *const half::bf16,
        ffn_norm2_weight: *const half::bf16,
        rows: i32,
        dim: i32,
        ffn_norm2_eps: f32,
        stream: crate::cuda::ffi::cudaStream_t,
    );
}

#[inline]
fn require_cuda_bf16(name: &str, tensors: &[&Tensor]) -> Result<()> {
    for (idx, t) in tensors.iter().enumerate() {
        if t.dtype() != DataType::BF16 || !matches!(t.device(), DeviceType::Cuda(_)) {
            return Err(Error::InvalidArgument(format!(
                "{} requires CUDA BF16 tensors; tensor #{} is {:?} on {:?}",
                name,
                idx,
                t.dtype(),
                t.device(),
            )).into());
        }
    }
    Ok(())
}

pub fn zimage_fused_adaln_modulation_split_scale_add_tanh_bf16_inplace(
    mod_out: &mut Tensor,
    dim: usize,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    require_cuda_bf16(
        "zimage_fused_adaln_modulation_split_scale_add_tanh_bf16_inplace",
        &[mod_out],
    )?;
    if mod_out.numel() != 4 * dim {
        return Err(Error::InvalidArgument(format!(
            "zimage fused adaLN expects mod_out numel {}, got {}",
            4 * dim,
            mod_out.numel(),
        )).into());
    }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let p = mod_out.as_bf16_mut()?.data_ptr_mut();
    unsafe {
        zimage_fused_adaln_modulation_split_scale_add_tanh_bf16_forward(
            p,
            dim as i32,
            stream,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn zimage_fused_qkv_split_head_rmsnorm_rope_interleaved_bf16(
    qkv_out: &Tensor,
    norm_q_weight: &Tensor,
    norm_k_weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    q: &mut Tensor,
    k: &mut Tensor,
    v: &mut Tensor,
    seq: usize,
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    require_cuda_bf16(
        "zimage_fused_qkv_split_head_rmsnorm_rope_interleaved_bf16",
        &[qkv_out, norm_q_weight, norm_k_weight, q, k, v],
    )?;
    if cos.dtype() != DataType::F32 || sin.dtype() != DataType::F32 || !cos.device().is_cuda() || !sin.device().is_cuda() {
        return Err(Error::InvalidArgument(
            "zimage fused qkv/norm/rope requires CUDA F32 cos/sin".into(),
        ).into());
    }
    let dim = n_heads * head_dim;
    if qkv_out.shape() != [seq, 3 * dim]
        || q.shape() != [seq, dim]
        || k.shape() != [seq, dim]
        || v.shape() != [seq, dim]
        || norm_q_weight.shape() != [head_dim]
        || norm_k_weight.shape() != [head_dim]
        || cos.shape() != [seq, head_dim / 2]
        || sin.shape() != [seq, head_dim / 2]
    {
        return Err(Error::InvalidArgument(format!(
            "zimage fused qkv/norm/rope shape mismatch: qkv={:?} q={:?} k={:?} v={:?} wq={:?} wk={:?} cos={:?} sin={:?}, expected seq={} n_heads={} head_dim={}",
            qkv_out.shape(), q.shape(), k.shape(), v.shape(),
            norm_q_weight.shape(), norm_k_weight.shape(), cos.shape(), sin.shape(),
            seq, n_heads, head_dim,
        )).into());
    }

    let stream = CudaConfig::resolve_stream(cuda_config);
    let q_ptr = q.as_bf16_mut()?.data_ptr_mut();
    let k_ptr = k.as_bf16_mut()?.data_ptr_mut();
    let v_ptr = v.as_bf16_mut()?.data_ptr_mut();
    let qkv_ptr = qkv_out.as_bf16()?.data_ptr();
    let wq_ptr = norm_q_weight.as_bf16()?.data_ptr();
    let wk_ptr = norm_k_weight.as_bf16()?.data_ptr();
    let cos_ptr = cos.as_f32()?.data_ptr();
    let sin_ptr = sin.as_f32()?.data_ptr();
    unsafe {
        zimage_fused_qkv_split_head_rmsnorm_rope_interleaved_bf16_forward(
            q_ptr,
            k_ptr,
            v_ptr,
            qkv_ptr,
            wq_ptr,
            wk_ptr,
            cos_ptr,
            sin_ptr,
            seq as i32,
            n_heads as i32,
            head_dim as i32,
            eps,
            stream,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn zimage_fused_post_attention_rmsnorm_gate_residual_and_ffn_prenorm_scale_bf16(
    x: &Tensor,
    to_out_result: &Tensor,
    gate_msa: &Tensor,
    scale_mlp: &Tensor,
    attention_norm2_weight: &Tensor,
    ffn_norm1_weight: &Tensor,
    residual_mid: &mut Tensor,
    ffn_in: &mut Tensor,
    attention_norm2_eps: f32,
    ffn_norm1_eps: f32,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    require_cuda_bf16(
        "zimage_fused_post_attention_rmsnorm_gate_residual_and_ffn_prenorm_scale_bf16",
        &[x, to_out_result, gate_msa, scale_mlp, attention_norm2_weight, ffn_norm1_weight, residual_mid, ffn_in],
    )?;
    let dim = attention_norm2_weight.shape()[0];
    let rows = x.numel() / dim;
    if x.shape() != to_out_result.shape()
        || residual_mid.shape() != x.shape()
        || ffn_in.shape() != x.shape()
        || gate_msa.numel() != dim
        || scale_mlp.numel() != dim
        || ffn_norm1_weight.numel() != dim
    {
        return Err(Error::InvalidArgument(format!(
            "zimage fused post-attn/pre-ffn shape mismatch: x={:?} to_out={:?} gate={:?} scale={:?} attn_w={:?} ffn_w={:?} residual={:?} ffn_in={:?}",
            x.shape(), to_out_result.shape(), gate_msa.shape(), scale_mlp.shape(),
            attention_norm2_weight.shape(), ffn_norm1_weight.shape(), residual_mid.shape(), ffn_in.shape(),
        )).into());
    }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let residual_ptr = residual_mid.as_bf16_mut()?.data_ptr_mut();
    let ffn_in_ptr = ffn_in.as_bf16_mut()?.data_ptr_mut();
    unsafe {
        zimage_fused_post_attention_rmsnorm_gate_residual_and_ffn_prenorm_scale_bf16_forward(
            residual_ptr,
            ffn_in_ptr,
            x.as_bf16()?.data_ptr(),
            to_out_result.as_bf16()?.data_ptr(),
            gate_msa.as_bf16()?.data_ptr(),
            scale_mlp.as_bf16()?.data_ptr(),
            attention_norm2_weight.as_bf16()?.data_ptr(),
            ffn_norm1_weight.as_bf16()?.data_ptr(),
            rows as i32,
            dim as i32,
            attention_norm2_eps,
            ffn_norm1_eps,
            stream,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn zimage_fused_ffn_down_rmsnorm_gate_residual_bf16(
    residual_mid: &Tensor,
    ffn_out: &Tensor,
    gate_mlp: &Tensor,
    ffn_norm2_weight: &Tensor,
    dst: &mut Tensor,
    ffn_norm2_eps: f32,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    require_cuda_bf16(
        "zimage_fused_ffn_down_rmsnorm_gate_residual_bf16",
        &[residual_mid, ffn_out, gate_mlp, ffn_norm2_weight, dst],
    )?;
    let dim = ffn_norm2_weight.shape()[0];
    let rows = residual_mid.numel() / dim;
    if residual_mid.shape() != ffn_out.shape()
        || dst.shape() != residual_mid.shape()
        || gate_mlp.numel() != dim
    {
        return Err(Error::InvalidArgument(format!(
            "zimage fused ffn-out/residual shape mismatch: residual={:?} ffn_out={:?} gate={:?} weight={:?} dst={:?}",
            residual_mid.shape(), ffn_out.shape(), gate_mlp.shape(), ffn_norm2_weight.shape(), dst.shape(),
        )).into());
    }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let dst_ptr = dst.as_bf16_mut()?.data_ptr_mut();
    unsafe {
        zimage_fused_ffn_down_rmsnorm_gate_residual_bf16_forward(
            dst_ptr,
            residual_mid.as_bf16()?.data_ptr(),
            ffn_out.as_bf16()?.data_ptr(),
            gate_mlp.as_bf16()?.data_ptr(),
            ffn_norm2_weight.as_bf16()?.data_ptr(),
            rows as i32,
            dim as i32,
            ffn_norm2_eps,
            stream,
        );
    }
    Ok(())
}
