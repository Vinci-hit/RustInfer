//! Four-stream BF16 mHC, retaining FP32 mapping weights and Sinkhorn state.
use super::v4_common::Validation;
use crate::{Cuda, ffi};
use half::bf16;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

type MixingOutputs<'a> = (&'a mut Tensor<f32, Cuda>, &'a mut Tensor<f32, Cuda>);

unsafe extern "C" {
    fn rustinfer_v4_mhc_pre_bf16(
        x: *const bf16,
        weight: *const f32,
        scale: *const f32,
        base: *const f32,
        partial: *mut f32,
        collapsed: *mut bf16,
        post: *mut f32,
        comb: *mut f32,
        n: i32,
        dim: i32,
        head: i32,
        norm_eps: f32,
        hc_eps: f32,
        iters: i32,
        stream: ffi::cudaStream_t,
    ) -> i32;
    fn rustinfer_v4_mhc_post_bf16(
        residual: *const bf16,
        branch: *const bf16,
        post: *const f32,
        comb: *const f32,
        output: *mut bf16,
        n: i32,
        dim: i32,
        stream: ffi::cudaStream_t,
    ) -> i32;
}

fn dimensions(n: usize, dim: usize) -> OpResult<usize> {
    if !(1..=i32::MAX as usize).contains(&n) || !(2..=8192).contains(&dim) || !dim.is_multiple_of(2)
    {
        return Err(OpError::Shape(
            "v4_mhc: N>0 and even 2<=D<=8192 required".into(),
        ));
    }
    let parts = (4 * dim).div_ceil(256);
    if n.saturating_mul(parts) > i32::MAX as usize {
        return Err(OpError::Shape("v4_mhc: CUDA grid exceeds i32::MAX".into()));
    }
    Ok(parts)
}

pub fn workspace_floats(n: usize, dim: usize, head: bool) -> OpResult<usize> {
    let parts = dimensions(n, dim)?;
    n.checked_mul(parts)
        .and_then(|v| v.checked_mul(if head { 5 } else { 25 }))
        .ok_or_else(|| OpError::Shape("v4_mhc: workspace size overflow".into()))
}

// Shared validation and launch for Pre and Head. Only Pre has post/comb outputs.
pub fn pre(
    stream: ffi::cudaStream_t,
    device: i32,
    x: &Tensor<bf16, Cuda>,
    weight: &Tensor<f32, Cuda>,
    scale: &Tensor<f32, Cuda>,
    base: &Tensor<f32, Cuda>,
    workspace: &mut Tensor<f32, Cuda>,
    collapsed: &mut Tensor<bf16, Cuda>,
    mixing: Option<MixingOutputs<'_>>,
    norm_eps: f32,
    hc_eps: f32,
    iters: usize,
) -> OpResult<()> {
    let head = mixing.is_none();
    let check = Validation::new(if head { "v4_mhc_head" } else { "v4_mhc_pre" }, device);
    let shape = x.shape().as_slice();
    let n = shape.first().copied().unwrap_or(0);
    let dim = shape.get(2).copied().unwrap_or(0);
    let words = workspace_floats(n, dim, head)?;
    let outputs = if head { 4 } else { 24 };
    if shape != [n, 4, dim]
        || weight.shape().as_slice() != [outputs, 4 * dim]
        || scale.shape().as_slice() != [if head { 1 } else { 3 }]
        || base.shape().as_slice() != [outputs]
        || collapsed.shape().as_slice() != [n, dim]
        || workspace.shape().as_slice().len() != 1
        || workspace.numel() < words
        || !norm_eps.is_finite()
        || norm_eps <= 0.0
        || !hc_eps.is_finite()
        || hc_eps <= 0.0
        || !(1..=20).contains(&iters)
    {
        return Err(OpError::Shape("v4_mhc: expected X [N,4,D], weight [24,4D]/head [4,4D], scale [3]/head [1], base [24]/head [4], collapsed [N,D], sufficient 1D FP32 scratch, finite positive epsilons, 1<=iters<=20".into()));
    }
    let reads = [
        check.tensor(x)?,
        check.tensor(weight)?,
        check.tensor(scale)?,
        check.tensor(base)?,
    ];
    let mut writes = [
        check.tensor(workspace)?,
        check.tensor(collapsed)?,
        (0, 0),
        (0, 0),
    ];
    let (post, comb) = if let Some((post, comb)) = mixing {
        if post.shape().as_slice() != [n, 4] || comb.shape().as_slice() != [n, 4, 4] {
            return Err(OpError::Shape(
                "v4_mhc_pre: expected post [N,4], comb [N,4,4]".into(),
            ));
        }
        writes[2] = check.tensor(post)?;
        writes[3] = check.tensor(comb)?;
        (post.data_ptr_mut(), comb.data_ptr_mut())
    } else {
        (std::ptr::null_mut(), std::ptr::null_mut())
    };
    check.disjoint(&reads, &writes)?;
    let status = unsafe {
        rustinfer_v4_mhc_pre_bf16(
            x.data_ptr(),
            weight.data_ptr(),
            scale.data_ptr(),
            base.data_ptr(),
            workspace.data_ptr_mut(),
            collapsed.data_ptr_mut(),
            post,
            comb,
            n as i32,
            dim as i32,
            i32::from(head),
            norm_eps,
            hc_eps,
            iters as i32,
            stream,
        )
    };
    check.launched(status)
}

pub fn post(
    stream: ffi::cudaStream_t,
    device: i32,
    residual: &Tensor<bf16, Cuda>,
    branch: &Tensor<bf16, Cuda>,
    post: &Tensor<f32, Cuda>,
    comb: &Tensor<f32, Cuda>,
    output: &mut Tensor<bf16, Cuda>,
) -> OpResult<()> {
    let check = Validation::new("v4_mhc_post", device);
    let shape = residual.shape().as_slice();
    let n = shape.first().copied().unwrap_or(0);
    let dim = shape.get(2).copied().unwrap_or(0);
    dimensions(n, dim)?;
    if shape != [n, 4, dim]
        || branch.shape().as_slice() != [n, dim]
        || post.shape().as_slice() != [n, 4]
        || comb.shape().as_slice() != [n, 4, 4]
        || output.shape().as_slice() != shape
    {
        return Err(OpError::Shape(
            "v4_mhc_post: expected residual/output [N,4,D], branch [N,D], post [N,4], comb [N,4,4]"
                .into(),
        ));
    }
    let reads = [
        check.tensor(residual)?,
        check.tensor(branch)?,
        check.tensor(post)?,
        check.tensor(comb)?,
    ];
    check.disjoint(&reads, &[check.tensor(output)?])?;
    let status = unsafe {
        rustinfer_v4_mhc_post_bf16(
            residual.data_ptr(),
            branch.data_ptr(),
            post.data_ptr(),
            comb.data_ptr(),
            output.data_ptr_mut(),
            n as i32,
            dim as i32,
            stream,
        )
    };
    check.launched(status)
}
