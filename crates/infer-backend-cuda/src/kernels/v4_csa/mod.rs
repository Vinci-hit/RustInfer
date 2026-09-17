//! Single-request CSA overlapping compression (projection GEMMs excluded).
use super::v4_common::Validation;
use crate::{Cuda, ffi};
use half::bf16;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn rustinfer_v4_csa_compress(
        values: *const f32,
        gates: *const f32,
        ape: *const f32,
        norm: *const f32,
        rope: *const f32,
        start: *const i32,
        state: *mut f32,
        cache: *mut bf16,
        tokens: i32,
        capacity: i32,
        eps: f32,
        stream: ffi::cudaStream_t,
    ) -> i32;
}

pub fn compress(
    stream: ffi::cudaStream_t,
    device: i32,
    values: &Tensor<f32, Cuda>,
    gates: &Tensor<f32, Cuda>,
    ape: &Tensor<f32, Cuda>,
    norm: &Tensor<f32, Cuda>,
    rope: &Tensor<f32, Cuda>,
    start: &Tensor<i32, Cuda>,
    state: &mut Tensor<f32, Cuda>,
    compressed: &mut Tensor<bf16, Cuda>,
    eps: f32,
) -> OpResult<()> {
    let check = Validation::new("v4_csa_compress", device);
    let tokens = values.shape().as_slice().first().copied().unwrap_or(0);
    let capacity = compressed.shape().as_slice().first().copied().unwrap_or(0);
    if !(1..=i32::MAX as usize).contains(&tokens)
        || !(1..=i32::MAX as usize / 4 + 1).contains(&capacity)
        || values.shape().as_slice() != [tokens, 1024]
        || gates.shape() != values.shape()
        || ape.shape().as_slice() != [4, 1024]
        || norm.shape().as_slice() != [512]
        || rope.shape().as_slice() != [capacity, 32, 2]
        || start.shape().as_slice() != [1]
        || state.shape().as_slice() != [3, 3, 512]
        || compressed.shape().as_slice() != [capacity, 512]
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(OpError::Shape("v4_csa_compress: expected values/gates [N,1024], ape [4,1024], norm [512], rope [C,32,2], start [1], state [3,3,512], compressed [C,512], positive finite eps".into()));
    }
    let reads = [
        check.tensor(values)?,
        check.tensor(gates)?,
        check.tensor(ape)?,
        check.tensor(norm)?,
        check.tensor(rope)?,
        check.tensor(start)?,
    ];
    check.disjoint(&reads, &[check.tensor(state)?, check.tensor(compressed)?])?;
    check.launched(unsafe {
        rustinfer_v4_csa_compress(
            values.data_ptr(),
            gates.data_ptr(),
            ape.data_ptr(),
            norm.data_ptr(),
            rope.data_ptr(),
            start.data_ptr(),
            state.data_ptr_mut(),
            compressed.data_ptr_mut(),
            tokens as i32,
            capacity as i32,
            eps,
            stream,
        )
    })
}
