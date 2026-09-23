//! Asynchronous quantized matmul: GEMV for one token, tiled GEMM otherwise.
use crate::{Cuda, CudaScope, aot::RawKernel};
use infer_core::{
    dtype::{DTypeId, Dtype, quant::BlockQuantFormat as F},
    exec::ExecScope,
    ports::{OpError, OpResult},
    quantized::BlockQuantWeight,
    tensor::Tensor,
};
use std::{ffi::c_void, sync::Arc};

struct MatmulSpec {
    format: F,
    dtype: u8,
    threads: u32,
    image: &'static [u8],
    name: &'static [u8],
}
include!(concat!(env!("OUT_DIR"), "/cute_gemv_kernels.rs"));
include!(concat!(env!("OUT_DIR"), "/cute_gemm_kernels.rs"));

#[derive(Debug)]
pub(crate) struct MatmulKernels {
    gemv: Vec<RawKernel>,
    gemm: Vec<RawKernel>,
}
impl MatmulKernels {
    pub(crate) fn load() -> OpResult<Self> {
        let load = |specs: &[MatmulSpec]| {
            specs
                .iter()
                .map(|s| RawKernel::load(s.image, s.name))
                .collect::<OpResult<Vec<_>>>()
        };
        Ok(Self {
            gemv: load(GEMV_SPECS)?,
            gemm: load(GEMM_SPECS)?,
        })
    }
}

fn int64(v: usize) -> OpResult<i64> {
    i64::try_from(v).map_err(|_| OpError::Shape("block matmul dimension/stride exceeds i64".into()))
}

pub(crate) fn matmul<T: Dtype>(
    scope: &CudaScope,
    input: &Tensor<T, Cuda>,
    weight: &BlockQuantWeight<Cuda>,
    bias: Option<&Tensor<T, Cuda>>,
    output: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let _guard = scope.enter();
    crate::require_scope_tensor(scope, input, "block matmul input")?;
    crate::require_scope_tensor(scope, weight.bytes(), "block matmul weight")?;
    crate::require_scope_tensor(scope, output, "block matmul output")?;
    if let Some(b) = bias {
        crate::require_scope_tensor(scope, b, "block matmul bias")?;
    }
    let dtype = match T::ID {
        DTypeId::F32 => 0,
        DTypeId::F16 => 1,
        DTypeId::BF16 => 2,
        _ => {
            return Err(OpError::unsupported(
                "cuda",
                "block matmul activation dtype",
            ));
        }
    };
    let size = if dtype == 0 { 4 } else { 2 };
    if T::SIZE_BYTES != size || std::mem::size_of::<T>() != size {
        return Err(OpError::Shape("block matmul dtype size mismatch".into()));
    }
    let [n, k] = weight.layout().shape();
    let shape = input.shape().as_slice();
    if shape.len() != 2
        || shape[1] != k
        || output.shape().as_slice() != [shape[0], n]
        || bias.is_some_and(|b| b.shape().as_slice() != [n])
    {
        return Err(OpError::Shape(
            "block matmul input/output/bias shape mismatch".into(),
        ));
    }
    if Arc::ptr_eq(input.storage(), output.storage())
        || Arc::ptr_eq(weight.bytes().storage(), output.storage())
        || bias.is_some_and(|b| Arc::ptr_eq(b.storage(), output.storage()))
    {
        return Err(OpError::Shape(
            "block matmul output aliases source storage".into(),
        ));
    }
    if !infer_core::types::matrix_elements_are_disjoint(
        shape[0],
        n,
        output.strides().as_slice()[0],
        output.strides().as_slice()[1],
    ) {
        return Err(OpError::Shape(
            "block matmul output elements overlap".into(),
        ));
    }
    let kernels = scope
        .device()
        .config
        .block_matmul
        .as_ref()
        .ok_or_else(|| OpError::unsupported("cuda", "block matmul AOT architecture"))?;
    let codebooks = scope
        .device()
        .config
        .block_embedding
        .as_ref()
        .ok_or_else(|| OpError::unsupported("cuda", "block matmul codebooks"))?;
    let (specs, loaded) = if shape[0] <= 1 {
        (GEMV_SPECS, &kernels.gemv)
    } else {
        (GEMM_SPECS, &kernels.gemm)
    };
    let index = specs
        .iter()
        .position(|s| s.format == weight.layout().format() && s.dtype == dtype)
        .ok_or_else(|| OpError::unsupported("cuda", "block matmul format/dtype"))?;
    let grid = n
        .div_ceil(4)
        .checked_mul(shape[0].div_ceil(8))
        .filter(|&g| g <= i32::MAX as usize)
        .ok_or_else(|| OpError::Shape("block matmul grid exceeds CUDA limit".into()))?;
    let mut cols = int64(k)?;
    let mut rows = int64(n)?;
    let mut row_bytes = int64(weight.layout().row_bytes())?;
    // Also bound the total byte extent used by the device descriptor.
    int64(weight.layout().byte_len())?;
    let mut input_stride = int64(input.strides().as_slice()[1])?;
    let mut output_stride = int64(output.strides().as_slice()[1])?;
    let mut bias_stride = int64(bias.map_or(0, |b| b.strides().as_slice()[0]))?;
    let mut has_bias = i64::from(bias.is_some());
    let mut tokens = int64(shape[0])?;
    let mut input_row_stride = int64(input.strides().as_slice()[0])?;
    let mut output_row_stride = int64(output.strides().as_slice()[0])?;
    if shape[0] == 0 {
        return Ok(());
    }
    let mut dst = output.data_ptr_mut();
    let mut src = weight.bytes().data_ptr();
    let mut x = input.data_ptr();
    // Always pass a valid aligned pointer; the no-bias path never reads it.
    let mut b = bias.map_or(x, |b| b.data_ptr());
    let mut tables = codebooks.tables_ptr();
    let mut args = [
        (&mut dst as *mut *mut T).cast::<c_void>(),
        (&mut src as *mut *const u8).cast(),
        (&mut x as *mut *const T).cast(),
        (&mut b as *mut *const T).cast(),
        (&mut tables as *mut *mut c_void).cast(),
        (&mut cols as *mut i64).cast(),
        (&mut rows as *mut i64).cast(),
        (&mut row_bytes as *mut i64).cast(),
        (&mut input_stride as *mut i64).cast(),
        (&mut output_stride as *mut i64).cast(),
        (&mut bias_stride as *mut i64).cast(),
        (&mut has_bias as *mut i64).cast(),
        (&mut tokens as *mut i64).cast(),
        (&mut input_row_stride as *mut i64).cast(),
        (&mut output_row_stride as *mut i64).cast(),
    ];
    // GEMV uses the first 12 arguments; GEMM appends M and the row strides.
    let argc = if shape[0] <= 1 { 12 } else { 15 };
    // SAFETY: AOT compiler checks ABI/block size; validation above establishes
    // shapes, dtype, device/config, nonoverlap and representable dimensions.
    // As with other async MathOps, tensors must outlive stream completion.
    unsafe {
        loaded[index].launch(
            grid as u32,
            specs[index].threads,
            scope.stream().0,
            &mut args[..argc],
        )
    }
}
