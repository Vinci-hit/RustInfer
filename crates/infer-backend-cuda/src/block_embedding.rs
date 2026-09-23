//! CuTe DSL AOT block embedding. IDs are synchronously validated before launch;
//! this checked entry point intentionally rejects CUDA graph capture for now.
use crate::{Cuda, CudaScope, aot::RawKernel, error::classify_sync_error, ffi};
use infer_core::{
    dtype::{DTypeId, Dtype, quant::BlockQuantFormat as F},
    exec::ExecScope,
    ports::{MemoryPort, OpError, OpResult},
    quantized::BlockQuantWeight,
    tensor::Tensor,
};
use std::{ffi::c_void, ptr, sync::Arc};

struct EmbeddingSpec {
    format: F,
    dtype: u8,
    threads: u32,
    image: &'static [u8],
    name: &'static [u8],
}
include!(concat!(env!("OUT_DIR"), "/cute_embedding_kernels.rs"));

#[derive(Debug)]
struct Codebooks(*mut c_void);
impl Drop for Codebooks {
    fn drop(&mut self) {
        if !self.0.is_null() {
            let code = unsafe { ffi::cudaFree(self.0) };
            if code != ffi::cudaError_cudaSuccess {
                tracing::error!(?code, "free embedding codebooks failed");
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct EmbeddingKernels {
    kernels: Vec<RawKernel>,
    tables: Codebooks,
}

impl EmbeddingKernels {
    pub(crate) fn tables_ptr(&self) -> *mut c_void {
        self.tables.0
    }

    /// Called only when the context's CuTe AOT target matches the current GPU.
    pub(crate) fn load() -> OpResult<Self> {
        let kernels = SPECS
            .iter()
            .map(|s| RawKernel::load(s.image, s.name))
            .collect::<OpResult<Vec<_>>>()?;
        let mut tables = Codebooks(ptr::null_mut());
        unsafe {
            let code = ffi::cudaMalloc(&mut tables.0, CODEBOOKS.len());
            if code != ffi::cudaError_cudaSuccess {
                return Err(crate::error::allocation_error(code, "embedding codebooks"));
            }
            let code = ffi::cudaMemcpy(
                tables.0,
                CODEBOOKS.as_ptr().cast(),
                CODEBOOKS.len(),
                ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            if code != ffi::cudaError_cudaSuccess {
                return Err(classify_sync_error(code, "upload embedding codebooks"));
            }
        }
        tracing::info!(
            sm = TARGET_SM,
            kernels = kernels.len(),
            "CuTe block embedding kernels loaded"
        );
        Ok(Self { kernels, tables })
    }
}

fn int64(v: usize) -> OpResult<i64> {
    i64::try_from(v)
        .map_err(|_| OpError::Shape("block embedding dimension/stride exceeds i64".into()))
}

pub(crate) fn embedding<T: Dtype>(
    scope: &CudaScope,
    weight: &BlockQuantWeight<Cuda>,
    ids: &Tensor<i32, Cuda>,
    output: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let _guard = scope.enter();
    crate::require_scope_tensor(scope, weight.bytes(), "block embedding weight")?;
    crate::require_scope_tensor(scope, ids, "block embedding ids")?;
    crate::require_scope_tensor(scope, output, "block embedding output")?;
    let dtype = match T::ID {
        DTypeId::F32 => 0,
        DTypeId::F16 => 1,
        DTypeId::BF16 => 2,
        _ => {
            return Err(OpError::unsupported(
                "cuda",
                "block embedding activation dtype",
            ));
        }
    };
    let element_bytes = if dtype == 0 { 4 } else { 2 };
    if T::SIZE_BYTES != element_bytes || std::mem::size_of::<T>() != element_bytes {
        return Err(OpError::Shape("block embedding dtype size mismatch".into()));
    }
    let [vocab, cols] = weight.layout().shape();
    if ids.ndim() != 1 || output.shape().as_slice() != [ids.numel(), cols] {
        return Err(OpError::Shape(
            "block embedding input/output shape mismatch".into(),
        ));
    }
    if Arc::ptr_eq(ids.storage(), output.storage())
        || Arc::ptr_eq(weight.bytes().storage(), output.storage())
    {
        return Err(OpError::Shape(
            "block embedding output aliases source storage".into(),
        ));
    }
    let strides = output.strides().as_slice();
    if !infer_core::types::matrix_elements_are_disjoint(ids.numel(), cols, strides[0], strides[1]) {
        return Err(OpError::Shape(
            "block embedding output elements overlap".into(),
        ));
    }
    let kernels = scope
        .device()
        .config
        .block_embedding
        .as_ref()
        .ok_or_else(|| OpError::unsupported("cuda", "block embedding AOT architecture"))?;
    let index = SPECS
        .iter()
        .position(|s| s.format == weight.layout().format() && s.dtype == dtype)
        .ok_or_else(|| OpError::unsupported("cuda", "block embedding format/dtype"))?;
    let grid = ids
        .numel()
        .checked_mul(weight.layout().blocks_per_row())
        .filter(|&v| v <= i32::MAX as usize)
        .ok_or_else(|| OpError::Shape("block embedding grid exceeds CUDA limit".into()))?;
    let mut cols = int64(cols)?;
    let mut row_bytes = int64(weight.layout().row_bytes())?;
    let mut id_stride = int64(ids.strides().as_slice()[0])?;
    let mut out_row_stride = int64(strides[0])?;
    let mut out_col_stride = int64(strides[1])?;
    let mut vocab = int64(vocab)?;
    if grid == 0 {
        return Ok(());
    }
    let stream = scope.stream().0;
    let mut capture = ffi::cudaStreamCaptureStatus_cudaStreamCaptureStatusNone;
    let code = unsafe { ffi::cudaStreamIsCapturing(stream, &mut capture) };
    if code != ffi::cudaError_cudaSuccess {
        return Err(classify_sync_error(code, "query embedding capture"));
    }
    if capture != ffi::cudaStreamCaptureStatus_cudaStreamCaptureStatusNone {
        return Err(OpError::unsupported(
            "cuda",
            "checked block embedding during graph capture",
        ));
    }
    // Packed host snapshot of potentially strided IDs, on the actual scope
    // stream. Unlike Tensor::to_host_vec, this also supports column views.
    let count = if id_stride == 0 { 1 } else { ids.numel() };
    let len = count
        .checked_mul(4)
        .ok_or_else(|| OpError::Shape("embedding ID buffer overflow".into()))?;
    let pitch = if id_stride == 0 {
        4
    } else {
        usize::try_from(id_stride)
            .unwrap()
            .checked_mul(4)
            .ok_or_else(|| OpError::Shape("embedding ID pitch overflow".into()))?
    };
    let mut staging = scope.device().alloc_host_buffer(len)?;
    let copy = unsafe {
        ffi::cudaMemcpy2DAsync(
            staging.bytes_mut().as_mut_ptr().cast(),
            4,
            ids.data_ptr().cast(),
            pitch,
            4,
            count,
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            stream,
        )
    };
    let sync = unsafe { ffi::cudaStreamSynchronize(stream) };
    if sync != ffi::cudaError_cudaSuccess {
        // Failed synchronization cannot prove pending DMA stopped. Retain both
        // ends until process teardown, matching the block-weight upload policy.
        std::mem::forget((staging, ids.clone()));
        return Err(OpError::Fatal(format!(
            "embedding ID sync failed; buffers retained: {}",
            crate::CudaError(sync)
        )));
    }
    if copy != ffi::cudaError_cudaSuccess {
        return Err(OpError::Kernel(format!(
            "embedding ID copy failed: {}",
            crate::CudaError(copy)
        )));
    }
    if staging.bytes().chunks_exact(4).any(|b| {
        let id = i32::from_ne_bytes(b.try_into().unwrap()) as i64;
        id < 0 || id >= vocab
    }) {
        return Err(OpError::Shape(
            "block embedding token ID outside vocabulary".into(),
        ));
    }
    let mut dst = output.data_ptr_mut();
    let mut src = weight.bytes().data_ptr();
    let mut token_ids = ids.data_ptr();
    let mut tables = kernels.tables.0;
    let mut args = [
        (&mut dst as *mut *mut T).cast::<c_void>(),
        (&mut src as *mut *const u8).cast(),
        (&mut token_ids as *mut *const i32).cast(),
        (&mut tables as *mut *mut c_void).cast(),
        (&mut cols as *mut i64).cast(),
        (&mut row_bytes as *mut i64).cast(),
        (&mut id_stride as *mut i64).cast(),
        (&mut out_row_stride as *mut i64).cast(),
        (&mut out_col_stride as *mut i64).cast(),
        (&mut vocab as *mut i64).cast(),
    ];
    // SAFETY: exact ABI checked by compiler; host checks cover device, shape,
    // offsets, strides, ID bounds and aliasing. Tensor storage must survive
    // stream completion, as for all asynchronous CUDA MathOps.
    unsafe { kernels.kernels[index].launch(grid as u32, SPECS[index].threads, stream, &mut args) }
}
