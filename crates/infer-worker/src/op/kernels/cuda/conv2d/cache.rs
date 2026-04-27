//! Descriptor / algorithm / workspace cache for cuDNN `conv2d_cudnn`.
//!
//! The naive per-call cuDNN path does a heavy amount of bookkeeping work on
//! every invocation:
//!
//! 1. `cudnnCreate*Descriptor` × 4–5 and `cudnnSet*Descriptor` × 4–5.
//! 2. `cudnnGetConvolutionForwardAlgorithm_v7` — heuristic search, hundreds
//!    of microseconds per call.
//! 3. `cudaMalloc` / `cudaFree` for the per-call workspace.
//! 4. Destroying the descriptors at the end.
//!
//! For a VAE decoder that issues ~34 convolutions per image, that adds up to
//! tens of milliseconds of pure host-side overhead per `decode()`.
//!
//! This module caches every piece of state that depends only on the tuple
//! `(input shape, weight shape, stride, padding, dtype)`, which is invariant
//! across denoising steps. The first call for a given shape populates an
//! entry; every subsequent call reuses it. A single shared workspace grows
//! on demand to fit the largest requirement.

use std::collections::HashMap;
use std::os::raw::c_void;

use crate::base::DataType;
use crate::base::error::{Error, Result};
use crate::cuda::ffi;

/// Key under which a fully configured cuDNN conv entry is cached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct ConvKey {
    pub in_shape: [i32; 4],  // [N, Cin, H, W]
    pub w_shape: [i32; 4],   // [Cout, Cin, kH, kW]
    pub stride: i32,
    pub padding: i32,
    pub dtype: DataType,
    /// Whether the caller supplies a bias tensor. Separate entries are
    /// kept for the bias vs. no-bias variants so the bias descriptor is
    /// only created when needed.
    pub has_bias: bool,
}

/// Fully configured cuDNN state for one `(shape, dtype)` combination.
///
/// Descriptors are raw pointers owned by cuDNN; we explicitly destroy them
/// in [`Conv2dCache::drop`].
pub(super) struct ConvEntry {
    pub input_desc: ffi::cudnnTensorDescriptor_t,
    pub output_desc: ffi::cudnnTensorDescriptor_t,
    pub filter_desc: ffi::cudnnFilterDescriptor_t,
    pub conv_desc: ffi::cudnnConvolutionDescriptor_t,
    /// `[1, Cout, 1, 1]` descriptor — `None` when `has_bias == false`.
    pub bias_desc: Option<ffi::cudnnTensorDescriptor_t>,
    pub algo: ffi::cudnnConvolutionFwdAlgo_t,
    pub ws_size: usize,
    pub cudnn_dtype: ffi::cudnnDataType_t,
}

/// Per-`CudaConfig` cache shared by every `conv2d_cudnn` invocation that
/// uses this config's cuDNN handle.
pub struct Conv2dCache {
    entries: HashMap<ConvKey, ConvEntry>,
    /// Shared workspace backing store, grown on demand.
    workspace_ptr: *mut c_void,
    workspace_size: usize,
}

impl std::fmt::Debug for Conv2dCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv2dCache")
            .field("entries", &self.entries.len())
            .field("workspace_size", &self.workspace_size)
            .finish()
    }
}

// SAFETY: Conv2dCache stores raw cuDNN descriptor pointers. These are
// opaque handles that cuDNN itself treats as thread-safe as long as the
// owning handle's stream is serialized. The outer `CudaConfig` already
// wraps us in a `Mutex` before exposing mutable access.
unsafe impl Send for Conv2dCache {}
unsafe impl Sync for Conv2dCache {}

impl Default for Conv2dCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            workspace_ptr: std::ptr::null_mut(),
            workspace_size: 0,
        }
    }
}

impl Conv2dCache {
    #[allow(dead_code)] // constructor 别名，留给未来使用；当前仅通过 Default::default() 创建
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Pointer / size of the shared workspace. The pointer is valid until
    /// the next call to [`Self::ensure_workspace`] that actually grows the
    /// buffer or until [`Self::drop`] runs.
    pub(super) fn workspace(&self) -> (*mut c_void, usize) {
        (self.workspace_ptr, self.workspace_size)
    }

    /// Grow the shared workspace to at least `needed` bytes. Free the old
    /// buffer if reallocation is required; do nothing if the buffer is
    /// already large enough.
    pub(super) fn ensure_workspace(&mut self, needed: usize) -> Result<()> {
        if needed <= self.workspace_size {
            return Ok(());
        }
        if !self.workspace_ptr.is_null() {
            unsafe {
                crate::cuda_check!(ffi::cudaFree(self.workspace_ptr))?;
            }
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            crate::cuda_check!(ffi::cudaMalloc(&mut ptr, needed))?;
        }
        self.workspace_ptr = ptr;
        self.workspace_size = needed;
        Ok(())
    }

    /// Fetch a cache entry, creating it if it doesn't exist yet.
    ///
    /// All `cudnnCreate*`/`cudnnSet*` calls happen here on cache miss; on
    /// hit we simply return a reference to the stored descriptors.
    pub(super) fn get_or_insert(
        &mut self,
        key: ConvKey,
        handle: ffi::cudnnHandle_t,
        out_shape: [i32; 4],
    ) -> Result<&ConvEntry> {
        if !self.entries.contains_key(&key) {
            let entry = Self::build_entry(key, handle, out_shape)?;
            self.entries.insert(key, entry);
        }
        Ok(&self.entries[&key])
    }

    /// Build a fresh cuDNN descriptor set + pick an algorithm for `key`.
    fn build_entry(
        key: ConvKey,
        handle: ffi::cudnnHandle_t,
        out_shape: [i32; 4],
    ) -> Result<ConvEntry> {
        let (cudnn_dtype, compute_type) = match key.dtype {
            DataType::F32 => (
                ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
                ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
            ),
            DataType::F16 => (
                ffi::cudnnDataType_t::CUDNN_DATA_HALF,
                ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
            ),
            DataType::BF16 => (
                ffi::cudnnDataType_t::CUDNN_DATA_BFLOAT16,
                ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
            ),
            other => {
                return Err(Error::InvalidArgument(format!(
                    "cuDNN conv2d: unsupported dtype {:?}", other
                )).into())
            }
        };
        let format = ffi::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;

        unsafe {
            let mut input_desc: ffi::cudnnTensorDescriptor_t = std::ptr::null_mut();
            let mut output_desc: ffi::cudnnTensorDescriptor_t = std::ptr::null_mut();
            let mut filter_desc: ffi::cudnnFilterDescriptor_t = std::ptr::null_mut();
            let mut conv_desc: ffi::cudnnConvolutionDescriptor_t = std::ptr::null_mut();

            cudnn_check(ffi::cudnnCreateTensorDescriptor(&mut input_desc))?;
            cudnn_check(ffi::cudnnCreateTensorDescriptor(&mut output_desc))?;
            cudnn_check(ffi::cudnnCreateFilterDescriptor(&mut filter_desc))?;
            cudnn_check(ffi::cudnnCreateConvolutionDescriptor(&mut conv_desc))?;

            cudnn_check(ffi::cudnnSetTensor4dDescriptor(
                input_desc, format, cudnn_dtype,
                key.in_shape[0], key.in_shape[1], key.in_shape[2], key.in_shape[3],
            ))?;
            cudnn_check(ffi::cudnnSetTensor4dDescriptor(
                output_desc, format, cudnn_dtype,
                out_shape[0], out_shape[1], out_shape[2], out_shape[3],
            ))?;
            cudnn_check(ffi::cudnnSetFilter4dDescriptor(
                filter_desc, cudnn_dtype, format,
                key.w_shape[0], key.w_shape[1], key.w_shape[2], key.w_shape[3],
            ))?;
            cudnn_check(ffi::cudnnSetConvolution2dDescriptor(
                conv_desc, key.padding, key.padding, key.stride, key.stride,
                1, 1, // dilation
                ffi::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                compute_type,
            ))?;
            // Enable Tensor Core math so BF16/FP16 inputs hit _hmma_ kernels
            // instead of the F32 implicit_gemm fallback.
            cudnn_check(ffi::cudnnSetConvolutionMathType(
                conv_desc, ffi::cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
            ))?;

            // Pick an algorithm once — cached for every subsequent call.
            let mut perf_results: [ffi::cudnnConvolutionFwdAlgoPerf_t; 1] = std::mem::zeroed();
            let mut returned_algo_count: i32 = 0;
            cudnn_check(ffi::cudnnGetConvolutionForwardAlgorithm_v7(
                handle, input_desc, filter_desc, conv_desc, output_desc,
                1, &mut returned_algo_count, perf_results.as_mut_ptr(),
            ))?;
            let algo = perf_results[0].algo;

            let mut ws_size: usize = 0;
            cudnn_check(ffi::cudnnGetConvolutionForwardWorkspaceSize(
                handle, input_desc, filter_desc, conv_desc, output_desc, algo, &mut ws_size,
            ))?;

            let bias_desc = if key.has_bias {
                let mut desc: ffi::cudnnTensorDescriptor_t = std::ptr::null_mut();
                cudnn_check(ffi::cudnnCreateTensorDescriptor(&mut desc))?;
                cudnn_check(ffi::cudnnSetTensor4dDescriptor(
                    desc, format, cudnn_dtype, 1, key.w_shape[0], 1, 1,
                ))?;
                Some(desc)
            } else {
                None
            };

            Ok(ConvEntry {
                input_desc,
                output_desc,
                filter_desc,
                conv_desc,
                bias_desc,
                algo,
                ws_size,
                cudnn_dtype,
            })
        }
    }
}

impl Drop for Conv2dCache {
    fn drop(&mut self) {
        for entry in self.entries.values_mut() {
            unsafe {
                ffi::cudnnDestroyTensorDescriptor(entry.input_desc);
                ffi::cudnnDestroyTensorDescriptor(entry.output_desc);
                ffi::cudnnDestroyFilterDescriptor(entry.filter_desc);
                ffi::cudnnDestroyConvolutionDescriptor(entry.conv_desc);
                if let Some(bd) = entry.bias_desc {
                    ffi::cudnnDestroyTensorDescriptor(bd);
                }
            }
        }
        if !self.workspace_ptr.is_null() {
            unsafe {
                let _ = ffi::cudaFree(self.workspace_ptr);
            }
        }
    }
}

#[inline]
fn cudnn_check(status: ffi::cudnnStatus_t) -> Result<()> {
    if status != ffi::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Err(Error::InvalidArgument(format!("cuDNN error: {:?}", status)).into());
    }
    Ok(())
}
