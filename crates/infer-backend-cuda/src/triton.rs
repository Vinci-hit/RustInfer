//! Pinned Triton AOT binaries, owned by one CUDA execution context.
//!
//! No Python, compilation, module loading or allocation occurs during launch.
//! The supported ABI is Triton 3.6.0, ordinary single-CTA kernels without
//! global/profile scratch. The compiler checks these constraints at build time.

use crate::ffi;
use infer_core::ports::{OpError, OpResult};
use infer_core::types::{DataType, Dtype};
use std::ffi::{CStr, c_char, c_void};
use std::ptr;

// Keep the small Driver API surface private. Runtime and driver stream handles
// refer to the same CUDA stream; all opaque handles remain in this module.
unsafe extern "C" {
    fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> i32;
    fn cuModuleGetFunction(
        function: *mut *mut c_void,
        module: *mut c_void,
        name: *const c_char,
    ) -> i32;
    fn cuModuleUnload(module: *mut c_void) -> i32;
    fn cuFuncSetAttribute(function: *mut c_void, attribute: i32, value: i32) -> i32;
    fn cuLaunchKernel(
        function: *mut c_void,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared: u32,
        stream: *mut c_void,
        args: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn cuGetErrorString(error: i32, description: *mut *const c_char) -> i32;
}

#[derive(Debug)]
pub(crate) struct KernelSpec {
    dtype: u8,
    block: usize,
    image: &'static [u8],
    name: &'static [u8],
    shared: u32,
    threads: u32,
}

include!(concat!(env!("OUT_DIR"), "/triton_kernels.rs"));

fn check(code: i32, operation: &str) -> OpResult<()> {
    if code == 0 {
        return Ok(());
    }
    let mut description = ptr::null();
    let text = unsafe {
        cuGetErrorString(code, &mut description);
        if description.is_null() {
            "unknown CUDA driver error".into()
        } else {
            CStr::from_ptr(description).to_string_lossy()
        }
    };
    let message = format!("Triton {operation} failed ({code}): {text}");
    // Driver calls can report a prior asynchronous fault. These CUDA driver
    // codes identify a lost/poisoned context, so the worker must not retry it.
    Err(if matches!(code, 214 | 700 | 702 | 709 | 710 | 714..=719) {
        OpError::Fatal(message)
    } else {
        OpError::Kernel(message)
    })
}

#[derive(Debug)]
struct LoadedKernel {
    module: *mut c_void,
    function: *mut c_void,
    spec: &'static KernelSpec,
}

impl LoadedKernel {
    fn load(spec: &'static KernelSpec) -> OpResult<Self> {
        let mut loaded = Self {
            module: ptr::null_mut(),
            function: ptr::null_mut(),
            spec,
        };
        unsafe {
            check(
                cuModuleLoadData(&mut loaded.module, spec.image.as_ptr().cast()),
                "load module",
            )?;
            // Resolving every function here also preloads kernels when CUDA's
            // lazy module loading is enabled, before any graph capture begins.
            check(
                cuModuleGetFunction(
                    &mut loaded.function,
                    loaded.module,
                    spec.name.as_ptr().cast(),
                ),
                "resolve kernel",
            )?;
            if spec.shared > 48 * 1024 {
                // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8.
                check(
                    cuFuncSetAttribute(loaded.function, 8, spec.shared as i32),
                    "configure shared memory",
                )?;
            }
        }
        Ok(loaded)
    }
}

impl Drop for LoadedKernel {
    fn drop(&mut self) {
        if !self.module.is_null()
            && let Err(error) = check(unsafe { cuModuleUnload(self.module) }, "unload module")
        {
            tracing::error!(?error, "Triton module teardown failed");
        }
    }
}

#[derive(Debug)]
pub(crate) struct TritonKernels {
    kernels: Vec<LoadedKernel>,
}

impl TritonKernels {
    /// Called after runtime context initialization, on its owning device.
    pub(crate) fn new(device_id: i32) -> OpResult<Option<Self>> {
        let mut major = 0;
        let mut minor = 0;
        for (value, attr) in [
            (
                &mut major,
                ffi::cudaDeviceAttr_cudaDevAttrComputeCapabilityMajor,
            ),
            (
                &mut minor,
                ffi::cudaDeviceAttr_cudaDevAttrComputeCapabilityMinor,
            ),
        ] {
            let status = unsafe { ffi::cudaDeviceGetAttribute(value, attr, device_id) };
            if status != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!(
                    "Triton device query: {}",
                    crate::CudaError(status)
                )));
            }
        }
        if major * 10 + minor != TARGET_SM {
            tracing::warn!(
                device_id,
                compiled_sm = TARGET_SM,
                major,
                minor,
                "Triton architecture mismatch; using native CUDA RMSNorm"
            );
            return Ok(None);
        }
        let kernels = KERNELS
            .iter()
            .map(LoadedKernel::load)
            .collect::<OpResult<_>>()?;
        tracing::info!(
            device_id,
            sm = TARGET_SM,
            "Triton AOT RMSNorm kernels loaded"
        );
        Ok(Some(Self { kernels }))
    }

    /// Returns false only when this shape/dtype has no compiled specialization.
    ///
    /// # Safety
    /// Pointers address contiguous `[rows, dim]` input/output and `[dim]`
    /// weights of T in this context. Buffers survive completion on `stream`;
    /// output may equal input but must not overlap weights or partially alias
    /// input. The owning CUDA device/context must be current.
    pub(crate) unsafe fn rmsnorm<T: Dtype>(
        &self,
        stream: ffi::cudaStream_t,
        output: *mut T,
        input: *const T,
        weight: *const T,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> OpResult<bool> {
        let dtype = match T::DATA_TYPE {
            DataType::F32 => 0,
            DataType::F16 => 1,
            DataType::BF16 => 2,
            _ => return Ok(false),
        };
        if !(1..=16384).contains(&dim) {
            return Ok(false);
        }
        let block = dim.next_power_of_two().max(32);
        let Some(kernel) = self
            .kernels
            .iter()
            .find(|k| k.spec.dtype == dtype && k.spec.block == block)
        else {
            return Ok(false);
        };
        let rows = u32::try_from(rows)
            .ok()
            .filter(|&n| n <= i32::MAX as u32)
            .ok_or_else(|| {
                OpError::Shape("Triton RMSNorm row count exceeds CUDA grid limit".into())
            })?;
        if rows == 0 {
            return Ok(true);
        }
        let mut output = output;
        let mut input = input;
        let mut weight = weight;
        let mut dim = dim as i32;
        let mut eps = eps;
        // The pinned compiler always appends these two ABI arguments, including
        // for kernels whose metadata guarantees that no scratch is used.
        let mut global_scratch: u64 = 0;
        let mut profile_scratch: u64 = 0;
        let mut args = [
            (&mut output as *mut *mut T).cast::<c_void>(),
            (&mut input as *mut *const T).cast(),
            (&mut weight as *mut *const T).cast(),
            (&mut dim as *mut i32).cast(),
            (&mut eps as *mut f32).cast(),
            (&mut global_scratch as *mut u64).cast(),
            (&mut profile_scratch as *mut u64).cast(),
        ];
        unsafe {
            check(
                cuLaunchKernel(
                    kernel.function,
                    rows,
                    1,
                    1,
                    kernel.spec.threads,
                    1,
                    1,
                    kernel.spec.shared,
                    stream.cast(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "launch RMSNorm",
            )?;
        }
        Ok(true)
    }
}
