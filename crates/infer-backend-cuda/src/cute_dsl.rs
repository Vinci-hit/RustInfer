//! Pinned CuTe DSL AOT manifests; driver lifecycle is shared with Triton.
use crate::aot::{AotKernels, KernelSpec};
use infer_core::ports::OpResult;

include!(concat!(env!("OUT_DIR"), "/cute_dsl_kernels.rs"));

pub(crate) fn load(device_id: i32) -> OpResult<Option<AotKernels>> {
    AotKernels::new(device_id, KERNELS, TARGET_SM, "CuTe DSL")
}
