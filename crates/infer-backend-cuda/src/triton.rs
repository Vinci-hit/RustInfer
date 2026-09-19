//! Pinned Triton AOT manifests; driver lifecycle is shared with TileLang.
use crate::aot::{AotKernels, KernelSpec};
use infer_core::ports::OpResult;

include!(concat!(env!("OUT_DIR"), "/triton_kernels.rs"));

pub(crate) fn load(device_id: i32) -> OpResult<Option<AotKernels>> {
    AotKernels::new(device_id, KERNELS, TARGET_SM, "Triton")
}
