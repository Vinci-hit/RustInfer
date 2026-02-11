use super::nccl;
use crate::base::error::Result;
use crate::cuda;
use std::ffi::CStr;
use thiserror::Error;

/// NCCL Error wrapper
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub struct NcclError(pub nccl::ncclResult_t);

impl std::fmt::Display for NcclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let err_str_ptr = unsafe { nccl::ncclGetErrorString(self.0) };
        if err_str_ptr.is_null() {
            write!(f, "Unknown NCCL Error: {:?}", self.0)
        } else {
            let err_str = unsafe { CStr::from_ptr(err_str_ptr) }.to_string_lossy();
            write!(f, "NCCL Error: {}", err_str)
        }
    }
}

/// Helper macro to check NCCL FFI call return values.
/// If not ncclSuccess, returns an NcclError.
#[macro_export]
macro_rules! nccl_check {
    ($expr:expr) => {{
        let result = $expr;
        if result != $crate::comm::nccl::ncclResult_t::ncclSuccess {
            Err($crate::base::error::Error::NcclError(
                $crate::comm::NcclError(result),
            ))
        } else {
            Ok(())
        }
    }};
}

/// Data types for NCCL operations
#[derive(Debug, Clone, Copy)]
pub enum NcclDataType {
    Float32,
    Float16,
    BFloat16,
}

impl From<NcclDataType> for nccl::ncclDataType_t {
    fn from(dt: NcclDataType) -> Self {
        match dt {
            NcclDataType::Float32 => nccl::ncclDataType_t::ncclFloat32,
            NcclDataType::Float16 => nccl::ncclDataType_t::ncclFloat16,
            NcclDataType::BFloat16 => nccl::ncclDataType_t::ncclBfloat16,
        }
    }
}

/// Reduction operations
#[derive(Debug, Clone, Copy, Default)]
pub enum NcclReduceOp {
    #[default]
    Sum,
    Prod,
    Max,
    Min,
}

impl From<NcclReduceOp> for nccl::ncclRedOp_t {
    fn from(op: NcclReduceOp) -> Self {
        match op {
            NcclReduceOp::Sum => nccl::ncclRedOp_t::ncclSum,
            NcclReduceOp::Prod => nccl::ncclRedOp_t::ncclProd,
            NcclReduceOp::Max => nccl::ncclRedOp_t::ncclMax,
            NcclReduceOp::Min => nccl::ncclRedOp_t::ncclMin,
        }
    }
}

/// Wrapper for ncclUniqueId - 128 bytes that must be broadcast to all ranks
#[derive(Clone)]
pub struct NcclUniqueId(pub nccl::ncclUniqueId);

impl NcclUniqueId {
    /// Generate a new unique ID (call on rank 0 only)
    pub fn new() -> Result<Self> {
        let mut id: nccl::ncclUniqueId = unsafe { std::mem::zeroed() };
        unsafe { crate::nccl_check!(nccl::ncclGetUniqueId(&mut id))? };
        Ok(Self(id))
    }

    /// Get as bytes for serialization/broadcast
    pub fn as_bytes(&self) -> &[u8; 128] {
        // ncclUniqueId.internal is char[128]
        unsafe { &*(&self.0.internal as *const _ as *const [u8; 128]) }
    }

    /// Create from bytes (after receiving broadcast)
    pub fn from_bytes(bytes: &[u8; 128]) -> Self {
        let mut id: nccl::ncclUniqueId = unsafe { std::mem::zeroed() };
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), id.internal.as_mut_ptr() as *mut u8, 128);
        }
        Self(id)
    }
}

/// NCCL Communicator - manages distributed communication context
#[derive(Debug)]
pub struct NcclComm {
    comm: nccl::ncclComm_t,
    rank: i32,
    world_size: i32,
}

impl NcclComm {
    /// Initialize communicator from unique ID
    ///
    /// # Arguments
    /// * `unique_id` - Shared unique ID (generated on rank 0, broadcast to all)
    /// * `rank` - This process's rank (0..world_size)
    /// * `world_size` - Total number of ranks
    pub fn init_rank(unique_id: &NcclUniqueId, rank: i32, world_size: i32) -> Result<Self> {
        let mut comm: nccl::ncclComm_t = std::ptr::null_mut();
        unsafe {
            crate::nccl_check!(nccl::ncclCommInitRank(&mut comm, world_size, unique_id.0, rank))?;
        }
        Ok(Self {
            comm,
            rank,
            world_size,
        })
    }

    /// AllReduce: reduce data across all ranks and distribute result
    ///
    /// # Arguments
    /// * `send_buf` - Input buffer (device pointer)
    /// * `recv_buf` - Output buffer (device pointer), can be same as send_buf
    /// * `count` - Number of elements
    /// * `dtype` - Data type
    /// * `op` - Reduction operation
    /// * `stream` - CUDA stream for async execution
    pub fn all_reduce(
        &self,
        send_buf: *const std::ffi::c_void,
        recv_buf: *mut std::ffi::c_void,
        count: usize,
        dtype: NcclDataType,
        op: NcclReduceOp,
        stream: cuda::ffi::cudaStream_t,
    ) -> Result<()> {
        unsafe {
            crate::nccl_check!(nccl::ncclAllReduce(
                send_buf,
                recv_buf,
                count,
                dtype.into(),
                op.into(),
                self.comm,
                stream,
            ))?;
        }
        Ok(())
    }

    /// Get this rank's ID
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Get total number of ranks
    pub fn world_size(&self) -> i32 {
        self.world_size
    }
}

impl Drop for NcclComm {
    fn drop(&mut self) {
        if !self.comm.is_null() {
            unsafe {
                let _ = nccl::ncclCommDestroy(self.comm);
            }
        }
    }
}

// NCCL communicators are thread-safe
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}
