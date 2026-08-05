use std::ffi::CStr;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use infer_core::exec::{ExecScope, RankPair};
use infer_core::ports::{CollectiveOps, CommAxis, OpError, OpResult, ReduceOp};
use infer_core::tensor::Tensor;
use infer_core::types::{DTypeId, Dtype};

use crate::{Cuda, CudaConfig, CudaScope, device_utils, ffi, require_scope_tensor};

pub const NCCL_UNIQUE_ID_BYTES: usize = 128;
const MIN_NCCL_VERSION: i32 = 22_403;
const _: () = assert!(std::mem::size_of::<ffi::ncclUniqueId>() == NCCL_UNIQUE_ID_BYTES);

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct NcclUniqueId([u8; NCCL_UNIQUE_ID_BYTES]);

impl fmt::Debug for NcclUniqueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("NcclUniqueId").field(&"<opaque>").finish()
    }
}

impl NcclUniqueId {
    pub fn generate() -> OpResult<Self> {
        require_nccl_version()?;
        let mut raw = MaybeUninit::<ffi::ncclUniqueId>::zeroed();
        nccl_check(
            unsafe { ffi::ncclGetUniqueId(raw.as_mut_ptr()) },
            "ncclGetUniqueId",
        )?;
        let raw = unsafe { raw.assume_init() };
        let mut bytes = [0u8; NCCL_UNIQUE_ID_BYTES];
        unsafe {
            ptr::copy_nonoverlapping(
                raw.internal.as_ptr().cast::<u8>(),
                bytes.as_mut_ptr(),
                NCCL_UNIQUE_ID_BYTES,
            );
        }
        Ok(Self(bytes))
    }

    pub fn from_bytes(bytes: &[u8]) -> OpResult<Self> {
        let bytes: [u8; NCCL_UNIQUE_ID_BYTES] = bytes.try_into().map_err(|_| {
            OpError::Shape(format!(
                "NCCL unique id must contain {NCCL_UNIQUE_ID_BYTES} bytes, got {}",
                bytes.len()
            ))
        })?;
        Ok(Self(bytes))
    }

    pub fn as_bytes(&self) -> &[u8; NCCL_UNIQUE_ID_BYTES] {
        &self.0
    }

    fn as_raw(self) -> ffi::ncclUniqueId {
        let mut raw = MaybeUninit::<ffi::ncclUniqueId>::zeroed();
        unsafe {
            ptr::copy_nonoverlapping(
                self.0.as_ptr(),
                (*raw.as_mut_ptr()).internal.as_mut_ptr().cast::<u8>(),
                NCCL_UNIQUE_ID_BYTES,
            );
            raw.assume_init()
        }
    }
}

pub struct NcclCommunicator {
    raw: AtomicPtr<ffi::ncclComm>,
    rank: usize,
    size: usize,
    device_id: i32,
    _config: Arc<CudaConfig>,
    issue_lock: Mutex<()>,
    failed: AtomicBool,
}

struct NcclDeviceRestore {
    previous: i32,
}

impl Drop for NcclDeviceRestore {
    fn drop(&mut self) {
        if self.previous >= 0 {
            let _ = device_utils::set_current_device(self.previous);
        }
    }
}

impl fmt::Debug for NcclCommunicator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NcclCommunicator")
            .field("rank", &self.rank)
            .field("size", &self.size)
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}

unsafe impl Send for NcclCommunicator {}
unsafe impl Sync for NcclCommunicator {}

impl NcclCommunicator {
    pub fn init_rank(
        device: &Cuda,
        rank: usize,
        size: usize,
        unique_id: NcclUniqueId,
    ) -> OpResult<Arc<Self>> {
        require_nccl_version()?;
        validate_rank(rank, size)?;
        let _restore = NcclDeviceRestore {
            previous: device_utils::current_device().unwrap_or(-1),
        };
        device_utils::set_current_device(device.device_id).map_err(|error| {
            OpError::Kernel(format!("set CUDA device {}: {error}", device.device_id))
        })?;
        let rank_count = checked_i32(size, "NCCL rank count")?;
        let rank_id = checked_i32(rank, "NCCL rank")?;
        let mut raw = ptr::null_mut();
        let result =
            unsafe { ffi::ncclCommInitRank(&mut raw, rank_count, unique_id.as_raw(), rank_id) };
        if result != ffi::ncclResult_t::ncclSuccess {
            if !raw.is_null() {
                unsafe {
                    ffi::ncclCommAbort(raw);
                }
            }
            return Err(nccl_error(result, "ncclCommInitRank"));
        }
        if raw.is_null() {
            return Err(OpError::Kernel(
                "ncclCommInitRank returned success with a null communicator".into(),
            ));
        }
        Ok(Arc::new(Self {
            raw: AtomicPtr::new(raw),
            rank,
            size,
            device_id: device.device_id,
            _config: device.config.clone(),
            issue_lock: Mutex::new(()),
            failed: AtomicBool::new(false),
        }))
    }

    /// Create a complete single-process communicator group in device-list
    /// order. Production TP workers and multi-GPU primitive tests use this;
    /// [`Self::init_rank`] remains available for future multi-process groups.
    pub fn init_all(devices: &[Cuda]) -> OpResult<Vec<Arc<Self>>> {
        require_nccl_version()?;
        if devices.is_empty() {
            return Err(OpError::Shape(
                "ncclCommInitAll requires at least one CUDA device".into(),
            ));
        }
        let _restore = NcclDeviceRestore {
            previous: device_utils::current_device().unwrap_or(-1),
        };
        let device_ids: Vec<i32> = devices.iter().map(|device| device.device_id).collect();
        let mut raw = vec![ptr::null_mut(); devices.len()];
        let device_count = checked_i32(devices.len(), "NCCL device count")?;
        let result =
            unsafe { ffi::ncclCommInitAll(raw.as_mut_ptr(), device_count, device_ids.as_ptr()) };
        if result != ffi::ncclResult_t::ncclSuccess || raw.iter().any(|comm| comm.is_null()) {
            for (&comm, device) in raw.iter().zip(devices) {
                if !comm.is_null() {
                    let _ = device_utils::set_current_device(device.device_id);
                    unsafe {
                        ffi::ncclCommAbort(comm);
                    }
                }
            }
            if result != ffi::ncclResult_t::ncclSuccess {
                return Err(nccl_error(result, "ncclCommInitAll"));
            }
            return Err(OpError::Kernel(
                "ncclCommInitAll returned success with a null communicator".into(),
            ));
        }
        Ok(raw
            .into_iter()
            .zip(devices.iter())
            .enumerate()
            .map(|(rank, (raw, device))| {
                Arc::new(Self {
                    raw: AtomicPtr::new(raw),
                    rank,
                    size: devices.len(),
                    device_id: device.device_id,
                    _config: device.config.clone(),
                    issue_lock: Mutex::new(()),
                    failed: AtomicBool::new(false),
                })
            })
            .collect())
    }

    pub fn rank_pair(&self) -> RankPair {
        RankPair {
            rank: self.rank,
            size: self.size,
        }
    }

    fn lock(&self) -> OpResult<MutexGuard<'_, ()>> {
        let guard = self.issue_lock.lock().map_err(|_| {
            self.failed.store(true, Ordering::Release);
            OpError::Fatal("NCCL communicator issue lock is poisoned".into())
        })?;
        self.ensure_healthy("collective issue")?;
        Ok(guard)
    }

    fn ensure_healthy(&self, op: &str) -> OpResult<()> {
        if self.failed.load(Ordering::Acquire) {
            Err(OpError::Fatal(format!(
                "NCCL communicator rank {}/{} is poisoned before {op}",
                self.rank, self.size
            )))
        } else {
            Ok(())
        }
    }

    fn check(&self, result: ffi::ncclResult_t, what: &str) -> OpResult<()> {
        if result == ffi::ncclResult_t::ncclSuccess {
            Ok(())
        } else {
            self.failed.store(true, Ordering::Release);
            Err(OpError::Fatal(format!(
                "{what} failed on NCCL rank {}/{}: {}",
                self.rank,
                self.size,
                nccl_error_string(result)
            )))
        }
    }

    pub(crate) fn check_async_error(&self) -> OpResult<()> {
        self.ensure_healthy("async error check")?;
        let _issue = self.lock()?;
        let mut async_error = ffi::ncclResult_t::ncclSuccess;
        self.check(
            unsafe {
                ffi::ncclCommGetAsyncError(self.raw.load(Ordering::Acquire), &mut async_error)
            },
            "ncclCommGetAsyncError",
        )?;
        self.check(async_error, "NCCL asynchronous operation")
    }

    fn abort(&self) -> OpResult<()> {
        if self.raw.load(Ordering::Acquire).is_null() {
            return Ok(());
        }
        let _restore = NcclDeviceRestore {
            previous: device_utils::current_device().unwrap_or(-1),
        };
        device_utils::set_current_device(self.device_id).map_err(|error| {
            OpError::Kernel(format!(
                "set CUDA device {} during NCCL shutdown: {error}",
                self.device_id
            ))
        })?;
        let raw = self.raw.swap(ptr::null_mut(), Ordering::AcqRel);
        if raw.is_null() {
            return Ok(());
        }
        self.failed.store(true, Ordering::Release);
        // Captured graphs retain NCCL execution plans. They must be destroyed
        // before every rank collectively tears down its communicator.
        self._config.invalidate_all_graphs();
        nccl_check(unsafe { ffi::ncclCommAbort(raw) }, "ncclCommAbort")
    }
}

impl Drop for NcclCommunicator {
    fn drop(&mut self) {
        // Drop is also the failure-path cleanup, so abort outstanding device
        // work instead of entering the normal quiescent destroy path. Worker
        // normal teardown calls `shutdown_comm` collectively before rank
        // threads are joined; this is only the idempotent fallback.
        if let Err(error) = self.abort() {
            tracing::error!(
                rank = self.rank,
                size = self.size,
                device = self.device_id,
                error = %error,
                "ncclCommAbort failed"
            );
        }
    }
}

impl CudaScope {
    pub fn with_tp_communicator(mut self, comm: Arc<NcclCommunicator>) -> OpResult<Self> {
        if self.topology.tp != comm.rank_pair() {
            return Err(OpError::Shape(format!(
                "TP communicator rank {}/{} does not match scope topology {}/{}",
                comm.rank, comm.size, self.topology.tp.rank, self.topology.tp.size
            )));
        }
        if self.device.device_id != comm.device_id {
            return Err(OpError::Shape(format!(
                "TP communicator device {} does not match scope device {}",
                comm.device_id, self.device.device_id
            )));
        }
        if !Arc::ptr_eq(&self.device.config, &comm._config) {
            return Err(OpError::Shape(
                "TP communicator and scope use different CUDA configs".into(),
            ));
        }
        self.tp_comm = Some(comm);
        Ok(self)
    }
}

impl CollectiveOps for Cuda {
    type Comm = NcclCommunicator;

    fn comm(scope: &Self::Scope, axis: CommAxis) -> Option<&Self::Comm> {
        match axis {
            CommAxis::Tp => scope.tp_comm.as_deref(),
            CommAxis::Pp | CommAxis::Dp => None,
        }
    }

    fn all_reduce<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        op: ReduceOp,
        buf: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        require_scope_tensor(scope, buf, "all_reduce")?;
        let Some(comm) = require_comm(scope, axis, "all_reduce")? else {
            return Ok(());
        };
        require_contiguous(buf, "all_reduce")?;
        let dtype = nccl_dtype::<T>()?;
        let _guard = scope.enter();
        let _issue = comm.lock()?;
        comm.check(
            unsafe {
                ffi::ncclAllReduce(
                    buf.data_ptr().cast(),
                    buf.data_ptr_mut().cast(),
                    buf.numel(),
                    dtype,
                    nccl_reduce_op(op),
                    comm.raw.load(Ordering::Acquire),
                    scope.stream.0,
                )
            },
            "ncclAllReduce",
        )
    }

    fn all_gather<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        dim: usize,
        shard: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        require_scope_tensor(scope, shard, "all_gather shard")?;
        require_scope_tensor(scope, out, "all_gather output")?;
        let Some(comm) = require_comm(scope, axis, "all_gather")? else {
            return out.copy_from(shard);
        };
        let (outer, local_count) = validate_gather_shape(shard, out, dim, comm.size)?;
        require_contiguous(out, "all_gather output")?;
        require_gather_inner_contiguous(shard, dim)?;
        validate_gather_alias(shard, out, dim, outer, local_count, comm.rank, comm.size)?;
        if outer == 0 || local_count == 0 {
            return Ok(());
        }
        let _guard = scope.enter();
        let _issue = comm.lock()?;
        let dtype = nccl_dtype::<T>()?;
        if outer > 1 {
            comm.check(
                unsafe { ffi::ncclGroupStart() },
                "ncclGroupStart(all_gather)",
            )?;
        }
        let mut first_error = None;
        for outer_index in 0..outer {
            // The validation pass above evaluated every offset with the same
            // checked arithmetic before the NCCL group began.
            let send_offset = gather_outer_offset(shard, dim, outer_index)
                .expect("all_gather send offset was validated before ncclGroupStart");
            let send = unsafe { shard.data_ptr().add(send_offset) };
            let recv = unsafe {
                out.data_ptr_mut()
                    .add(outer_index * local_count * comm.size)
            };
            if let Err(error) = comm.check(
                unsafe {
                    ffi::ncclAllGather(
                        send.cast(),
                        recv.cast(),
                        local_count,
                        dtype,
                        comm.raw.load(Ordering::Acquire),
                        scope.stream.0,
                    )
                },
                "ncclAllGather",
            ) && first_error.is_none()
            {
                // Every rank must submit the same number of calls inside a
                // group even when one enqueue reports an error. Preserve the
                // first failure, finish the sequence, then end the group.
                first_error = Some(error);
            }
        }
        let end = if outer > 1 {
            comm.check(unsafe { ffi::ncclGroupEnd() }, "ncclGroupEnd(all_gather)")
        } else {
            Ok(())
        };
        if let Some(error) = first_error {
            Err(error)
        } else {
            end
        }
    }

    fn reduce_scatter<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        _op: ReduceOp,
        _dim: usize,
        buf: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let Some(_comm) = require_comm(scope, axis, "reduce_scatter")? else {
            return out.copy_from(buf);
        };
        Err(OpError::unsupported("cuda", "NCCL reduce_scatter"))
    }

    fn broadcast<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        root: usize,
        buf: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        require_scope_tensor(scope, buf, "broadcast")?;
        let Some(comm) = require_comm(scope, axis, "broadcast")? else {
            if root != 0 {
                return Err(OpError::Shape(format!(
                    "single-rank broadcast root must be 0, got {root}"
                )));
            }
            return Ok(());
        };
        if root >= comm.size {
            return Err(OpError::Shape(format!(
                "broadcast root {root} outside communicator size {}",
                comm.size
            )));
        }
        require_contiguous(buf, "broadcast")?;
        let dtype = nccl_dtype::<T>()?;
        let root = checked_i32(root, "broadcast root")?;
        let _guard = scope.enter();
        let _issue = comm.lock()?;
        comm.check(
            unsafe {
                ffi::ncclBroadcast(
                    buf.data_ptr().cast(),
                    buf.data_ptr_mut().cast(),
                    buf.numel(),
                    dtype,
                    root,
                    comm.raw.load(Ordering::Acquire),
                    scope.stream.0,
                )
            },
            "ncclBroadcast",
        )
    }

    fn send<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        _peer: usize,
        _buf: &Tensor<T, Self>,
    ) -> OpResult<()> {
        let _ = require_comm(scope, axis, "send")?;
        Err(OpError::unsupported("cuda", "NCCL send"))
    }

    fn recv<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        _peer: usize,
        _buf: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _ = require_comm(scope, axis, "recv")?;
        Err(OpError::unsupported("cuda", "NCCL recv"))
    }

    fn all_to_all<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        send_chunks: &[Tensor<T, Self>],
        recv_chunks: &mut [Tensor<T, Self>],
    ) -> OpResult<()> {
        let Some(_comm) = require_comm(scope, axis, "all_to_all")? else {
            if send_chunks.len() != recv_chunks.len() {
                return Err(OpError::Shape(format!(
                    "all_to_all: send_chunks={} recv_chunks={}",
                    send_chunks.len(),
                    recv_chunks.len()
                )));
            }
            for (send, recv) in send_chunks.iter().zip(recv_chunks.iter_mut()) {
                recv.copy_from(send)?;
            }
            return Ok(());
        };
        Err(OpError::unsupported("cuda", "NCCL all_to_all"))
    }

    fn barrier(scope: &Self::Scope, axis: CommAxis) -> OpResult<()> {
        if require_comm(scope, axis, "barrier")?.is_none() {
            Ok(())
        } else {
            Err(OpError::unsupported("cuda", "NCCL barrier"))
        }
    }

    fn shutdown_comm(scope: &Self::Scope, axis: CommAxis) -> OpResult<()> {
        let Some(comm) = Self::comm(scope, axis) else {
            return Ok(());
        };
        comm.abort()
    }
}

fn validate_rank(rank: usize, size: usize) -> OpResult<()> {
    if size == 0 || rank >= size {
        return Err(OpError::Shape(format!(
            "invalid NCCL rank {rank} for communicator size {size}"
        )));
    }
    Ok(())
}

fn checked_i32(value: usize, what: &str) -> OpResult<i32> {
    i32::try_from(value).map_err(|_| OpError::Shape(format!("{what} {value} exceeds i32")))
}

fn nccl_check(result: ffi::ncclResult_t, what: &str) -> OpResult<()> {
    if result == ffi::ncclResult_t::ncclSuccess {
        Ok(())
    } else {
        Err(nccl_error(result, what))
    }
}

fn nccl_error(result: ffi::ncclResult_t, what: &str) -> OpError {
    OpError::Kernel(format!("{what} failed: {}", nccl_error_string(result)))
}

fn require_nccl_version() -> OpResult<()> {
    let mut version = 0;
    nccl_check(
        unsafe { ffi::ncclGetVersion(&mut version) },
        "ncclGetVersion",
    )?;
    if version < MIN_NCCL_VERSION {
        return Err(OpError::Kernel(format!(
            "NCCL {version} is too old; RustInfer requires version code >= {MIN_NCCL_VERSION} (2.24.3)"
        )));
    }
    Ok(())
}

fn nccl_error_string(result: ffi::ncclResult_t) -> String {
    let ptr = unsafe { ffi::ncclGetErrorString(result) };
    if ptr.is_null() {
        format!("NCCL error {result:?}")
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

fn nccl_dtype<T: Dtype>() -> OpResult<ffi::ncclDataType_t> {
    match T::ID {
        DTypeId::F32 => Ok(ffi::ncclDataType_t::ncclFloat32),
        DTypeId::F16 => Ok(ffi::ncclDataType_t::ncclFloat16),
        DTypeId::BF16 => Ok(ffi::ncclDataType_t::ncclBfloat16),
        DTypeId::F8E4M3 => Ok(ffi::ncclDataType_t::ncclFloat8e4m3),
        DTypeId::F8E5M2 => Ok(ffi::ncclDataType_t::ncclFloat8e5m2),
        DTypeId::I32 => Ok(ffi::ncclDataType_t::ncclInt32),
        DTypeId::I8 => Ok(ffi::ncclDataType_t::ncclInt8),
        DTypeId::U8 => Ok(ffi::ncclDataType_t::ncclUint8),
        DTypeId::U32 => Ok(ffi::ncclDataType_t::ncclUint32),
        _ => Err(OpError::unsupported("cuda", "NCCL custom dtype")),
    }
}

fn nccl_reduce_op(op: ReduceOp) -> ffi::ncclRedOp_t {
    match op {
        ReduceOp::Sum => ffi::ncclRedOp_t::ncclSum,
        ReduceOp::Max => ffi::ncclRedOp_t::ncclMax,
        ReduceOp::Min => ffi::ncclRedOp_t::ncclMin,
        ReduceOp::Avg => ffi::ncclRedOp_t::ncclAvg,
    }
}

fn require_comm<'a>(
    scope: &'a CudaScope,
    axis: CommAxis,
    op: &str,
) -> OpResult<Option<&'a NcclCommunicator>> {
    let topology = scope.topology();
    let size = topology.group_size(axis);
    if size == 1 {
        return Ok(None);
    }
    let comm = Cuda::comm(scope, axis).ok_or_else(|| {
        OpError::Kernel(format!(
            "{op} requires a {axis:?} communicator for group size {size}"
        ))
    })?;
    let expected = RankPair {
        rank: topology.rank_in(axis),
        size,
    };
    if comm.rank_pair() != expected {
        return Err(OpError::Shape(format!(
            "{op}: communicator rank {}/{} does not match {axis:?} topology {}/{}",
            comm.rank, comm.size, expected.rank, expected.size
        )));
    }
    comm.ensure_healthy(op)?;
    Ok(Some(comm))
}

fn require_contiguous<T: Dtype>(tensor: &Tensor<T, Cuda>, what: &str) -> OpResult<()> {
    if tensor.is_contiguous() {
        Ok(())
    } else {
        Err(OpError::Shape(format!(
            "{what} requires a contiguous tensor, got shape {:?}",
            tensor.shape().as_slice()
        )))
    }
}

fn validate_gather_shape<T: Dtype>(
    shard: &Tensor<T, Cuda>,
    out: &Tensor<T, Cuda>,
    dim: usize,
    size: usize,
) -> OpResult<(usize, usize)> {
    let shard_shape = shard.shape().as_slice();
    let out_shape = out.shape().as_slice();
    if dim >= shard_shape.len() || shard_shape.len() != out_shape.len() {
        return Err(OpError::Shape(format!(
            "all_gather dim {dim} invalid for shapes {:?} -> {:?}",
            shard_shape, out_shape
        )));
    }
    for axis in 0..shard_shape.len() {
        let expected = if axis == dim {
            shard_shape[axis]
                .checked_mul(size)
                .ok_or_else(|| OpError::Shape("all_gather output dimension overflows".into()))?
        } else {
            shard_shape[axis]
        };
        if out_shape[axis] != expected {
            return Err(OpError::Shape(format!(
                "all_gather shape mismatch at dim {axis}: expected {expected}, got {}",
                out_shape[axis]
            )));
        }
    }
    let outer = checked_shape_product(&shard_shape[..dim], "all_gather outer size")?;
    let local_count = checked_shape_product(&shard_shape[dim..], "all_gather send count")?;
    Ok((outer, local_count))
}

/// NCCL receives one packed block per call. Dimensions at and after the gather
/// axis must therefore form a contiguous block, while dimensions before it may
/// carry gaps. The latter is what permits a `[rows, local_vocab]` column view
/// into a packed `[rows, global_vocab]` logits tensor.
fn require_gather_inner_contiguous<T: Dtype>(shard: &Tensor<T, Cuda>, dim: usize) -> OpResult<()> {
    if shard.numel() == 0 {
        return Ok(());
    }
    let shape = shard.shape().as_slice();
    let strides = shard.strides().as_slice();
    let mut expected = 1usize;
    for axis in (dim..shape.len()).rev() {
        let extent = shape[axis];
        if extent > 1 && strides[axis] != expected {
            return Err(OpError::Shape(format!(
                "all_gather shard dimensions {dim}.. must be contiguous, got shape {:?} strides {:?}",
                shape, strides
            )));
        }
        expected = expected
            .checked_mul(extent)
            .ok_or_else(|| OpError::Shape("all_gather shard contiguous span overflows".into()))?;
    }
    Ok(())
}

/// Element offset for one NCCL send block. A contiguous shard naturally
/// produces `row * local_count`; a column view uses its real outer stride (for
/// example `row * global_vocab`).
fn gather_outer_offset<T: Dtype>(
    shard: &Tensor<T, Cuda>,
    dim: usize,
    flat_index: usize,
) -> OpResult<usize> {
    let shape = shard.shape().as_slice();
    let strides = shard.strides().as_slice();
    let mut remainder = flat_index;
    let mut offset = 0usize;
    for axis in (0..dim).rev() {
        let extent = shape[axis];
        if extent == 0 {
            return Err(OpError::Shape(
                "all_gather outer index addresses an empty dimension".into(),
            ));
        }
        let index = remainder % extent;
        remainder /= extent;
        let term = index
            .checked_mul(strides[axis])
            .ok_or_else(|| OpError::Shape("all_gather shard outer offset overflows".into()))?;
        offset = offset
            .checked_add(term)
            .ok_or_else(|| OpError::Shape("all_gather shard outer offset overflows".into()))?;
    }
    Ok(offset)
}

/// NCCL permits an in-place AllGather only when the send block is exactly the
/// calling rank's slot inside the receive block. Tensor views can alias through
/// their shared `Storage`, so reject every other overlap before entering the
/// collective.
fn validate_gather_alias<T: Dtype>(
    shard: &Tensor<T, Cuda>,
    out: &Tensor<T, Cuda>,
    dim: usize,
    outer: usize,
    local_count: usize,
    rank: usize,
    size: usize,
) -> OpResult<()> {
    let aliases_output = Arc::ptr_eq(shard.storage(), out.storage()) && local_count != 0;
    for outer_index in 0..outer {
        let send_offset = gather_outer_offset(shard, dim, outer_index)?;
        if aliases_output {
            validate_gather_alias_offset(
                shard.offset_elems(),
                out.offset_elems(),
                out.numel(),
                send_offset,
                local_count,
                rank,
                size,
                outer_index,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_gather_alias_offset(
    shard_offset: usize,
    out_offset: usize,
    out_numel: usize,
    send_offset: usize,
    local_count: usize,
    rank: usize,
    size: usize,
    outer_index: usize,
) -> OpResult<()> {
    let global_count = local_count
        .checked_mul(size)
        .ok_or_else(|| OpError::Shape("all_gather receive count overflows".into()))?;
    let out_end = out_offset
        .checked_add(out_numel)
        .ok_or_else(|| OpError::Shape("all_gather output range overflows".into()))?;

    let send_start = shard_offset
        .checked_add(send_offset)
        .ok_or_else(|| OpError::Shape("all_gather send range overflows".into()))?;
    let send_end = send_start
        .checked_add(local_count)
        .ok_or_else(|| OpError::Shape("all_gather send range overflows".into()))?;
    let overlaps_output = send_start < out_end && out_offset < send_end;
    if !overlaps_output {
        return Ok(());
    }

    let expected = outer_index
        .checked_mul(global_count)
        .and_then(|offset| {
            rank.checked_mul(local_count)
                .and_then(|slot| offset.checked_add(slot))
        })
        .and_then(|offset| out_offset.checked_add(offset))
        .ok_or_else(|| OpError::Shape("all_gather in-place offset overflows".into()))?;
    if send_start != expected {
        return Err(OpError::Shape(format!(
            "all_gather shard overlaps its output at outer index {outer_index}; NCCL in-place requires rank {rank}/{size} send data to start at element {expected}, got {send_start}"
        )));
    }
    Ok(())
}

fn checked_shape_product(shape: &[usize], what: &str) -> OpResult<usize> {
    shape.iter().try_fold(1usize, |product, &extent| {
        product
            .checked_mul(extent)
            .ok_or_else(|| OpError::Shape(format!("{what} overflows")))
    })
}

#[cfg(test)]
mod tests {
    use super::validate_gather_alias_offset;

    #[test]
    fn gather_alias_accepts_exact_rank_slots_for_strided_rows() {
        for (outer_index, send_offset) in [0, 4, 8].into_iter().enumerate() {
            validate_gather_alias_offset(2, 0, 12, send_offset, 2, 1, 2, outer_index).unwrap();
        }
    }

    #[test]
    fn gather_alias_rejects_any_other_output_overlap() {
        let error = validate_gather_alias_offset(0, 0, 12, 0, 2, 1, 2, 0)
            .expect_err("rank 1 may not send from rank 0's receive slot");
        assert!(
            error
                .to_string()
                .contains("NCCL in-place requires rank 1/2")
        );
    }

    #[test]
    fn gather_alias_allows_non_overlapping_storage_regions() {
        validate_gather_alias_offset(32, 0, 12, 0, 2, 1, 2, 0).unwrap();
    }
}
