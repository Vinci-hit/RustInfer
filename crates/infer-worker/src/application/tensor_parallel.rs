//! CUDA resources for the tensor-parallel ranks owned by one worker process.
//!
//! The constructor implemented today is deliberately single-process: it uses
//! `ncclCommInitAll` and owns the complete global TP group.  Callers consume a
//! placement plus rank resources, so a future distributed rendezvous can add a
//! second constructor without changing model loading or the runtime peer loop.

use std::sync::Arc;
use std::time::{Duration, Instant};

use infer_core::ports::{OpError, OpResult};

use crate::domain::TensorParallelPlacement;
use crate::infrastructure::cuda::{Cuda, CudaMemoryPlan, NcclCommunicator};

use super::runtime::RuntimePeerWatchdog;

#[derive(Debug)]
pub struct TpRankResource {
    pub global_rank: usize,
    pub global_size: usize,
    pub local_rank: usize,
    pub cuda: Cuda,
    pub communicator: Option<Arc<NcclCommunicator>>,
}

#[derive(Debug)]
pub struct LocalTpBootstrap {
    placement: TensorParallelPlacement,
    ranks: Vec<TpRankResource>,
}

impl LocalTpBootstrap {
    /// Build every rank of one TP group inside the current process.
    pub fn single_process(
        global_size: usize,
        base_device_id: i32,
        memory_plan: CudaMemoryPlan,
        startup_timeout: Duration,
    ) -> OpResult<Self> {
        let placement = TensorParallelPlacement::single_process(global_size)?;
        let mut devices = Vec::with_capacity(placement.local_rank_count());
        for local_rank in 0..placement.local_rank_count() {
            let global_rank = placement.global_rank(local_rank)?;
            let device_offset = i32::try_from(local_rank).map_err(|_| {
                OpError::Shape(format!(
                    "local TP rank {local_rank} does not fit a CUDA device id"
                ))
            })?;
            let device_id = base_device_id.checked_add(device_offset).ok_or_else(|| {
                OpError::Shape(format!(
                    "CUDA device id overflow: base={base_device_id} local TP rank={local_rank}"
                ))
            })?;
            let cuda = Cuda::with_memory_plan(device_id, memory_plan).map_err(|error| {
                OpError::Kernel(format!(
                    "create CUDA device for global TP rank {global_rank}/{global_size} on cuda:{device_id}: {error}"
                ))
            })?;
            devices.push(cuda);
        }

        let communicators =
            Self::initialize_communicators(&devices, placement.global_size(), startup_timeout)?;
        if !communicators.is_empty() && communicators.len() != devices.len() {
            return Err(OpError::Kernel(format!(
                "TP communicator/device mismatch: {} communicators for {} devices",
                communicators.len(),
                devices.len()
            )));
        }

        let ranks = devices
            .into_iter()
            .enumerate()
            .map(|(local_rank, cuda)| {
                let global_rank = placement
                    .global_rank(local_rank)
                    .expect("validated local TP rank");
                let communicator = if communicators.is_empty() {
                    None
                } else {
                    Some(communicators[local_rank].clone())
                };
                TpRankResource {
                    global_rank,
                    global_size: placement.global_size(),
                    local_rank,
                    cuda,
                    communicator,
                }
            })
            .collect();

        Ok(Self { placement, ranks })
    }

    fn initialize_communicators(
        devices: &[Cuda],
        global_size: usize,
        startup_timeout: Duration,
    ) -> OpResult<Vec<Arc<NcclCommunicator>>> {
        if global_size == 1 {
            return Ok(Vec::new());
        }
        if startup_timeout.is_zero() {
            return Err(OpError::Shape(
                "TP startup timeout must be greater than zero".into(),
            ));
        }

        let watchdog = RuntimePeerWatchdog::fail_stop()?;
        let deadline = Instant::now()
            .checked_add(startup_timeout)
            .ok_or_else(|| OpError::Fatal("NCCL initialization deadline overflowed".into()))?;
        watchdog.arm(0, "NCCL group initialization", deadline)?;
        let initialized = NcclCommunicator::init_all(devices);
        watchdog.disarm(0)?;
        drop(watchdog);
        initialized
    }

    pub const fn placement(&self) -> TensorParallelPlacement {
        self.placement
    }

    pub fn devices(&self) -> Vec<Cuda> {
        self.ranks.iter().map(|rank| rank.cuda.clone()).collect()
    }

    pub fn into_ranks(self) -> Vec<TpRankResource> {
        self.ranks
    }
}
