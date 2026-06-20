use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

use infer_protocol::scheduler_to_worker_control::LoadModel;

use crate::device::MemoryPort;
use crate::error::OpResult;

/// Communication axis for tensor/pipeline/data/expert parallelism. Lives here
/// (next to `TopologyShape`, its only structural user) rather than with the
/// `CollectiveOps` trait, so the exec vocabulary has no upward dependency on the
/// op-port layer. The `CollectiveOps` trait re-exports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommAxis {
    Tp,
    Pp,
    Dp,
    Ep,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DeviceId(pub i32);

pub trait ExecDevice: MemoryPort {
    type Scope: ExecScope<Device = Self>;

    fn device_id(&self) -> DeviceId {
        DeviceId(<Self as crate::device::Device>::device_id(self))
    }

    fn enter_device(&self) -> DeviceId {
        <Self as ExecDevice>::device_id(self)
    }

    fn restore_device(&self, _previous: DeviceId) {}
}

pub trait ExecHostDevice: ExecDevice {}

pub trait ExecScope: Send + Sync + Sized + 'static {
    type Device: ExecDevice<Scope = Self>;
    type Stream: Stream;

    fn device(&self) -> &Self::Device;
    fn enter(&self) -> ActiveGuard<'_, Self::Device>;
    fn stream(&self) -> &Self::Stream;
    fn rank(&self) -> Rank;
    fn topology(&self) -> TopologyShape;
    fn quant_tier(&self) -> QuantTier;
    fn workspace(&self) -> &Workspace<Self::Device>;
    fn supports_graphs(&self) -> bool {
        false
    }
    fn synchronize(&self) -> OpResult<()>;
}

pub trait Stream: Send + Sync + 'static {}

pub struct ActiveGuard<'a, D: ExecDevice> {
    _scope: &'a D::Scope,
    _prev_device: DeviceId,
    _not_send: PhantomData<Rc<()>>,
}

impl<'a, D: ExecDevice> ActiveGuard<'a, D> {
    pub fn new(scope: &'a D::Scope, prev_device: DeviceId) -> Self {
        Self {
            _scope: scope,
            _prev_device: prev_device,
            _not_send: PhantomData,
        }
    }
}

impl<D: ExecDevice> Drop for ActiveGuard<'_, D> {
    fn drop(&mut self) {
        self._scope.device().restore_device(self._prev_device);
    }
}

#[derive(Debug)]
pub struct Workspace<D: ExecDevice> {
    _ptr: Option<NonNull<u8>>,
    _size: usize,
    _d: PhantomData<D>,
}

unsafe impl<D: ExecDevice> Send for Workspace<D> {}
unsafe impl<D: ExecDevice> Sync for Workspace<D> {}

impl<D: ExecDevice> Workspace<D> {
    pub const fn empty() -> Self {
        Self {
            _ptr: None,
            _size: 0,
            _d: PhantomData,
        }
    }

    pub fn from_raw(ptr: Option<NonNull<u8>>, size: usize) -> Self {
        Self {
            _ptr: ptr,
            _size: if ptr.is_some() { size } else { 0 },
            _d: PhantomData,
        }
    }

    pub fn ptr(&self) -> Option<NonNull<u8>> {
        self._ptr
    }

    pub fn size(&self) -> usize {
        self._size
    }
}

pub struct StepCtx<'a, D: ExecDevice> {
    scope: &'a D::Scope,
    plan: &'a crate::plan::BatchPlan,
    _marker: PhantomData<D>,
}

impl<'a, D: ExecDevice> StepCtx<'a, D> {
    pub fn new(scope: &'a D::Scope, plan: &'a crate::plan::BatchPlan) -> Self {
        Self {
            scope,
            plan,
            _marker: PhantomData,
        }
    }

    pub fn scope(&self) -> &D::Scope {
        self.scope
    }

    pub fn plan(&self) -> &crate::plan::BatchPlan {
        self.plan
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rank {
    pub tp_rank: usize,
    pub pp_rank: usize,
    pub dp_rank: usize,
    pub node_rank: usize,
    pub world_rank: usize,
}

impl Rank {
    pub const SINGLE: Rank = Rank {
        tp_rank: 0,
        pp_rank: 0,
        dp_rank: 0,
        node_rank: 0,
        world_rank: 0,
    };
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RankPair {
    pub rank: usize,
    pub size: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TopologyShape {
    pub tp: RankPair,
    pub pp: RankPair,
    pub dp: RankPair,
    pub node: RankPair,
}

impl TopologyShape {
    pub const SINGLE: TopologyShape = TopologyShape {
        tp: RankPair { rank: 0, size: 1 },
        pp: RankPair { rank: 0, size: 1 },
        dp: RankPair { rank: 0, size: 1 },
        node: RankPair { rank: 0, size: 1 },
    };

    pub fn from_load_model(load: &LoadModel) -> Self {
        Self {
            tp: RankPair {
                rank: load.tp_rank,
                size: load.tp_size.max(1),
            },
            pp: RankPair {
                rank: load.pp_rank,
                size: load.pp_size.max(1),
            },
            dp: RankPair { rank: 0, size: 1 },
            node: RankPair { rank: 0, size: 1 },
        }
    }

    pub fn world_size(&self) -> usize {
        self.tp.size * self.pp.size * self.dp.size * self.node.size
    }

    pub fn rank_in(&self, axis: CommAxis) -> usize {
        match axis {
            CommAxis::Tp => self.tp.rank,
            CommAxis::Pp => self.pp.rank,
            CommAxis::Dp => self.dp.rank,
            CommAxis::Ep => self.tp.rank,
        }
    }

    pub fn group_size(&self, axis: CommAxis) -> usize {
        match axis {
            CommAxis::Tp => self.tp.size,
            CommAxis::Pp => self.pp.size,
            CommAxis::Dp => self.dp.size,
            CommAxis::Ep => self.tp.size,
        }
    }

    pub fn is_pp_first(&self) -> bool {
        self.pp.rank == 0
    }

    pub fn is_pp_last(&self) -> bool {
        self.pp.rank + 1 == self.pp.size
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuantTier {
    None,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MaskHandle(pub u64);

#[derive(Clone, Copy, Debug, Default)]
pub struct HostStream;

impl Stream for HostStream {}

#[derive(Debug)]
pub struct HostScope<D: ExecDevice> {
    device: D,
    stream: HostStream,
    rank: Rank,
    topology: TopologyShape,
    quant_tier: QuantTier,
    workspace: Workspace<D>,
}

impl<D: ExecDevice> HostScope<D> {
    pub fn new(device: D) -> Self {
        Self {
            device,
            stream: HostStream,
            rank: Rank::SINGLE,
            topology: TopologyShape::SINGLE,
            quant_tier: QuantTier::None,
            workspace: Workspace::empty(),
        }
    }

    pub fn with_topology(mut self, topology: TopologyShape) -> Self {
        self.rank = Rank {
            tp_rank: topology.tp.rank,
            pp_rank: topology.pp.rank,
            dp_rank: topology.dp.rank,
            node_rank: topology.node.rank,
            world_rank: 0,
        };
        self.topology = topology;
        self
    }

    pub fn device(&self) -> &D {
        &self.device
    }
}

impl<D> ExecScope for HostScope<D>
where
    D: ExecDevice<Scope = HostScope<D>>,
{
    type Device = D;
    type Stream = HostStream;

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn enter(&self) -> ActiveGuard<'_, Self::Device> {
        let previous = self.device.enter_device();
        ActiveGuard::new(self, previous)
    }

    fn stream(&self) -> &Self::Stream {
        &self.stream
    }

    fn rank(&self) -> Rank {
        self.rank
    }

    fn topology(&self) -> TopologyShape {
        self.topology
    }

    fn quant_tier(&self) -> QuantTier {
        self.quant_tier
    }

    fn workspace(&self) -> &Workspace<Self::Device> {
        &self.workspace
    }

    fn synchronize(&self) -> OpResult<()> {
        self.device.synchronize()
    }
}
