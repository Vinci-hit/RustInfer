use crate::ports::OpResult;
use infer_core::dtype::Dtype;
use infer_core::exec::ExecDevice as Device;
use infer_core::tensor::Tensor;

// CommAxis now lives in infer-core's exec vocabulary (next to TopologyShape);
// re-exported here so `ports::collective::CommAxis` / `ports::CommAxis` resolve.
pub use infer_core::exec::CommAxis;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Avg,
}

#[derive(Debug, Clone, Copy)]
pub struct SingleRankComm;

pub trait CollectiveOps: Device {
    type Comm: Send + Sync;

    fn comm(scope: &Self::Scope, axis: CommAxis) -> Option<&Self::Comm>;

    fn all_reduce<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        op: ReduceOp,
        buf: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    /// Gather rank shards along `dim`. `shard` may be an inner-contiguous,
    /// outer-strided view into the calling rank's slot of `out`; backends must
    /// preserve this legal in-place collective layout.
    fn all_gather<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        dim: usize,
        shard: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    fn reduce_scatter<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        op: ReduceOp,
        dim: usize,
        buf: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    fn broadcast<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        root: usize,
        buf: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    fn send<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        peer: usize,
        buf: &Tensor<T, Self>,
    ) -> OpResult<()>;

    fn recv<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        peer: usize,
        buf: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    fn all_to_all<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        send_chunks: &[Tensor<T, Self>],
        recv_chunks: &mut [Tensor<T, Self>],
    ) -> OpResult<()>;

    fn barrier(scope: &Self::Scope, axis: CommAxis) -> OpResult<()>;

    /// Release one communication axis after all in-flight operations have
    /// completed. Multi-rank runtimes call this collectively before rank
    /// threads are joined; single-rank/backends without a communicator need no
    /// action.
    fn shutdown_comm(_scope: &Self::Scope, _axis: CommAxis) -> OpResult<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardSpec {
    Replicated,
    ColumnParallel { dim: usize },
    RowParallel { dim: usize },
    VocabParallel { dim: usize },
}

pub trait ShardedLoad: Device {
    fn load_shard<T: Dtype>(
        &self,
        rank: infer_core::exec::Rank,
        name: &str,
        spec: ShardSpec,
    ) -> OpResult<Tensor<T, Self>>;
}
