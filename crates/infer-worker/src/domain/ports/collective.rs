use crate::domain::dtype::Dtype;
use crate::domain::exec::Device;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommAxis {
    Tp,
    Pp,
    Dp,
    Ep,
}

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
}

impl<D: Device> CollectiveOps for D {
    type Comm = SingleRankComm;

    fn comm(_scope: &Self::Scope, _axis: CommAxis) -> Option<&Self::Comm> {
        None
    }

    fn all_reduce<T: Dtype>(
        _scope: &Self::Scope,
        _axis: CommAxis,
        _op: ReduceOp,
        _buf: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        Ok(())
    }

    fn all_gather<T: Dtype>(
        _scope: &Self::Scope,
        _axis: CommAxis,
        _dim: usize,
        shard: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        out.copy_from(shard)
    }

    fn reduce_scatter<T: Dtype>(
        _scope: &Self::Scope,
        _axis: CommAxis,
        _op: ReduceOp,
        _dim: usize,
        buf: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        out.copy_from(buf)
    }

    fn broadcast<T: Dtype>(
        _scope: &Self::Scope,
        _axis: CommAxis,
        _root: usize,
        _buf: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        Ok(())
    }

    fn send<T: Dtype>(
        _scope: &Self::Scope,
        _axis: CommAxis,
        _peer: usize,
        buf: &Tensor<T, Self>,
    ) -> OpResult<()> {
        Err(OpError::unsupported(buf.device().name(), "send"))
    }

    fn recv<T: Dtype>(
        _scope: &Self::Scope,
        _axis: CommAxis,
        _peer: usize,
        buf: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        Err(OpError::unsupported(buf.device().name(), "recv"))
    }

    fn all_to_all<T: Dtype>(
        _scope: &Self::Scope,
        _axis: CommAxis,
        send_chunks: &[Tensor<T, Self>],
        recv_chunks: &mut [Tensor<T, Self>],
    ) -> OpResult<()> {
        if send_chunks.len() != recv_chunks.len() {
            return Err(OpError::Shape(format!(
                "all_to_all: send_chunks={} recv_chunks={}",
                send_chunks.len(),
                recv_chunks.len()
            )));
        }
        for (src, dst) in send_chunks.iter().zip(recv_chunks.iter_mut()) {
            dst.copy_from(src)?;
        }
        Ok(())
    }

    fn barrier(_scope: &Self::Scope, _axis: CommAxis) -> OpResult<()> {
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
        rank: crate::domain::exec::Rank,
        name: &str,
        spec: ShardSpec,
    ) -> OpResult<Tensor<T, Self>>;
}
