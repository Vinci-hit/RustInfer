use crate::ports::collective::CollectiveOps;
use crate::ports::fused_ops::FusedOps;

pub trait LlmBackend: FusedOps + CollectiveOps {}
impl<D: FusedOps + CollectiveOps> LlmBackend for D {}
