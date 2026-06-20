use crate::domain::ports::collective::CollectiveOps;
use crate::domain::ports::diffusion_ops::DiffusionOps;
use crate::domain::ports::fused_ops::FusedOps;

pub trait LlmBackend: FusedOps + CollectiveOps {}
impl<D: FusedOps + CollectiveOps> LlmBackend for D {}

pub trait DiffusionBackend: DiffusionOps + CollectiveOps {}
impl<D: DiffusionOps + CollectiveOps> DiffusionBackend for D {}

pub trait Backend: LlmBackend + DiffusionBackend {}
impl<D: LlmBackend + DiffusionBackend> Backend for D {}
