use crate::ports::collective::CollectiveOps;
use crate::ports::fused_ops::FusedOps;
use crate::ports::pipeline_ops::DecodePipelineOps;

pub trait LlmBackend: FusedOps + CollectiveOps + DecodePipelineOps {}
impl<D: FusedOps + CollectiveOps + DecodePipelineOps> LlmBackend for D {}
