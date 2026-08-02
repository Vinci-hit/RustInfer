use crate::ports::collective::CollectiveOps;
use crate::ports::fused_ops::FusedOps;
use crate::ports::pipeline_ops::DecodePipelineOps;
use crate::ports::vocab_ops::VocabOps;

pub trait LlmBackend: FusedOps + CollectiveOps + DecodePipelineOps + VocabOps {}
impl<D: FusedOps + CollectiveOps + DecodePipelineOps + VocabOps> LlmBackend for D {}
