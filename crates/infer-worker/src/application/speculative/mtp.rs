//! Compatibility name for the Qwen3.5 MTP proposer.
pub type MtpProposer<T, D, H> =
    super::conditioned::ConditionedProposer<T, D, crate::components::mtp::MtpHead<T, D, H>>;
