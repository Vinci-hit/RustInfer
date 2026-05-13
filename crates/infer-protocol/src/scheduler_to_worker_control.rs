use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerHello {
    pub protocol_version: u32,
    pub heartbeat_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModel {
    pub model_instance_id: String,
    pub model_path: String,
    pub model_type: String,
    pub device: String,
    pub max_batch_tokens: usize,
    pub max_batch_seqs: usize,
    pub max_model_len: usize,
    pub mem_fraction_static: f32,
    pub tp_rank: usize,
    pub tp_size: usize,
    pub pp_rank: usize,
    pub pp_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerControlMessage {
    Hello(SchedulerHello),
    LoadModel(LoadModel),
}
