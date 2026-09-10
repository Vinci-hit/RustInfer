//! Beam requests use the normal frontend lifecycle and a dedicated worker command.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamOptions {
    pub width: usize,
    pub length_penalty: f64,
}

impl BeamOptions {
    pub fn validate(&self, capacity: usize) -> Result<(), String> {
        if self.width == 0 || self.width > capacity {
            return Err(format!("beam_width must be between 1 and {capacity}"));
        }
        if !self.length_penalty.is_finite() || self.length_penalty < 0.0 {
            return Err("length_penalty must be finite and non-negative".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamCommand {
    pub sequence_id: u64,
    pub free_indices: Vec<u32>,
    pub request: crate::server_to_scheduler::InferenceRequest,
}

/// Tagged to remain distinguishable from legacy positional StepOutput frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BeamOutput {
    Finished {
        sequence_id: u64,
        allocated_kv_tokens: Option<u32>,
        response: crate::scheduler_to_server::InferenceResponse,
    },
}

impl BeamOutput {
    pub fn with_kv_usage(mut self, tokens: u32) -> Self {
        let Self::Finished {
            allocated_kv_tokens,
            ..
        } = &mut self;
        *allocated_kv_tokens = Some(tokens);
        self
    }
}
