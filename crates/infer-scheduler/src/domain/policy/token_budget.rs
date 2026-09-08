//! Token budget calculator.

/// Token and sequence budget for one scheduling iteration.
#[derive(Debug, Clone, Copy)]
pub struct TokenBudget {
    /// Max total tokens (prefill + decode) this iteration.
    pub max_tokens: usize,
    /// Available KV slots, including future decode reservations for new requests.
    pub max_kv_tokens: usize,
    /// Max total sequences (prefill + decode) this iteration.
    pub max_seqs: usize,
}

impl TokenBudget {
    /// Whether there's any budget left.
    pub fn has_budget(&self) -> bool {
        self.max_tokens > 0 && self.max_seqs > 0
    }
}
