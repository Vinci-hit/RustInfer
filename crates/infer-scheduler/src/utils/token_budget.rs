//! Token budget calculator.

/// Token and sequence budget for one scheduling iteration.
#[derive(Debug, Clone, Copy)]
pub struct TokenBudget {
    /// Max total tokens (prefill + decode) this iteration.
    pub max_tokens: usize,
    /// Max total sequences (prefill + decode) this iteration.
    pub max_seqs: usize,
}

impl TokenBudget {
    /// Compute the remaining budget after accounting for running sequences.
    pub fn remaining(&self, running_seqs: usize, running_decode_tokens: usize) -> TokenBudget {
        TokenBudget {
            max_tokens: self.max_tokens.saturating_sub(running_decode_tokens),
            max_seqs: self.max_seqs.saturating_sub(running_seqs),
        }
    }

    /// Whether there's any budget left.
    pub fn has_budget(&self) -> bool {
        self.max_tokens > 0 && self.max_seqs > 0
    }
}
