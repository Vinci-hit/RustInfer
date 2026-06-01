//! Core traits for cache management.

/// Result of a prefix match operation.
#[derive(Debug, Clone)]
pub struct PrefixMatch {
    /// Number of tokens that hit the cache (always block-aligned in paged mode).
    pub num_cached_tokens: usize,
}

impl PrefixMatch {
    /// No prefix match found.
    pub fn none() -> Self {
        Self {
            num_cached_tokens: 0,
        }
    }
}
