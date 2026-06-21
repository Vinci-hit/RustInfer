//! Prefix-cache match — a pure domain value object.
//!
//! Describes how much of a request's prompt was satisfied by an existing
//! prefix in the KV cache. This is a domain concept (it parameterizes the
//! `Queued -> Prefilling` transition); the infrastructure `RadixTree`
//! *produces* it but does not own its definition. Keeping it here keeps the
//! dependency arrow pointing inward (infra -> domain), per the hexagonal
//! layering the crate documents.

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
