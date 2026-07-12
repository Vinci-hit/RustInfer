//! Cache-management helper types.
//!
//! `PrefixMatch` is a pure domain value object; its canonical definition lives
//! in [`crate::domain::prefix`]. It is re-exported here so existing
//! `infrastructure::kv_cache::traits::PrefixMatch` import paths keep resolving,
//! while the dependency arrow now points infra -> domain (not the reverse).

pub use crate::domain::prefix::PrefixMatch;
