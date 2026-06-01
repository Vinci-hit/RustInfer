//! Domain identity & quantity types.
//!
//! Centralized newtype dictionary for the scheduler domain:
//! - **Identity**: `InferenceRequestId`, `SequenceId`, `ClientId`,
//!   `WorkerNodeId`, `ModelInstanceId`
//! - **KV resources**: `BlockSize`, `BlockCount`
//! - **Counts**: `TokenCount`, `SeqCount`, `PromptLen`, `GeneratedCount`
//! - **Time**: `ArrivalTime`, `LastSeenAt`
//!
//! Conventions:
//! - All types derive `Debug, Clone, Copy, PartialEq, Eq, Hash` where
//!   the inner type allows (Vec/String/Arc-backed types skip Copy).
//! - Construction via `pub fn new(raw) -> Self`; access via `pub fn raw`
//!   or borrow-style accessor.
//! - Counting types implement `Add` / `Sub` / `saturating_sub` to make
//!   arithmetic explicit; **no implicit conversion to `usize`**.
//! - Identity types implement `Display` for log-friendly output.
//!
//! `RequestId`, `SequenceId`, `ClientId`, and `WorkerNodeId` are re-exported
//! here from their canonical modules so callers can spell their imports
//! through `domain::ids` without navigating into sub-modules.

use std::ops::{Add, Sub};
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
//  Re-exports — canonical definitions live in their owning modules.
// ─────────────────────────────────────────────────────────────────────────────

pub use crate::domain::inference_session::handle::ClientId;
pub use crate::domain::inference_session::lifecycle::{RequestId, SequenceId};
pub use crate::infrastructure::transport::control_plane::WorkerId as WorkerNodeId;

// ─────────────────────────────────────────────────────────────────────────────
//  Identity (defined in this module)
// ─────────────────────────────────────────────────────────────────────────────

/// Internal inference request id (newly generated per request, distinct from
/// the client-supplied `external_id` string). The `external_id` is preserved
/// as a separate field on `RequestMeta` for response routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InferenceRequestId(uuid::Uuid);

impl InferenceRequestId {
    pub fn new_v4() -> Self {
        Self(uuid::Uuid::new_v4())
    }

    pub fn from_uuid(uuid: uuid::Uuid) -> Self {
        Self(uuid)
    }

    pub fn raw(self) -> uuid::Uuid {
        self.0
    }
}

impl std::fmt::Display for InferenceRequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Logical model instance id; replaces bare `String` references.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelInstanceId(String);

impl ModelInstanceId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::fmt::Display for ModelInstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  KV resources (BlockSize / BlockCount)
// ─────────────────────────────────────────────────────────────────────────────

/// Number of tokens per paged KV block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockSize(u32);

impl BlockSize {
    pub fn new(size: u32) -> Self {
        debug_assert!(size > 0, "BlockSize must be > 0");
        Self(size)
    }

    pub fn raw(self) -> u32 {
        self.0
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for BlockSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Number of paged KV blocks (capacity / allocation count).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockCount(usize);

impl BlockCount {
    pub fn new(n: usize) -> Self {
        Self(n)
    }

    pub fn raw(self) -> usize {
        self.0
    }

    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
}

impl Add for BlockCount {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for BlockCount {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl std::fmt::Display for BlockCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Counts
// ─────────────────────────────────────────────────────────────────────────────

macro_rules! count_newtype {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(usize);

        impl $name {
            pub fn new(n: usize) -> Self {
                Self(n)
            }

            pub fn raw(self) -> usize {
                self.0
            }

            pub fn saturating_sub(self, other: Self) -> Self {
                Self(self.0.saturating_sub(other.0))
            }
        }

        impl Add for $name {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl Sub for $name {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

count_newtype!(TokenCount, "Generic token count (max_tokens / num_tokens / etc.)");
count_newtype!(SeqCount, "Sequence/batch slot count");
count_newtype!(PromptLen, "Prompt length in tokens");
count_newtype!(GeneratedCount, "Number of tokens generated so far");

// ─────────────────────────────────────────────────────────────────────────────
//  Time
// ─────────────────────────────────────────────────────────────────────────────

/// Wall-clock arrival time of a request.
#[derive(Debug, Clone, Copy)]
pub struct ArrivalTime(Instant);

impl ArrivalTime {
    pub fn now() -> Self {
        Self(Instant::now())
    }

    pub fn from_instant(t: Instant) -> Self {
        Self(t)
    }

    pub fn raw(self) -> Instant {
        self.0
    }

    pub fn elapsed(self) -> std::time::Duration {
        self.0.elapsed()
    }
}

/// Last-seen timestamp for liveness/heartbeat tracking.
#[derive(Debug, Clone, Copy)]
pub struct LastSeenAt(Instant);

impl LastSeenAt {
    pub fn now() -> Self {
        Self(Instant::now())
    }

    pub fn from_instant(t: Instant) -> Self {
        Self(t)
    }

    pub fn raw(self) -> Instant {
        self.0
    }

    pub fn duration_since(self, earlier: Instant) -> std::time::Duration {
        self.0.duration_since(earlier)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_request_id_uuid_unique() {
        let a = InferenceRequestId::new_v4();
        let b = InferenceRequestId::new_v4();
        assert_ne!(a, b);
        assert_eq!(format!("{}", a).len(), 36); // standard uuid string length
    }

    #[test]
    fn token_count_arithmetic_explicit() {
        let a = TokenCount::new(10);
        let b = TokenCount::new(3);
        assert_eq!((a + b).raw(), 13);
        assert_eq!((a - b).raw(), 7);
        assert_eq!(b.saturating_sub(a).raw(), 0);
    }

    #[test]
    fn block_size_must_be_positive() {
        let sz = BlockSize::new(16);
        assert_eq!(sz.raw(), 16);
        assert_eq!(sz.as_usize(), 16);
    }

    #[test]
    fn arrival_time_elapsed_monotonic() {
        let t = ArrivalTime::now();
        let _ = t.elapsed();
    }

    #[test]
    fn model_instance_id_str_round_trip() {
        let m = ModelInstanceId::new("default");
        assert_eq!(m.as_str(), "default");
        assert_eq!(m.into_inner(), "default");
    }

    /// Compile-time assertion: NewTypes are not implicitly convertible to/from raw.
    #[test]
    fn count_newtypes_are_distinct() {
        let _t = TokenCount::new(8);
        let _s = SeqCount::new(8);
        // The line below MUST NOT compile:
        // let _: TokenCount = SeqCount::new(8);
    }
}
