//! Speculative decoding components, independent of ordinary token sampling.

mod verifier;

pub use verifier::{GreedyVerifier, validate_sampling};

mod mtp;
pub mod prefill;
mod session;
pub use mtp::MtpProposer;
pub use session::{MtpLimits, MtpSession, MtpStep};
#[cfg(any(feature = "cuda", test))]
mod commit;
#[cfg(feature = "cuda")]
pub mod serving;
