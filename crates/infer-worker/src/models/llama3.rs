//! Llama3 model — a dense decoder assembled from reusable components.
//!
//! `Llama3Model` is the shared [`Decoder`](crate::models::decoder::Decoder);
//! the loader builds it with each block's `Attention` Q/K norms absent.

pub use crate::models::decoder::Decoder as Llama3Model;
