//! Qwen3 model — the same assembled [`Decoder`](crate::models::decoder::Decoder)
//! as Llama3, with each block's `Attention` Q/K RMSNorms populated by the loader.

pub use crate::models::decoder::Decoder as Qwen3Model;
