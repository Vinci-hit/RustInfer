//! Qwen3-based TextEncoder for Z-Image.
//!
//! Thin wrapper around [`Qwen3`] that:
//! - Loads from the `text_encoder/` subfolder (same weight format)
//! - Applies the Qwen3 chat template to the prompt
//! - Runs `layer_num - 1` layers via [`Qwen3::forward_prefill_hidden_states`]
//! - Returns `[actual_tokens, hidden_dim]` (padding filtered out)

use std::path::Path;

use crate::base::{DataType, DeviceType};
use crate::base::error::{Error, Result};
use crate::model::llm::qwen3::Qwen3;
use crate::model::runtime::InferenceState;
use crate::tensor::Tensor;

/// Maximum sequence length for text encoder input (matches Python: max_sequence_length=512).
const TEXT_ENCODER_MAX_SEQ_LEN: usize = 512;

/// Padding token ID for Qwen3 (<|endoftext|>).
const PAD_TOKEN_ID: i32 = 151643;

/// Qwen3-based text encoder for Z-Image.
///
/// Internally holds a full [`Qwen3`] model, but only uses `layer_num - 1`
/// layers to produce `hidden_states[-2]`.
pub struct Qwen3TextEncoder {
    /// The underlying Qwen3 model (reused as-is).
    pub model: Qwen3,
    /// Number of layers to run (= layer_num - 1).
    pub output_layer_count: usize,
}

impl Qwen3TextEncoder {
    /// Load text encoder from a diffusers-format model directory.
    ///
    /// - `text_encoder_dir`: path to `model_dir/text_encoder/` (contains config.json + safetensors)
    /// - `tokenizer_dir`: path to `model_dir/tokenizer/` — **must be symlinked or copied
    ///   into** `text_encoder_dir` as `tokenizer.json` for `Qwen3::new` to find it.
    ///   Alternatively, you can create the tokenizer.json symlink before calling this.
    ///
    /// Since `Qwen3::new` expects `tokenizer.json` inside `model_dir`, we create a
    /// temporary symlink if it doesn't exist.
    pub fn new<P: AsRef<Path>>(
        text_encoder_dir: P,
        tokenizer_dir: P,
        device_type: DeviceType,
    ) -> Result<Self> {
        let te_dir = text_encoder_dir.as_ref();
        let tok_dir = tokenizer_dir.as_ref();

        // Qwen3::new expects tokenizer.json inside model_dir.
        // Create a symlink if it isn't there yet. This is idempotent
        // under concurrent test execution: if another thread wins the
        // race and creates the symlink first, we swallow the EEXIST
        // and move on.
        //
        // We **don't** remove the symlink afterwards — it's cheap,
        // harmless (points into the immutable model dir), and removing
        // it would race with any other `Qwen3TextEncoder::new` running
        // in parallel.
        let tokenizer_json_in_te = te_dir.join("tokenizer.json");
        let tokenizer_json_src = tok_dir.join("tokenizer.json");
        if !tokenizer_json_in_te.exists() && tokenizer_json_src.exists() {
            match std::os::unix::fs::symlink(&tokenizer_json_src, &tokenizer_json_in_te) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => {
                    return Err(Error::InternalError(format!(
                        "Failed to symlink tokenizer.json into text_encoder dir: {}", e
                    )).into());
                }
            }
        }

        let model = Qwen3::new(te_dir, device_type)?;
        let output_layer_count = model.config.layer_num.saturating_sub(1);

        Ok(Self { model, output_layer_count })
    }

    /// Move all weights to GPU.
    #[cfg(feature = "cuda")]
    pub fn to_cuda(&mut self, device_id: i32) -> Result<()> {
        self.model.layers.to_cuda(device_id)?;
        Ok(())
    }

    /// Create inference state (KV cache + workspace buffers).
    pub fn create_state(&self) -> Result<InferenceState> {
        self.model.create_state()
    }

    // ─────────────────── Tokenizer + Chat Template ───────────────────

    /// Apply chat template and tokenize with padding.
    ///
    /// Template: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
    ///
    /// Returns `(padded_token_ids, attention_mask)` both of length `max_seq_len`.
    pub fn tokenize(&self, prompt: &str, max_seq_len: usize) -> Result<(Vec<i32>, Vec<i32>)> {
        let formatted = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        );

        let token_ids = self.model.tokenizer.encode(&formatted)?;
        let actual_len = token_ids.len().min(max_seq_len);

        let mut padded_tokens = vec![PAD_TOKEN_ID; max_seq_len];
        let mut attention_mask = vec![0i32; max_seq_len];

        padded_tokens[..actual_len].copy_from_slice(&token_ids[..actual_len]);
        for i in 0..actual_len {
            attention_mask[i] = 1;
        }

        Ok((padded_tokens, attention_mask))
    }

    // ─────────────────── Encode ──────────────────────────────────────

    /// Encode a text prompt into hidden states.
    ///
    /// Returns `Tensor` of shape `[actual_tokens, hidden_dim]` — only
    /// non-padding tokens (filtered by attention_mask).
    pub fn encode(
        &self,
        state: &mut InferenceState,
        prompt: &str,
    ) -> Result<Tensor> {
        let (tokens, attention_mask) = self.tokenize(prompt, TEXT_ENCODER_MAX_SEQ_LEN)?;

        let actual_len = attention_mask.iter().filter(|&&m| m == 1).count();
        if actual_len == 0 {
            return Err(Error::InvalidArgument("Empty prompt after tokenization".to_string()).into());
        }

        // Only feed actual (non-padding) tokens to the model
        let mut input_tokens = Tensor::new(&[actual_len], DataType::I32, DeviceType::Cpu)?;
        input_tokens.as_i32_mut()?.as_slice_mut()?.copy_from_slice(&tokens[..actual_len]);

        // Reuse Qwen3's prefill logic, running output_layer_count layers
        let hidden_states = self.model.forward_prefill_hidden_states(
            state,
            &input_tokens,
            actual_len,
            self.output_layer_count,
        )?;

        // hidden_states: [actual_len, dim] — already filtered (no padding fed in)
        Ok(hidden_states)
    }
}

// ──────────────────────────── Tests ──────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn get_text_encoder_dir() -> &'static Path {
        Path::new("/root/z-image-turbo/text_encoder")
    }

    fn get_tokenizer_dir() -> &'static Path {
        Path::new("/root/z-image-turbo/tokenizer")
    }

    #[test]
    #[ignore = "需要 Z-Image 模型权重"]
    fn test_text_encoder_tokenize() -> Result<()> {
        let te_dir = get_text_encoder_dir();
        let tok_dir = get_tokenizer_dir();
        if !te_dir.exists() || !tok_dir.exists() { return Ok(()); }

        let encoder = Qwen3TextEncoder::new(te_dir, tok_dir, DeviceType::Cpu)?;

        let (tokens, mask) = encoder.tokenize("a beautiful sunset over the ocean", 512)?;
        let actual_len = mask.iter().filter(|&&m| m == 1).count();
        println!("Tokenized: {} actual tokens (padded to {})", actual_len, tokens.len());
        assert!(actual_len > 5, "Token count too small: {}", actual_len);
        assert_eq!(tokens.len(), 512);
        Ok(())
    }

    #[test]
    #[ignore = "需要 Z-Image 模型权重，CPU 较慢"]
    fn test_text_encoder_cpu_encode() -> Result<()> {
        let te_dir = get_text_encoder_dir();
        let tok_dir = get_tokenizer_dir();
        if !te_dir.exists() || !tok_dir.exists() { return Ok(()); }

        let encoder = Qwen3TextEncoder::new(te_dir, tok_dir, DeviceType::Cpu)?;
        let mut state = encoder.create_state()?;

        let start = std::time::Instant::now();
        let hidden = encoder.encode(&mut state, "a beautiful sunset over the ocean")?;
        let elapsed = start.elapsed();

        println!("Text encoder output shape: {:?}", hidden.shape());
        println!("Time: {:.1}ms", elapsed.as_millis());
        assert_eq!(hidden.shape().len(), 2);
        assert_eq!(hidden.shape()[1], encoder.model.config.dim);

        let data = hidden.as_f32()?.as_slice()?;
        for &val in data.iter().take(100) {
            assert!(val.is_finite(), "Non-finite value: {}", val);
        }
        Ok(())
    }

    #[test]
    #[ignore = "需要 Z-Image 模型权重 + CUDA"]
    #[cfg(feature = "cuda")]
    fn test_text_encoder_cuda_encode() -> Result<()> {
        let te_dir = get_text_encoder_dir();
        let tok_dir = get_tokenizer_dir();
        if !te_dir.exists() || !tok_dir.exists() { return Ok(()); }

        let encoder = Qwen3TextEncoder::new(te_dir, tok_dir, DeviceType::Cuda(0))?;
        let mut state = encoder.create_state()?;

        let start = std::time::Instant::now();
        let hidden = encoder.encode(&mut state, "a beautiful sunset over the ocean")?;
        let elapsed = start.elapsed();

        let hidden_cpu = hidden.to_cpu()?;
        println!("CUDA text encoder shape: {:?}, time: {:.1}ms",
            hidden_cpu.shape(), elapsed.as_millis());
        assert_eq!(hidden_cpu.shape()[1], encoder.model.config.dim);
        Ok(())
    }
}
