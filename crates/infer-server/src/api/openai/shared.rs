//! Logic shared by the `/v1/chat/completions` and `/v1/completions` handlers.
//!
//! Both endpoints validate sampling params, cap `max_tokens` to the context
//! window, and (for the non-stream path) decode the engine's token ids into
//! text the exact same way — only the request/response *shapes* differ. That
//! shared behavior lives here so a fix (e.g. the vLLM max_tokens semantics)
//! happens in one place instead of drifting between two copies.

use tokenizers::Tokenizer;

use crate::error::AppError;

/// Validate the sampling params common to both endpoints (`temperature`,
/// `top_p`, `max_tokens`). Endpoint-specific checks (non-empty messages /
/// prompt) stay in each handler's own `validate_request`.
pub fn validate_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
) -> Result<(), AppError> {
    if let Some(temp) = temperature
        && !(0.0..=2.0).contains(&temp)
    {
        return Err(AppError::bad_request("temperature must be between 0 and 2"));
    }
    if let Some(top_p) = top_p
        && !(0.0..=1.0).contains(&top_p)
    {
        return Err(AppError::bad_request("top_p must be between 0 and 1"));
    }
    if let Some(max_tokens) = max_tokens
        && max_tokens == 0
    {
        return Err(AppError::bad_request("max_tokens must be greater than 0"));
    }
    Ok(())
}

/// Reject sampling knobs the engine does not implement yet, rather than
/// silently ignoring them (which would surprise clients relying on them).
pub fn reject_unsupported_sampling(
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<u64>,
) -> Result<(), AppError> {
    if let Some(value) = frequency_penalty
        && value != 0.0
    {
        return Err(AppError::bad_request(
            "frequency_penalty is not supported yet",
        ));
    }
    if let Some(value) = presence_penalty
        && value != 0.0
    {
        return Err(AppError::bad_request(
            "presence_penalty is not supported yet",
        ));
    }
    if seed.is_some() {
        return Err(AppError::bad_request("seed is not supported for LLM yet"));
    }
    Ok(())
}

/// vLLM `max_tokens` semantics: when the client omits `max_tokens`, default to
/// `max_model_len - prompt_len`; when it provides one, cap it to that same
/// bound so `prompt + output ≤ ctx`. Errors if the prompt alone already fills
/// (or overflows) the context window — otherwise the worker would later abort
/// in `SeqStep::validate` with `kv_len_after > max_seq_len`.
pub fn cap_max_tokens(
    prompt_tokens: u32,
    requested_max_tokens: Option<usize>,
    max_model_len: usize,
) -> Result<usize, AppError> {
    let prompt_len = prompt_tokens as usize;
    if prompt_len >= max_model_len {
        return Err(AppError::bad_request(format!(
            "prompt_tokens {} exceeds max_model_len {}",
            prompt_len, max_model_len
        )));
    }
    let remaining = max_model_len - prompt_len;
    Ok(match requested_max_tokens {
        Some(n) => n.min(remaining),
        None => remaining,
    })
}

/// Non-stream decode tail shared by both handlers: surface an engine error,
/// then decode the output token ids into text. Returns
/// `(generated_text, completion_tokens, finish_reason)`.
pub fn decode_completion(
    tokenizer: &Tokenizer,
    engine_resp: infer_protocol::scheduler_to_server::InferenceResponse,
) -> Result<(String, u32, String), AppError> {
    if let infer_protocol::scheduler_to_server::ResponseStatus::Error = engine_resp.status {
        return Err(AppError::internal(anyhow::anyhow!(
            "Engine error: {}",
            engine_resp.error.unwrap_or_else(|| "Unknown".to_string())
        )));
    }

    let output_ids_u32: Vec<u32> = engine_resp
        .output_token_ids
        .iter()
        .map(|&id| id as u32)
        .collect();
    let generated_text = tokenizer
        .decode(&output_ids_u32, true)
        .map_err(|e| AppError::internal(anyhow::anyhow!("Decode error: {}", e)))?;
    let completion_tokens = engine_resp.output_token_ids.len() as u32;
    let finish_reason = engine_resp
        .finish_reason
        .unwrap_or_else(|| "stop".to_string());

    Ok((generated_text, completion_tokens, finish_reason))
}
