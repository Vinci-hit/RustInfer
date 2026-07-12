//! Logic shared by the `/v1/chat/completions` and `/v1/completions` handlers.
//!
//! Both endpoints validate sampling params, cap `max_tokens` to the context
//! window, and (for the non-stream path) decode the engine's token ids into
//! text the exact same way — only the request/response *shapes* differ. That
//! shared behavior lives here so a fix (e.g. the vLLM max_tokens semantics)
//! happens in one place instead of drifting between two copies.

use tokenizers::Tokenizer;

use crate::error::AppError;

use super::types::StopSequence;

/// Validate the sampling params common to both endpoints (`temperature`,
/// `top_p`, `max_tokens`). Endpoint-specific checks (non-empty messages /
/// prompt) stay in each handler's own `validate_request`.
pub fn validate_sampling(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
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
    if let Some(top_k) = top_k
        && top_k < -1
    {
        return Err(AppError::bad_request("top_k must be -1 or non-negative"));
    }
    Ok(())
}

/// Tokenize user-provided stop strings once at the HTTP boundary. The
/// scheduler intentionally does not load a tokenizer; it receives token-id
/// sequences and performs suffix matching on generated ids.
pub fn tokenize_stop_sequences(
    tokenizer: &Tokenizer,
    stop: Option<&StopSequence>,
) -> Result<Vec<Vec<i32>>, AppError> {
    let Some(stop) = stop else {
        return Ok(Vec::new());
    };

    let mut encoded = Vec::new();
    match stop {
        StopSequence::Single(value) => encode_stop(tokenizer, value, &mut encoded)?,
        StopSequence::Multiple(values) => {
            for value in values {
                encode_stop(tokenizer, value, &mut encoded)?;
            }
        }
    }
    Ok(encoded)
}

fn encode_stop(
    tokenizer: &Tokenizer,
    value: &str,
    encoded: &mut Vec<Vec<i32>>,
) -> Result<(), AppError> {
    if value.is_empty() {
        return Err(AppError::bad_request("stop sequences must not be empty"));
    }
    let encoding = tokenizer
        .encode(value, false)
        .map_err(|e| AppError::internal(anyhow::anyhow!("stop tokenization failed: {}", e)))?;
    if encoding.get_ids().is_empty() {
        return Err(AppError::bad_request(
            "stop sequence did not produce any tokens",
        ));
    }
    let token_ids = encoding
        .get_ids()
        .iter()
        .copied()
        .map(|id| {
            i32::try_from(id).map_err(|_| {
                AppError::internal(anyhow::anyhow!("token id {} exceeds protocol range", id))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    encoded.push(token_ids);
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn tokenizer() -> Tokenizer {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".to_string(), 0),
                    ("END".to_string(), 1),
                    ("NOW".to_string(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));
        tokenizer
    }

    #[test]
    fn tokenizes_stop_strings_without_template_special_tokens() {
        let stop = StopSequence::Single("END NOW".to_string());
        assert_eq!(
            tokenize_stop_sequences(&tokenizer(), Some(&stop)).unwrap(),
            vec![vec![1, 2]]
        );
    }

    #[test]
    fn rejects_empty_stop_and_invalid_top_k() {
        let stop = StopSequence::Single(String::new());
        assert!(tokenize_stop_sequences(&tokenizer(), Some(&stop)).is_err());
        assert!(validate_sampling(None, None, Some(-2), None).is_err());
        assert!(validate_sampling(None, None, Some(-1), None).is_ok());
        assert!(validate_sampling(None, None, Some(0), None).is_ok());
    }
}
