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

/// Bound CPU work independently of the larger image data-URL body limit.
pub const MAX_TEXT_BYTES: usize = 1024 * 1024;

pub fn validate_mtp_request(
    draft_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    has_image: bool,
) -> Result<(), AppError> {
    if draft_tokens > 0 {
        if has_image {
            return Err(AppError::bad_request(
                "MTP currently supports text input only",
            ));
        }
        if temperature.unwrap_or(1.0) > 0.0 && top_k != Some(1) && top_p.unwrap_or(1.0) > 0.0 {
            return Err(AppError::bad_request(
                "MTP currently requires greedy sampling (temperature=0, top_k=1, or top_p=0)",
            ));
        }
    }
    Ok(())
}

pub fn validate_text_bytes(bytes: usize) -> Result<(), AppError> {
    if bytes > MAX_TEXT_BYTES {
        return Err(AppError::bad_request("request text exceeds 1 MiB"));
    }
    Ok(())
}

pub async fn prepare_stop_sequences(
    tokenizer: std::sync::Arc<Tokenizer>,
    stop: Option<StopSequence>,
    permit: crate::middleware::admission::AdmissionPermit,
) -> Result<Vec<Vec<i32>>, AppError> {
    let bytes = match &stop {
        None => return Ok(Vec::new()),
        Some(StopSequence::Single(s)) => s.len(),
        Some(StopSequence::Multiple(values)) => {
            values.iter().fold(0usize, |n, s| n.saturating_add(s.len()))
        }
    };
    validate_text_bytes(bytes)?;
    tokio::task::spawn_blocking(move || {
        let result = tokenize_stop_sequences(&tokenizer, stop.as_ref());
        drop(permit);
        result
    })
    .await
    .map_err(|e| AppError::internal(anyhow::anyhow!(e)))?
}

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
    #[test]
    fn mtp_request_limits_do_not_change_ordinary_sampling() {
        use super::validate_mtp_request;
        assert!(validate_mtp_request(0, None, None, None, true).is_ok());
        assert!(validate_mtp_request(3, Some(0.0), None, None, false).is_ok());
        assert!(validate_mtp_request(3, None, None, Some(1), false).is_ok());
        assert!(validate_mtp_request(3, None, Some(0.0), None, false).is_ok());
        assert!(validate_mtp_request(3, None, None, None, false).is_err());
        assert!(validate_mtp_request(3, Some(0.0), None, None, true).is_err());
    }
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

/// Beam search ranks complete hypotheses, so partial streaming is not supported.
#[allow(clippy::too_many_arguments)]
pub(crate) fn beam_options(
    width: Option<usize>,
    length_penalty: Option<f64>,
    stream: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    has_images: bool,
    config: &infer_protocol::RustInferConfig,
) -> Result<Option<infer_protocol::beam::BeamOptions>, AppError> {
    let Some(width) = width else {
        if length_penalty.is_some() {
            return Err(AppError::bad_request("length_penalty requires beam_width"));
        }
        return Ok(None);
    };
    if stream || has_images || config.tensor_parallel_size != 1 || config.mtp_num_draft_tokens != 0
    {
        return Err(AppError::bad_request(
            "beam search requires non-streaming text, TP=1 and MTP disabled",
        ));
    }
    if temperature.is_some_and(|t| t != 0.0 && t != 1.0)
        || top_p.is_some_and(|p| p != 1.0)
        || top_k.is_some_and(|k| k > 0)
    {
        return Err(AppError::bad_request(
            "beam search does not use temperature, top_p or top_k filters",
        ));
    }
    let options = infer_protocol::beam::BeamOptions {
        width,
        length_penalty: length_penalty.unwrap_or(1.0),
    };
    options
        .validate(config.max_batch_seqs.min(config.max_batch_tokens))
        .map_err(AppError::bad_request)?;
    Ok(Some(options))
}

#[cfg(test)]
mod beam_tests {
    use super::*;
    #[test]
    fn beam_validation_rejects_incompatible_modes_and_limits() {
        let mut config: infer_protocol::RustInferConfig =
            serde_json::from_value(serde_json::json!({})).unwrap();
        config.max_batch_seqs = 4;
        let options = beam_options(Some(4), None, false, None, None, None, false, &config)
            .unwrap()
            .unwrap();
        assert_eq!(options.width, 4);
        assert_eq!(options.length_penalty, 1.0);
        assert!(beam_options(Some(5), None, false, None, None, None, false, &config).is_err());
        assert!(beam_options(Some(4), None, true, None, None, None, false, &config).is_err());
        assert!(beam_options(Some(4), None, false, None, None, None, true, &config).is_err());
        assert!(
            beam_options(Some(4), Some(-1.0), false, None, None, None, false, &config).is_err()
        );
        assert!(beam_options(Some(4), None, false, None, Some(0.9), None, false, &config).is_err());
        assert!(beam_options(None, Some(1.0), false, None, None, None, false, &config).is_err());
        config.mtp_num_draft_tokens = 1;
        assert!(beam_options(Some(4), None, false, None, None, None, false, &config).is_err());
    }
}
