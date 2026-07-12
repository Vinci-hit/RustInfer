//! POST /v1/completions handler
//!
//! Text completion 接口，接收 prompt 直接 tokenize 生成，不经过 chat template。

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};

use crate::client::InferClient;
use crate::error::AppError;
use crate::state::SharedState;

use super::shared;
use super::streaming;
use super::types::*;

/// POST /v1/completions
#[axum::debug_handler]
pub async fn completions(
    State(state): State<SharedState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, AppError> {
    // 0. 准入控制：过载时在入口返回 429（详见 chat_completions）。permit move 进
    //    流式 SSE 或随非流式响应构建结束而释放。
    let permit = state
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| AppError::too_many("server overloaded, please retry later"))?;

    // 1. 校验
    validate_request(&req, state.tokenizer.get_vocab_size(true))?;
    let response_model = state.model_info.model_id.clone();

    // 2. 获取 input_ids（直接 tokenize prompt，不经过 chat template）
    let (input_ids, prompt_tokens) = match &req.prompt {
        CompletionPrompt::Text(text) => {
            let encoding = state
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| AppError::internal(anyhow::anyhow!("Tokenize error: {}", e)))?;
            let ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
            let len = ids.len() as u32;
            (ids, len)
        }
        CompletionPrompt::Tokens(ids) => {
            let len = ids.len() as u32;
            (ids.clone(), len)
        }
    };

    // 3. 构建 InferenceRequest
    //
    // vLLM 语义：max_tokens 默认/上限均为 max_model_len - prompt_len。
    // 详见 shared::cap_max_tokens。
    let effective_max_tokens =
        shared::cap_max_tokens(prompt_tokens, req.max_tokens, state.config.max_model_len)?;
    let stop_sequences = shared::tokenize_stop_sequences(&state.tokenizer, req.stop.as_ref())?;
    let request_id = uuid::Uuid::new_v4().to_string();
    let engine_req = infer_protocol::server_to_scheduler::InferenceRequest {
        request_id: request_id.clone(),
        modality: infer_protocol::server_to_scheduler::InferenceModality::Llm,
        input_ids,
        max_tokens: effective_max_tokens,
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(-1),
        stream: req.stream,
        priority: 0,
        stop_sequences,
        ignore_eos: req.ignore_eos || state.config.ignore_eos,
        diffusion: None,
    };

    // 4. 分流
    if req.stream {
        let rx = state
            .client
            .infer_stream(engine_req)
            .await
            .map_err(AppError::from_submit)?;

        let include_usage = req
            .stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);

        let sse = streaming::stream_completion(
            request_id,
            response_model,
            prompt_tokens,
            rx,
            state.tokenizer.clone(),
            include_usage,
            permit,
        );

        Ok(sse.into_response())
    } else {
        let engine_resp = state
            .client
            .infer(engine_req)
            .await
            .map_err(AppError::from_submit)?;

        let (generated_text, completion_tokens, finish_reason) =
            shared::decode_completion(&state.tokenizer, engine_resp)?;

        let response = CompletionResponse {
            id: format!("cmpl-{}", request_id),
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: response_model,
            choices: vec![CompletionChoice {
                index: 0,
                text: generated_text,
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens.saturating_add(completion_tokens),
            },
        };

        Ok(Json(response).into_response())
    }
}

fn validate_request(req: &CompletionRequest, vocab_size: usize) -> Result<(), AppError> {
    match &req.prompt {
        CompletionPrompt::Text(text) if text.is_empty() => {
            return Err(AppError::bad_request("prompt must not be empty"));
        }
        CompletionPrompt::Tokens(ids) if ids.is_empty() => {
            return Err(AppError::bad_request(
                "prompt token array must not be empty",
            ));
        }
        CompletionPrompt::Tokens(ids) => {
            if let Some((index, token_id)) = ids
                .iter()
                .copied()
                .enumerate()
                .find(|(_, token_id)| *token_id < 0 || (*token_id as usize) >= vocab_size)
            {
                return Err(AppError::bad_request(format!(
                    "prompt token at index {} is outside valid range [0, {}): {}",
                    index, vocab_size, token_id
                )));
            }
        }
        _ => {}
    }
    shared::validate_sampling(req.temperature, req.top_p, req.top_k, req.max_tokens)?;
    shared::reject_unsupported_sampling(req.frequency_penalty, req.presence_penalty, req.seed)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_request;
    use crate::api::openai::types::CompletionRequest;

    fn request(json: &str) -> CompletionRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn token_prompt_must_be_inside_tokenizer_vocabulary() {
        assert!(validate_request(&request(r#"{"prompt":[0,99]}"#), 100).is_ok());
        assert!(validate_request(&request(r#"{"prompt":[-1]}"#), 100).is_err());
        assert!(validate_request(&request(r#"{"prompt":[100]}"#), 100).is_err());
    }
}
