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
    validate_request(&req)?;
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
    // vLLM 语义：未传 max_tokens 时默认为 max_model_len - prompt_len；
    // 传了也会被 cap 到该上限。否则 worker 解到 prompt+max_tokens > max_seq_len
    // 会在 SeqStep validate 处抛 kv_len_after > max。
    let max_model_len = state.config.max_model_len;
    let prompt_len_usize = prompt_tokens as usize;
    if prompt_len_usize >= max_model_len {
        return Err(AppError::bad_request(format!(
            "prompt_tokens {} exceeds max_model_len {}",
            prompt_len_usize, max_model_len
        )));
    }
    let remaining = max_model_len - prompt_len_usize;
    let effective_max_tokens = match req.max_tokens {
        Some(n) => n.min(remaining),
        None => remaining,
    };
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
        stop_sequences: req.stop.map(|s| s.into_vec()).unwrap_or_default(),
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

        if let infer_protocol::scheduler_to_server::ResponseStatus::Error = engine_resp.status {
            return Err(AppError::internal(anyhow::anyhow!(
                "Engine error: {}",
                engine_resp.error.unwrap_or_else(|| "Unknown".to_string())
            )));
        }

        // Decode
        let output_ids_u32: Vec<u32> = engine_resp
            .output_token_ids
            .iter()
            .map(|&id| id as u32)
            .collect();
        let generated_text = state
            .tokenizer
            .decode(&output_ids_u32, true)
            .map_err(|e| AppError::internal(anyhow::anyhow!("Decode error: {}", e)))?;
        let completion_tokens = engine_resp.output_token_ids.len() as u32;

        let finish_reason = engine_resp
            .finish_reason
            .unwrap_or_else(|| "stop".to_string());

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

fn validate_request(req: &CompletionRequest) -> Result<(), AppError> {
    match &req.prompt {
        CompletionPrompt::Text(text) if text.is_empty() => {
            return Err(AppError::bad_request("prompt must not be empty"));
        }
        CompletionPrompt::Tokens(ids) if ids.is_empty() => {
            return Err(AppError::bad_request(
                "prompt token array must not be empty",
            ));
        }
        _ => {}
    }

    if let Some(temp) = req.temperature
        && (!(0.0..=2.0).contains(&temp))
    {
        return Err(AppError::bad_request("temperature must be between 0 and 2"));
    }

    if let Some(top_p) = req.top_p
        && !(0.0..=1.0).contains(&top_p)
    {
        return Err(AppError::bad_request("top_p must be between 0 and 1"));
    }

    if let Some(max_tokens) = req.max_tokens
        && max_tokens == 0
    {
        return Err(AppError::bad_request("max_tokens must be greater than 0"));
    }

    reject_unsupported_sampling(req.frequency_penalty, req.presence_penalty, req.seed)?;

    Ok(())
}

fn reject_unsupported_sampling(
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
