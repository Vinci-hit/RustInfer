//! POST /v1/chat/completions handler
//!
//! 支持流式 (SSE) 和非流式两种模式。

use axum::{
    Extension, Json,
    extract::State,
    response::{IntoResponse, Response},
};

use crate::chat::get_template;
use crate::client::InferClient;
use crate::error::AppError;
use crate::state::SharedState;
use std::time::Instant;

use super::shared;
use super::streaming;
use super::types::*;

/// POST /v1/chat/completions
#[axum::debug_handler]
pub async fn chat_completions(
    State(state): State<SharedState>,
    Extension(permit): Extension<crate::middleware::admission::AdmissionPermit>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    let request_start = Instant::now();

    // 1. 校验请求
    validate_request(&req)?;
    let response_model = state.model_info.model_id.clone();

    let mut image_permit = if req.messages.iter().any(InputChatMessage::has_image) {
        Some(
            state
                .image_admission
                .clone()
                .try_acquire_owned()
                .map_err(|_| AppError::too_many("image request capacity exhausted"))?,
        )
    } else {
        None
    };
    let (input_ids, multimodal) = if image_permit.is_some() {
        let processor = state
            .image_processor
            .clone()
            .ok_or_else(|| AppError::bad_request("model does not support images"))?;
        let tokenizer = state.tokenizer.clone();
        let messages = req.messages.clone();
        let processing_admission = permit.clone();
        // Keep the image capacity reserved even if the HTTP future is dropped
        // while the non-cancellable CPU decoder is still running.
        let processing_permit = image_permit.take();
        let (prepared, returned_permit) = tokio::task::spawn_blocking(move || {
            let result = processor.prepare(&messages, &tokenizer);
            drop(processing_admission);
            (result, processing_permit)
        })
        .await
        .map_err(|e| AppError::internal(anyhow::anyhow!(e)))?;
        image_permit = returned_permit;
        prepared.map_err(|e| AppError::bad_request(e.to_string()))?
    } else {
        let messages = req.messages.clone();
        let tokenizer = state.tokenizer.clone();
        let model_type = state.model_type.clone();
        let processing_admission = permit.clone();
        let ids = tokio::task::spawn_blocking(move || {
            let messages = messages
                .iter()
                .map(InputChatMessage::text_message)
                .collect::<Result<Vec<_>, _>>()
                .map_err(AppError::bad_request)?;
            let prompt = get_template(&model_type)
                .apply(&messages)
                .map_err(|e| AppError::bad_request(format!("Template error: {e}")))?;
            let encoding = tokenizer
                .encode(prompt, true)
                .map_err(|e| AppError::internal(anyhow::anyhow!(e.to_string())))?;
            let ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
            if model_type == "qwen3_5" && ids.contains(&infer_protocol::multimodal::IMAGE_TOKEN_ID)
            {
                return Err(AppError::bad_request(
                    "image placeholders require image data",
                ));
            }
            drop(processing_admission);
            Ok::<_, AppError>(ids)
        })
        .await
        .map_err(|e| AppError::internal(anyhow::anyhow!(e)))??;
        (ids, None)
    };
    let prompt_tokens = input_ids.len() as u32;

    // 4. 构建 InferenceRequest
    //
    // vLLM 语义对齐：max_tokens 默认/上限均为 max_model_len - prompt_len，
    // 保证 prompt+output ≤ ctx 窗口，否则 worker 的 SeqStep validate 会因
    // kv_len_after > max_seq_len 而中止。详见 shared::cap_max_tokens。
    let effective_max_tokens =
        shared::cap_max_tokens(prompt_tokens, req.max_tokens, state.config.max_model_len)?;
    let stop_sequences =
        shared::prepare_stop_sequences(state.tokenizer.clone(), req.stop.clone(), permit.clone())
            .await?;
    let request_id = uuid::Uuid::new_v4().to_string();
    let tokenize_elapsed = request_start.elapsed();
    tracing::debug!(
        request_id = %request_id,
        prompt_tokens,
        tokenize_ms = tokenize_elapsed.as_secs_f64() * 1000.0,
        "TTFT_TRACE: server tokenized"
    );
    let engine_req = infer_protocol::server_to_scheduler::InferenceRequest {
        multimodal,
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

    // 5. 根据 stream 字段分流
    if req.stream {
        // 流式路径 → SSE
        let rx = state
            .client
            .infer_stream(engine_req)
            .await
            .map_err(AppError::from_submit)?;
        tracing::debug!(
            request_id = %request_id,
            submit_ms = request_start.elapsed().as_secs_f64() * 1000.0,
            "TTFT_TRACE: stream submitted to scheduler"
        );

        let include_usage = req
            .stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);

        let sse = streaming::stream_chat_completion(
            request_id,
            response_model,
            prompt_tokens,
            rx,
            state.tokenizer.clone(),
            include_usage,
            request_start,
            (permit, image_permit),
        );

        Ok(sse.into_response())
    } else {
        // 非流式路径
        let engine_resp = state
            .client
            .infer(engine_req)
            .await
            .map_err(AppError::from_submit)?;
        tracing::debug!(
            request_id = %request_id,
            elapsed_ms = request_start.elapsed().as_millis(),
            "chat response received"
        );

        let (generated_text, completion_tokens, finish_reason) =
            shared::decode_completion(&state.tokenizer, engine_resp)?;

        // 构造 OpenAI 格式响应
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", request_id),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: response_model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: generated_text,
                },
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

/// 校验请求参数
fn validate_request(req: &ChatCompletionRequest) -> Result<(), AppError> {
    if req.messages.is_empty() {
        return Err(AppError::bad_request("messages must not be empty"));
    }
    let mut bytes = 0usize;
    for message in &req.messages {
        bytes = bytes.saturating_add(message.role.len());
        match &message.content {
            MessageContent::Text(text) => bytes = bytes.saturating_add(text.len()),
            MessageContent::Parts(parts) => {
                for part in parts {
                    if let ContentPart::Text { text } = part {
                        bytes = bytes.saturating_add(text.len());
                    }
                }
            }
        }
    }
    shared::validate_text_bytes(bytes)?;
    shared::validate_sampling(req.temperature, req.top_p, req.top_k, req.max_tokens)?;
    shared::reject_unsupported_sampling(req.frequency_penalty, req.presence_penalty, req.seed)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_budget_is_aggregate_and_excludes_image_payloads() {
        let half = "x".repeat(shared::MAX_TEXT_BYTES / 2);
        let request = |messages| {
            serde_json::from_value::<ChatCompletionRequest>(
                serde_json::json!({"messages": messages}),
            )
            .unwrap()
        };
        let oversized = request(serde_json::json!([
            {"role":"user", "content": half},
            {"role":"user", "content":[{"type":"text", "text":half}]}
        ]));
        assert!(validate_request(&oversized).is_err());
        let image = request(serde_json::json!([{"role":"user", "content":[
            {"type":"text", "text":"describe"},
            {"type":"image_url", "image_url":{"url":"x".repeat(shared::MAX_TEXT_BYTES + 1)}}
        ]}]));
        assert!(validate_request(&image).is_ok());
    }
}
