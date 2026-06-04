//! POST /v1/chat/completions handler
//!
//! 支持流式 (SSE) 和非流式两种模式。

use axum::{
    extract::State,
    response::{IntoResponse, Response},
    Json,
};

use crate::client::InferClient;
use crate::error::AppError;
use crate::state::SharedState;
use crate::chat::get_template;
use std::time::Instant;

use super::streaming;
use super::types::*;

/// POST /v1/chat/completions
#[axum::debug_handler]
pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    let request_start = Instant::now();

    // 1. 校验请求
    validate_request(&req)?;

    // 2. 应用 chat template → 生成 prompt 文本
    //    模板必须基于服务端实际加载的 model_type，而不是客户端请求里的
    //    req.model（客户端可能填任意名字，填错会套错模板导致输出乱码 + 不停止）。
    let template = get_template(&state.config.model_type);
    let prompt = template.apply(&req.messages)
        .map_err(|e| AppError::bad_request(format!("Template error: {}", e)))?;

    // 3. Tokenize
    let encoding = state.tokenizer.encode(prompt.as_str(), true)
        .map_err(|e| AppError::internal(anyhow::anyhow!("Tokenize error: {}", e)))?;
    let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let prompt_tokens = input_ids.len() as u32;

    // 4. 构建 InferenceRequest
    let request_id = uuid::Uuid::new_v4().to_string();
    let tokenize_elapsed = request_start.elapsed();
    tracing::info!(
        request_id = %request_id,
        prompt_tokens,
        tokenize_ms = tokenize_elapsed.as_secs_f64() * 1000.0,
        "TTFT_TRACE: server tokenized"
    );
    let engine_req = infer_protocol::server_to_scheduler::InferenceRequest {
        request_id: request_id.clone(),
        modality: infer_protocol::server_to_scheduler::InferenceModality::Llm,
        input_ids,
        max_tokens: req.max_tokens.unwrap_or(2048),
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(-1),
        stream: req.stream,
        priority: 0,
        stop_sequences: req.stop.map(|s| s.into_vec()).unwrap_or_default(),
        ignore_eos: req.ignore_eos || state.config.ignore_eos,
        diffusion: None,
    };

    // 5. 根据 stream 字段分流
    if req.stream {
        // 流式路径 → SSE
        let rx = state.client.infer_stream(engine_req).await
            .map_err(AppError::internal)?;
        tracing::info!(
            request_id = %request_id,
            submit_ms = request_start.elapsed().as_secs_f64() * 1000.0,
            "TTFT_TRACE: stream submitted to scheduler"
        );

        let include_usage = req.stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);

        let sse = streaming::stream_chat_completion(
            request_id,
            req.model.clone(),
            prompt_tokens,
            rx,
            state.tokenizer.clone(),
            include_usage,
        );

        Ok(sse.into_response())
    } else {
        // 非流式路径
        let engine_resp = state.client.infer(engine_req).await
            .map_err(AppError::internal)?;
        tracing::debug!(
            request_id = %request_id,
            elapsed_ms = request_start.elapsed().as_millis(),
            "chat response received"
        );

        // 检查错误
        if let infer_protocol::scheduler_to_server::ResponseStatus::Error = engine_resp.status {
            return Err(AppError::internal(anyhow::anyhow!(
                "Engine error: {}",
                engine_resp.error.unwrap_or_else(|| "Unknown".to_string())
            )));
        }

        // Decode output tokens → 文本
        let output_ids_u32: Vec<u32> = engine_resp.output_token_ids.iter()
            .map(|&id| id as u32).collect();
        let generated_text = state.tokenizer.decode(&output_ids_u32, true)
            .map_err(|e| AppError::internal(anyhow::anyhow!("Decode error: {}", e)))?;
        let completion_tokens = engine_resp.output_token_ids.len() as u32;

        // 确定 finish_reason
        let finish_reason = engine_resp.finish_reason
            .unwrap_or_else(|| "stop".to_string());

        // 构造 OpenAI 格式响应
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", request_id),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: req.model,
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
                total_tokens: prompt_tokens + completion_tokens,
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

    if let Some(temp) = req.temperature
        && (!(0.0..=2.0).contains(&temp)) {
            return Err(AppError::bad_request("temperature must be between 0 and 2"));
        }

    if let Some(top_p) = req.top_p
        && !(0.0..=1.0).contains(&top_p) {
            return Err(AppError::bad_request("top_p must be between 0 and 1"));
        }

    if let Some(max_tokens) = req.max_tokens
        && max_tokens == 0 {
            return Err(AppError::bad_request("max_tokens must be greater than 0"));
        }

    Ok(())
}
