//! POST /v1/chat/completions handler
//!
//! 支持流式 (SSE) 和非流式两种模式。

use axum::{
    Json,
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
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    let request_start = Instant::now();

    // 0. 准入控制：在 tokenize 和任何内部排队之前抢占一个并发名额。过载时立即
    //    返回 429，而不是把请求压入无界队列。permit 的生命周期 == 请求生命周期：
    //    非流式分支持有到响应构建完毕；流式分支把 permit move 进 SSE 流，随流结束
    //    或客户端断开而释放。
    let permit = state
        .admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| AppError::too_many("server overloaded, please retry later"))?;

    // 1. 校验请求
    validate_request(&req)?;
    let response_model = state.model_info.model_id.clone();

    // 2. 应用 chat template → 生成 prompt 文本
    //    模板必须基于服务端实际加载的 model_type，而不是客户端请求里的
    //    req.model（客户端可能填任意名字，填错会套错模板导致输出乱码 + 不停止）。
    let template = get_template(&state.model_type);
    let prompt = template
        .apply(&req.messages)
        .map_err(|e| AppError::bad_request(format!("Template error: {}", e)))?;

    // 3. Tokenize
    let encoding = state
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| AppError::internal(anyhow::anyhow!("Tokenize error: {}", e)))?;
    let input_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let prompt_tokens = input_ids.len() as u32;

    // 4. 构建 InferenceRequest
    //
    // vLLM 语义对齐：max_tokens 默认/上限均为 max_model_len - prompt_len，
    // 保证 prompt+output ≤ ctx 窗口，否则 worker 的 SeqStep validate 会因
    // kv_len_after > max_seq_len 而中止。详见 shared::cap_max_tokens。
    let effective_max_tokens =
        shared::cap_max_tokens(prompt_tokens, req.max_tokens, state.config.max_model_len)?;
    let stop_sequences = shared::tokenize_stop_sequences(&state.tokenizer, req.stop.as_ref())?;
    let request_id = uuid::Uuid::new_v4().to_string();
    let tokenize_elapsed = request_start.elapsed();
    tracing::debug!(
        request_id = %request_id,
        prompt_tokens,
        tokenize_ms = tokenize_elapsed.as_secs_f64() * 1000.0,
        "TTFT_TRACE: server tokenized"
    );
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
            permit,
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
    shared::validate_sampling(req.temperature, req.top_p, req.top_k, req.max_tokens)?;
    shared::reject_unsupported_sampling(req.frequency_penalty, req.presence_penalty, req.seed)?;
    Ok(())
}
