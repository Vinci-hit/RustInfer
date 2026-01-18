//! HTTP Handlers
//!
//! 实现 OpenAI-compatible API endpoints with SSE streaming.

use crate::http::{ChatCompletionRequest, ChatCompletionChunk};
use crate::state::AppState;
use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    http::StatusCode,
    Json,
};
use futures::stream::Stream;
use infer_protocol::{
    SchedulerCommand, SchedulerOutput, InferRequest,
    SamplingParams, StoppingCriteria, FinishReason,
};
use std::convert::Infallible;
use std::sync::Arc;
use uuid::Uuid;

/// Chat Completions Handler
///
/// 处理 POST /v1/chat/completions 请求
#[axum::debug_handler]
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // 检查是否是流式请求
    if req.stream {
        // 返回 SSE 流
        match create_sse_stream(state, req).await {
            Ok(stream) => Sse::new(stream).into_response(),
            Err(e) => {
                tracing::error!("Failed to create SSE stream: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
            }
        }
    } else {
        // 非流式模式：等待完整响应
        match create_non_stream_response(state, req).await {
            Ok(response) => Json(response).into_response(),
            Err(e) => {
                tracing::error!("Failed to create non-stream response: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
            }
        }
    }
}

/// 创建 SSE 流
async fn create_sse_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> anyhow::Result<impl Stream<Item = Result<Event, Infallible>>> {
    // 1. 生成 request_id
    let request_id = Uuid::new_v4().to_string();
    
    // 2. 转换消息为 Prompt
    let prompt = req.to_prompt();
    
    // 3. Tokenize (可能耗时，考虑 spawn_blocking)
    let token_ids = state.tokenizer.encode(&prompt)?;
    
    tracing::info!(
        "New request: id={}, stream={}, prompt_len={} chars, token_count={}",
        request_id, req.stream, prompt.len(), token_ids.len()
    );
    
    // 4. 转换停止词为 Token IDs
    let stop_token_ids = if let Some(stop_words) = &req.stop {
        let mut stop_ids = Vec::new();
        for word in stop_words {
            if let Ok(ids) = state.tokenizer.encode(word) {
                stop_ids.extend(ids);
            }
        }
        stop_ids
    } else {
        Vec::new()
    };
    
    // 5. 构建 InferRequest
    let infer_req = InferRequest {
        request_id: request_id.clone(),
        input_token_ids: token_ids,
        sampling_params: SamplingParams {
            temperature: req.temperature.unwrap_or(1.0),
            top_p: req.top_p.unwrap_or(1.0),
            top_k: req.top_k.unwrap_or(-1),
            repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
            frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
            seed: req.seed,
        },
        stopping_criteria: StoppingCriteria {
            max_new_tokens: req.max_tokens.unwrap_or(512),
            stop_token_ids,
            ignore_eos: false,
        },
        adapter_id: None,
    };
    
    // 6. 注册到 Request Map
    let mut output_rx = state.register_request(request_id.clone()).await;
    
    // 7. 发送给 Scheduler
    let cmd = SchedulerCommand::AddRequest(infer_req);
    state.scheduler_tx.send(cmd)?;
    
    // 8. 创建 SSE 流
    let model_name = state.config.model.model_name.clone();
    let request_id_clone = request_id.clone();
    let state_clone = state.clone();
    
    let stream = async_stream::stream! {
        // 创建流式解码器
        let mut decoder = state_clone.tokenizer.create_decoder();
        
        // 首次发送角色信息
        let first_chunk = ChatCompletionChunk {
            id: request_id_clone.clone(),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model_name.clone(),
            choices: vec![crate::http::response::ChunkChoice {
                index: 0,
                delta: crate::http::response::ChunkDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        
        match serde_json::to_string(&first_chunk) {
            Ok(json) => yield Ok(Event::default().data(json)),
            Err(e) => {
                tracing::error!("Failed to serialize first chunk: {:?}", e);
            }
        }
        
        // 监听输出
        while let Some(output) = output_rx.recv().await {
            match output {
                SchedulerOutput::Step(step) => {
                    // 解码 Token
                    if let Some(text) = decoder.decode(step.new_token_id) {
                        // 发送 SSE 事件
                        let chunk = ChatCompletionChunk::new(
                            request_id_clone.clone(),
                            model_name.clone(),
                            Some(text),
                        );
                        
                        match serde_json::to_string(&chunk) {
                            Ok(json) => yield Ok(Event::default().data(json)),
                            Err(e) => {
                                tracing::error!("Failed to serialize chunk: {:?}", e);
                            }
                        }
                    }
                }
                SchedulerOutput::Finish(finish) => {
                    // 刷新剩余的 buffer
                    if let Some(text) = decoder.flush() {
                        let chunk = ChatCompletionChunk::new(
                            request_id_clone.clone(),
                            model_name.clone(),
                            Some(text),
                        );
                        
                        match serde_json::to_string(&chunk) {
                            Ok(json) => yield Ok(Event::default().data(json)),
                            Err(e) => {
                                tracing::error!("Failed to serialize final chunk: {:?}", e);
                            }
                        }
                    }
                    
                    // 发送完成事件
                    let finish_reason = match finish.reason {
                        FinishReason::Stop => "stop",
                        FinishReason::Length => "length",
                        FinishReason::Abort => "abort",
                    };
                    
                    let done_chunk = ChatCompletionChunk::done(
                        request_id_clone.clone(),
                        model_name.clone(),
                        finish_reason,
                    );
                    
                    match serde_json::to_string(&done_chunk) {
                        Ok(json) => yield Ok(Event::default().data(json)),
                        Err(e) => {
                            tracing::error!("Failed to serialize done chunk: {:?}", e);
                        }
                    }
                    
                    // 发送 [DONE] 标记
                    yield Ok(Event::default().data("[DONE]"));
                    
                    break;
                }
                SchedulerOutput::Error(error) => {
                    tracing::error!("Scheduler error for {}: {}", request_id_clone, error.message);
                    
                    // 发送错误信息
                    let error_data = format!("{{\"error\": \"{}\"}}", error.message);
                    yield Ok(Event::default().data(error_data));
                    
                    break;
                }
            }
        }
        
        // 清理 Request Map
        state_clone.unregister_request(&request_id_clone).await;
        
        tracing::info!("Request {} completed", request_id_clone);
    };
    
    Ok(stream)
}

/// 创建非流式响应
async fn create_non_stream_response(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> anyhow::Result<crate::http::response::ChatCompletionResponse> {
    // 1. 生成 request_id
    let request_id = Uuid::new_v4().to_string();

    // 2. 转换消息为 Prompt
    let prompt = req.to_prompt();

    // 3. Tokenize
    let token_ids = state.tokenizer.encode(&prompt)?;
    let prompt_tokens = token_ids.len();

    tracing::info!(
        "New request: id={}, stream=false, prompt_len={} chars, token_count={}",
        request_id, prompt.len(), token_ids.len()
    );

    // 4. 转换停止词为 Token IDs
    let stop_token_ids = if let Some(stop_words) = &req.stop {
        let mut stop_ids = Vec::new();
        for word in stop_words {
            if let Ok(ids) = state.tokenizer.encode(word) {
                stop_ids.extend(ids);
            }
        }
        stop_ids
    } else {
        Vec::new()
    };

    // 5. 构建 InferRequest
    let infer_req = InferRequest {
        request_id: request_id.clone(),
        input_token_ids: token_ids,
        sampling_params: SamplingParams {
            temperature: req.temperature.unwrap_or(1.0),
            top_p: req.top_p.unwrap_or(1.0),
            top_k: req.top_k.unwrap_or(-1),
            repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
            frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
            seed: req.seed,
        },
        stopping_criteria: StoppingCriteria {
            max_new_tokens: req.max_tokens.unwrap_or(512),
            stop_token_ids,
            ignore_eos: false,
        },
        adapter_id: None,
    };

    // 6. 注册到 Request Map
    let mut output_rx = state.register_request(request_id.clone()).await;

    // 7. 发送给 Scheduler
    let cmd = SchedulerCommand::AddRequest(infer_req);
    state.scheduler_tx.send(cmd)?;

    // 8. 等待并收集所有输出
    let mut generated_tokens = Vec::new();
    let mut finish_reason = "stop";

    while let Some(output) = output_rx.recv().await {
        match output {
            SchedulerOutput::Step(step) => {
                generated_tokens.push(step.new_token_id);
            }
            SchedulerOutput::Finish(finish) => {
                finish_reason = match finish.reason {
                    FinishReason::Stop => "stop",
                    FinishReason::Length => "length",
                    FinishReason::Abort => "abort",
                };
                break;
            }
            SchedulerOutput::Error(error) => {
                tracing::error!("Scheduler error for {}: {}", request_id, error.message);
                state.unregister_request(&request_id).await;
                return Err(anyhow::anyhow!("Scheduler error: {}", error.message));
            }
        }
    }

    // 9. 解码生成的 tokens
    let generated_text = state.tokenizer.decode(&generated_tokens)?;

    // 10. 清理 Request Map
    state.unregister_request(&request_id).await;

    tracing::info!(
        "Request {} completed: {} tokens generated",
        request_id,
        generated_tokens.len()
    );

    // 11. 构建响应
    Ok(crate::http::response::ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: state.config.model.model_name.clone(),
        choices: vec![crate::http::response::ChatChoice {
            index: 0,
            message: crate::http::request::ChatMessage {
                role: "assistant".to_string(),
                content: generated_text,
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: crate::http::response::Usage {
            prompt_tokens,
            completion_tokens: generated_tokens.len(),
            total_tokens: prompt_tokens + generated_tokens.len(),
        },
    })
}
