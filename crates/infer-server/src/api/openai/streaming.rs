//! SSE 流式输出逻辑
//!
//! 将 `mpsc::Receiver<StreamChunk>` 转换为 Axum SSE 流。
//! UTF-8 安全的渐进式解码由 [`super::decoder::IncrementalDecoder`] 负责，本模块只
//! 关心 SSE chunk 的协议封装与流终止语义。

use crate::client::StreamHandle;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::Stream;
use infer_protocol::scheduler_to_server::ChunkType;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

use super::decoder::IncrementalDecoder;
use super::types::*;

fn json_event<T: serde::Serialize>(request_id: &str, payload: &T) -> Event {
    match serde_json::to_string(payload) {
        Ok(data) => Event::default().data(data),
        Err(error) => {
            tracing::error!(
                request_id = %request_id,
                error = %error,
                "failed to serialize SSE payload"
            );
            Event::default().data("[DONE]")
        }
    }
}

/// 构建 Chat Completion SSE 流
pub fn stream_chat_completion(
    request_id: String,
    model: String,
    prompt_tokens: u32,
    mut stream_handle: StreamHandle,
    tokenizer: Arc<Tokenizer>,
    include_usage: bool,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let stream_started = Instant::now();
        let created = chrono::Utc::now().timestamp();
        let chunk_id = format!("chatcmpl-{}", request_id);
        let mut completion_tokens: u32 = 0;
        let mut first_content_sent = false;
        let mut decoder = IncrementalDecoder::new(tokenizer);

        // 第一个 chunk: 发送 role
        let first_chunk = ChatCompletionChunk {
            id: chunk_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        yield Ok(json_event(&request_id, &first_chunk));

        // 闭包工厂：构造一个内容增量 chunk。重复使用避免代码碎片化。
        let make_content_chunk = |chunk_id: &str, model: &str, content: String| ChatCompletionChunk {
            id: chunk_id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: None, content: Some(content) },
                finish_reason: None,
            }],
            usage: None,
        };

        while let Some(chunk) = stream_handle.recv().await {
            match chunk.chunk_type {
                ChunkType::Token => {
                    completion_tokens += 1;
                    let Some(token_id) = chunk.token_id else { continue };

                    match decoder.push(token_id as u32) {
                        Ok(Some(delta)) => {
                            if !first_content_sent {
                                first_content_sent = true;
                                tracing::debug!(
                                    request_id = %request_id,
                                    elapsed_ms = stream_started.elapsed().as_millis(),
                                    "chat stream first content chunk"
                                );
                            }
                            let payload = make_content_chunk(&chunk_id, &model, delta);
                            yield Ok(json_event(&request_id, &payload));
                        }
                        Ok(None) => {
                            // 当前 token 仅扩展了未确认尾部（如多字节字符的中间字节），不下发。
                        }
                        Err(e) => {
                            tracing::error!(
                                request_id = %request_id,
                                error = %e,
                                "tokenizer decode failed; terminating stream"
                            );
                            stream_handle.mark_finished();
                            // 给客户端一个最低限度的错误信号：一条带 finish_reason 的 chunk + [DONE]
                            let err_chunk = ChatCompletionChunk {
                                id: chunk_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model.clone(),
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: Delta::default(),
                                    finish_reason: Some("error".to_string()),
                                }],
                                usage: None,
                            };
                            yield Ok(json_event(&request_id, &err_chunk));
                            yield Ok(Event::default().data("[DONE]"));
                            return;
                        }
                    }
                }
                ChunkType::Done => {
                    // 提前标记完成：scheduler 已发 Done，逻辑上请求已结束。
                    // 若放在 yield 之后，客户端在收到 finish_chunk 后立即关闭连接（常见行为）
                    // 会触发 stream future drop → StreamHandle::Drop → 向 scheduler 发出
                    // 一个不必要的 cancel。
                    stream_handle.mark_finished();

                    // 流结束前 flush decoder：把任何被滞留的「未确认」字符吐出去。
                    // 走到这里说明流自然结束，残留的 U+FFFD 是真实的不可解码序列，
                    // 应当下发给客户端而不是丢弃。
                    if let Ok(Some(tail)) = decoder.flush() {
                        let payload = make_content_chunk(&chunk_id, &model, tail);
                        yield Ok(json_event(&request_id, &payload));
                    }

                    let finish_reason = chunk.finish_reason.unwrap_or_else(|| "stop".to_string());
                    let finish_chunk = ChatCompletionChunk {
                        id: chunk_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta::default(),
                            finish_reason: Some(finish_reason),
                        }],
                        usage: None,
                    };
                    yield Ok(json_event(&request_id, &finish_chunk));

                    if include_usage {
                        let usage_chunk = ChatCompletionChunk {
                            id: chunk_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![],
                            usage: Some(Usage {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens: prompt_tokens.saturating_add(completion_tokens),
                            }),
                        };
                        yield Ok(json_event(&request_id, &usage_chunk));
                    }

                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                ChunkType::Error => {
                    stream_handle.mark_finished();
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text(""))
}

/// 构建 Text Completion SSE 流
pub fn stream_completion(
    request_id: String,
    model: String,
    prompt_tokens: u32,
    mut stream_handle: StreamHandle,
    tokenizer: Arc<Tokenizer>,
    include_usage: bool,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let created = chrono::Utc::now().timestamp();
        let chunk_id = format!("cmpl-{}", request_id);
        let mut completion_tokens: u32 = 0;
        let mut decoder = IncrementalDecoder::new(tokenizer);

        let make_content_chunk = |chunk_id: &str, model: &str, text: String| CompletionChunk {
            id: chunk_id.to_string(),
            object: "text_completion".to_string(),
            created,
            model: model.to_string(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text,
                finish_reason: None,
            }],
            usage: None,
        };

        while let Some(chunk) = stream_handle.recv().await {
            match chunk.chunk_type {
                ChunkType::Token => {
                    completion_tokens += 1;
                    let Some(token_id) = chunk.token_id else { continue };

                    match decoder.push(token_id as u32) {
                        Ok(Some(delta)) => {
                            let payload = make_content_chunk(&chunk_id, &model, delta);
                            yield Ok(json_event(&request_id, &payload));
                        }
                        Ok(None) => {}
                        Err(e) => {
                            tracing::error!(
                                request_id = %request_id,
                                error = %e,
                                "tokenizer decode failed; terminating stream"
                            );
                            stream_handle.mark_finished();
                            let err_chunk = CompletionChunk {
                                id: chunk_id.clone(),
                                object: "text_completion".to_string(),
                                created,
                                model: model.clone(),
                                choices: vec![CompletionChunkChoice {
                                    index: 0,
                                    text: String::new(),
                                    finish_reason: Some("error".to_string()),
                                }],
                                usage: None,
                            };
                            yield Ok(json_event(&request_id, &err_chunk));
                            yield Ok(Event::default().data("[DONE]"));
                            return;
                        }
                    }
                }
                ChunkType::Done => {
                    stream_handle.mark_finished();

                    if let Ok(Some(tail)) = decoder.flush() {
                        let payload = make_content_chunk(&chunk_id, &model, tail);
                        yield Ok(json_event(&request_id, &payload));
                    }

                    let finish_reason = chunk.finish_reason.unwrap_or_else(|| "stop".to_string());
                    let finish_chunk = CompletionChunk {
                        id: chunk_id.clone(),
                        object: "text_completion".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![CompletionChunkChoice {
                            index: 0,
                            text: String::new(),
                            finish_reason: Some(finish_reason),
                        }],
                        usage: None,
                    };
                    yield Ok(json_event(&request_id, &finish_chunk));

                    if include_usage {
                        let usage_chunk = CompletionChunk {
                            id: chunk_id.clone(),
                            object: "text_completion".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![],
                            usage: Some(Usage {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens: prompt_tokens.saturating_add(completion_tokens),
                            }),
                        };
                        yield Ok(json_event(&request_id, &usage_chunk));
                    }

                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                ChunkType::Error => {
                    stream_handle.mark_finished();
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text(""))
}
