//! SSE 流式输出逻辑
//!
//! 将 `mpsc::Receiver<StreamChunk>` 转换为 Axum SSE 流。
//! 处理增量 UTF-8 decode，确保不发出不完整的字符。

use crate::client::StreamHandle;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::Stream;
use infer_protocol::scheduler_to_server::ChunkType;
use std::convert::Infallible;
use std::time::Duration;
use tokenizers::Tokenizer;

use super::types::*;

fn incremental_decode_delta(previous: &mut String, full_text: String) -> Option<String> {
    if full_text == *previous {
        return None;
    }

    let mut common_len = 0usize;
    let mut prev_chars = previous.char_indices();
    let mut full_chars = full_text.char_indices();

    loop {
        match (prev_chars.next(), full_chars.next()) {
            (Some((prev_idx, prev_ch)), Some((full_idx, full_ch))) if prev_ch == full_ch => {
                common_len = prev_idx + prev_ch.len_utf8();
                debug_assert_eq!(common_len, full_idx + full_ch.len_utf8());
            }
            _ => break,
        }
    }

    let delta = full_text[common_len..].to_string();
    *previous = full_text;

    if delta.is_empty() { None } else { Some(delta) }
}

/// 构建 Chat Completion SSE 流
pub fn stream_chat_completion(
    request_id: String,
    model: String,
    prompt_tokens: u32,
    mut stream_handle: StreamHandle,
    tokenizer: Tokenizer,
    include_usage: bool,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let created = chrono::Utc::now().timestamp();
        let chunk_id = format!("chatcmpl-{}", request_id);
        let mut completion_tokens: u32 = 0;

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
        yield Ok(Event::default().data(serde_json::to_string(&first_chunk).unwrap()));

        // 逐 chunk 接收并转发。
        let mut token_buffer: Vec<u32> = Vec::new();
        let mut decoded_text = String::new();

        while let Some(chunk) = stream_handle.recv().await {
            match chunk.chunk_type {
                ChunkType::Token => {
                    completion_tokens += 1;

                    if let Some(token_id) = chunk.token_id {
                        token_buffer.push(token_id as u32);

                        // 增量 decode：尝试 decode 整个 buffer，取新增部分
                        let full_text = tokenizer.decode(&token_buffer, true)
                            .unwrap_or_default();

                        if let Some(new_text) = incremental_decode_delta(&mut decoded_text, full_text) {
                            let content_chunk = ChatCompletionChunk {
                                id: chunk_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model.clone(),
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: Delta {
                                        role: None,
                                        content: Some(new_text),
                                    },
                                    finish_reason: None,
                                }],
                                usage: None,
                            };
                            yield Ok(Event::default().data(
                                serde_json::to_string(&content_chunk).unwrap()
                            ));
                        }
                    }
                }
                ChunkType::Done => {
                    // 发送 finish chunk
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
                    yield Ok(Event::default().data(
                        serde_json::to_string(&finish_chunk).unwrap()
                    ));

                    // 如果 include_usage，发送 usage chunk
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
                                total_tokens: prompt_tokens + completion_tokens,
                            }),
                        };
                        yield Ok(Event::default().data(
                            serde_json::to_string(&usage_chunk).unwrap()
                        ));
                    }

                    stream_handle.mark_finished();
                    // [DONE] 标记
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                ChunkType::Error => {
                    stream_handle.mark_finished();
                    // 错误时也发 [DONE] 关闭流
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
    tokenizer: Tokenizer,
    include_usage: bool,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let created = chrono::Utc::now().timestamp();
        let chunk_id = format!("cmpl-{}", request_id);
        let mut completion_tokens: u32 = 0;

        let mut token_buffer: Vec<u32> = Vec::new();
        let mut decoded_text = String::new();

        while let Some(chunk) = stream_handle.recv().await {
            match chunk.chunk_type {
                ChunkType::Token => {
                    completion_tokens += 1;

                    if let Some(token_id) = chunk.token_id {
                        token_buffer.push(token_id as u32);

                        let full_text = tokenizer.decode(&token_buffer, true)
                            .unwrap_or_default();

                        if let Some(new_text) = incremental_decode_delta(&mut decoded_text, full_text) {
                            let content_chunk = CompletionChunk {
                                id: chunk_id.clone(),
                                object: "text_completion".to_string(),
                                created,
                                model: model.clone(),
                                choices: vec![CompletionChunkChoice {
                                    index: 0,
                                    text: new_text,
                                    finish_reason: None,
                                }],
                                usage: None,
                            };
                            yield Ok(Event::default().data(
                                serde_json::to_string(&content_chunk).unwrap()
                            ));
                        }
                    }
                }
                ChunkType::Done => {
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
                    yield Ok(Event::default().data(
                        serde_json::to_string(&finish_chunk).unwrap()
                    ));

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
                                total_tokens: prompt_tokens + completion_tokens,
                            }),
                        };
                        yield Ok(Event::default().data(
                            serde_json::to_string(&usage_chunk).unwrap()
                        ));
                    }

                    stream_handle.mark_finished();
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
