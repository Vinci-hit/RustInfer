//! SSE 流式输出逻辑
//!
//! 将 `mpsc::Receiver<StreamChunk>` 转换为 Axum SSE 流。
//! UTF-8 安全的渐进式解码由 [`super::decoder::IncrementalDecoder`] 负责，本模块只
//! 关心 SSE chunk 的协议封装与流终止语义。
//!
//! Chat 和 Completion 两个端点的流循环逻辑完全一致（token → 增量解码 → 下发，
//! Done → flush → finish chunk → 可选 usage → `[DONE]`），只有 chunk 的 JSON
//! 形状不同。差异被抽象进 [`ChunkShape`]，循环本身只写一遍。

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
            // Emit a distinguishable `error` event rather than `[DONE]`: `[DONE]`
            // is the OpenAI stream *terminator* and would masquerade a failure as
            // successful completion, leaving the client unable to tell that the
            // response was truncated.
            let body = serde_json::json!({
                "error": {
                    "message": format!("failed to serialize streaming payload: {error}"),
                    "type": "internal_error",
                }
            });
            Event::default()
                .event("error")
                .data(body.to_string())
        }
    }
}

/// The per-endpoint chunk shape. Each method builds one SSE payload; the stream
/// loop in [`run_stream`] is written once against this trait. A chunk carries
/// the shared header fields (`id`/`created`/`model`) captured at construction.
trait ChunkShape {
    type Chunk: serde::Serialize + Send;

    /// Opening chunk, if the endpoint emits one before any content (chat sends
    /// the `assistant` role delta; completion sends nothing).
    fn opening(&self) -> Option<Self::Chunk>;
    /// A content delta chunk.
    fn content(&self, delta: String) -> Self::Chunk;
    /// The terminal chunk carrying `finish_reason` and no content.
    fn finish(&self, finish_reason: String) -> Self::Chunk;
    /// The optional usage-only chunk (empty `choices`).
    fn usage(&self, usage: Usage) -> Self::Chunk;
}

struct ChatShape {
    chunk_id: String,
    created: i64,
    model: String,
}

impl ChunkShape for ChatShape {
    type Chunk = ChatCompletionChunk;

    fn opening(&self) -> Option<Self::Chunk> {
        Some(ChatCompletionChunk {
            id: self.chunk_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
            usage: None,
        })
    }

    fn content(&self, delta: String) -> Self::Chunk {
        ChatCompletionChunk {
            id: self.chunk_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: None, content: Some(delta) },
                finish_reason: None,
            }],
            usage: None,
        }
    }

    fn finish(&self, finish_reason: String) -> Self::Chunk {
        ChatCompletionChunk {
            id: self.chunk_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(finish_reason),
            }],
            usage: None,
        }
    }

    fn usage(&self, usage: Usage) -> Self::Chunk {
        ChatCompletionChunk {
            id: self.chunk_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![],
            usage: Some(usage),
        }
    }
}

struct CompletionShape {
    chunk_id: String,
    created: i64,
    model: String,
}

impl ChunkShape for CompletionShape {
    type Chunk = CompletionChunk;

    fn opening(&self) -> Option<Self::Chunk> {
        None
    }

    fn content(&self, text: String) -> Self::Chunk {
        CompletionChunk {
            id: self.chunk_id.clone(),
            object: "text_completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![CompletionChunkChoice { index: 0, text, finish_reason: None }],
            usage: None,
        }
    }

    fn finish(&self, finish_reason: String) -> Self::Chunk {
        CompletionChunk {
            id: self.chunk_id.clone(),
            object: "text_completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text: String::new(),
                finish_reason: Some(finish_reason),
            }],
            usage: None,
        }
    }

    fn usage(&self, usage: Usage) -> Self::Chunk {
        CompletionChunk {
            id: self.chunk_id.clone(),
            object: "text_completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![],
            usage: Some(usage),
        }
    }
}

/// The one SSE loop, shared by both endpoints. `shape` supplies the chunk JSON;
/// `on_first_content` fires the first time a content delta is produced (chat
/// uses it for the TTFT trace; completion passes a no-op).
#[allow(clippy::too_many_arguments)]
fn run_stream<S, F>(
    request_id: String,
    prompt_tokens: u32,
    mut stream_handle: StreamHandle,
    tokenizer: Arc<Tokenizer>,
    include_usage: bool,
    permit: tokio::sync::OwnedSemaphorePermit,
    shape: S,
    mut on_first_content: F,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    S: ChunkShape + Send + 'static,
    F: FnMut() + Send + 'static,
{
    let stream = async_stream::stream! {
        // Hold the admission permit for the lifetime of the stream. Binding it
        // INSIDE the generator is what ties its release to stream completion or
        // client disconnect (generator drop); binding it outside would drop it
        // when this fn returns, making the gate a no-op for streaming.
        let _permit = permit;
        let mut completion_tokens: u32 = 0;
        let mut first_content_sent = false;
        // Set on the Done/Error arms. False after the loop means the channel
        // closed without a terminal chunk (ZMQ thread dropped the stream:
        // slow-consumer cancel or scheduler link death) — surfaced to the
        // client below instead of silently closing the connection.
        let mut ended_clean = false;
        let mut decoder = IncrementalDecoder::new(tokenizer);

        if let Some(opening) = shape.opening() {
            yield Ok(json_event(&request_id, &opening));
        }

        while let Some(chunk) = stream_handle.recv().await {
            match chunk.chunk_type {
                ChunkType::Token => {
                    completion_tokens += 1;
                    let Some(token_id) = chunk.token_id else { continue };

                    match decoder.push(token_id as u32) {
                        Ok(Some(delta)) => {
                            if !first_content_sent {
                                first_content_sent = true;
                                on_first_content();
                            }
                            yield Ok(json_event(&request_id, &shape.content(delta)));
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
                            yield Ok(json_event(&request_id, &shape.finish("error".to_string())));
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
                    ended_clean = true;

                    // 流结束前 flush decoder：把任何被滞留的「未确认」字符吐出去。
                    // 走到这里说明流自然结束，残留的 U+FFFD 是真实的不可解码序列，
                    // 应当下发给客户端而不是丢弃。
                    if let Ok(Some(tail)) = decoder.flush() {
                        yield Ok(json_event(&request_id, &shape.content(tail)));
                    }

                    let finish_reason = chunk.finish_reason.unwrap_or_else(|| "stop".to_string());
                    yield Ok(json_event(&request_id, &shape.finish(finish_reason)));

                    if include_usage {
                        let usage = Usage {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens.saturating_add(completion_tokens),
                        };
                        yield Ok(json_event(&request_id, &shape.usage(usage)));
                    }

                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                ChunkType::Error => {
                    stream_handle.mark_finished();
                    ended_clean = true;
                    // Surface the failure instead of masquerading as success —
                    // same policy as the serializer-failure path in
                    // `json_event`: an explicit `error` SSE event carrying the
                    // engine message, a finish chunk with `finish_reason:
                    // "error"`, then the `[DONE]` terminator.
                    let message = chunk
                        .finish_reason
                        .unwrap_or_else(|| "inference engine error".to_string());
                    let body = serde_json::json!({
                        "error": { "message": message, "type": "engine_error" }
                    });
                    yield Ok(Event::default().event("error").data(body.to_string()));
                    yield Ok(json_event(&request_id, &shape.finish("error".to_string())));
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }

        if !ended_clean {
            // Channel closed mid-stream with no terminal chunk. mark_finished
            // suppresses the Drop-cancel (the server side already dropped the
            // request when it closed the channel).
            stream_handle.mark_finished();
            let body = serde_json::json!({
                "error": {
                    "message": "stream aborted by server (slow consumer or scheduler failure)",
                    "type": "engine_error",
                }
            });
            yield Ok(Event::default().event("error").data(body.to_string()));
            yield Ok(json_event(&request_id, &shape.finish("error".to_string())));
            yield Ok(Event::default().data("[DONE]"));
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text(""))
}

/// 构建 Chat Completion SSE 流
#[allow(clippy::too_many_arguments)]
pub fn stream_chat_completion(
    request_id: String,
    model: String,
    prompt_tokens: u32,
    stream_handle: StreamHandle,
    tokenizer: Arc<Tokenizer>,
    include_usage: bool,
    request_start: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let shape = ChatShape {
        chunk_id: format!("chatcmpl-{}", request_id),
        created: chrono::Utc::now().timestamp(),
        model,
    };
    let stream_started = Instant::now();
    let trace_id = request_id.clone();
    run_stream(
        request_id,
        prompt_tokens,
        stream_handle,
        tokenizer,
        include_usage,
        permit,
        shape,
        move || {
            tracing::debug!(
                request_id = %trace_id,
                server_ttft_ms = request_start.elapsed().as_secs_f64() * 1000.0,
                since_stream_ms = stream_started.elapsed().as_secs_f64() * 1000.0,
                "TTFT_TRACE: chat first content chunk"
            );
        },
    )
}

/// 构建 Text Completion SSE 流
pub fn stream_completion(
    request_id: String,
    model: String,
    prompt_tokens: u32,
    stream_handle: StreamHandle,
    tokenizer: Arc<Tokenizer>,
    include_usage: bool,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let shape = CompletionShape {
        chunk_id: format!("cmpl-{}", request_id),
        created: chrono::Utc::now().timestamp(),
        model,
    };
    run_stream(
        request_id,
        prompt_tokens,
        stream_handle,
        tokenizer,
        include_usage,
        permit,
        shape,
        || {},
    )
}
