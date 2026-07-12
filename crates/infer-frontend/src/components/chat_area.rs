use crate::api::client::{ApiClient, ChatSseParser, ChatStreamEvent};
use crate::api::types::{ChatMessage, ChatRequest, Usage};
use crate::state::conversation::{Conversation, Message, MessageMetrics};
use dioxus::prelude::*;
use futures_util::StreamExt;

fn handle_stream_events(
    events: Vec<Result<ChatStreamEvent, String>>,
    conversations: &mut Signal<Vec<Conversation>>,
    active_id: &str,
    assistant_id: &str,
    full_content: &mut String,
    final_usage: &mut Option<Usage>,
) -> Result<bool, String> {
    for event in events {
        match event? {
            ChatStreamEvent::Done => return Ok(true),
            ChatStreamEvent::Error(message) => return Err(message),
            ChatStreamEvent::Chunk(chunk) => {
                if let Some(content) = chunk
                    .choices
                    .first()
                    .and_then(|choice| choice.delta.content.as_deref())
                {
                    full_content.push_str(content);
                    let mut convs = conversations.write();
                    if let Some(message) = convs
                        .iter_mut()
                        .find(|conversation| conversation.id == active_id)
                        .and_then(|conversation| {
                            conversation
                                .messages
                                .iter_mut()
                                .find(|message| message.id == assistant_id)
                        })
                    {
                        message.content.clone_from(full_content);
                    }
                }
                if let Some(usage) = chunk.usage {
                    *final_usage = Some(usage);
                }
            }
        }
    }

    Ok(false)
}

#[component]
pub fn ChatArea(conversations: Signal<Vec<Conversation>>, active_id: Signal<String>) -> Element {
    let mut is_generating = use_signal(|| false);
    let mut live_status = use_signal(|| "Connected".to_string());
    let mut stream_error = use_signal(|| None::<String>);

    // 获取当前活跃对话
    let active_conv = conversations
        .read()
        .iter()
        .find(|c| c.id == active_id())
        .cloned();

    let messages: Vec<Message> = active_conv
        .as_ref()
        .map(|c| c.messages.clone())
        .unwrap_or_default();

    let model = active_conv
        .as_ref()
        .map(|c| c.model.clone())
        .unwrap_or_else(|| "llama3".to_string());

    let mut send_message = move |text: String| {
        if text.trim().is_empty() || is_generating() {
            return;
        }

        let model = model.clone();
        stream_error.set(None);
        live_status.set("Generating response".to_string());

        // 添加用户消息
        let user_msg = Message::user(text.clone());
        {
            let mut convs = conversations.write();
            if let Some(conv) = convs.iter_mut().find(|c| c.id == active_id()) {
                conv.messages.push(user_msg);
                conv.updated_at = chrono::Utc::now().timestamp();
                conv.auto_title();
            }
        }

        // 添加 streaming 占位消息
        let assistant_msg = Message::assistant_streaming();
        let assistant_id = assistant_msg.id.clone();
        {
            let mut convs = conversations.write();
            if let Some(conv) = convs.iter_mut().find(|c| c.id == active_id()) {
                conv.messages.push(assistant_msg);
            }
        }

        is_generating.set(true);

        // 构建 API 请求
        let api_messages: Vec<ChatMessage> = {
            let convs = conversations.read();
            convs
                .iter()
                .find(|c| c.id == active_id())
                .map(|c| {
                    c.messages
                        .iter()
                        .filter(|m| !m.is_streaming)
                        .map(|m| ChatMessage {
                            role: m.role.clone(),
                            content: m.content.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default()
        };

        let request = ChatRequest {
            model,
            messages: api_messages,
            max_tokens: Some(2048),
            stream: true,
            temperature: None,
        };

        let active_id_val = active_id();

        spawn(async move {
            let client = ApiClient::default();

            match client.chat_completion_stream(request).await {
                Ok(response) => {
                    // Consume the browser response body as bytes so each SSE
                    // event updates the conversation as soon as it arrives.
                    let mut body = response.bytes_stream();
                    let mut parser = ChatSseParser::default();
                    let mut full_content = String::new();
                    let mut final_usage = None::<Usage>;
                    let mut failure = None;
                    let mut reached_done = false;

                    while let Some(next) = body.next().await {
                        let bytes = match next {
                            Ok(bytes) => bytes,
                            Err(error) => {
                                failure = Some(format!("response stream failed: {error}"));
                                break;
                            }
                        };

                        match handle_stream_events(
                            parser.push(&bytes),
                            &mut conversations,
                            &active_id_val,
                            &assistant_id,
                            &mut full_content,
                            &mut final_usage,
                        ) {
                            Ok(done) => {
                                if done {
                                    reached_done = true;
                                    break;
                                }
                            }
                            Err(error) => {
                                failure = Some(error);
                                break;
                            }
                        }
                    }

                    if failure.is_none() && !reached_done {
                        if let Err(error) = handle_stream_events(
                            parser.finish(),
                            &mut conversations,
                            &active_id_val,
                            &assistant_id,
                            &mut full_content,
                            &mut final_usage,
                        ) {
                            failure = Some(error);
                        }
                    }

                    // 标记 streaming 完成
                    let metrics = final_usage.map(|u| MessageMetrics {
                        prefill_ms: u.performance.as_ref().map(|p| p.prefill_ms).unwrap_or(0),
                        decode_ms: u.performance.as_ref().map(|p| p.decode_ms).unwrap_or(0),
                        tokens_per_second: u
                            .performance
                            .as_ref()
                            .map(|p| p.tokens_per_second)
                            .unwrap_or(0.0),
                        total_tokens: u.completion_tokens,
                    });

                    let mut convs = conversations.write();
                    if let Some(conv) = convs.iter_mut().find(|c| c.id == active_id_val) {
                        if let Some(msg) = conv.messages.iter_mut().find(|m| m.id == assistant_id) {
                            msg.is_streaming = false;
                            msg.metrics = metrics;
                            if msg.content.is_empty() {
                                msg.content = if failure.is_some() {
                                    "The response could not be completed.".to_string()
                                } else {
                                    full_content
                                };
                            }
                        }
                    }

                    drop(convs);
                    if let Some(error) = failure {
                        live_status.set("Response failed".to_string());
                        stream_error.set(Some(error));
                    } else {
                        live_status.set("Response complete".to_string());
                    }
                }
                Err(e) => {
                    let error = e.to_string();
                    let mut convs = conversations.write();
                    if let Some(conv) = convs.iter_mut().find(|c| c.id == active_id_val) {
                        if let Some(msg) = conv.messages.iter_mut().find(|m| m.id == assistant_id) {
                            msg.content = "The response could not be completed.".to_string();
                            msg.is_streaming = false;
                        }
                    }
                    drop(convs);
                    live_status.set("Response failed".to_string());
                    stream_error.set(Some(error));
                }
            }
            is_generating.set(false);
        });
    };

    rsx! {
        div {
            class: "flex-1 flex flex-col min-w-0 glass-panel rounded-2xl overflow-hidden",

            // Chat header
            div {
                class: "px-6 py-4 border-b border-white/5 flex items-center justify-between",

                div {
                    h2 {
                        class: "text-base font-semibold text-[var(--color-text-primary)]",
                        "{active_conv.as_ref().map(|c| c.title.as_str()).unwrap_or(\"RustInfer Chat\")}"
                    }
                    p {
                        class: "text-xs text-[var(--color-text-muted)] mt-0.5",
                        "Powered by local LLM inference"
                    }
                }

                // Connection status indicator
                div {
                    class: "flex items-center gap-2",
                    role: "status",
                    aria_live: "polite",
                    aria_atomic: "true",
                    div {
                        aria_hidden: "true",
                        class: if stream_error().is_some() {
                            "w-2 h-2 rounded-full bg-[var(--color-error)]"
                        } else if is_generating() {
                            "w-2 h-2 rounded-full bg-[var(--color-warning)] animate-pulse"
                        } else {
                            "w-2 h-2 rounded-full bg-[var(--color-success)]"
                        }
                    }
                    span { class: "text-xs text-[var(--color-text-muted)]", "{live_status}" }
                }
            }

            if let Some(error) = stream_error() {
                div {
                    class: "px-6 py-3 border-b border-red-400/20 bg-red-500/10 text-sm text-red-200",
                    role: "alert",
                    aria_live: "assertive",
                    aria_atomic: "true",
                    "Response failed: {error}"
                }
            }

            // Messages area
            div {
                class: "flex-1 overflow-y-auto px-6 py-4 space-y-4",
                id: "messages-container",

                if messages.is_empty() {
                    // Empty state
                    div {
                        class: "flex flex-col items-center justify-center h-full text-center animate-fade-in",

                        div {
                            class: "w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 flex items-center justify-center mb-4",
                            svg {
                                class: "w-8 h-8 text-indigo-400",
                                fill: "none",
                                stroke: "currentColor",
                                stroke_width: "1.5",
                                view_box: "0 0 24 24",
                                path { d: "M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" }
                            }
                        }

                        h3 { class: "text-lg font-semibold text-[var(--color-text-primary)] mb-2", "Start a conversation" }
                        p { class: "text-sm text-[var(--color-text-muted)] max-w-sm",
                            "Send a message to begin chatting with the local LLM. Responses stream in real-time."
                        }
                    }
                } else {
                    for msg in messages.iter() {
                        crate::components::message_bubble::MessageBubble {
                            key: "{msg.id}",
                            message: msg.clone()
                        }
                    }
                }
            }

            // Input area
            crate::components::message_input::MessageInput {
                on_send: move |text: String| send_message(text),
                is_disabled: is_generating(),
            }
        }
    }
}
