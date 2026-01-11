use dioxus::prelude::*;
use crate::state::conversation::{Message, MessageMetrics};
use crate::api::client::{ApiClient, ChatRequest, ChatMessage as ApiChatMessage};

#[component]
pub fn ChatInterface() -> Element {
    let mut messages = use_signal(Vec::<Message>::new);
    let mut input_text = use_signal(String::new);
    let mut is_generating = use_signal(|| false);

    let mut send_message = move || {
        let text = input_text().clone();

        if text.trim().is_empty() {
            return;
        }

        // Add user message
        let user_msg = Message::user(text.clone());
        messages.write().push(user_msg);

        // Clear input
        input_text.set(String::new());
        is_generating.set(true);

        // Build API request
        let api_messages: Vec<ApiChatMessage> = messages.read().iter().map(|m| {
            ApiChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            }
        }).collect();

        let request = ChatRequest {
            model: "llama3".to_string(),
            messages: api_messages,
            max_tokens: Some(512),
            stream: false,
        };

        let api_client = ApiClient::new();

        spawn(async move {
            match api_client.chat_completion(request).await {
                Ok(response) => {
                    if let Some(choice) = response.choices.first() {
                        let metrics = response.usage.performance.map(|p| MessageMetrics {
                            prefill_ms: p.prefill_ms,
                            decode_ms: p.decode_ms,
                            tokens_per_second: p.tokens_per_second,
                            total_tokens: response.usage.completion_tokens,
                        });

                        let assistant_msg = Message::assistant(
                            choice.message.content.clone(),
                            metrics,
                        );

                        messages.write().push(assistant_msg);
                    }
                    is_generating.set(false);
                }
                Err(e) => {
                    let error_msg = Message::assistant(
                        format!("Error: {}", e),
                        None,
                    );
                    messages.write().push(error_msg);
                    is_generating.set(false);
                }
            }
        });
    };

    rsx! {
        div {
            class: "flex flex-col h-full bg-gray-800 rounded-lg shadow-lg",

            // Header
            div {
                class: "bg-gray-700 p-4 rounded-t-lg border-b border-gray-600",
                h1 { class: "text-2xl font-bold", "RustInfer Chat" }
                p { class: "text-sm text-gray-400", "Multi-round conversation with Llama3" }
            }

            // Messages
            div {
                class: "flex-1 overflow-y-auto p-4 space-y-4",
                for msg in messages.read().iter() {
                    crate::components::message_bubble::MessageBubble {
                        key: "{msg.id}",
                        message: msg.clone()
                    }
                }

                if is_generating() {
                    crate::components::streaming_indicator::StreamingIndicator {}
                }
            }

            // Input
            div {
                class: "bg-gray-700 p-4 rounded-b-lg border-t border-gray-600",
                div {
                    class: "flex gap-2",

                    input {
                        class: "flex-1 bg-gray-600 text-white rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500",
                        r#type: "text",
                        placeholder: "Type your message...",
                        value: "{input_text()}",
                        oninput: move |e| {
                            input_text.set(e.value());
                        },
                        onkeypress: move |e| {
                            if e.key() == Key::Enter && !is_generating() && !input_text().trim().is_empty() {
                                send_message();
                            }
                        },
                        disabled: is_generating(),
                    }

                    button {
                        class: "bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded font-semibold disabled:opacity-50 disabled:cursor-not-allowed",
                        onclick: move |_| send_message(),
                        disabled: is_generating() || input_text().trim().is_empty(),
                        "Send"
                    }
                }
            }
        }
    }
}
