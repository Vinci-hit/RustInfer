use dioxus::prelude::*;

/// Component for displaying a streaming message
#[component]
pub fn StreamingMessage(content: String, is_streaming: bool) -> Element {
    rsx! {
        div {
            class: "flex flex-col mr-auto max-w-2xl",

            div {
                class: "bg-gray-700 rounded-lg p-4 shadow-md",

                div {
                    class: "text-xs font-semibold mb-1 text-gray-300 uppercase",
                    "assistant"
                }

                div {
                    class: "markdown-content leading-relaxed",
                    dangerous_inner_html: "{crate::utils::markdown::render_markdown(&content)}"
                }

                if is_streaming {
                    div {
                        class: "mt-2 flex items-center gap-2 text-gray-400 text-xs",
                        span { class: "w-2 h-2 bg-blue-500 rounded-full animate-pulse" }
                        span { "Generating..." }
                    }
                }
            }
        }
    }
}

/// Stream response from Server-Sent Events endpoint
/// Returns a signal that updates with each chunk
pub fn use_stream_response(url: String) -> (Signal<String>, Signal<bool>) {
    let mut content = use_signal(String::new);
    let is_active = use_signal(|| false);

    // TODO: Implement actual SSE streaming when server supports it
    // For now, this is a placeholder for the streaming functionality

    (content, is_active)
}

