use dioxus::prelude::*;
use crate::state::conversation::Message;

#[component]
pub fn MessageBubble(message: Message) -> Element {
    let is_user = message.role == "user";
    let bg_class = if is_user { "bg-blue-600" } else { "bg-gray-700" };
    let align_class = if is_user { "ml-auto" } else { "mr-auto" };

    rsx! {
        div {
            class: "flex flex-col {align_class} max-w-2xl",

            div {
                class: "{bg_class} rounded-lg p-4 shadow-md",

                div {
                    class: "text-xs font-semibold mb-1 text-gray-300 uppercase",
                    "{message.role}"
                }

                div {
                    class: "whitespace-pre-wrap leading-relaxed",
                    "{message.content}"
                }

                // Performance metrics (only for assistant)
                if let Some(metrics) = &message.metrics {
                    div {
                        class: "mt-3 text-xs text-gray-400 border-t border-gray-600 pt-2 space-y-1",

                        div { class: "flex justify-between",
                            span { "⚡ Prefill: {metrics.prefill_ms}ms" }
                            span { "🔄 Decode: {metrics.decode_ms}ms" }
                        }

                        div { class: "flex justify-between",
                            span { "🚀 Speed: {metrics.tokens_per_second:.2} tok/s" }
                            span { "📝 Tokens: {metrics.total_tokens}" }
                        }
                    }
                }
            }
        }
    }
}
