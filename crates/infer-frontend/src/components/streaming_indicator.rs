use dioxus::prelude::*;

pub fn StreamingIndicator() -> Element {
    rsx! {
        div {
            class: "flex items-center gap-2 text-gray-400 p-4",

            div { class: "flex gap-1",
                span { class: "w-2 h-2 bg-blue-500 rounded-full animate-bounce" }
                span { class: "w-2 h-2 bg-blue-500 rounded-full animate-bounce", style: "animation-delay: 0.1s;" }
                span { class: "w-2 h-2 bg-blue-500 rounded-full animate-bounce", style: "animation-delay: 0.2s;" }
            }

            span { class: "text-sm", "AI is generating response..." }
        }
    }
}
