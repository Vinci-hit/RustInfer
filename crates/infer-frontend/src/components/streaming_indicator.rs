use dioxus::prelude::*;

#[component]
pub fn StreamingIndicator() -> Element {
    rsx! {
        div {
            class: "flex items-center gap-2 py-1",

            div { class: "flex gap-1.5",
                span { class: "w-2 h-2 bg-indigo-400 rounded-full animate-pulse-dot" }
                span { class: "w-2 h-2 bg-indigo-400 rounded-full animate-pulse-dot", style: "animation-delay: 0.2s;" }
                span { class: "w-2 h-2 bg-indigo-400 rounded-full animate-pulse-dot", style: "animation-delay: 0.4s;" }
            }

            span {
                class: "text-xs text-[var(--color-text-muted)]",
                "Thinking..."
            }
        }
    }
}
