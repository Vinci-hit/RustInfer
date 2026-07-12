use crate::state::conversation::Message;
use crate::utils::markdown::render_markdown;
use dioxus::prelude::*;

#[component]
pub fn MessageBubble(message: Message) -> Element {
    let is_user = message.role == "user";
    let is_streaming = message.is_streaming;

    rsx! {
        div {
            class: "animate-slide-up",

            div {
                class: if is_user {
                    "flex justify-end"
                } else {
                    "flex justify-start"
                },

                div {
                    class: if is_user {
                        "max-w-[75%] rounded-2xl rounded-br-md px-5 py-3 bg-gradient-to-br from-indigo-600 to-purple-600 text-white shadow-lg shadow-indigo-500/20"
                    } else {
                        "max-w-[85%] rounded-2xl rounded-bl-md px-5 py-3 bg-[var(--color-surface-2)] border border-[var(--color-border)] text-[var(--color-text-primary)]"
                    },

                    // Role label
                    div {
                        class: "flex items-center gap-2 mb-2",

                        // Avatar
                        div {
                            class: if is_user {
                                "w-5 h-5 rounded-full bg-white/20 flex items-center justify-center"
                            } else {
                                "w-5 h-5 rounded-full bg-gradient-to-br from-emerald-400 to-cyan-400 flex items-center justify-center"
                            },
                            span {
                                class: "text-[10px] font-bold",
                                if is_user { "U" } else { "AI" }
                            }
                        }

                        span {
                            class: if is_user {
                                "text-xs font-medium text-white/70 uppercase"
                            } else {
                                "text-xs font-medium text-[var(--color-text-muted)] uppercase"
                            },
                            if is_user { "You" } else { "Assistant" }
                        }
                    }

                    // Content
                    if is_user {
                        div {
                            class: "text-sm leading-relaxed whitespace-pre-wrap",
                            "{message.content}"
                        }
                    } else if is_streaming && message.content.is_empty() {
                        crate::components::streaming_indicator::StreamingIndicator {}
                    } else {
                        div {
                            class: "text-sm leading-relaxed markdown-body",
                            dangerous_inner_html: "{render_markdown(&message.content)}"
                        }
                        if is_streaming {
                            span {
                                class: "inline-block w-2 h-4 bg-[var(--color-accent)] animate-pulse rounded-sm ml-1"
                            }
                        }
                    }

                    // Performance metrics badge
                    if let Some(metrics) = &message.metrics {
                        div {
                            class: "mt-3 pt-2 border-t border-white/10 flex flex-wrap gap-3",

                            div {
                                class: "flex items-center gap-1 text-xs",
                                span { class: "text-emerald-400", "⚡" }
                                span {
                                    class: if is_user { "text-white/60" } else { "text-[var(--color-text-muted)]" },
                                    "{metrics.tokens_per_second:.1} tok/s"
                                }
                            }

                            div {
                                class: "flex items-center gap-1 text-xs",
                                span { class: "text-blue-400", "◆" }
                                span {
                                    class: if is_user { "text-white/60" } else { "text-[var(--color-text-muted)]" },
                                    "Prefill {metrics.prefill_ms}ms"
                                }
                            }

                            div {
                                class: "flex items-center gap-1 text-xs",
                                span { class: "text-purple-400", "◇" }
                                span {
                                    class: if is_user { "text-white/60" } else { "text-[var(--color-text-muted)]" },
                                    "Decode {metrics.decode_ms}ms"
                                }
                            }

                            div {
                                class: "flex items-center gap-1 text-xs",
                                span { class: "text-yellow-400", "●" }
                                span {
                                    class: if is_user { "text-white/60" } else { "text-[var(--color-text-muted)]" },
                                    "{metrics.total_tokens} tokens"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
