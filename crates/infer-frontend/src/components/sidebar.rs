use dioxus::prelude::*;
use crate::state::conversation::Conversation;

#[component]
pub fn Sidebar(
    conversations: Signal<Vec<Conversation>>,
    active_id: Signal<String>,
    on_new_chat: EventHandler<()>,
    on_select: EventHandler<String>,
    on_delete: EventHandler<String>,
    collapsed: Signal<bool>,
) -> Element {
    let is_collapsed = collapsed();

    rsx! {
        aside {
            class: if is_collapsed {
                "w-0 lg:w-16 transition-all duration-300 overflow-hidden flex flex-col glass-panel rounded-2xl"
            } else {
                "w-72 transition-all duration-300 flex flex-col glass-panel rounded-2xl"
            },

            // Header
            div {
                class: "p-4 border-b border-white/5",

                div {
                    class: "flex items-center justify-between",

                    if !is_collapsed {
                        h1 {
                            class: "text-lg font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent",
                            "RustInfer"
                        }
                    }

                    button {
                        class: "p-2 rounded-lg hover:bg-white/5 transition-colors text-[var(--color-text-secondary)]",
                        onclick: move |_| collapsed.set(!is_collapsed),
                        // Hamburger / Close icon
                        if is_collapsed {
                            svg {
                                class: "w-5 h-5",
                                fill: "none",
                                stroke: "currentColor",
                                stroke_width: "2",
                                view_box: "0 0 24 24",
                                path { d: "M4 6h16M4 12h16M4 18h16" }
                            }
                        } else {
                            svg {
                                class: "w-5 h-5",
                                fill: "none",
                                stroke: "currentColor",
                                stroke_width: "2",
                                view_box: "0 0 24 24",
                                path { d: "M11 19l-7-7 7-7M18 19l-7-7 7-7" }
                            }
                        }
                    }
                }
            }

            if !is_collapsed {
                // New Chat button
                div {
                    class: "p-3",
                    button {
                        class: "w-full flex items-center gap-2 px-4 py-2.5 rounded-xl border border-dashed border-[var(--color-border)] hover:border-[var(--color-accent)] hover:bg-white/5 transition-all text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]",
                        onclick: move |_| on_new_chat.call(()),

                        svg {
                            class: "w-4 h-4",
                            fill: "none",
                            stroke: "currentColor",
                            stroke_width: "2",
                            view_box: "0 0 24 24",
                            path { d: "M12 4v16m8-8H4" }
                        }
                        span { class: "text-sm font-medium", "New Chat" }
                    }
                }

                // Conversation list
                div {
                    class: "flex-1 overflow-y-auto px-3 space-y-1",

                    for conv in conversations.read().iter().rev() {
                        {
                            let conv_id = conv.id.clone();
                            let conv_id2 = conv.id.clone();
                            let is_active = conv.id == active_id();
                            let title = conv.title.clone();

                            rsx! {
                                div {
                                    key: "{conv_id}",
                                    class: if is_active {
                                        "group flex items-center gap-2 px-3 py-2.5 rounded-xl bg-white/10 border border-white/10 animate-fade-in cursor-pointer"
                                    } else {
                                        "group flex items-center gap-2 px-3 py-2.5 rounded-xl hover:bg-white/5 transition-colors cursor-pointer"
                                    },
                                    onclick: move |_| on_select.call(conv_id.clone()),

                                    // Chat icon
                                    svg {
                                        class: "w-4 h-4 shrink-0 text-[var(--color-text-muted)]",
                                        fill: "none",
                                        stroke: "currentColor",
                                        stroke_width: "2",
                                        view_box: "0 0 24 24",
                                        path { d: "M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" }
                                    }

                                    span {
                                        class: "flex-1 text-sm truncate text-[var(--color-text-secondary)]",
                                        "{title}"
                                    }

                                    // Delete button (show on hover)
                                    button {
                                        class: "opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-500/20 hover:text-red-400 transition-all",
                                        onclick: move |e| {
                                            e.stop_propagation();
                                            on_delete.call(conv_id2.clone());
                                        },
                                        svg {
                                            class: "w-3.5 h-3.5",
                                            fill: "none",
                                            stroke: "currentColor",
                                            stroke_width: "2",
                                            view_box: "0 0 24 24",
                                            path { d: "M6 18L18 6M6 6l12 12" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Model selector at bottom
                div {
                    class: "p-3 border-t border-white/5",
                    crate::components::model_selector::ModelSelector {}
                }
            }
        }
    }
}
