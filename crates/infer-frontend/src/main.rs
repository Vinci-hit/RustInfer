#![allow(dead_code)]

use dioxus::prelude::*;

mod api;
mod components;
mod state;
mod utils;

use state::conversation::Conversation;

const CSS: Asset = asset!("/assets/output.css");

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    // 对话列表状态
    let mut conversations = use_signal(|| {
        let initial = Conversation::new("llama3".to_string());
        vec![initial]
    });

    // 当前活跃对话 ID
    let mut active_id = use_signal(|| {
        conversations.read().first().map(|c| c.id.clone()).unwrap_or_default()
    });

    // Sidebar 折叠状态
    let sidebar_collapsed = use_signal(|| false);

    // 新建对话
    let on_new_chat = move |_: ()| {
        let new_conv = Conversation::new("llama3".to_string());
        let new_id = new_conv.id.clone();
        conversations.write().push(new_conv);
        active_id.set(new_id);
    };

    // 选择对话
    let on_select = move |id: String| {
        active_id.set(id);
    };

    // 删除对话
    let on_delete = move |id: String| {
        let mut convs = conversations.write();
        convs.retain(|c| c.id != id);
        // 如果删除了当前活跃对话，切换到第一个
        if active_id() == id {
            if let Some(first) = convs.first() {
                active_id.set(first.id.clone());
            } else {
                // 没有对话了，创建一个新的
                let new_conv = Conversation::new("llama3".to_string());
                active_id.set(new_conv.id.clone());
                convs.push(new_conv);
            }
        }
    };

    rsx! {
        document::Stylesheet { href: CSS }

        // Google Fonts
        document::Link {
            rel: "preconnect",
            href: "https://fonts.googleapis.com"
        }
        document::Link {
            rel: "stylesheet",
            href: "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
        }

        div {
            class: "h-screen w-screen overflow-hidden bg-[var(--color-surface-0)] p-3 flex gap-3",

            // Background gradient decoration
            div {
                class: "fixed inset-0 pointer-events-none",
                div { class: "absolute top-0 left-1/4 w-96 h-96 bg-indigo-600/5 rounded-full blur-3xl" }
                div { class: "absolute bottom-0 right-1/4 w-96 h-96 bg-purple-600/5 rounded-full blur-3xl" }
            }

            // Sidebar
            components::sidebar::Sidebar {
                conversations: conversations,
                active_id: active_id,
                on_new_chat: on_new_chat,
                on_select: on_select,
                on_delete: on_delete,
                collapsed: sidebar_collapsed,
            }

            // Chat Area (主区域)
            components::chat_area::ChatArea {
                conversations: conversations,
                active_id: active_id,
            }

            // Metrics Panel (右侧)
            div {
                class: "hidden lg:block w-72 shrink-0",
                components::metrics_panel::MetricsPanel {}
            }
        }
    }
}
