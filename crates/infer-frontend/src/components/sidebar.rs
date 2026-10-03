use super::icon::Icon;
use crate::state::{
    conversation::Conversation,
    settings::Theme,
    workspace::{Connection, Workspace},
};
use dioxus::prelude::*;

#[component]
pub fn Sidebar(
    conversations: Signal<Vec<Conversation>>,
    active_id: Signal<String>,
    on_new_chat: EventHandler<()>,
    on_select: EventHandler<String>,
    on_delete: EventHandler<String>,
    mobile_open: bool,
    on_settings: EventHandler<()>,
    on_close: EventHandler<()>,
) -> Element {
    let mut workspace = use_context::<Workspace>();
    let mut search = use_signal(String::new);
    let mut deleting = use_signal(|| None::<String>);
    let query = search().to_lowercase();
    let mut filtered: Vec<_> = conversations
        .read()
        .iter()
        .filter(|c| {
            c.title.to_lowercase().contains(&query)
                || c.messages
                    .iter()
                    .any(|m| m.content.to_lowercase().contains(&query))
        })
        .cloned()
        .collect();
    filtered.sort_by_key(|c| std::cmp::Reverse(c.updated_at));
    let count = filtered.len();
    let connection = (workspace.connection)();
    let theme_icon = if workspace.settings.read().theme == Theme::Dark {
        "sun"
    } else {
        "moon"
    };
    rsx! {
        aside { class: if mobile_open { "sidebar is-open" } else { "sidebar" }, aria_label: "对话导航",
            div { class: "brand",
                div { class: "brand-symbol", "R" }
                div { class: "brand-copy", strong { "RustInfer" } span { "本地智能工作台" } }
                button { class: "icon-button mobile-only", aria_label: "关闭导航", onclick: move |_| on_close.call(()), Icon { name: "x", size: 18 } }
            }
            button { class: "new-chat", onclick: move |_| on_new_chat.call(()), Icon { name: "plus", size: 18 } "新建对话" span { "⌘ N" } }
            label { class: "conversation-search",
                Icon { name: "search", size: 16 }
                input { placeholder: "搜索对话", aria_label: "搜索对话", value: "{search}", oninput: move |e| search.set(e.value()) }
            }
            div { class: "sidebar-section-title", span { if query.is_empty() { "最近对话" } else { "搜索结果" } } span { "{count}" } }
            nav { class: "conversation-list", aria_label: "历史会话",
                if filtered.is_empty() { p { class: "sidebar-empty", "没有找到相关对话" } }
                for conversation in filtered {
                    {
                        let id = conversation.id.clone();
                        let select_id = id.clone();
                        let delete_id = id.clone();
                        let active = id == active_id();
                        let generating = conversation.messages.iter().any(|m| m.is_streaming);
                        rsx! {
                            div { class: if active { "conversation-item active" } else { "conversation-item" }, key: "{id}",
                                button { class: "conversation-select", aria_current: if active { "page" } else { "false" }, onclick: move |_| on_select.call(select_id.clone()),
                                    Icon { name: "chat", size: 16 }
                                    span { "{conversation.title}" }
                                    if generating { span { class: "stream-dot", aria_label: "正在生成" } }
                                }
                                button { class: "conversation-delete", title: "删除对话", aria_label: "删除对话", disabled: generating, onclick: move |_| deleting.set(Some(delete_id.clone())), Icon { name: "trash", size: 14 } }
                            }
                        }
                    }
                }
            }
            div { class: "sidebar-bottom",
                div { class: "local-note", Icon { name: "shield", size: 17 } div { strong { "数据，由你掌控" } p { "对话保存在当前浏览器" } } }
                div { class: "server-indicator",
                    span { class: match connection { Connection::Online => "status-dot online", Connection::Connecting => "status-dot connecting", _ => "status-dot offline" } }
                    span { match connection { Connection::Online => "推理服务已连接", Connection::Connecting => "正在连接服务", _ => "推理服务未连接" } }
                    button { class: "icon-button", title: "重新连接", aria_label: "重新连接", onclick: move |_| { let next = *workspace.reconnect.peek() + 1; workspace.reconnect.set(next); }, Icon { name: "refresh", size: 14 } }
                }
                div { class: "sidebar-footer",
                    button { class: "sidebar-settings", onclick: move |_| on_settings.call(()), Icon { name: "settings", size: 17 } "连接与设置" }
                    button { class: "icon-button", title: "切换明暗主题", aria_label: "切换明暗主题", onclick: move |_| {
                        let theme = workspace.settings.peek().theme.clone();
                        workspace.settings.write().theme = if theme == Theme::Dark { Theme::Light } else { Theme::Dark };
                    }, Icon { name: theme_icon, size: 18 } }
                }
            }
            if let Some(id) = deleting() {
                div { class: "delete-confirm", id: "delete-conversation-dialog", role: "alertdialog",
                    aria_modal: "true", aria_label: "确认删除对话", aria_describedby: "delete-dialog-description", tabindex: "-1",
                    p { id: "delete-dialog-description", "删除这段对话？此操作无法撤销。" }
                    div { button { id: "cancel-delete-conversation", class: "button-secondary", onclick: move |_| deleting.set(None), "取消" }
                        button { class: "button-danger", onclick: move |_| { on_delete.call(id.clone()); deleting.set(None); }, "删除" }
                    }
                }
            }
        }
    }
}
