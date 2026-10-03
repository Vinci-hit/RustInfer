use super::icon::Icon;
use crate::{
    state::{conversation::Message, workspace::Workspace},
    utils::markdown::render_markdown,
};
use dioxus::prelude::*;

#[component]
pub fn MessageBubble(
    message: Message,
    model: String,
    can_retry: bool,
    on_retry: EventHandler<()>,
) -> Element {
    let mut workspace = use_context::<Workspace>();
    let mut copied = use_signal(|| false);
    let is_user = message.role == "user";
    let markdown = render_markdown(&message.content);
    let text = message.content.clone();
    let model = message.model.as_deref().unwrap_or(&model);
    let copy_icon = if copied() { "check" } else { "copy" };
    rsx! {
        article { class: if is_user { "message user-message" } else { "message assistant-message" },
            div { class: if is_user { "message-avatar user-avatar" } else { "message-avatar assistant-avatar" }, if is_user { "你" } else { "R" } }
            div { class: "message-body",
                div { class: "message-meta", strong { if is_user { "你" } else { "RustInfer" } } if !is_user { span { "{model}" } } }
                if !message.attachments.is_empty() {
                    div { class: "message-images", for attachment in &message.attachments {
                        a { href: "{attachment.data_url}", target: "_blank", rel: "noopener noreferrer", title: "{attachment.name}", img { src: "{attachment.data_url}", alt: "{attachment.name}", loading: "lazy" } }
                    } }
                }
                if is_user { div { class: "user-text", "{message.content}" } }
                else if message.is_streaming && message.content.is_empty() {
                    div { class: "thinking", role: "status", span {} span {} span {} "正在思考" }
                } else {
                    div { class: "markdown-body", dangerous_inner_html: "{markdown}" }
                    if message.is_streaming { span { class: "typing-cursor", aria_label: "正在生成" } }
                }
                if let Some(error) = &message.error { div { class: "message-error", role: "alert", Icon { name: "info", size: 15 } span { "{error}" } } }
                if message.interrupted { p { class: "message-interrupted", "已停止生成" } }
                if !is_user && !message.is_streaming {
                    div { class: "message-actions",
                        button { class: "message-action", aria_label: "复制回复", title: "复制回复", disabled: text.is_empty(), onclick: move |_| {
                            let content = text.clone();
                            spawn(async move {
                                let eval = document::eval("const text = await dioxus.recv(); try { await navigator.clipboard.writeText(text); return true; } catch { return false; }");
                                let _ = eval.send(content);
                                if eval.join::<bool>().await.unwrap_or(false) {
                                    copied.set(true); gloo_timers::future::TimeoutFuture::new(2000).await; copied.set(false);
                                } else { workspace.notice.set(Some("复制失败，请选择回复文字手动复制。".into())); }
                            });
                        }, Icon { name: copy_icon, size: 14 } if copied() { "已复制" } }
                        if can_retry { button { class: "message-action", title: "重新生成", aria_label: "重新生成", onclick: move |_| on_retry.call(()), Icon { name: "refresh", size: 14 } "重新生成" } }
                        if let Some(metrics) = &message.metrics { span { class: "message-token-count", "{metrics.total_tokens} tokens" } }
                    }
                }
            }
        }
    }
}
