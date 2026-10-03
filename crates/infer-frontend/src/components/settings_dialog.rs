use super::icon::Icon;
use crate::state::workspace::{Connection, Workspace};
use dioxus::prelude::*;

#[component]
pub fn SettingsDialog(on_close: EventHandler<()>) -> Element {
    let mut workspace = use_context::<Workspace>();
    let mut endpoint = use_signal(|| workspace.settings.peek().api_base_url.clone());
    let mut error = use_signal(|| None::<String>);
    let save = move |_| {
        let value = endpoint().trim().trim_end_matches('/').to_string();
        let valid = reqwest::Url::parse(&value).is_ok_and(|url| {
            matches!(url.scheme(), "http" | "https")
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none()
                && url.fragment().is_none()
        });
        if !valid {
            error.set(Some(
                "请输入完整的 http:// 或 https:// 服务地址，不包含账号、查询参数或锚点。".into(),
            ));
            return;
        }
        workspace.settings.write().api_base_url = value;
        let revision = *workspace.reconnect.peek() + 1;
        workspace.reconnect.set(revision);
        on_close.call(());
    };
    rsx! {
        div { class: "modal-backdrop", onclick: move |_| on_close.call(()),
            section { class: "settings-dialog", role: "dialog", aria_modal: "true", aria_labelledby: "settings-title", onclick: move |e| e.stop_propagation(),
                onmounted: move |_| { document::eval("requestAnimationFrame(() => document.getElementById('api-endpoint')?.focus());"); },
                header { div { div { class: "settings-symbol", Icon { name: "globe", size: 22 } } h2 { id: "settings-title", "连接你的推理服务" } }
                    button { id: "close-settings", class: "icon-button", aria_label: "关闭连接设置", onclick: move |_| on_close.call(()), Icon { name: "x", size: 20 } }
                }
                p { class: "settings-description", "将 RustInfer 连接到正在运行的服务。消息和附件只会发送到你配置的地址。" }
                label { class: "settings-label", r#for: "api-endpoint", "服务地址" }
                input { id: "api-endpoint", class: "endpoint-input", r#type: "url", placeholder: "http://localhost:8000", value: "{endpoint}", oninput: move |e| { endpoint.set(e.value()); error.set(None); }, onkeydown: move |e| { if e.key() == Key::Escape { on_close.call(()); } } }
                p { class: "field-description", "支持带路径前缀的地址；末尾的 /v1 会自动处理。" }
                if let Some(error) = error() { p { class: "settings-error", role: "alert", "{error}" } }
                div { class: "connection-detail",
                    span { class: match (workspace.connection)() { Connection::Online => "status-dot online", Connection::Connecting => "status-dot connecting", _ => "status-dot offline" } }
                    div { strong { match (workspace.connection)() { Connection::Online => "当前服务已连接", Connection::Connecting => "正在检查服务", _ => "当前服务未连接" } }
                        if let Connection::Offline(reason) = (workspace.connection)() { p { "{reason}" } }
                    }
                }
                div { class: "settings-privacy", Icon { name: "shield", size: 19 } p { "会话和偏好设置保存在此浏览器。清除浏览器数据会删除本地记录，可在对话右上角导出备份。" } }
                footer { button { class: "button-secondary", onclick: move |_| on_close.call(()), "取消" } button { class: "button-primary", onclick: save, "保存并连接", Icon { name: "arrow-right", size: 16 } } }
            }
        }
    }
}
