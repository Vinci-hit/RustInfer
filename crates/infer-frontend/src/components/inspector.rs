use super::icon::Icon;
use crate::state::workspace::{Connection, Workspace};
use dioxus::prelude::*;

#[component]
pub fn Inspector(
    conversations: Signal<Vec<crate::state::conversation::Conversation>>,
    active_id: Signal<String>,
    on_close: EventHandler<()>,
) -> Element {
    let mut workspace = use_context::<Workspace>();
    let settings = (workspace.settings)();
    let model = conversations
        .read()
        .iter()
        .find(|c| c.id == active_id())
        .map(|c| c.model.clone())
        .unwrap_or_default();
    let capabilities = workspace
        .models
        .read()
        .iter()
        .find(|m| m.id == model)
        .and_then(|m| m.capabilities.clone())
        .unwrap_or_else(|| (workspace.capabilities)());
    let connected = matches!((workspace.connection)(), Connection::Online);
    rsx! {
        aside { class: "inspector", aria_label: "会话配置",
            div { class: "inspector-header", div { Icon { name: "sliders", size: 18 } h2 { "会话配置" } }
                button { class: "icon-button", aria_label: "关闭会话配置", onclick: move |_| on_close.call(()), Icon { name: "x", size: 17 } }
            }
            div { class: "inspector-scroll",
                section { class: "inspector-section",
                    div { class: "section-heading", h3 { "系统提示词" } span { "可选" } }
                    p { class: "field-description", "赋予模型一个角色，让回答更符合你的期待。" }
                    textarea { class: "system-prompt", aria_label: "系统提示词", rows: "4", placeholder: "例如：你是一位资深 Rust 工程师，请用简洁的中文回答…", value: "{settings.system_prompt}", oninput: move |e| workspace.settings.write().system_prompt = e.value() }
                }
                section { class: "inspector-section",
                    div { class: "section-heading", h3 { "生成参数" } button { class: "text-button", onclick: move |_| {
                        let mut settings = workspace.settings.write(); settings.temperature = 0.7; settings.top_p = 0.95; settings.max_tokens = 2048;
                    }, "重置" } }
                    label { class: "parameter-field", div { span { "创造性" } output { "{settings.temperature:.1}" } }
                        input { r#type: "range", min: "0", max: "2", step: "0.1", value: "{settings.temperature}", disabled: capabilities.greedy_sampling, aria_label: "创造性 Temperature", oninput: move |e| { if let Ok(value) = e.value().parse::<f32>() { workspace.settings.write().temperature = value.clamp(0.0,2.0); } } }
                        div { class: "range-labels", span { "更严谨" } span { "更有创意" } }
                    }
                    label { class: "parameter-field", div { span { "采样范围" } output { "{settings.top_p:.2}" } }
                        input { r#type: "range", min: "0.05", max: "1", step: "0.05", value: "{settings.top_p}", disabled: capabilities.greedy_sampling, aria_label: "采样范围 Top P", oninput: move |e| { if let Ok(value) = e.value().parse::<f32>() { workspace.settings.write().top_p = value.clamp(0.05,1.0); } } }
                        div { class: "range-labels", span { "更聚焦" } span { "更丰富" } }
                    }
                    label { class: "parameter-field token-field", span { "最大输出长度" }
                        select { aria_label: "最大输出长度", value: "{settings.max_tokens}", onchange: move |e| { if let Ok(value) = e.value().parse::<usize>() { workspace.settings.write().max_tokens = value; } },
                            option { value: "512", "512 tokens" } option { value: "1024", "1,024 tokens" } option { value: "2048", "2,048 tokens" } option { value: "4096", "4,096 tokens" } option { value: "8192", "8,192 tokens" }
                        }
                    }
                    if capabilities.thinking {
                        label { class: "parameter-field token-field", span { "深度思考" }
                            input { r#type: "checkbox", checked: settings.enable_thinking, aria_label: "深度思考", onchange: move |e| workspace.settings.write().enable_thinking = e.checked() }
                        }
                    }
                    if capabilities.greedy_sampling { p { class: "field-description", "当前服务使用推测解码，自动采用确定性采样。" } }
                }
                section { class: "inspector-section",
                    div { class: "section-heading", h3 { "服务能力" } Icon { name: "spark", size: 15 } }
                    p { class: "field-description", "根据当前服务实际支持的能力启用。" }
                    div { class: "capability-list",
                        CapabilityRow { icon: "chat", label: "文字对话", enabled: connected && capabilities.inputs.text, supported: true }
                        CapabilityRow { icon: "image", label: "图片理解", enabled: connected && capabilities.inputs.image, supported: true }
                        CapabilityRow { icon: "mic", label: "语音对话", enabled: capabilities.realtime, supported: false }
                        CapabilityRow { icon: "file", label: "文件解析", enabled: capabilities.inputs.file, supported: false }
                        CapabilityRow { icon: "video", label: "视频理解", enabled: capabilities.inputs.video, supported: false }
                    }
                }
                super::metrics_panel::MetricsPanel {}
            }
            div { class: "inspector-footer", Icon { name: "info", size: 14 } "参数将应用于下一次生成" }
        }
    }
}

#[component]
fn CapabilityRow(
    icon: &'static str,
    label: &'static str,
    enabled: bool,
    supported: bool,
) -> Element {
    rsx! { div { class: "capability-row", div { Icon { name: icon, size: 16 } span { "{label}" } }
        span { class: if enabled && supported { "capability-status available" } else { "capability-status" },
            if enabled && supported { "可用" } else if supported { "未启用" } else { "待接入" }
        }
    } }
}
