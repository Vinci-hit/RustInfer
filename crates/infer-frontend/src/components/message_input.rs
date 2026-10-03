use dioxus::prelude::*;
use serde::Deserialize;

use crate::api::types::ServerCapabilities;
use crate::components::icon::Icon;
use crate::state::conversation::Attachment;

const MEDIA_SCRIPT: &str = include_str!("../../assets/composer.js");
const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;

#[derive(Clone, PartialEq)]
pub struct ComposerSubmission {
    pub text: String,
    pub attachments: Vec<Attachment>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BrowserComposerEvent {
    Attachment { attachment: Attachment },
    Error { message: String },
    Drag { active: bool },
    Loading { active: bool },
    DismissMenu,
}

fn format_size(size: u64) -> String {
    if size >= 1024 * 1024 {
        format!("{:.1} MB", size as f64 / (1024.0 * 1024.0))
    } else {
        format!("{} KB", size.div_ceil(1024))
    }
}

fn resize_composer(id: &str) {
    let eval = document::eval(
        r#"const id = await dioxus.recv();
        requestAnimationFrame(() => {
            const input = document.getElementById(id)?.querySelector('textarea');
            if (!input) return;
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 200) + 'px';
        });"#,
    );
    let _ = eval.send(id);
}

fn choose_images(id: &str) {
    let eval = document::eval(
        r#"const id = await dioxus.recv();
        document.getElementById(id)?.querySelector('input[type=file]')?.click();"#,
    );
    let _ = eval.send(id);
}

#[component]
pub fn MessageInput(
    on_send: Callback<ComposerSubmission, bool>,
    on_stop: EventHandler<()>,
    is_generating: bool,
    capabilities: ServerCapabilities,
    draft: Signal<String>,
    disabled: bool,
) -> Element {
    let composer_id = use_signal(|| format!("composer-{}", uuid::Uuid::new_v4()));
    let mut attachments = use_signal(Vec::<Attachment>::new);
    let mut attachment_error = use_signal(String::new);
    let mut is_dragging = use_signal(|| false);
    let mut is_loading = use_signal(|| false);
    let mut menu_open = use_signal(|| false);

    // This also responds to a draft inserted by a welcome-screen suggestion.
    use_effect(move || {
        let _ = draft();
        resize_composer(&composer_id());
    });
    use_drop(move || {
        let eval = document::eval(
            r#"const id = await dioxus.recv();
            window.__rustinferComposers?.get(id)?.();"#,
        );
        let _ = eval.send(composer_id.peek().clone());
    });

    let max_images = capabilities.limits.max_images.min(4);
    let max_image_bytes = capabilities.limits.max_image_bytes.min(MAX_IMAGE_BYTES);
    let image_supported = capabilities.inputs.image && max_images > 0 && max_image_bytes > 0;
    let has_attachments = !attachments.read().is_empty();
    let attachments_supported = !has_attachments || image_supported;
    let can_send = !disabled
        && !is_generating
        && attachments_supported
        && !is_loading()
        && (!draft.read().trim().is_empty() || has_attachments);
    let supported_mime_types = capabilities
        .limits
        .image_mime_types
        .iter()
        .filter(|mime| matches!(mime.as_str(), "image/png" | "image/jpeg"))
        .cloned()
        .collect::<Vec<_>>();
    let accepted_types = supported_mime_types.join(",");
    let limit_label = format!("PNG、JPG · 最多 {max_images} 张");
    let image_hint = if image_supported {
        format!(
            "添加图片 · {limit_label} · 每张不超过 {}",
            format_size(max_image_bytes as u64)
        )
    } else if capabilities.inputs.image {
        "此对话的图片额度已用完，请新建对话后上传".to_string()
    } else {
        "当前模型未开启图片理解，请切换支持视觉的模型".to_string()
    };
    let current_error = if has_attachments && !image_supported {
        image_hint.clone()
    } else {
        attachment_error()
    };

    let mut do_send = move || {
        if disabled || is_generating || is_loading() {
            return;
        }
        let text = draft().trim().to_owned();
        let files = attachments();
        if text.is_empty() && files.is_empty() {
            return;
        }
        if !files.is_empty() && !image_supported {
            attachment_error.set("当前模型无法接收这些图片，请切换视觉模型或移除图片。".into());
            return;
        }
        if files.len() > max_images {
            attachment_error.set(format!(
                "此对话还可添加 {max_images} 张图片，请移除多余图片。"
            ));
            return;
        }
        if files.iter().any(|file| {
            file.size > max_image_bytes as u64 || !supported_mime_types.contains(&file.mime_type)
        }) {
            attachment_error.set("图片超出当前模型限制，请移除后重新上传。".into());
            return;
        }
        if !on_send.call(ComposerSubmission {
            text,
            attachments: files,
        }) {
            return;
        }
        draft.set(String::new());
        attachments.set(Vec::new());
        attachment_error.set(String::new());
        menu_open.set(false);
    };

    rsx! {
        div { class: "composer-region",
            div {
                id: "{composer_id()}",
                class: if is_dragging() { "composer composer-dragging" } else { "composer" },
                "data-disabled": "{disabled}",
                "data-attachment-count": "{attachments.read().len()}",
                "data-max-images": "{max_images}",
                "data-max-image-bytes": "{max_image_bytes}",
                "data-max-image-dimension": "{capabilities.limits.max_image_dimension}",
                "data-image-types": "{accepted_types}",
                onmounted: move |_| {
                    let mut eval = document::eval(MEDIA_SCRIPT);
                    let _ = eval.send(composer_id());
                    spawn(async move {
                        while let Ok(event) = eval.recv::<BrowserComposerEvent>().await {
                            match event {
                                BrowserComposerEvent::Attachment { attachment } => {
                                    // Enforce hard client bounds again at the Rust boundary.
                                    if attachments.peek().len() >= 4 {
                                        attachment_error.set("一次最多添加 4 张图片。".into());
                                    } else if attachment.size > MAX_IMAGE_BYTES as u64
                                        || !matches!(attachment.mime_type.as_str(), "image/png" | "image/jpeg")
                                    {
                                        attachment_error.set("请选择 10 MB 以内的 PNG 或 JPG 图片。".into());
                                    } else {
                                        attachments.write().push(attachment);
                                    }
                                }
                                BrowserComposerEvent::Error { message } => attachment_error.set(message),
                                BrowserComposerEvent::Drag { active } => is_dragging.set(active),
                                BrowserComposerEvent::Loading { active } => {
                                    if active && !is_loading() {
                                        attachment_error.set(String::new());
                                    }
                                    is_loading.set(active);
                                }
                                BrowserComposerEvent::DismissMenu => menu_open.set(false),
                            }
                        }
                    });
                },
                input {
                    r#type: "file",
                    accept: "image/png,image/jpeg",
                    multiple: true,
                    hidden: true,
                    disabled: disabled || !image_supported,
                    tabindex: "-1",
                    aria_label: "选择 PNG 或 JPG 图片",
                }
                if is_dragging() {
                    div { class: "composer-drop-hint", aria_hidden: "true",
                        Icon { name: "image", size: 24 }
                        span { "松开以添加图片" }
                    }
                }
                if has_attachments {
                    div { class: "composer-attachments", aria_label: "待发送的图片",
                        for attachment in attachments() {
                            {
                                let id = attachment.id.clone();
                                let size = format_size(attachment.size);
                                rsx! {
                                    div { class: "attachment-preview", key: "{attachment.id}",
                                        img {
                                            class: "attachment-image",
                                            src: "{attachment.data_url}",
                                            alt: "{attachment.name}",
                                        }
                                        div { class: "attachment-info",
                                            span { class: "attachment-name", "{attachment.name}" }
                                            span { class: "attachment-size", "{size}" }
                                        }
                                        button {
                                            class: "attachment-remove",
                                            r#type: "button",
                                            aria_label: "移除图片 {attachment.name}",
                                            title: "移除图片",
                                            onclick: move |_| {
                                                attachments.write().retain(|attachment| attachment.id != id);
                                                attachment_error.set(String::new());
                                            },
                                            Icon { name: "x", size: 14 }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                textarea {
                    class: "composer-textarea",
                    id: "message-composer",
                    rows: "2",
                    placeholder: "问一个问题，或一起实现一个想法…",
                    aria_label: "消息内容",
                    aria_describedby: "composer-help",
                    value: "{draft()}",
                    disabled,
                    oninput: move |event| draft.set(event.value()),
                }
                div { class: "composer-toolbar",
                    div { class: "composer-tools",
                        div { class: "composer-attach-control",
                            button {
                                class: "composer-tool",
                                r#type: "button",
                                id: "{composer_id()}-attachment-trigger",
                                title: "添加图片与附件",
                                aria_controls: "{composer_id()}-attachment-menu",
                                aria_label: "添加图片与附件",
                                aria_expanded: "{menu_open()}",
                                aria_haspopup: "menu",
                                disabled,
                                onclick: move |_| menu_open.toggle(),
                                Icon { name: "plus", size: 20 }
                            }
                            if menu_open() {
                                div { class: "composer-menu", id: "{composer_id()}-attachment-menu", role: "menu", aria_label: "附件类型", tabindex: "-1",
                                    button {
                                        class: "composer-menu-item",
                                        r#type: "button",
                                        role: "menuitem",
                                        disabled: !image_supported,
                                        title: "{image_hint}",
                                        onclick: move |_| {
                                            choose_images(&composer_id());
                                            menu_open.set(false);
                                        },
                                        Icon { name: "image", size: 18 }
                                        span { class: "composer-menu-copy",
                                            span { "上传图片" }
                                            small { if image_supported { "{limit_label}" } else { "当前模型未启用" } }
                                        }
                                        span { class: "composer-menu-status", if image_supported { "可用" } else { "未启用" } }
                                    }
                                    button { class: "composer-menu-item", r#type: "button", role: "menuitem", disabled: true,
                                        title: "当前服务暂不支持文档上传",
                                        Icon { name: "file", size: 18 }
                                        span { class: "composer-menu-copy", span { "上传文档" } small { "PDF、文本与其他文件" } }
                                        span { class: "composer-menu-status", "未启用" }
                                    }
                                    button { class: "composer-menu-item", r#type: "button", role: "menuitem", disabled: true,
                                        title: "当前服务暂不支持音频输入",
                                        Icon { name: "audio-lines", size: 18 }
                                        span { class: "composer-menu-copy", span { "上传音频" } small { "录音、语音与音频文件" } }
                                        span { class: "composer-menu-status", "未启用" }
                                    }
                                    button { class: "composer-menu-item", r#type: "button", role: "menuitem", disabled: true,
                                        title: "当前服务暂不支持视频理解",
                                        Icon { name: "video", size: 18 }
                                        span { class: "composer-menu-copy", span { "上传视频" } small { "视频内容与关键帧" } }
                                        span { class: "composer-menu-status", "未启用" }
                                    }
                                }
                            }
                        }
                        button {
                            class: "composer-tool",
                            r#type: "button",
                            title: "{image_hint}",
                            aria_label: "添加图片",
                            disabled: disabled || !image_supported,
                            onclick: move |_| choose_images(&composer_id()),
                            Icon { name: "image", size: 18 }
                        }
                        span { class: "composer-divider", aria_hidden: "true" }
                        span { class: "composer-mode", "文本对话" }
                        if is_loading() {
                            span { class: "composer-counter", role: "status", "正在读取图片…" }
                        } else if has_attachments {
                            span { class: "composer-counter", "{attachments.read().len()}/{max_images} 图片" }
                        }
                    }
                    div { class: "composer-actions",
                        button {
                            class: "composer-tool voice-button",
                            r#type: "button",
                            disabled: true,
                            aria_label: "语音输入尚未启用",
                            title: "当前服务尚未启用语音转写",
                            Icon { name: "mic", size: 18 }
                        }
                        if is_generating {
                            button {
                                class: "send-button stop-button",
                                r#type: "button",
                                title: "停止生成",
                                aria_label: "停止生成",
                                onclick: move |_| on_stop.call(()),
                                Icon { name: "square", size: 16 }
                            }
                        } else {
                            button {
                                class: "send-button",
                                "data-composer-send": "true",
                                r#type: "button",
                                title: "发送消息",
                                aria_label: "发送消息",
                                disabled: !can_send,
                                onclick: move |_| do_send(),
                                Icon { name: "arrow-up", size: 20 }
                            }
                        }
                    }
                }
            }
            if !current_error.is_empty() {
                p { class: "composer-error", role: "alert", "{current_error}" }
            }
            div { class: "composer-footer", id: "composer-help",
                span { "让想法开始发生。" }
                span { class: "composer-shortcut",
                    kbd { "Enter" } " 发送 " span { "·" } " " kbd { "Shift + Enter" } " 换行"
                }
            }
        }
    }
}
