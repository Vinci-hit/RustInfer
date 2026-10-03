use super::{
    icon::Icon,
    message_input::{ComposerSubmission, MessageInput},
};
use crate::api::{
    client::{ApiClient, ChatSseParser, ChatStreamEvent},
    types::{
        ChatContent, ChatMessage, ChatRequest, ContentPart, ImageUrl, ServerCapabilities, Usage,
    },
};
use crate::state::{
    conversation::{Attachment, Conversation, Message, MessageMetrics},
    settings::AppSettings,
    workspace::{Connection, Workspace},
};
use dioxus::core::Task;
use dioxus::prelude::*;
use futures_util::StreamExt;

fn apply_events(
    events: Vec<Result<ChatStreamEvent, String>>,
    conversations: &mut Signal<Vec<Conversation>>,
    conversation_id: &str,
    assistant_id: &str,
    usage: &mut Option<Usage>,
) -> Result<bool, String> {
    for event in events {
        match event? {
            ChatStreamEvent::Done => return Ok(true),
            ChatStreamEvent::Error(error) => return Err(error),
            ChatStreamEvent::Chunk(chunk) => {
                if let Some(text) = chunk
                    .choices
                    .first()
                    .and_then(|c| c.delta.content.as_deref())
                {
                    if let Some(message) = conversations
                        .write()
                        .iter_mut()
                        .find(|c| c.id == conversation_id)
                        .and_then(|c| c.messages.iter_mut().find(|m| m.id == assistant_id))
                    {
                        message.content.push_str(text);
                    }
                }
                if chunk.usage.is_some() {
                    *usage = chunk.usage;
                }
            }
        }
    }
    Ok(false)
}

/// Validate the exact history that will be sent before changing the conversation
/// or clearing the composer. Retry uses the same path as a new user message.
fn prepare_request(
    conversation: &Conversation,
    submission: Option<&ComposerSubmission>,
    settings: &AppSettings,
    capabilities: &ServerCapabilities,
) -> Result<ChatRequest, String> {
    let end = if submission.is_some() {
        conversation.messages.len()
    } else {
        if conversation
            .messages
            .last()
            .is_none_or(|message| message.role != "assistant")
        {
            return Err("没有可重新生成的回复。".into());
        }
        conversation.messages.len() - 1
    };
    if settings.max_tokens == 0 {
        return Err("最大输出长度必须大于 0，请调整会话配置。".into());
    }
    let temperature = if capabilities.greedy_sampling {
        0.0
    } else {
        settings.temperature
    };
    let top_p = if capabilities.greedy_sampling {
        0.0
    } else {
        settings.top_p
    };
    if !temperature.is_finite()
        || !(0.0..=2.0).contains(&temperature)
        || !top_p.is_finite()
        || !(0.0..=1.0).contains(&top_p)
    {
        return Err("采样参数无效，请检查温度和 Top P 设置。".into());
    }

    let mut messages = Vec::new();
    let mut image_count = 0usize;
    let mut text_bytes = 0usize;
    let mut append = |role: &str, text: &str, attachments: &[Attachment]| -> Result<(), String> {
        text_bytes = text_bytes
            .saturating_add(role.len())
            .saturating_add(text.len());
        if text_bytes > 1024 * 1024 {
            return Err("这段对话的文字已超过 1 MiB，请新建对话后继续。".into());
        }
        if !text.is_empty() && !capabilities.inputs.text {
            return Err("当前模型没有启用文字输入。".into());
        }
        let content = if attachments.is_empty() {
            ChatContent::Text(text.to_owned())
        } else {
            if !capabilities.inputs.image || role != "user" {
                return Err("当前模型不支持此对话中的图片，请切换视觉模型或新建文字对话。".into());
            }
            image_count = image_count.saturating_add(attachments.len());
            if image_count > capabilities.limits.max_images {
                return Err(format!(
                    "每次请求最多包含 {} 张历史与新图片，请新建对话后继续。",
                    capabilities.limits.max_images
                ));
            }
            let mut parts = Vec::new();
            if !text.is_empty() {
                parts.push(ContentPart::Text {
                    text: text.to_owned(),
                });
            }
            for attachment in attachments {
                let limit = capabilities.limits.max_image_bytes;
                let encoded_limit = limit.div_ceil(3).saturating_mul(4);
                let prefix = format!("data:{};base64,", attachment.mime_type);
                let payload = attachment.data_url.strip_prefix(&prefix);
                if !matches!(attachment.mime_type.as_str(), "image/png" | "image/jpeg")
                    || !capabilities
                        .limits
                        .image_mime_types
                        .contains(&attachment.mime_type)
                    || attachment.size == 0
                    || attachment.size > limit as u64
                    || payload.is_none_or(|data| data.is_empty() || data.len() > encoded_limit)
                {
                    return Err(format!(
                        "图片「{}」不符合当前模型的格式或大小限制，请新建对话或重新上传。",
                        attachment.name
                    ));
                }
                parts.push(ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: attachment.data_url.clone(),
                        detail: Some("auto".into()),
                    },
                });
            }
            ChatContent::Parts(parts)
        };
        messages.push(ChatMessage {
            role: role.to_owned(),
            content,
        });
        Ok(())
    };
    if !settings.system_prompt.trim().is_empty() {
        append("system", &settings.system_prompt, &[])?;
    }
    for message in conversation.messages[..end]
        .iter()
        .filter(|message| eligible_history(message))
    {
        append(&message.role, &message.content, &message.attachments)?;
    }
    if let Some(submission) = submission {
        append("user", &submission.text, &submission.attachments)?;
    }
    if !messages.iter().any(|message| message.role == "user") {
        return Err("请先输入一条消息。".into());
    }
    Ok(ChatRequest {
        model: conversation.model.clone(),
        messages,
        max_tokens: Some(settings.max_tokens),
        stream: true,
        enable_thinking: capabilities.thinking.then_some(settings.enable_thinking),
        temperature: Some(temperature),
        top_p: Some(top_p),
        ..Default::default()
    })
}

fn eligible_history(message: &Message) -> bool {
    !message.is_streaming
        && message.error.is_none()
        && !message.interrupted
        && (!message.content.is_empty() || !message.attachments.is_empty())
}

fn cancel_generation(
    generation: &mut Signal<Option<(String, String)>>,
    generation_task: &mut Signal<Option<Task>>,
    conversations: &mut Signal<Vec<Conversation>>,
) {
    if let Some(task) = generation_task.take() {
        task.cancel();
    }
    if let Some((id, message_id)) = generation.take() {
        if let Some(conversation) = conversations
            .write()
            .iter_mut()
            .find(|conversation| conversation.id == id)
        {
            conversation.updated_at = chrono::Utc::now().timestamp();
            if let Some(message) = conversation
                .messages
                .iter_mut()
                .find(|message| message.id == message_id)
            {
                message.is_streaming = false;
                message.interrupted = true;
            }
        }
    }
}

#[component]
pub fn ChatArea(
    mut conversations: Signal<Vec<Conversation>>,
    active_id: Signal<String>,
    mut inspector_open: Signal<bool>,
    on_menu: EventHandler<()>,
    on_settings: EventHandler<()>,
) -> Element {
    let mut workspace = use_context::<Workspace>();
    let mut generation = use_signal(|| None::<(String, String)>);
    let mut generation_task = use_signal(|| None::<Task>);
    let mut generation_endpoint = use_signal(String::new);
    let mut draft = use_signal(String::new);
    let mut draft_owner = use_signal(|| active_id.peek().clone());
    let mut send_error = use_signal(|| None::<String>);
    let mut follow_bottom = use_signal(|| true);

    use_effect(move || {
        let id = active_id();
        let previous = draft_owner.peek().clone();
        if previous != id {
            if let Some(conversation) = conversations
                .write()
                .iter_mut()
                .find(|conversation| conversation.id == previous)
            {
                conversation.draft = draft.peek().clone();
            }
        }
        let saved = conversations
            .peek()
            .iter()
            .find(|conversation| conversation.id == id)
            .map(|conversation| conversation.draft.clone())
            .unwrap_or_default();
        draft_owner.set(id);
        draft.set(saved);
        send_error.set(None);
        follow_bottom.set(true);
    });
    use_effect(move || {
        let text = draft();
        if let Some(conversation) = conversations
            .write()
            .iter_mut()
            .find(|c| c.id == *draft_owner.peek())
        {
            conversation.draft = text;
        }
    });
    use_effect(move || {
        let _state = conversations.read();
        let following = *follow_bottom.peek();
        if following {
            document::eval("requestAnimationFrame(() => { const el = document.getElementById('messages-container'); if (el) el.scrollTop = el.scrollHeight; });");
        }
    });

    // A deleted conversation or a changed server must not leave a hidden
    // request consuming tokens and blocking all later submissions.
    use_effect(move || {
        let endpoint = workspace.settings.read().api_base_url.clone();
        if let Some((id, _)) = generation() {
            let exists = conversations
                .read()
                .iter()
                .any(|conversation| conversation.id == id);
            if !exists || endpoint != *generation_endpoint.peek() {
                cancel_generation(&mut generation, &mut generation_task, &mut conversations);
            }
        }
    });

    let send = use_callback(move |submission: Option<ComposerSubmission>| -> bool {
        if generation.peek().is_some() {
            return false;
        }
        if !matches!(*workspace.connection.peek(), Connection::Online) {
            send_error.set(Some("推理服务尚未连接，请检查连接设置。".into()));
            return false;
        }
        let id = active_id();
        let Some(conversation) = conversations.peek().iter().find(|c| c.id == id).cloned() else {
            return false;
        };
        let model = workspace
            .models
            .peek()
            .iter()
            .find(|m| m.id == conversation.model)
            .cloned();
        let Some(model) = model else {
            send_error.set(Some("请先连接推理服务并选择可用模型。".into()));
            return false;
        };
        let capabilities = model
            .capabilities
            .unwrap_or_else(|| (workspace.capabilities)());
        if submission.as_ref().is_some_and(|submission| {
            submission.text.trim().is_empty() && submission.attachments.is_empty()
        }) {
            return false;
        }
        let settings = (workspace.settings)();
        let request =
            match prepare_request(&conversation, submission.as_ref(), &settings, &capabilities) {
                Ok(request) => request,
                Err(error) => {
                    send_error.set(Some(error));
                    return false;
                }
            };
        let mut assistant = Message::assistant_streaming();
        assistant.model = Some(request.model.clone());
        let assistant_id = assistant.id.clone();
        let is_new_message = submission.is_some();
        {
            let mut list = conversations.write();
            let Some(conversation) = list.iter_mut().find(|c| c.id == id) else {
                return false;
            };
            if let Some(submission) = submission {
                let mut user = Message::user(submission.text);
                user.attachments = submission.attachments;
                conversation.messages.push(user);
                conversation.draft.clear();
            } else {
                conversation.messages.pop();
            }
            conversation.auto_title();
            conversation.updated_at = chrono::Utc::now().timestamp();
            conversation.messages.push(assistant);
        }
        send_error.set(None);
        follow_bottom.set(true);
        generation_endpoint.set(settings.api_base_url.clone());
        generation.set(Some((id.clone(), assistant_id.clone())));
        if is_new_message {
            draft.set(String::new());
        }
        let task = spawn(async move {
            let client = ApiClient::new(&settings.api_base_url);
            let mut usage = None;
            let outcome: Result<(), String> = async {
                let response = client
                    .chat_completion_stream(request)
                    .await
                    .map_err(|e| e.to_string())?;
                let mut body = response.bytes_stream();
                let mut parser = ChatSseParser::default();
                while let Some(bytes) = body.next().await {
                    let bytes = bytes.map_err(|e| format!("响应中断：{e}"))?;
                    if apply_events(
                        parser.push(&bytes),
                        &mut conversations,
                        &id,
                        &assistant_id,
                        &mut usage,
                    )? {
                        return Ok(());
                    }
                }
                if apply_events(
                    parser.finish(),
                    &mut conversations,
                    &id,
                    &assistant_id,
                    &mut usage,
                )? {
                    Ok(())
                } else {
                    Err("连接提前结束，回复可能不完整。你可以重新生成。".into())
                }
            }
            .await;
            if let Some(conversation) = conversations.write().iter_mut().find(|c| c.id == id) {
                conversation.updated_at = chrono::Utc::now().timestamp();
                if let Some(message) = conversation
                    .messages
                    .iter_mut()
                    .find(|m| m.id == assistant_id)
                {
                    message.is_streaming = false;
                    message.error = outcome.err();
                    message.metrics = usage.map(|u| MessageMetrics {
                        prefill_ms: u.performance.as_ref().map_or(0, |p| p.prefill_ms),
                        decode_ms: u.performance.as_ref().map_or(0, |p| p.decode_ms),
                        tokens_per_second: u
                            .performance
                            .as_ref()
                            .map_or(0.0, |p| p.tokens_per_second),
                        total_tokens: u.completion_tokens,
                    });
                }
            }
            generation.set(None);
            generation_task.set(None);
        });
        generation_task.set(Some(task));
        true
    });
    let stop = move |_: ()| {
        cancel_generation(&mut generation, &mut generation_task, &mut conversations);
    };
    let active = conversations
        .read()
        .iter()
        .find(|c| c.id == active_id())
        .cloned();
    let model = active.as_ref().map(|c| c.model.clone()).unwrap_or_default();
    let messages = active
        .as_ref()
        .map(|c| c.messages.clone())
        .unwrap_or_default();
    let generating = generation().is_some();
    let capabilities = workspace
        .models
        .read()
        .iter()
        .find(|m| m.id == model)
        .and_then(|m| m.capabilities.clone())
        .unwrap_or_else(|| (workspace.capabilities)());
    let mut composer_capabilities = capabilities.clone();
    composer_capabilities.limits.max_images = capabilities.limits.max_images.saturating_sub(
        messages
            .iter()
            .filter(|message| eligible_history(message))
            .map(|message| message.attachments.len())
            .sum::<usize>(),
    );
    let connected = matches!((workspace.connection)(), Connection::Online);
    let selected = workspace.models.read().iter().any(|m| m.id == model);
    let count = messages.iter().filter(|m| m.role == "user").count();
    let export = move |_: MouseEvent| {
        if let Some(conversation) = conversations
            .peek()
            .iter()
            .find(|c| c.id == *active_id.peek())
        {
            let data = serde_json::json!({"name":format!("RustInfer-{}.json", conversation.id), "content":serde_json::to_string_pretty(conversation).unwrap_or_default()});
            let eval = document::eval("const data = await dioxus.recv(); const url = URL.createObjectURL(new Blob([data.content], {type:'application/json'})); const a = document.createElement('a'); a.href=url; a.download=data.name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);");
            let _ = eval.send(data);
        }
    };
    rsx! {
        header { class: "workspace-header",
            div { class: "header-leading",
                button { class: "icon-button mobile-only", aria_label: "打开导航", onclick: move |_| on_menu.call(()), Icon { name: "panel", size: 20 } }
                span { class: "workspace-breadcrumb", "工作台" } span { class: "breadcrumb-divider", "/" } strong { "对话" }
            }
            div { class: "header-actions",
                super::model_selector::ModelSelector { model: model.clone(), disabled: generating,
                    on_change: move |model| { if let Some(c) = conversations.write().iter_mut().find(|c| c.id == active_id()) { c.model = model; } }
                }
                button { class: "icon-button export-button", title: "导出当前对话", aria_label: "导出当前对话", disabled: messages.is_empty() || generating, onclick: export, Icon { name: "download", size: 18 } }
                button { class: if inspector_open() { "icon-button selected" } else { "icon-button" }, title: "会话配置", aria_label: "会话配置", aria_expanded: "{inspector_open}", onclick: move |_| { let next = !inspector_open(); inspector_open.set(next); workspace.settings.write().metrics_visible = next; }, Icon { name: "sliders", size: 19 } }
            }
        }
        main { class: "chat-workspace",
            if !connected {
                div { class: "connection-banner", role: "status",
                    span { class: if matches!((workspace.connection)(), Connection::Connecting) { "status-dot connecting" } else { "status-dot offline" } }
                    span { if matches!((workspace.connection)(), Connection::Connecting) { "正在连接你的推理服务…" } else { "连接推理服务，开始你的第一段对话。" } }
                    button { onclick: move |_| on_settings.call(()), "连接设置", Icon { name: "arrow-right", size: 14 } }
                }
            }
            div { class: if messages.is_empty() { "messages-scroll empty" } else { "messages-scroll" }, id: "messages-container",
                onscroll: move |_| {
                    spawn(async move {
                        let eval = document::eval("const e = document.getElementById('messages-container'); return !e || e.scrollHeight-e.scrollTop-e.clientHeight < 100;");
                        if let Ok(following) = eval.join::<bool>().await { follow_bottom.set(following); }
                    });
                },
                if messages.is_empty() {
                    section { class: "welcome",
                        div { class: "welcome-emblem", aria_hidden: "true", div { class: "emblem-core", "R" } span { class: "emblem-orbit orbit-one" } span { class: "emblem-orbit orbit-two" } span { class: "emblem-point" } }
                        div { class: "welcome-heading", p { class: "welcome-greeting", "让想法，自由发生" } h1 { "你的灵感，", br {} "从这里开始。" } }
                        p { class: "welcome-description", "写代码、梳理思路，或探索一张图片。", br {} "与你的本地模型，一起把想法变成可能。" }
                        div { class: "suggestion-grid",
                            button { class: "suggestion", onclick: move |_| draft.set("请帮我写一个 Rust 异步任务调度器，并解释关键设计。".into()),
                                span { class: "suggestion-icon code", Icon { name: "code", size: 20 } } strong { "一起写代码" } p { "从一个想法，到第一行实现" } Icon { name: "arrow-right", size: 16 }
                            }
                            button { class: "suggestion", onclick: move |_| draft.set("我有一个新想法，请通过几个问题帮我梳理目标、核心功能和下一步。".into()),
                                span { class: "suggestion-icon idea", Icon { name: "spark", size: 20 } } strong { "让思路更清晰" } p { "拆解问题，发现新的可能" } Icon { name: "arrow-right", size: 16 }
                            }
                            button { class: "suggestion", onclick: move |_| { draft.set("请描述这张图片的内容，并提取其中的关键信息。".into()); if !capabilities.inputs.image { workspace.notice.set(Some("连接支持视觉输入的模型后，即可添加图片。".into())); } },
                                span { class: "suggestion-icon vision", Icon { name: "image", size: 20 } } strong { "换个视角看世界" } p { "读懂图片里的细节与信息" } Icon { name: "arrow-right", size: 16 }
                            }
                        }
                        div { class: "welcome-footnote", Icon { name: "shield", size: 14 } "只连接你配置的推理服务" }
                    }
                } else {
                    div { class: "conversation-heading", h1 { "{active.as_ref().map(|c| c.title.as_str()).unwrap_or(\"新对话\")}" } span { "{count} 轮对话" } }
                    div { class: "message-list", role: "log", aria_label: "对话消息",
                        for (index, message) in messages.iter().enumerate() {
                            super::message_bubble::MessageBubble { key: "{message.id}", message: message.clone(), model: model.clone(),
                                can_retry: index + 1 == messages.len() && message.role == "assistant" && !generating && connected,
                                on_retry: move |_| { send.call(None); },
                            }
                        }
                    }
                }
            }
            if !follow_bottom() && !messages.is_empty() {
                button { class: "scroll-bottom", onclick: move |_| { follow_bottom.set(true); document::eval("const e = document.getElementById('messages-container'); if(e) e.scrollTo({top:e.scrollHeight,behavior:'smooth'});"); }, Icon { name: "chevron-down", size: 16 } "回到最新消息" }
            }
            if let Some(error) = send_error() { div { class: "send-error", role: "alert", "{error}" } }
            MessageInput { key: "{active_id}", on_send: move |submission| send.call(Some(submission)), on_stop: stop,
                is_generating: generating, capabilities: composer_capabilities, draft, disabled: !connected || !selected }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_message() -> Message {
        let mut message = Message::user("Describe this".into());
        message.attachments.push(Attachment {
            id: "image-1".into(),
            name: "photo.png".into(),
            mime_type: "image/png".into(),
            size: 5,
            data_url: "data:image/png;base64,aGVsbG8=".into(),
        });
        message
    }

    fn completed_response(text: &str) -> Message {
        let mut message = Message::assistant_streaming();
        message.content = text.into();
        message.is_streaming = false;
        message
    }

    #[test]
    fn retry_revalidates_images_without_destroying_previous_response() {
        let mut conversation = Conversation::new("vision".into());
        conversation.messages = vec![image_message(), completed_response("previous answer")];
        let before = conversation.clone();
        let error = prepare_request(
            &conversation,
            None,
            &AppSettings::default(),
            &ServerCapabilities::default(),
        )
        .expect_err("text-only models cannot retry image histories");
        assert!(error.contains("图片"));
        assert_eq!(conversation, before);
        let mut vision = ServerCapabilities::default();
        vision.inputs.image = true;
        let request =
            prepare_request(&conversation, None, &AppSettings::default(), &vision).unwrap();
        assert_eq!(request.messages.len(), 1);
        assert!(matches!(request.messages[0].content, ChatContent::Parts(_)));
    }

    #[test]
    fn image_budget_counts_history_plus_new_submission() {
        let mut conversation = Conversation::new("vision".into());
        conversation.messages.push(image_message());
        let mut capabilities = ServerCapabilities::default();
        capabilities.inputs.image = true;
        capabilities.limits.max_images = 1;
        let submission = ComposerSubmission {
            text: "another image".into(),
            attachments: image_message().attachments,
        };
        assert!(prepare_request(
            &conversation,
            Some(&submission),
            &AppSettings::default(),
            &capabilities
        )
        .is_err());
        capabilities.limits.max_images = 2;
        capabilities.limits.max_image_bytes = 4;
        assert!(prepare_request(
            &conversation,
            Some(&submission),
            &AppSettings::default(),
            &capabilities
        )
        .is_err());
    }

    #[test]
    fn history_omits_failed_interrupted_and_streaming_assistant_turns() {
        let mut conversation = Conversation::new("text".into());
        let mut failed = completed_response("do not reuse this error");
        failed.error = Some("failed".into());
        let mut stopped = completed_response("unfinished content");
        stopped.interrupted = true;
        conversation.messages = vec![
            Message::user("hello".into()),
            completed_response("valid answer"),
            failed,
            stopped,
            Message::assistant_streaming(),
        ];
        let submission = ComposerSubmission {
            text: "continue".into(),
            attachments: vec![],
        };
        let settings = AppSettings {
            system_prompt: "Be helpful".into(),
            ..Default::default()
        };
        let request = prepare_request(
            &conversation,
            Some(&submission),
            &settings,
            &ServerCapabilities::default(),
        )
        .unwrap();
        assert_eq!(
            request
                .messages
                .iter()
                .map(|message| message.content.text())
                .collect::<Vec<_>>(),
            vec!["Be helpful", "hello", "valid answer", "continue"]
        );
    }

    #[test]
    fn thinking_control_is_sent_only_when_service_advertises_it() {
        let conversation = Conversation::new("text".into());
        let submission = ComposerSubmission {
            text: "hello".into(),
            attachments: vec![],
        };
        let mut settings = AppSettings::default();
        let mut caps = ServerCapabilities::default();
        assert_eq!(
            prepare_request(&conversation, Some(&submission), &settings, &caps)
                .unwrap()
                .enable_thinking,
            None
        );
        caps.thinking = true;
        assert_eq!(
            prepare_request(&conversation, Some(&submission), &settings, &caps)
                .unwrap()
                .enable_thinking,
            Some(false)
        );
        settings.enable_thinking = true;
        assert_eq!(
            prepare_request(&conversation, Some(&submission), &settings, &caps)
                .unwrap()
                .enable_thinking,
            Some(true)
        );
    }

    #[test]
    fn speculative_requests_force_greedy_and_reject_zero_output_budget() {
        let conversation = Conversation::new("text".into());
        let submission = ComposerSubmission {
            text: "hello".into(),
            attachments: vec![],
        };
        let mut capabilities = ServerCapabilities::default();
        capabilities.greedy_sampling = true;
        let mut settings = AppSettings::default();
        let request =
            prepare_request(&conversation, Some(&submission), &settings, &capabilities).unwrap();
        assert_eq!(request.temperature, Some(0.0));
        assert_eq!(request.top_p, Some(0.0));
        settings.max_tokens = 0;
        assert!(
            prepare_request(&conversation, Some(&submission), &settings, &capabilities).is_err()
        );
    }
}
