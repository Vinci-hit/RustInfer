#![allow(dead_code)]

use dioxus::core::Task;
use dioxus::prelude::*;
mod api;
mod components;
mod state;
mod utils;

use api::{
    client::ApiClient,
    types::{ModelObject, ServerCapabilities},
};
use state::{
    conversation::Conversation,
    settings::{AppSettings, Theme},
    workspace::{load_local, save_local, Connection, Workspace},
};

const CSS: Asset = asset!("/assets/output.css");
const BEHAVIOR: Asset = asset!("/assets/workspace.js");

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let settings =
        use_signal(|| load_local::<AppSettings>("rustinfer.settings.v1").unwrap_or_default());
    let mut conversations = use_signal(|| {
        let mut saved =
            load_local::<Vec<Conversation>>("rustinfer.conversations.v1").unwrap_or_default();
        for conversation in &mut saved {
            for message in &mut conversation.messages {
                if message.is_streaming {
                    message.is_streaming = false;
                    message.interrupted = true;
                }
            }
        }
        if saved.is_empty() {
            saved.push(Conversation::new(settings.peek().default_model.clone()));
        }
        saved
    });
    let mut active_id = use_signal(|| {
        let saved = load_local::<String>("rustinfer.active.v1").unwrap_or_default();
        if conversations.peek().iter().any(|c| c.id == saved) {
            saved
        } else {
            conversations.peek()[0].id.clone()
        }
    });
    let mut workspace = Workspace {
        settings,
        connection: use_signal(|| Connection::Connecting),
        models: use_signal(Vec::<ModelObject>::new),
        capabilities: use_signal(ServerCapabilities::default),
        reconnect: use_signal(|| 0u32),
        notice: use_signal(|| None::<String>),
    };
    use_context_provider(|| workspace);
    let mut sidebar_open = use_signal(|| false);
    let mut inspector_open = use_signal(|| {
        #[cfg(target_arch = "wasm32")]
        if web_sys::window()
            .and_then(|w| w.inner_width().ok())
            .and_then(|w| w.as_f64())
            .is_some_and(|w| w <= 1080.0)
        {
            return false;
        }
        settings.peek().metrics_visible
    });
    let mut settings_open = use_signal(|| false);
    let mut connection_task = use_signal(|| None::<Task>);

    // Depend only on endpoint/reconnect, so sampling controls do not refetch models.
    let endpoint = use_memo(move || settings.read().api_base_url.clone());
    use_effect(move || {
        let base = endpoint();
        let _revision = (workspace.reconnect)();
        if let Some(task) = connection_task.peek().as_ref() {
            task.cancel();
        }
        workspace.connection.set(Connection::Connecting);
        workspace.models.set(Vec::new());
        workspace.capabilities.set(ServerCapabilities::default());
        let task = spawn(async move {
            let client = ApiClient::new(&base);
            loop {
                let (health, models, capabilities) = futures_util::join!(
                    client.readiness_check(),
                    client.list_models(),
                    client.get_capabilities()
                );
                match (health, models) {
                    (Ok(()), Ok(models)) if !models.is_empty() => {
                        let fallback = models[0].id.clone();
                        for conversation in conversations.write().iter_mut() {
                            if conversation.model.is_empty() {
                                conversation.model.clone_from(&fallback);
                            }
                        }
                        workspace.models.set(models);
                        workspace.connection.set(Connection::Online);
                        match capabilities {
                            Ok(capabilities) => workspace.capabilities.set(capabilities),
                            Err(error) => {
                                workspace.capabilities.set(ServerCapabilities::default());
                                workspace.notice.set(Some(format!(
                                    "能力检测失败，暂按纯文字模式连接：{error}"
                                )));
                            }
                        }
                    }
                    (Err(error), _) | (_, Err(error)) => workspace
                        .connection
                        .set(Connection::Offline(error.to_string())),
                    _ => workspace
                        .connection
                        .set(Connection::Offline("服务尚未加载模型".into())),
                }
                gloo_timers::future::TimeoutFuture::new(15_000).await;
            }
        });
        connection_task.set(Some(task));
    });
    // Keep user turns safe even during generation. Streaming content is excluded
    // from the persistence key, avoiding a synchronous storage write per token.
    let persistent_snapshot = use_memo(move || {
        let mut snapshot = conversations.read().clone();
        for message in snapshot.iter_mut().flat_map(|c| &mut c.messages) {
            if message.is_streaming {
                message.content.clear();
                message.is_streaming = false;
                message.interrupted = true;
            }
        }
        snapshot
    });
    use_effect(move || {
        if let Err(error) = save_local("rustinfer.conversations.v1", &persistent_snapshot()) {
            workspace.notice.set(Some(error));
        }
    });
    use_effect(move || {
        if let Err(error) = save_local("rustinfer.settings.v1", &*settings.read()) {
            workspace.notice.set(Some(error));
        }
    });
    use_effect(move || {
        let _ = save_local("rustinfer.active.v1", &active_id());
    });

    let new_chat = move |_: ()| {
        let current_model = conversations
            .peek()
            .iter()
            .find(|c| c.id == *active_id.peek())
            .map(|c| c.model.clone());
        let models = workspace.models.peek();
        let model = current_model
            .filter(|id| models.iter().any(|m| m.id == *id))
            .or_else(|| models.first().map(|m| m.id.clone()))
            .unwrap_or_default();
        let conversation = Conversation::new(model);
        active_id.set(conversation.id.clone());
        conversations.write().push(conversation);
        sidebar_open.set(false);
    };
    let delete_chat = move |id: String| {
        let mut list = conversations.write();
        list.retain(|c| c.id != id);
        if list.is_empty() {
            list.push(Conversation::new(
                workspace
                    .models
                    .peek()
                    .first()
                    .map(|m| m.id.clone())
                    .unwrap_or_default(),
            ));
        }
        if active_id() == id {
            active_id.set(list.last().unwrap().id.clone());
        }
    };
    let theme = if settings.read().theme == Theme::Dark {
        "dark"
    } else {
        "light"
    };
    rsx! {
        document::Title { "RustInfer · 本地智能工作台" }
        document::Stylesheet { href: CSS }
        document::Script { src: BEHAVIOR }
        div { class: "workspace", "data-theme": theme,
            a { class: "skip-link", href: "#message-composer", "跳到消息输入" }
            if sidebar_open() {
                button { class: "sidebar-backdrop", aria_label: "关闭导航", onclick: move |_| sidebar_open.set(false) }
            }
            components::sidebar::Sidebar {
                conversations, active_id, on_new_chat: new_chat,
                on_select: move |id| { active_id.set(id); sidebar_open.set(false); },
                on_delete: delete_chat, mobile_open: sidebar_open(),
                on_settings: move |_| settings_open.set(true),
                on_close: move |_| sidebar_open.set(false),
            }
            div { class: "workspace-main",
                components::chat_area::ChatArea {
                    conversations, active_id, inspector_open,
                    on_menu: move |_| sidebar_open.set(true),
                    on_settings: move |_| settings_open.set(true),
                }
            }
            if inspector_open() {
                components::inspector::Inspector { conversations, active_id, on_close: move |_| inspector_open.set(false) }
            }
            if let Some(notice) = (workspace.notice)() {
                div { class: "toast", role: "status",
                    components::icon::Icon { name: "info", size: 18 }
                    span { "{notice}" }
                    button { class: "icon-button", aria_label: "关闭提示", onclick: move |_| workspace.notice.set(None), components::icon::Icon { name: "x", size: 16 } }
                }
            }
            if settings_open() {
                components::settings_dialog::SettingsDialog { on_close: move |_| settings_open.set(false) }
            }
        }
    }
}
