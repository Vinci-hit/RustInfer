use crate::api::client::ApiClient;
use crate::api::types::ModelObject;
use dioxus::prelude::*;

#[component]
pub fn ModelSelector() -> Element {
    let mut models = use_signal(Vec::<ModelObject>::new);
    let mut selected_model = use_signal(|| "llama3".to_string());
    let mut is_loading = use_signal(|| true);

    // 获取模型列表
    use_future(move || async move {
        let client = ApiClient::default();
        if let Ok(model_list) = client.list_models().await {
            if !model_list.is_empty() {
                selected_model.set(model_list[0].id.clone());
            }
            models.set(model_list);
        }
        is_loading.set(false);
    });

    rsx! {
        div {
            class: "space-y-2",

            label {
                class: "text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider",
                "Model"
            }

            if is_loading() {
                div {
                    class: "px-3 py-2 rounded-lg bg-[var(--color-surface-2)] text-sm text-[var(--color-text-muted)] animate-pulse",
                    "Loading..."
                }
            } else {
                select {
                    class: "w-full px-3 py-2 rounded-lg bg-[var(--color-surface-2)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] focus:outline-none focus:border-[var(--color-accent)] transition-colors appearance-none cursor-pointer",
                    value: "{selected_model()}",
                    onchange: move |e| {
                        selected_model.set(e.value());
                    },
                    for model in models.read().iter() {
                        option {
                            value: "{model.id}",
                            "{model.id}"
                        }
                    }
                    // Fallback if no models loaded
                    if models.read().is_empty() {
                        option { value: "llama3", "llama3 (default)" }
                    }
                }
            }
        }
    }
}
