use super::icon::Icon;
use crate::state::workspace::Workspace;
use dioxus::prelude::*;

#[component]
pub fn ModelSelector(model: String, on_change: EventHandler<String>, disabled: bool) -> Element {
    let workspace = use_context::<Workspace>();
    let models = workspace.models.read();
    let known = models.iter().any(|m| m.id == model);
    rsx! {
        label { class: "model-selector", title: "选择推理模型",
            Icon { name: "cpu", size: 17 }
            select { aria_label: "选择模型", value: "{model}", disabled: disabled || models.is_empty(), onchange: move |e| on_change.call(e.value()),
                if models.is_empty() { option { value: "{model}", if model.is_empty() { "等待连接模型" } else { "{model}" } } }
                else if !known { option { value: "{model}", disabled: true, "请选择可用模型" } }
                for available in models.iter() { option { value: "{available.id}", "{available.id}" } }
            }
            Icon { name: "chevron-down", size: 14 }
        }
    }
}
