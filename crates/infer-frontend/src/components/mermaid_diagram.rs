use dioxus::prelude::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "/public/mermaid-init.js")]
extern "C" {
    fn initMermaid();
    fn renderMermaid(element_id: &str);
}

#[component]
pub fn MermaidDiagram(code: String) -> Element {
    let id = use_signal(|| format!("mermaid-{}", uuid::Uuid::new_v4()));

    // Initialize Mermaid on mount
    use_effect(move || {
        initMermaid();
    });

    // Render diagram when code changes
    use_effect(move || {
        let element_id = id();
        spawn(async move {
            // Small delay to ensure DOM is ready
            gloo_timers::future::TimeoutFuture::new(100).await;
            renderMermaid(&element_id);
        });
    });

    rsx! {
        div {
            id: "{id}",
            class: "mermaid bg-gray-800 p-4 rounded-lg my-4",
            "{code}"
        }
    }
}
