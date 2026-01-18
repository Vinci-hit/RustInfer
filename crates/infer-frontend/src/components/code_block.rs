use dioxus::prelude::*;
use web_sys::window;

#[component]
pub fn CodeBlock(code: String, language: String) -> Element {
    let mut copied = use_signal(|| false);
    let code_clone = code.clone();

    let copy_code = move |_| {
        let code = code_clone.clone();
        spawn(async move {
            if let Some(window) = window() {
                let navigator = window.navigator();
                let clipboard = navigator.clipboard();
                let promise = clipboard.write_text(&code);
                let _ = wasm_bindgen_futures::JsFuture::from(promise).await;

                copied.set(true);

                // Reset after 2 seconds
                gloo_timers::future::TimeoutFuture::new(2000).await;
                copied.set(false);
            }
        });
    };

    rsx! {
        div {
            class: "relative group code-block-wrapper",

            // Copy button
            button {
                class: "absolute top-2 right-2 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-all opacity-0 group-hover:opacity-100",
                onclick: copy_code,

                if copied() {
                    span { class: "text-green-400", "✓ Copied!" }
                } else {
                    span { class: "text-gray-300", "📋 Copy" }
                }
            }

            // Language badge
            div {
                class: "absolute top-2 left-2 px-2 py-1 bg-blue-600 rounded text-xs font-mono text-white opacity-70",
                "{language}"
            }

            // Code content
            pre {
                class: "bg-gray-900 rounded-lg p-4 pt-10 overflow-x-auto",
                code {
                    class: "language-{language} text-sm font-mono text-gray-100",
                    "{code}"
                }
            }
        }
    }
}
