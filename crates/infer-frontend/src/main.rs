use dioxus::prelude::*;

mod api;
mod components;
mod state;
mod utils;

fn main() {
    dioxus::launch(App);
}

#[derive(Clone, Copy, PartialEq)]
enum Page {
    Chat,
    Admin,
}

#[component]
fn App() -> Element {
    let mut current_page = use_signal(|| Page::Chat);

    rsx! {
        document::Link { rel: "stylesheet", href: asset!("/assets/output.css") }
        document::Link {
            rel: "stylesheet",
            href: "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css",
            integrity: "sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV",
            crossorigin: "anonymous"
        }
        document::Script { src: asset!("/public/katex-init.js"), r#type: "module" }
        document::Script { src: asset!("/public/mermaid-init.js"), r#type: "module" }

        div {
            class: "min-h-screen bg-gray-900 text-white",

            // Navigation Bar
            nav {
                class: "bg-gray-800 border-b border-gray-700 shadow-lg",
                div { class: "container mx-auto px-4",
                    div { class: "flex items-center justify-between h-16",
                        // Logo/Title
                        div { class: "flex items-center gap-3",
                            div { class: "text-2xl", "🦀" }
                            h1 { class: "text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent",
                                "RustInfer"
                            }
                        }

                        // Navigation Tabs
                        div { class: "flex gap-2",
                            button {
                                class: if *current_page.read() == Page::Chat {
                                    "px-6 py-2 rounded-lg bg-blue-600 text-white font-medium transition-all"
                                } else {
                                    "px-6 py-2 rounded-lg bg-gray-700 text-gray-300 hover:bg-gray-600 font-medium transition-all"
                                },
                                onclick: move |_| current_page.set(Page::Chat),
                                "💬 Chat"
                            }

                            button {
                                class: if *current_page.read() == Page::Admin {
                                    "px-6 py-2 rounded-lg bg-purple-600 text-white font-medium transition-all"
                                } else {
                                    "px-6 py-2 rounded-lg bg-gray-700 text-gray-300 hover:bg-gray-600 font-medium transition-all"
                                },
                                onclick: move |_| current_page.set(Page::Admin),
                                "📊 Admin"
                            }
                        }
                    }
                }
            }

            // Page Content
            match *current_page.read() {
                Page::Chat => rsx! {
                    div { class: "container mx-auto p-4 h-[calc(100vh-4rem)]",
                        div { class: "grid grid-cols-3 gap-4 h-full",
                            // Chat (2/3 width)
                            div { class: "col-span-2 flex flex-col",
                                components::chat_interface::ChatInterface {}
                            }

                            // Metrics panel (1/3 width)
                            div { class: "col-span-1",
                                components::metrics_panel::MetricsPanel {}
                            }
                        }
                    }
                },
                Page::Admin => rsx! {
                    components::admin_console::AdminConsole {}
                }
            }
        }
    }
}
