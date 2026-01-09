use dioxus::prelude::*;

mod api;
mod components;
mod state;

fn main() {
    dioxus::launch(App);
}

fn App() -> Element {
    rsx! {
        div {
            class: "min-h-screen bg-gray-900 text-white",

            div { class: "container mx-auto p-4 h-screen",
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
        }
    }
}
