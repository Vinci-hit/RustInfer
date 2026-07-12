use dioxus::prelude::*;

#[component]
pub fn MessageInput(on_send: EventHandler<String>, is_disabled: bool) -> Element {
    let mut input_text = use_signal(String::new);

    let mut do_send = move || {
        let text = input_text().clone();
        if !text.trim().is_empty() && !is_disabled {
            on_send.call(text);
            input_text.set(String::new());
        }
    };

    rsx! {
        div {
            class: "px-6 py-4 border-t border-white/5",

            div {
                class: "flex items-end gap-3 p-1 rounded-2xl bg-[var(--color-surface-1)] border border-[var(--color-border)] focus-within:border-[var(--color-accent)] transition-colors",

                // Text input
                textarea {
                    class: "flex-1 bg-transparent text-sm text-[var(--color-text-primary)] placeholder:text-[var(--color-text-muted)] resize-none px-4 py-3 focus:outline-none max-h-36 min-h-[44px]",
                    rows: "1",
                    placeholder: "Type a message... (Enter to send)",
                    value: "{input_text()}",
                    disabled: is_disabled,
                    oninput: move |e| {
                        input_text.set(e.value());
                    },
                    onkeypress: move |e| {
                        if e.key() == Key::Enter && !e.modifiers().shift() {
                            e.prevent_default();
                            do_send();
                        }
                    },
                }

                // Send button
                button {
                    class: "shrink-0 m-1.5 p-2.5 rounded-xl btn-primary disabled:opacity-30",
                    onclick: move |_| do_send(),
                    disabled: is_disabled || input_text().trim().is_empty(),

                    svg {
                        class: "w-4 h-4",
                        fill: "none",
                        stroke: "currentColor",
                        stroke_width: "2",
                        view_box: "0 0 24 24",
                        path { d: "M5 12h14M12 5l7 7-7 7" }
                    }
                }
            }

            // Hint text
            p {
                class: "text-[10px] text-[var(--color-text-muted)] mt-2 text-center",
                "Press Enter to send, Shift+Enter for new line"
            }
        }
    }
}
