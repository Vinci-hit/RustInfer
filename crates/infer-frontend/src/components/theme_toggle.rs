use dioxus::prelude::*;

#[component]
pub fn ThemeToggle() -> Element {
    let mut is_dark = use_signal(|| true);

    rsx! {
        button {
            class: "p-2 rounded-lg hover:bg-white/5 transition-colors",
            onclick: move |_| {
                is_dark.set(!is_dark());
            },
            title: if is_dark() { "Switch to light mode" } else { "Switch to dark mode" },

            if is_dark() {
                // Moon icon
                svg {
                    class: "w-5 h-5 text-yellow-400",
                    fill: "none",
                    stroke: "currentColor",
                    stroke_width: "2",
                    view_box: "0 0 24 24",
                    path { d: "M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" }
                }
            } else {
                // Sun icon
                svg {
                    class: "w-5 h-5 text-orange-400",
                    fill: "none",
                    stroke: "currentColor",
                    stroke_width: "2",
                    view_box: "0 0 24 24",
                    circle { cx: "12", cy: "12", r: "5" }
                    path { d: "M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" }
                }
            }
        }
    }
}
