use dioxus::prelude::*;

#[component]
pub fn Icon(name: &'static str, #[props(default = 20)] size: u32) -> Element {
    let path = match name {
        "plus" => "M12 5v14M5 12h14",
        "chat" => "M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5Z",
        "search" => "M21 21l-4.5-4.5M19 10.5a8.5 8.5 0 1 1-17 0 8.5 8.5 0 0 1 17 0Z",
        "panel" => "M9 3v18M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z",
        "settings" | "sliders" => "M4 7h9m4 0h3M4 17h3m4 0h9M13 4v6M7 14v6",
        "image" => "M4 3h16a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1ZM3 16l5-5 4 4 4-6 5 7M8 7h.01",
        "mic" => "M12 15a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v7a3 3 0 0 0 3 3ZM5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8",
        "audio-lines" | "wave" => "M3 10v4M7 6v12M11 3v18M15 7v10M19 5v14M23 10v4",
        "arrow-up" => "M12 19V5M5 12l7-7 7 7",
        "arrow-right" => "M5 12h14m-6-6 6 6-6 6",
        "square" => "M6 6h12v12H6Z",
        "x" => "m6 6 12 12M6 18 18 6",
        "file" => "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8ZM14 2v6h6M8 13h8M8 17h5",
        "video" => "M16 8l6-4v16l-6-4M3 5h11a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z",
        "chevron-down" => "m6 9 6 6 6-6",
        "copy" => "M9 9h12v12H9ZM15 5V3H3v12h2",
        "check" => "m5 12 4 4L19 6",
        "refresh" => "M20 7v5h-5M4 17v-5h5M6 7a7 7 0 0 1 12-2l2 3M4 16l2 3a7 7 0 0 0 12-2",
        "trash" => "M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7",
        "download" => "M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5",
        "sun" => "M12 2v2M12 20v2M2 12h2M20 12h2m-3-7 1.4-1.4M3.6 20.4 5 19M5 5 3.6 3.6M19 19l1.4 1.4M17 12a5 5 0 1 1-10 0 5 5 0 0 1 10 0Z",
        "moon" => "M21 12.8A9 9 0 0 1 11.2 3 9 9 0 1 0 21 12.8Z",
        "code" => "m8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18",
        "spark" => "m12 3 2.6 6.4L21 12l-6.4 2.6L12 21l-2.6-6.4L3 12l6.4-2.6ZM20 2v4m-2-2h4",
        "cpu" => "M7 7h10v10H7ZM9 1v3m6-3v3M9 20v3m6-3v3M1 9h3m-3 6h3m16-6h3m-3 6h3M5 4h14a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1Z",
        "shield" => "M12 3 3 7v6c0 5 9 9 9 9s9-4 9-9V7ZM8 12l3 3 5-5",
        "globe" => "M21 12A9 9 0 1 1 3 12a9 9 0 0 1 18 0ZM3 12h18M12 3c5 5 5 13 0 18-5-5-5-13 0-18Z",
        "info" => "M12 11v6m0-10h.01M22 12A10 10 0 1 1 2 12a10 10 0 0 1 20 0Z",
        _ => "M5 12h14",
    };
    rsx! { svg { width: "{size}", height: "{size}", view_box: "0 0 24 24", fill: "none", stroke: "currentColor", stroke_width: "1.7", stroke_linecap: "round", stroke_linejoin: "round", "aria-hidden": "true", path { d: path } } }
}
