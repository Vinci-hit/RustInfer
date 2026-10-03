use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum Theme {
    Dark,
    #[default]
    Light,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AppSettings {
    pub theme: Theme,
    pub sidebar_collapsed: bool,
    pub metrics_visible: bool,
    pub api_base_url: String,
    pub default_model: String,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
    pub system_prompt: String,
    pub enable_thinking: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            theme: Theme::Light,
            sidebar_collapsed: false,
            metrics_visible: true,
            api_base_url: default_api_url(),
            default_model: String::new(),
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: 2048,
            system_prompt: String::new(),
            enable_thinking: false,
        }
    }
}

fn default_api_url() -> String {
    #[cfg(target_arch = "wasm32")]
    if let Some(window) = web_sys::window() {
        let location = window.location();
        let hostname = location.hostname().unwrap_or_else(|_| "localhost".into());
        let protocol = location.protocol().unwrap_or_else(|_| "http:".into());
        return format!("{protocol}//{hostname}:8080");
    }
    "http://localhost:8080".into()
}
