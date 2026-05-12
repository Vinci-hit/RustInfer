use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Theme {
    Dark,
    Light,
}

impl Default for Theme {
    fn default() -> Self {
        Theme::Dark
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub theme: Theme,
    pub sidebar_collapsed: bool,
    pub metrics_visible: bool,
    pub api_base_url: String,
    pub default_model: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            theme: Theme::Dark,
            sidebar_collapsed: false,
            metrics_visible: true,
            api_base_url: "http://localhost:8000".to_string(),
            default_model: "llama3".to_string(),
        }
    }
}
