use super::settings::AppSettings;
use crate::api::types::{ModelObject, ServerCapabilities};
use dioxus::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Connection {
    Connecting,
    Online,
    Offline(String),
}

#[derive(Clone, Copy)]
pub struct Workspace {
    pub settings: Signal<AppSettings>,
    pub connection: Signal<Connection>,
    pub models: Signal<Vec<ModelObject>>,
    pub capabilities: Signal<ServerCapabilities>,
    pub reconnect: Signal<u32>,
    pub notice: Signal<Option<String>>,
}

pub fn load_local<T: serde::de::DeserializeOwned>(key: &str) -> Option<T> {
    #[cfg(target_arch = "wasm32")]
    {
        let storage = web_sys::window()?.local_storage().ok()??;
        let raw = storage.get_item(key).ok()??;
        serde_json::from_str(&raw).ok()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = key;
        None
    }
}

pub fn save_local<T: serde::Serialize>(key: &str, value: &T) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    {
        let storage = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
            .ok_or_else(|| "浏览器存储不可用，本次会话暂时无法保存。".to_string())?;
        let data = serde_json::to_string(value).map_err(|e| e.to_string())?;
        storage.set_item(key, &data).map_err(|_| {
            "浏览器存储空间不足，请导出重要对话并删除部分历史记录。当前会话仍可继续使用。".into()
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (key, value);
        Ok(())
    }
}
