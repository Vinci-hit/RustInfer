use infer_core::base::DeviceType;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub model_path: String,
    pub device: DeviceType,
    pub max_tokens: usize,
}
