//! Transport contracts. Plain text keeps the existing wire format; new modalities
//! are sent only after the service advertises their capability.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: ChatContent,
}

impl ChatMessage {
    pub fn text(role: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: ChatContent::Text(text.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl ChatContent {
    pub fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

impl From<String> for ChatContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    InputAudio {
        input_audio: InputAudio,
    },
    File {
        file: FileContent,
    },
    /// Reserved RustInfer extension; not implemented by the current server.
    VideoUrl {
        video_url: VideoUrl,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputAudio {
    /// Base64 encoded audio bytes, without a data URL prefix.
    pub data: String,
    pub format: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FileContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoUrl {
    pub url: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: u32,
    pub total_tokens: Option<u32>,
    pub performance: Option<Performance>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Performance {
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Deserialize)]
pub struct StreamChunk {
    pub id: Option<String>,
    pub choices: Vec<StreamChoice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct StreamChoice {
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ModelListResponse {
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub owned_by: Option<String>,
    #[serde(default)]
    pub capabilities: Option<ServerCapabilities>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Modalities {
    pub text: bool,
    pub image: bool,
    pub audio: bool,
    pub file: bool,
    pub video: bool,
}

impl Default for Modalities {
    fn default() -> Self {
        Self {
            text: true,
            image: false,
            audio: false,
            file: false,
            video: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AttachmentLimits {
    /// Image limits apply across all messages in a single chat request.
    pub max_images: usize,
    pub max_image_bytes: usize,
    pub image_mime_types: Vec<String>,
    pub max_image_dimension: usize,
    pub max_visual_tokens: usize,
    pub max_file_bytes: usize,
}

impl Default for AttachmentLimits {
    fn default() -> Self {
        Self {
            max_images: 4,
            max_image_bytes: 10 * 1024 * 1024,
            image_mime_types: vec!["image/png".into(), "image/jpeg".into()],
            max_image_dimension: 8192,
            max_visual_tokens: 1024,
            // No file service is available unless explicitly advertised.
            max_file_bytes: 0,
        }
    }
}

/// Missing discovery endpoints must never enable unimplemented modalities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerCapabilities {
    pub version: u32,
    pub inputs: Modalities,
    pub outputs: Modalities,
    pub transcription: bool,
    pub speech: bool,
    pub realtime: bool,
    pub file_upload: bool,
    pub greedy_sampling: bool,
    pub thinking: bool,
    pub limits: AttachmentLimits,
}

impl Default for ServerCapabilities {
    fn default() -> Self {
        Self {
            version: 1,
            inputs: Modalities::default(),
            outputs: Modalities::default(),
            transcription: false,
            speech: false,
            realtime: false,
            file_upload: false,
            greedy_sampling: false,
            thinking: false,
            limits: AttachmentLimits::default(),
        }
    }
}

/// Reserved audio transcription request. File bytes are supplied to the client.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionRequest {
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct SpeechResponse {
    pub bytes: Vec<u8>,
    pub mime_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UploadedFile {
    pub id: String,
    pub filename: String,
    pub bytes: u64,
    pub purpose: String,
}

/// Versioned RustInfer session proposal. A backend adapter may translate this
/// to its provider; this is not a promise of a particular provider's API shape.
#[derive(Debug, Clone, Serialize)]
pub struct RealtimeSessionRequest {
    pub model: String,
    pub modalities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RealtimeSession {
    pub id: String,
    pub transport: String,
    pub url: String,
    /// Session-scoped credential; never persist in browser storage.
    pub token: Option<String>,
    pub expires_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeClientEvent {
    InputAudioAppend { audio: String },
    InputAudioCommit,
    ResponseCreate,
    ResponseCancel,
    SessionClose,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeServerEvent {
    TranscriptDelta { delta: String },
    AudioDelta { audio: String, format: String },
    ResponseDone,
    Error { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_messages_keep_the_existing_wire_format() {
        let json = serde_json::to_value(ChatMessage::text("user", "你好")).unwrap();
        assert_eq!(json, serde_json::json!({"role": "user", "content": "你好"}));
        let parsed: ChatMessage = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.content.text(), "你好");
    }

    #[test]
    fn image_messages_use_server_content_part_shape() {
        let message = ChatMessage {
            role: "user".into(),
            content: ChatContent::Parts(vec![
                ContentPart::Text {
                    text: "describe".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,aGVsbG8=".into(),
                        detail: Some("auto".into()),
                    },
                },
            ]),
        };
        let value = serde_json::to_value(&message).unwrap();
        assert_eq!(value["content"][1]["type"], "image_url");
        assert_eq!(value["content"][1]["image_url"]["detail"], "auto");
        assert_eq!(message.content.text(), "describe");
    }

    #[test]
    fn absent_capabilities_are_conservatively_text_only() {
        let capabilities: ServerCapabilities = serde_json::from_str("{}").unwrap();
        assert!(capabilities.inputs.text && capabilities.outputs.text);
        assert!(!capabilities.inputs.image && !capabilities.inputs.audio);
        assert!(!capabilities.transcription && !capabilities.speech && !capabilities.realtime);
        assert!(!capabilities.file_upload);
        let model: ModelObject = serde_json::from_str(r#"{"id":"old-server-model"}"#).unwrap();
        assert!(model.capabilities.is_none());
    }
}
