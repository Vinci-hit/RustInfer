//! Additive feature discovery for clients. Advertising a transport type does not
//! create an inference implementation; unsupported modalities stay disabled.
use axum::{Json, extract::State};
use serde::Serialize;

use crate::{
    chat::multimodal::{MAX_IMAGE_BYTES, MAX_IMAGE_DIMENSION},
    state::SharedState,
};

#[derive(Debug, Clone, Serialize)]
pub struct Modalities {
    pub text: bool,
    pub image: bool,
    pub audio: bool,
    pub file: bool,
    pub video: bool,
}

impl Modalities {
    fn text_only() -> Self {
        Self {
            text: true,
            image: false,
            audio: false,
            file: false,
            video: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AttachmentLimits {
    pub max_images: usize,
    pub max_image_bytes: usize,
    pub image_mime_types: [&'static str; 2],
    pub max_image_dimension: usize,
    pub max_visual_tokens: usize,
    pub max_file_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
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

impl ServerCapabilities {
    pub fn for_state(state: &SharedState) -> Self {
        let mut caps = Self::new(
            state.image_processor.is_some(),
            state.config.speculative_draft_tokens() > 0,
        );
        caps.thinking = state.gguf_text.is_some();
        caps
    }

    fn new(has_image_processor: bool, speculative: bool) -> Self {
        let mut inputs = Modalities::text_only();
        // The chat validator forbids image requests with speculative decoding.
        inputs.image = has_image_processor && !speculative;
        Self {
            version: 1,
            inputs,
            outputs: Modalities::text_only(),
            transcription: false,
            speech: false,
            realtime: false,
            file_upload: false,
            greedy_sampling: speculative,
            thinking: false,
            limits: AttachmentLimits {
                max_images: infer_protocol::multimodal::MAX_IMAGES,
                max_image_bytes: MAX_IMAGE_BYTES,
                image_mime_types: ["image/png", "image/jpeg"],
                max_image_dimension: MAX_IMAGE_DIMENSION as usize,
                max_visual_tokens: infer_protocol::multimodal::MAX_VISUAL_TOKENS,
                max_file_bytes: 0,
            },
        }
    }
}

pub async fn get_capabilities(State(state): State<SharedState>) -> Json<ServerCapabilities> {
    Json(ServerCapabilities::for_state(&state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_capability_tracks_actual_request_support() {
        assert!(!ServerCapabilities::new(false, false).inputs.image);
        assert!(ServerCapabilities::new(true, false).inputs.image);
        let speculative = ServerCapabilities::new(true, true);
        assert!(!speculative.inputs.image);
        assert!(speculative.greedy_sampling);
    }

    #[test]
    fn discovery_never_advertises_reserved_endpoints() {
        let caps = serde_json::to_value(ServerCapabilities::new(true, false)).unwrap();
        assert_eq!(caps["version"], 1);
        assert_eq!(caps["inputs"]["text"], true);
        assert_eq!(caps["inputs"]["image"], true);
        for flag in ["audio", "file", "video"] {
            assert_eq!(caps["inputs"][flag], false);
        }
        for flag in ["transcription", "speech", "realtime", "file_upload"] {
            assert_eq!(caps[flag], false);
        }
        assert_eq!(caps["limits"]["max_images"], 4);
        assert_eq!(caps["limits"]["max_image_bytes"], 10 * 1024 * 1024);
        assert_eq!(caps["limits"]["max_image_dimension"], 8192);
        assert_eq!(
            caps["limits"]["image_mime_types"],
            serde_json::json!(["image/png", "image/jpeg"])
        );
    }
}
