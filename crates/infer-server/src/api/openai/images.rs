//! POST /v1/images/generations handler

use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image::{codecs::jpeg::JpegEncoder, codecs::png::PngEncoder, ColorType, ImageEncoder};

use crate::client::InferClient;
use crate::error::AppError;
use crate::state::SharedState;

use super::types::*;

const DIFFUSION_TEXT_ENCODER_MAX_SEQ_LEN: usize = 512;

/// POST /v1/images/generations
#[axum::debug_handler]
pub async fn image_generations(
    State(state): State<SharedState>,
    Json(req): Json<ImageGenerationRequest>,
) -> Result<Response, AppError> {
    validate_image_request(&req)?;

    let n = req.n.unwrap_or(1);
    let response_format = req.response_format.unwrap_or(ImageResponseFormat::B64Json);
    let output_format = req.output_format.unwrap_or(ImageOutputFormat::Png);
    let jpeg_quality = req.jpeg_quality.unwrap_or(90);
    let (width, height) = parse_size(req.size.as_deref().unwrap_or("1024x1024"))?;

    if matches!(response_format, ImageResponseFormat::Binary) && n != 1 {
        return Err(AppError::bad_request("response_format=binary requires n=1"));
    }

    let prompt_input_ids = tokenize_diffusion_prompt(&state.tokenizer, &req.prompt)?;
    let negative_prompt_input_ids = match &req.negative_prompt {
        Some(negative_prompt) if !negative_prompt.trim().is_empty() => {
            Some(tokenize_diffusion_prompt(&state.tokenizer, negative_prompt)?)
        }
        _ => None,
    };

    let mut encoded_images = Vec::with_capacity(n);
    for _ in 0..n {
        let request_id = uuid::Uuid::new_v4().to_string();
        let engine_req = infer_protocol::server_to_scheduler::InferenceRequest {
            request_id,
            modality: infer_protocol::server_to_scheduler::InferenceModality::Diffusion,
            input_ids: vec![],
            max_tokens: 1,
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            stream: false,
            priority: 0,
            stop_sequences: vec![],
            diffusion: Some(infer_protocol::server_to_scheduler::DiffusionRequest {
                prompt: req.prompt.clone(),
                prompt_input_ids: prompt_input_ids.clone(),
                negative_prompt: req.negative_prompt.clone(),
                negative_prompt_input_ids: negative_prompt_input_ids.clone(),
                height,
                width,
                num_inference_steps: req.num_inference_steps.unwrap_or(8),
                sigmas: req.sigmas.clone(),
                guidance_scale: req.guidance_scale.unwrap_or(0.0),
                seed: req.seed,
                output_format: "rgb8".to_string(),
            }),
        };

        let engine_resp = state.client.infer(engine_req).await
            .map_err(AppError::internal)?;

        if let infer_protocol::scheduler_to_server::ResponseStatus::Error = engine_resp.status {
            return Err(AppError::internal(anyhow::anyhow!(
                "Engine error: {}",
                engine_resp.error.unwrap_or_else(|| "Unknown".to_string())
            )));
        }

        let image = engine_resp.images.first()
            .ok_or_else(|| AppError::internal(anyhow::anyhow!("Engine returned no image")))?;
        let encoded = encode_backend_image(image, output_format, jpeg_quality)?;
        encoded_images.push(encoded);
    }

    match response_format {
        ImageResponseFormat::Binary => {
            let image = encoded_images.into_iter().next()
                .ok_or_else(|| AppError::internal(anyhow::anyhow!("No encoded image")))?;
            let content_type = match output_format {
                ImageOutputFormat::Png => "image/png",
                ImageOutputFormat::Jpeg => "image/jpeg",
            };
            let mut response = Response::new(Body::from(image));
            *response.status_mut() = StatusCode::OK;
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static(content_type),
            );
            Ok(response)
        }
        ImageResponseFormat::B64Json => {
            let created = chrono::Utc::now().timestamp();
            let mime_type = match output_format {
                ImageOutputFormat::Png => "image/png".to_string(),
                ImageOutputFormat::Jpeg => "image/jpeg".to_string(),
            };
            let data = encoded_images
                .into_iter()
                .map(|bytes| ImageData {
                    b64_json: Some(BASE64.encode(bytes)),
                    mime_type: Some(mime_type.clone()),
                    revised_prompt: None,
                })
                .collect();

            Ok(Json(ImageGenerationResponse { created, data }).into_response())
        }
    }
}

fn validate_image_request(req: &ImageGenerationRequest) -> Result<(), AppError> {
    if req.prompt.trim().is_empty() {
        return Err(AppError::bad_request("prompt must not be empty"));
    }
    if let Some(n) = req.n
        && (n == 0 || n > 16) {
            return Err(AppError::bad_request("n must be between 1 and 16"));
        }
    if let Some(steps) = req.num_inference_steps
        && steps == 0 {
            return Err(AppError::bad_request("num_inference_steps must be greater than 0"));
        }
    if let Some(quality) = req.jpeg_quality
        && (quality == 0 || quality > 100) {
            return Err(AppError::bad_request("jpeg_quality must be between 1 and 100"));
        }
    if let Some(size) = &req.size {
        parse_size(size)?;
    }
    Ok(())
}

fn parse_size(size: &str) -> Result<(u32, u32), AppError> {
    let Some((w, h)) = size.split_once('x') else {
        return Err(AppError::bad_request("size must be formatted as WIDTHxHEIGHT, e.g. 1024x1024"));
    };
    let width: u32 = w.parse()
        .map_err(|_| AppError::bad_request("size width must be an integer"))?;
    let height: u32 = h.parse()
        .map_err(|_| AppError::bad_request("size height must be an integer"))?;
    if width == 0 || height == 0 || !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(AppError::bad_request("image width/height must be positive multiples of 16"));
    }
    Ok((width, height))
}

fn tokenize_diffusion_prompt(tokenizer: &tokenizers::Tokenizer, prompt: &str) -> Result<Vec<i32>, AppError> {
    let formatted = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    );
    let encoding = tokenizer.encode(formatted.as_str(), true)
        .map_err(|e| AppError::internal(anyhow::anyhow!("Diffusion prompt tokenize error: {}", e)))?;
    let ids: Vec<i32> = encoding
        .get_ids()
        .iter()
        .take(DIFFUSION_TEXT_ENCODER_MAX_SEQ_LEN)
        .map(|&id| id as i32)
        .collect();
    if ids.is_empty() {
        return Err(AppError::bad_request("prompt produced no tokens"));
    }
    Ok(ids)
}

fn encode_backend_image(
    image: &infer_protocol::scheduler_to_server::ImageOutput,
    format: ImageOutputFormat,
    jpeg_quality: u8,
) -> Result<Vec<u8>, AppError> {
    if image.format != "rgb8" || image.channels != 3 {
        return Err(AppError::internal(anyhow::anyhow!(
            "unsupported backend image format: format={} channels={}",
            image.format,
            image.channels,
        )));
    }
    let expected = image.width as usize * image.height as usize * 3;
    if image.data.len() != expected {
        return Err(AppError::internal(anyhow::anyhow!(
            "invalid backend image size: got {} bytes, expected {}",
            image.data.len(), expected
        )));
    }

    let mut out = Vec::new();
    match format {
        ImageOutputFormat::Png => {
            let encoder = PngEncoder::new(&mut out);
            encoder.write_image(&image.data, image.width, image.height, ColorType::Rgb8.into())
                .map_err(|e| AppError::internal(anyhow::anyhow!("PNG encode error: {}", e)))?;
        }
        ImageOutputFormat::Jpeg => {
            let mut encoder = JpegEncoder::new_with_quality(&mut out, jpeg_quality);
            encoder.encode(&image.data, image.width, image.height, ColorType::Rgb8.into())
                .map_err(|e| AppError::internal(anyhow::anyhow!("JPEG encode error: {}", e)))?;
        }
    }
    Ok(out)
}
