//! Bounded Qwen3.5 image preparation at the HTTP boundary.
use anyhow::{Result, bail, ensure};
use base64::Engine;
use infer_protocol::multimodal::*;
use std::{io::Cursor, path::Path, sync::Arc};
use tokenizers::Tokenizer;

use crate::api::openai::types::{ChatMessage, ContentPart, InputChatMessage, MessageContent};

pub struct Qwen35Processor {
    min_pixels: usize,
    max_pixels: usize,
}

impl Qwen35Processor {
    pub fn load(path: &Path) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path.join("config.json"))?)?;
        let p: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            path.join("preprocessor_config.json"),
        )?)?;
        ensure!(
            config["image_token_id"] == IMAGE_TOKEN_ID
                && config["vision_start_token_id"] == VISION_START_ID
                && config["vision_end_token_id"] == VISION_END_ID,
            "unsupported image token configuration"
        );
        ensure!(
            p["patch_size"] == 16
                && p["temporal_patch_size"] == 2
                && p["merge_size"] == 2
                && p["image_mean"] == serde_json::json!([0.5, 0.5, 0.5])
                && p["image_std"] == serde_json::json!([0.5, 0.5, 0.5]),
            "unsupported Qwen3.5 image processor configuration"
        );
        let min_pixels = p["size"]["shortest_edge"].as_u64().unwrap_or(65536) as usize;
        let max_pixels = (p["size"]["longest_edge"].as_u64().unwrap_or(16777216) as usize)
            .min(MAX_IMAGE_TOKENS * 32 * 32);
        ensure!(
            min_pixels > 0 && min_pixels <= max_pixels,
            "invalid image pixel limits"
        );
        Ok(Self {
            min_pixels,
            max_pixels,
        })
    }

    pub fn prepare(
        &self,
        messages: &[InputChatMessage],
        tokenizer: &Tokenizer,
    ) -> Result<(Vec<i32>, Option<Arc<MultimodalInput>>)> {
        let mut text_messages = Vec::with_capacity(messages.len());
        let mut images = Vec::new();
        for message in messages {
            let mut content = String::new();
            match &message.content {
                MessageContent::Text(text) => content.push_str(text),
                MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            ContentPart::Text { text } => content.push_str(text),
                            ContentPart::ImageUrl { image_url } => {
                                ensure!(
                                    message.role == "user",
                                    "images are supported in user messages"
                                );
                                ensure!(images.len() < MAX_IMAGES, "too many images");
                                ensure!(
                                    image_url.detail.as_deref().is_none_or(|v| v == "auto"),
                                    "image detail currently supports auto only"
                                );
                                let image = self.decode(&image_url.url)?;
                                ensure!(
                                    images.iter().map(ImageInput::num_tokens).sum::<usize>()
                                        + image.num_tokens()
                                        <= MAX_VISUAL_TOKENS,
                                    "request exceeds visual token budget"
                                );
                                content.push_str("<|vision_start|>");
                                for _ in 0..image.num_tokens() {
                                    content.push_str("<|image_pad|>");
                                }
                                content.push_str("<|vision_end|>");
                                images.push(image);
                            }
                        }
                    }
                }
            }
            text_messages.push(ChatMessage {
                role: message.role.clone(),
                content,
            });
        }
        let prompt = super::get_template("qwen3_5").apply(&text_messages)?;
        let encoded = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let ids: Vec<i32> = encoded.get_ids().iter().map(|&id| id as i32).collect();
        if images.is_empty() {
            ensure!(
                !ids.contains(&IMAGE_TOKEN_ID),
                "image placeholders require image data"
            );
            return Ok((ids, None));
        }
        let mut spans = Vec::new();
        let mut cursor = 0usize;
        for (i, image) in images.iter().enumerate() {
            let start = ids[cursor..]
                .iter()
                .position(|&id| id == IMAGE_TOKEN_ID)
                .map(|v| v + cursor)
                .ok_or_else(|| anyhow::anyhow!("image placeholder missing"))?;
            spans.push(ImageSpan {
                image_index: i as u32,
                token_start: start as u32,
                token_len: image.num_tokens() as u32,
            });
            cursor = start + image.num_tokens();
        }
        let mm = MultimodalInput {
            images,
            spans,
            original_prompt_len: ids.len().try_into()?,
        };
        mm.validate_tokens(&ids).map_err(anyhow::Error::msg)?;
        Ok((ids, Some(Arc::new(mm))))
    }

    fn decode(&self, url: &str) -> Result<ImageInput> {
        const MAX_BYTES: usize = 10 * 1024 * 1024;
        let (header, payload) = url
            .split_once(',')
            .ok_or_else(|| anyhow::anyhow!("expected a PNG/JPEG base64 data URL"))?;
        ensure!(
            matches!(header, "data:image/png;base64" | "data:image/jpeg;base64"),
            "expected a PNG/JPEG base64 data URL"
        );
        ensure!(
            payload.len() <= MAX_BYTES.div_ceil(3) * 4,
            "encoded image exceeds 10 MiB"
        );
        let bytes = base64::engine::general_purpose::STANDARD.decode(payload)?;
        ensure!(bytes.len() <= MAX_BYTES, "encoded image exceeds 10 MiB");
        let mut reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
        ensure!(
            matches!(
                reader.format(),
                Some(image::ImageFormat::Png | image::ImageFormat::Jpeg)
            ),
            "unsupported image format"
        );
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(8192);
        limits.max_image_height = Some(8192);
        limits.max_alloc = Some(64 * 1024 * 1024);
        reader.limits(limits);
        let image = reader.decode()?.to_rgb8();
        self.prepare_rgb(
            image.as_raw(),
            image.height() as usize,
            image.width() as usize,
        )
    }

    pub fn prepare_rgb(&self, rgb: &[u8], h: usize, w: usize) -> Result<ImageInput> {
        ensure!(
            h > 0 && w > 0 && rgb.len() == h * w * 3,
            "invalid RGB image dimensions"
        );
        let (rh, rw) = smart_resize(h, w, self.min_pixels, self.max_pixels)?;
        let resized = resize_rgb(rgb, h, w, rh, rw);
        let gh = rh / 16;
        let gw = rw / 16;
        let mut patches = Vec::with_capacity(gh * gw * PATCH_WIDTH * 2);
        for by in 0..gh / 2 {
            for bx in 0..gw / 2 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        for channel in 0..3 {
                            for _time in 0..2 {
                                for py in 0..16 {
                                    for px in 0..16 {
                                        let y = (by * 2 + dy) * 16 + py;
                                        let x = (bx * 2 + dx) * 16 + px;
                                        let value = (resized[(y * rw + x) * 3 + channel] as f32
                                            - 127.5)
                                            / 127.5;
                                        patches.extend_from_slice(
                                            &half::bf16::from_f32(value).to_bits().to_le_bytes(),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let image = ImageInput {
            grid_thw: [1, gh as u32, gw as u32],
            patches,
        };
        image.validate().map_err(anyhow::Error::msg)?;
        Ok(image)
    }
}

fn smart_resize(
    h: usize,
    w: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    ensure!(
        h.max(w) as f64 / h.min(w) as f64 <= 200.0,
        "image aspect ratio exceeds 200"
    );
    let mut rh = ((h as f64 / 32.0).round_ties_even() as usize * 32).max(32);
    let mut rw = ((w as f64 / 32.0).round_ties_even() as usize * 32).max(32);
    if rh * rw > max_pixels {
        let beta = (h as f64 * w as f64 / max_pixels as f64).sqrt();
        rh = ((h as f64 / beta / 32.0).floor() as usize * 32).max(32);
        rw = ((w as f64 / beta / 32.0).floor() as usize * 32).max(32);
    } else if rh * rw < min_pixels {
        let beta = (min_pixels as f64 / (h as f64 * w as f64)).sqrt();
        rh = (h as f64 * beta / 32.0).ceil() as usize * 32;
        rw = (w as f64 * beta / 32.0).ceil() as usize * 32;
    }
    if rh * rw > max_pixels {
        bail!("resized image exceeds pixel budget");
    }
    Ok((rh, rw))
}

// Antialiased separable bicubic with pixel-center coordinates and clipped borders.
// uint8 is rounded after each pass, matching torchvision's uint8 CPU resize.
fn resize_rgb(src: &[u8], h: usize, w: usize, rh: usize, rw: usize) -> Vec<u8> {
    fn coefficients(input: usize, output: usize) -> (Vec<Vec<(usize, i64)>>, u32) {
        let scale = input as f64 / output as f64;
        let filter_scale = scale.max(1.0);
        let weights: Vec<Vec<(usize, f64)>> = (0..output)
            .map(|i| {
                let center = (i as f64 + 0.5) * scale;
                let start = ((center - 2.0 * filter_scale + 0.5).floor() as isize).max(0) as usize;
                let end = ((center + 2.0 * filter_scale + 0.5).floor() as usize).min(input);
                let mut weights: Vec<_> = (start..end)
                    .map(|j| {
                        let x = ((j as f64 + 0.5 - center) / filter_scale).abs();
                        let a = if x < 1.0 {
                            ((1.5 * x - 2.5) * x) * x + 1.0
                        } else if x < 2.0 {
                            ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
                        } else {
                            0.0
                        };
                        (j, a)
                    })
                    .collect();
                let sum = weights.iter().map(|v| v.1).sum::<f64>();
                for v in &mut weights {
                    v.1 /= sum;
                }
                weights
            })
            .collect();
        let max = weights.iter().flatten().map(|v| v.1).fold(0.0f64, f64::max);
        let precision = (0..22)
            .find(|&p| (0.5 + max * (1u32 << (p + 1)) as f64) as i32 >= 32768)
            .unwrap_or(22);
        (
            weights
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|(i, a)| (i, (a * (1u32 << precision) as f64).round() as i64))
                        .collect()
                })
                .collect(),
            precision,
        )
    }
    if h == rh && w == rw {
        return src.to_vec();
    }
    let (cx, px) = coefficients(w, rw);
    let (cy, py) = coefficients(h, rh);
    let mut tmp = vec![0u8; h * rw * 3];
    for y in 0..h {
        for x in 0..rw {
            for c in 0..3 {
                tmp[(y * rw + x) * 3 + c] = ((cx[x]
                    .iter()
                    .map(|&(j, a)| src[(y * w + j) * 3 + c] as i64 * a)
                    .sum::<i64>()
                    + (1i64 << (px - 1)))
                    >> px)
                    .clamp(0, 255) as u8;
            }
        }
    }
    let mut out = vec![0u8; rh * rw * 3];
    for y in 0..rh {
        for x in 0..rw {
            for c in 0..3 {
                out[(y * rw + x) * 3 + c] = ((cy[y]
                    .iter()
                    .map(|&(j, a)| tmp[(j * rw + x) * 3 + c] as i64 * a)
                    .sum::<i64>()
                    + (1i64 << (py - 1)))
                    >> py)
                    .clamp(0, 255) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry_and_patch_order() {
        let p = Qwen35Processor {
            min_pixels: 1024,
            max_pixels: MAX_IMAGE_TOKENS * 1024,
        };
        assert_eq!(smart_resize(270, 350, 65536, 524288).unwrap(), (256, 352));
        let mut rgb = vec![0u8; 32 * 32 * 3];
        for y in 0..32 {
            for x in 0..32 {
                rgb[(y * 32 + x) * 3] = if x >= 16 { 255 } else { 0 };
            }
        }
        let image = p.prepare_rgb(&rgb, 32, 32).unwrap();
        let values: Vec<_> = image
            .patches
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect();
        assert_eq!(image.grid_thw, [1, 2, 2]);
        assert_eq!(values[0], -1.0);
        assert_eq!(values[PATCH_WIDTH], 1.0);
        assert_eq!(values[2 * PATCH_WIDTH], -1.0);
        assert_eq!(values[3 * PATCH_WIDTH], 1.0);
        assert_eq!(&values[..256], &values[256..512]);
    }
    #[test]
    #[ignore = "requires QWEN35_MODEL_PATH and QWEN35_VISION_REFERENCE"]
    fn processor_matches_hf_fixture() {
        let model = std::env::var("QWEN35_MODEL_PATH").unwrap();
        let dir = std::env::var("QWEN35_VISION_REFERENCE").unwrap();
        let dir = Path::new(&dir);
        let p = Qwen35Processor::load(Path::new(&model)).unwrap();
        let url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD
                .encode(std::fs::read(dir.join("image.png")).unwrap())
        );
        let messages:Vec<InputChatMessage>=serde_json::from_value(serde_json::json!([{"role":"user","content":[
            {"type":"image_url","image_url":{"url":url}}, {"type":"text","text":"Describe the image briefly."}]}])).unwrap();
        let tokenizer = Tokenizer::from_file(Path::new(&model).join("tokenizer.json")).unwrap();
        let (ids, mm) = p.prepare(&messages, &tokenizer).unwrap();
        let mm = mm.unwrap();
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(
            ids,
            serde_json::from_value::<Vec<i32>>(meta["input_ids"].clone()).unwrap()
        );
        assert_eq!(
            mm.positions().unwrap().axes,
            serde_json::from_value::<Vec<[i32; 3]>>(meta["positions"].clone()).unwrap()
        );
        let expected = std::fs::read(dir.join("patches.bf16")).unwrap();
        let actual = &mm.images[0].patches;
        std::fs::write(dir.join("rust-patches.bf16"), actual).unwrap();
        assert_eq!(actual.len(), expected.len());
        let different = actual
            .chunks_exact(2)
            .zip(expected.chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "processor BF16 patches: {different}/{} differ",
            actual.len() / 2
        );
        assert_eq!(different, 0);
    }
    #[test]
    #[ignore = "requires QWEN35_MODEL_PATH and QWEN35_VISION_REFERENCE processor cases"]
    fn processor_matches_hf_resize_cases() {
        let model = std::env::var("QWEN35_MODEL_PATH").unwrap();
        let dir = std::env::var("QWEN35_VISION_REFERENCE").unwrap();
        let dir = Path::new(&dir);
        let p = Qwen35Processor::load(Path::new(&model)).unwrap();
        let cases: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(dir.join("processor_cases.json")).unwrap())
                .unwrap();
        for (i, case) in cases.iter().enumerate() {
            let rgb = std::fs::read(dir.join(format!("processor{i}.rgb"))).unwrap();
            let image = p
                .prepare_rgb(
                    &rgb,
                    case["height"].as_u64().unwrap() as usize,
                    case["width"].as_u64().unwrap() as usize,
                )
                .unwrap();
            assert_eq!(
                image.grid_thw,
                serde_json::from_value::<[u32; 3]>(case["grid"].clone()).unwrap()
            );
            let reference = std::fs::read(dir.join(format!("processor{i}.bf16"))).unwrap();
            let differences = image
                .patches
                .iter()
                .zip(&reference)
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(image.patches.len(), reference.len());
            eprintln!(
                "processor {i} {:?}: {differences} differing bytes",
                image.grid_thw
            );
            assert_eq!(differences, 0);
        }
    }
}
