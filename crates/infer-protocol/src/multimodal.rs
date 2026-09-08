//! Immutable, prepared image inputs. Pixels are BF16 patch rows, not JPEG bytes.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const IMAGE_TOKEN_ID: i32 = 248056;
pub const VISION_START_ID: i32 = 248053;
pub const VISION_END_ID: i32 = 248054;
pub const PATCH_WIDTH: usize = 3 * 2 * 16 * 16;
pub const MAX_IMAGE_TOKENS: usize = 512;
pub const MAX_VISUAL_TOKENS: usize = 1024;
pub const MAX_IMAGES: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInput {
    pub grid_thw: [u32; 3],
    /// Little-endian BF16, [T*H*W, 1536], in spatial merge order.
    #[serde(with = "serde_bytes")]
    pub patches: Vec<u8>,
}

impl ImageInput {
    pub fn num_tokens(&self) -> usize {
        self.grid_thw.iter().map(|&v| v as usize).product::<usize>() / 4
    }

    pub fn validate(&self) -> Result<(), String> {
        let [t, h, w] = self.grid_thw;
        if t != 1 || h == 0 || w == 0 || h % 2 != 0 || w % 2 != 0 {
            return Err("image grid must be [1, even height, even width]".into());
        }
        let patches = (h as usize)
            .checked_mul(w as usize)
            .ok_or("image grid overflow")?;
        if patches > MAX_IMAGE_TOKENS * 4 || self.patches.len() != patches * PATCH_WIDTH * 2 {
            return Err("image patch shape or token budget is invalid".into());
        }
        if self
            .patches
            .chunks_exact(2)
            .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
        {
            return Err("image patches must be finite".into());
        }
        Ok(())
    }

    /// Include geometry: identical patch bytes with different grids are different inputs.
    pub fn cache_key(&self) -> [u8; 32] {
        let mut hash = Sha256::new();
        for n in self.grid_thw {
            hash.update(n.to_le_bytes());
        }
        hash.update(&self.patches);
        hash.finalize().into()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSpan {
    pub image_index: u32,
    pub token_start: u32,
    pub token_len: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalInput {
    pub images: Vec<ImageInput>,
    pub spans: Vec<ImageSpan>,
    /// Never extended when a preempted request appends its generated tokens.
    pub original_prompt_len: u32,
}

impl MultimodalInput {
    pub fn validate(&self) -> Result<(), String> {
        if self.original_prompt_len > i32::MAX as u32
            || self.images.is_empty()
            || self.images.len() > MAX_IMAGES
            || self.spans.len() != self.images.len()
        {
            return Err("image count or span count is invalid".into());
        }
        let mut end = 0usize;
        let mut total = 0usize;
        for (i, (image, span)) in self.images.iter().zip(&self.spans).enumerate() {
            image.validate()?;
            let start = span.token_start as usize;
            let len = span.token_len as usize;
            if span.image_index as usize != i
                || len != image.num_tokens()
                || start < end
                || start
                    .checked_add(len)
                    .is_none_or(|v| v >= self.original_prompt_len as usize)
                || start == 0
            {
                return Err("image span does not match the prompt or feature layout".into());
            }
            end = start + len;
            total += len;
        }
        if total > MAX_VISUAL_TOKENS {
            return Err("request exceeds visual token budget".into());
        }
        Ok(())
    }

    pub fn validate_tokens(&self, ids: &[i32]) -> Result<(), String> {
        self.validate()?;
        if ids.len() < self.original_prompt_len as usize {
            return Err("truncated multimodal prompt".into());
        }
        for span in &self.spans {
            let start = span.token_start as usize;
            let end = start + span.token_len as usize;
            if ids[start - 1] != VISION_START_ID
                || ids[end] != VISION_END_ID
                || ids[start..end].iter().any(|&id| id != IMAGE_TOKEN_ID)
            {
                return Err("image placeholders do not match visual spans".into());
            }
        }
        if ids[..self.original_prompt_len as usize]
            .iter()
            .filter(|&&id| id == IMAGE_TOKEN_ID)
            .count()
            != self
                .spans
                .iter()
                .map(|s| s.token_len as usize)
                .sum::<usize>()
        {
            return Err("unbound image placeholder".into());
        }
        Ok(())
    }

    pub fn positions(&self) -> Result<PromptPositions, String> {
        self.validate()?;
        let mut axes = Vec::with_capacity(self.original_prompt_len as usize);
        let mut next = 0i32;
        for span in &self.spans {
            while axes.len() < span.token_start as usize {
                axes.push([next; 3]);
                next += 1;
            }
            let [_, h, w] = self.images[span.image_index as usize].grid_thw;
            for y in 0..h / 2 {
                for x in 0..w / 2 {
                    axes.push([next, next + y as i32, next + x as i32]);
                }
            }
            next += h.max(w) as i32 / 2;
        }
        while axes.len() < self.original_prompt_len as usize {
            axes.push([next; 3]);
            next += 1;
        }
        Ok(PromptPositions {
            axes,
            rope_delta: next - self.original_prompt_len as i32,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PromptPositions {
    pub axes: Vec<[i32; 3]>,
    pub rope_delta: i32,
}

impl PromptPositions {
    /// The same rule covers decode and generated tokens replayed during prefill.
    pub fn at(&self, physical: usize) -> [i32; 3] {
        self.axes
            .get(physical)
            .copied()
            .unwrap_or([physical as i32 + self.rope_delta; 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positions_keep_spatial_axes_and_generated_tail_across_recompute() {
        let mm = MultimodalInput {
            images: vec![ImageInput {
                grid_thw: [1, 4, 6],
                patches: vec![0; 24 * PATCH_WIDTH * 2],
            }],
            spans: vec![ImageSpan {
                image_index: 0,
                token_start: 2,
                token_len: 6,
            }],
            original_prompt_len: 10,
        };
        let p = mm.positions().unwrap();
        assert_eq!(
            p.axes[2..8],
            [
                [2, 2, 2],
                [2, 2, 3],
                [2, 2, 4],
                [2, 3, 2],
                [2, 3, 3],
                [2, 3, 4]
            ]
        );
        assert_eq!(p.at(8), [5; 3]);
        assert_eq!(p.rope_delta, -3);
        assert_eq!(p.at(10), [7; 3]);
        assert_eq!(p.at(13), [10; 3]);
        let mut invalid = mm.clone();
        invalid.spans[0].token_len -= 1;
        assert!(invalid.validate().is_err());
    }
    #[test]
    fn binary_payload_roundtrip_and_invalid_image_inputs() {
        let image = ImageInput {
            grid_thw: [1, 2, 2],
            patches: vec![0; 4 * PATCH_WIDTH * 2],
        };
        let wire = rmp_serde::to_vec_named(&image).unwrap();
        assert!(wire.len() < image.patches.len() + 100);
        let decoded: ImageInput = rmp_serde::from_slice(&wire).unwrap();
        assert_eq!(decoded.cache_key(), image.cache_key());
        let mut invalid = image.clone();
        invalid.grid_thw = [1, 3, 2];
        assert!(invalid.validate().is_err());
        invalid = image.clone();
        invalid.patches[1] = 0x7f;
        invalid.patches[0] = 0x80;
        assert!(invalid.validate().is_err());
        let mm = MultimodalInput {
            images: vec![image],
            spans: vec![ImageSpan {
                image_index: 0,
                token_start: 1,
                token_len: 1,
            }],
            original_prompt_len: 4,
        };
        assert!(
            mm.validate_tokens(&[VISION_START_ID, IMAGE_TOKEN_ID, VISION_END_ID, 1])
                .is_ok()
        );
        assert!(
            mm.validate_tokens(&[
                VISION_START_ID,
                IMAGE_TOKEN_ID,
                VISION_END_ID,
                IMAGE_TOKEN_ID
            ])
            .is_err()
        );
        assert!(
            mm.validate_tokens(&[1, IMAGE_TOKEN_ID, VISION_END_ID, 1])
                .is_err()
        );
    }
    #[test]
    fn prefill_image_payload_is_sent_only_at_request_start() {
        use crate::scheduler_to_worker_data::*;
        let mm = std::sync::Arc::new(MultimodalInput {
            images: vec![ImageInput {
                grid_thw: [1, 2, 2],
                patches: vec![0; 4 * PATCH_WIDTH * 2],
            }],
            spans: vec![ImageSpan {
                image_index: 0,
                token_start: 1,
                token_len: 1,
            }],
            original_prompt_len: 4,
        });
        let mut cmd = PrefillBatchCmd {
            input_ids: vec![VISION_START_ID, IMAGE_TOKEN_ID],
            q_start_loc: vec![0],
            segments: vec![PrefillSegmentMeta {
                multimodal: Some(mm),
                has_multimodal: true,
                sequence_id: 1,
                block_table: vec![],
                block_size: 1,
                prompt_len: 4,
                segment_start: 0,
                segment_end: 2,
                max_tokens: 8,
                sampling_params: SamplingParams {
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: -1,
                },
                completion: PrefillSegmentCompletion::ContinuePrefill,
                ignore_eos: false,
                prefix_hint: None,
            }],
        };
        cmd.validate(32, 4).unwrap();
        cmd.segments[0].has_multimodal = false;
        assert!(cmd.validate(32, 4).is_err());
        cmd.segments[0].has_multimodal = true;
        let payload = cmd.segments[0].multimodal.take();
        assert!(cmd.validate(32, 4).is_err());
        cmd.segments[0].segment_start = 2;
        cmd.segments[0].segment_end = 4;
        cmd.segments[0].completion = PrefillSegmentCompletion::FinishPrefillAndStartDecode;
        cmd.input_ids = vec![VISION_END_ID, 1];
        cmd.validate(32, 4).unwrap();
        cmd.segments[0].multimodal = payload;
        assert!(cmd.validate(32, 4).is_err());
    }
    #[test]
    fn legacy_inference_request_defaults_to_text() {
        use crate::server_to_scheduler::{DiffusionRequest, InferenceModality, InferenceRequest};
        let legacy = (
            "legacy",
            InferenceModality::Llm,
            vec![1i32, 2],
            8usize,
            0.0f32,
            1.0f32,
            -1i32,
            false,
            0i32,
            Vec::<Vec<i32>>::new(),
            false,
            None::<DiffusionRequest>,
        );
        let bytes = rmp_serde::to_vec(&legacy).unwrap();
        let request: InferenceRequest = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(request.input_ids, vec![1, 2]);
        assert!(request.multimodal.is_none());
    }
}
