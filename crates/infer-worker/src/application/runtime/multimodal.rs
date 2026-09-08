//! Request geometry is durable; encoder outputs use a bounded, evictable cache.
use super::*;
use infer_protocol::multimodal::{IMAGE_TOKEN_ID, MultimodalInput, PromptPositions};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const ENCODER_CACHE_BYTES: usize = 64 * 1024 * 1024;

struct VisualRequest {
    input: Arc<MultimodalInput>,
    positions: PromptPositions,
    keys: Vec<[u8; 32]>,
}

struct EncoderEntry<T: Dtype, D: LlmBackend> {
    embedding: Tensor<T, D>,
    users: HashSet<u64>,
    touched: u64,
}

pub(super) struct EmbeddingOverride<T: Dtype, D: LlmBackend> {
    pub start: usize,
    pub embedding: Tensor<T, D>,
}

pub(super) struct VisualState<T: Dtype, D: LlmBackend> {
    requests: HashMap<u64, VisualRequest>,
    cache: HashMap<[u8; 32], EncoderEntry<T, D>>,
    bytes: usize,
    clock: u64,
    pub overrides: Vec<EmbeddingOverride<T, D>>,
    pub angles: Option<(Tensor<f32, D>, Tensor<f32, D>)>,
    pub encoder_runs: usize,
    pub cache_hits: usize,
}

impl<T: Dtype, D: LlmBackend> Default for VisualState<T, D> {
    fn default() -> Self {
        Self {
            requests: HashMap::new(),
            cache: HashMap::new(),
            bytes: 0,
            clock: 0,
            overrides: Vec::new(),
            angles: None,
            encoder_runs: 0,
            cache_hits: 0,
        }
    }
}

impl<T: Dtype, D: LlmBackend> VisualState<T, D> {
    /// Generated tokens have identical temporal/height/width coordinates, so
    /// decode can use the ordinary RoPE cache with a request-specific position.
    /// Prompt tokens (including a one-token prefill chunk) still need MRoPE.
    fn decode_position(&self, id: u64, physical: usize) -> Option<i32> {
        let state = self.requests.get(&id)?;
        (physical >= state.input.original_prompt_len as usize)
            .then(|| state.positions.at(physical)[0])
    }

    fn requires_prefill(&self, req: &StepRequest) -> bool {
        req.seqs.iter().any(|seq| {
            self.requests.contains_key(&seq.sequence_id)
                && (seq.input_ids.len() != 1
                    || seq.kv_write_start < 0
                    || self
                        .decode_position(seq.sequence_id, seq.kv_write_start as usize)
                        .is_none())
        })
    }

    fn evict_for(&mut self, needed: usize, capacity: usize) -> OpResult<()> {
        if needed > capacity {
            return Err(OpError::Shape(
                "image embedding exceeds encoder cache capacity".into(),
            ));
        }
        while self.bytes > capacity - needed {
            let victim = self
                .cache
                .iter()
                .filter(|(_, e)| e.users.is_empty())
                .min_by_key(|(_, e)| e.touched)
                .map(|(key, _)| *key)
                .ok_or_else(|| {
                    OpError::Shape("encoder cache capacity exhausted by live prefills".into())
                })?;
            let entry = self.cache.remove(&victim).unwrap();
            self.bytes -= entry.embedding.numel() * T::SIZE_BYTES;
        }
        Ok(())
    }
    pub fn release(&mut self, id: u64) {
        self.requests.remove(&id);
        self.unpin(id);
    }
    fn unpin(&mut self, id: u64) {
        for entry in self.cache.values_mut() {
            entry.users.remove(&id);
        }
    }
    pub fn retain(&mut self, live: &HashSet<u64>) {
        let expired: Vec<_> = self
            .requests
            .keys()
            .filter(|id| !live.contains(id))
            .copied()
            .collect();
        for id in expired {
            self.release(id);
        }
    }
}

impl<T: Dtype, D: LlmBackend, M: DecoderModel<T, D>> Runtime<T, D, M> {
    pub fn has_multimodal_sequence(&self, id: u64) -> bool {
        self.visual.requests.contains_key(&id)
    }
    pub fn request_is_multimodal(&self, req: &StepRequest) -> bool {
        req.seqs
            .iter()
            .any(|s| self.has_multimodal_sequence(s.sequence_id))
    }

    pub(super) fn requires_multimodal_prefill(&self, req: &StepRequest) -> bool {
        self.visual.requires_prefill(req)
    }

    pub(super) fn multimodal_decode_position(&self, id: u64, physical: usize) -> Option<i32> {
        self.visual.decode_position(id, physical)
    }
    pub fn visual_cache_stats(&self) -> (usize, usize, usize) {
        (
            self.visual.encoder_runs,
            self.visual.cache_hits,
            self.visual.bytes,
        )
    }

    pub fn encoder_cache_reserve_bytes(&self) -> usize {
        if self.model.multimodal_rope().is_some() {
            ENCODER_CACHE_BYTES
        } else {
            0
        }
    }

    /// Exercise the largest accepted image before sizing the persistent KV pool.
    pub fn profile_vision(&self) -> OpResult<()> {
        if self.model.multimodal_rope().is_none() {
            return Ok(());
        }
        let image = infer_protocol::multimodal::ImageInput {
            grid_thw: [1, 32, 64],
            patches: vec![
                0;
                4 * infer_protocol::multimodal::MAX_IMAGE_TOKENS
                    * infer_protocol::multimodal::PATCH_WIDTH
                    * 2
            ],
        };
        let output = self.model.encode_image(&image, &self.scope)?;
        self.scope.synchronize()?;
        drop(output);
        Ok(())
    }

    pub fn register_multimodal(&mut self, id: u64, input: Arc<MultimodalInput>) -> OpResult<()> {
        if self.model.multimodal_rope().is_none() {
            return Err(OpError::unsupported("model", "image inputs"));
        }
        input.validate().map_err(OpError::Shape)?;
        if input.original_prompt_len as usize > self.max_seq_len {
            return Err(OpError::Shape("image prompt exceeds context".into()));
        }
        let positions = input.positions().map_err(OpError::Shape)?;
        let keys = input.images.iter().map(|image| image.cache_key()).collect();
        self.visual.release(id);
        self.visual.requests.insert(
            id,
            VisualRequest {
                input,
                positions,
                keys,
            },
        );
        Ok(())
    }

    pub(super) fn prepare_text_abc(&mut self, req: &StepRequest) -> OpResult<()> {
        if self.request_is_multimodal(req) {
            return Err(OpError::unsupported(
                "multimodal",
                "ABC execution; use eager step",
            ));
        }
        self.visual.overrides.clear();
        self.visual.angles = None;
        Ok(())
    }

    pub(super) fn prepare_decode_abc(&mut self, req: &StepRequest) -> OpResult<()> {
        if self.requires_multimodal_prefill(req) || !req.draft_tokens.is_empty() {
            return Err(OpError::unsupported(
                "decode ABC",
                "multimodal prefill or speculative decoding; use eager step",
            ));
        }
        self.visual.overrides.clear();
        self.visual.angles = None;
        Ok(())
    }

    pub(super) fn prepare_multimodal(&mut self, req: &StepRequest) -> OpResult<()> {
        self.visual.overrides.clear();
        self.visual.angles = None;
        if !self.request_is_multimodal(req) {
            return Ok(());
        }
        if !req.draft_tokens.is_empty() {
            return Err(OpError::unsupported("multimodal", "speculative decoding"));
        }
        // Once every image request is decoding, build_plan has already put
        // each request's scalar MRoPE position in the persistent RoPE index.
        // This uses the same captured kernels and addresses as text decode.
        if !self.requires_multimodal_prefill(req) {
            return Ok(());
        }
        let (dim, theta, sections) = self
            .model
            .multimodal_rope()
            .ok_or_else(|| OpError::Shape("missing MRoPE config".into()))?;
        let half = dim / 2;
        let tokens = req.seqs.iter().map(|s| s.input_ids.len()).sum::<usize>();
        let mut sin = Vec::with_capacity(tokens * half);
        let mut cos = Vec::with_capacity(tokens * half);
        let mut base = 0usize;
        for seq in &req.seqs {
            let physical = usize::try_from(seq.kv_write_start)
                .map_err(|_| OpError::Shape("negative image token position".into()))?;
            let end = physical + seq.input_ids.len();
            let mut jobs = Vec::new();
            if let Some(state) = self.visual.requests.get(&seq.sequence_id) {
                for span in &state.input.spans {
                    let start = span.token_start as usize;
                    let stop = start + span.token_len as usize;
                    let lo = physical.max(start);
                    let hi = end.min(stop);
                    if lo < hi {
                        if seq.input_ids[lo - physical..hi - physical]
                            .iter()
                            .any(|&id| id != IMAGE_TOKEN_ID)
                        {
                            return Err(OpError::Shape("prefill image token/span mismatch".into()));
                        }
                        jobs.push((
                            state.input.clone(),
                            span.image_index as usize,
                            state.keys[span.image_index as usize],
                            lo,
                            hi,
                            start,
                        ));
                    }
                }
            }
            for (input, index, key, lo, hi, start) in jobs {
                self.visual.clock += 1;
                if self.visual.cache.contains_key(&key) {
                    self.visual.cache_hits += 1;
                } else {
                    let image = &input.images[index];
                    let needed = image.num_tokens() * self.dims.dim * T::SIZE_BYTES;
                    self.visual.evict_for(needed, ENCODER_CACHE_BYTES)?;
                    let embedding = self.model.encode_image(image, &self.scope)?;
                    if embedding.shape().as_slice() != [image.num_tokens(), self.dims.dim] {
                        return Err(OpError::Shape(
                            "vision encoder returned incompatible embedding shape".into(),
                        ));
                    }
                    self.visual.encoder_runs += 1;
                    self.visual.bytes += needed;
                    self.visual.cache.insert(
                        key,
                        EncoderEntry {
                            embedding,
                            users: HashSet::new(),
                            touched: 0,
                        },
                    );
                }
                let entry = self.visual.cache.get_mut(&key).unwrap();
                entry.users.insert(seq.sequence_id);
                entry.touched = self.visual.clock;
                self.visual.overrides.push(EmbeddingOverride {
                    start: base + lo - physical,
                    embedding: entry.embedding.narrow(0, lo - start, hi - lo)?,
                });
                // Once the last row is queued on this stream, future chunks need
                // no encoder rows for this image. Keep the entry as evictable data.
                if end >= start + input.images[index].num_tokens() {
                    entry.users.remove(&seq.sequence_id);
                }
            }
            for p in physical..end {
                let axes = self
                    .visual
                    .requests
                    .get(&seq.sequence_id)
                    .map(|s| s.positions.at(p))
                    .unwrap_or([p as i32; 3]);
                for j in 0..half {
                    let axis = if j % 3 == 1 && j < sections[1] * 3 {
                        1
                    } else if j % 3 == 2 && j < sections[2] * 3 {
                        2
                    } else {
                        0
                    };
                    let inv = 1.0f32 / (theta as f32).powf(2.0 * j as f32 / dim as f32);
                    let angle = axes[axis] as f32 * inv;
                    // HF computes trig in FP32 then casts to decoder activation dtype.
                    sin.push(T::read_f64(&T::write_f64(angle.sin() as f64)) as f32);
                    cos.push(T::read_f64(&T::write_f64(angle.cos() as f64)) as f32);
                }
            }
            base += seq.input_ids.len();
        }
        self.visual.angles = Some((
            Tensor::from_host_slice(&sin, [tokens, half], self.scope.device())?,
            Tensor::from_host_slice(&cos, [tokens, half], self.scope.device())?,
        ));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::cpu::Cpu;

    #[test]
    fn decode_offsets_follow_requests_and_prompt_chunks_stay_eager() {
        use crate::domain::plan::{SeqStep, StopCriteria};
        use infer_protocol::multimodal::{ImageInput, ImageSpan, PATCH_WIDTH};
        let mut state = VisualState::<f32, Cpu>::default();
        for (id, grid) in [(10, [1, 4, 8]), (20, [1, 8, 8])] {
            let image = ImageInput {
                grid_thw: grid,
                patches: vec![0; grid[1] as usize * grid[2] as usize * PATCH_WIDTH * 2],
            };
            let input = Arc::new(MultimodalInput {
                spans: vec![ImageSpan {
                    image_index: 0,
                    token_start: 2,
                    token_len: image.num_tokens() as u32,
                }],
                original_prompt_len: image.num_tokens() as u32 + 4,
                images: vec![image],
            });
            state.requests.insert(
                id,
                VisualRequest {
                    positions: input.positions().unwrap(),
                    input,
                    keys: vec![],
                },
            );
        }
        let row = |id, physical, len| SeqStep {
            sequence_id: id,
            input_ids: vec![1; len],
            positions: (physical..physical + len as i32).collect(),
            kv_write_start: physical,
            kv_len_after: physical + len as i32,
            block_table: vec![],
        };
        let mut req = StepRequest {
            seqs: vec![row(20, 20, 1), row(99, 30, 1), row(10, 12, 1)],
            sampling: vec![],
            stop: StopCriteria {
                eos_ids: vec![],
                generated_counts: vec![],
                max_tokens: vec![],
                ignore_eos: vec![],
            },
            draft_tokens: vec![],
        };
        assert!(!state.requires_prefill(&req));
        let positions = |state: &VisualState<f32, Cpu>, req: &StepRequest| {
            req.seqs
                .iter()
                .map(|s| {
                    state
                        .decode_position(s.sequence_id, s.kv_write_start as usize)
                        .unwrap_or(s.positions[0])
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(positions(&state, &req), [8, 30, 8]);
        req.seqs.swap(0, 2);
        req.seqs[0] = row(10, 13, 1);
        assert_eq!(positions(&state, &req), [9, 30, 8]);
        for (start, len) in [(2, 1), (11, 1), (12, 2)] {
            req.seqs[0] = row(10, start, len);
            assert!(state.requires_prefill(&req));
        }
        state.release(10);
        assert_eq!(state.decode_position(10, 12), None);
        assert!(!state.requires_prefill(&req));
    }

    #[test]
    fn encoder_cache_evicts_old_unused_entries_and_unpins_on_release() {
        let mut state = VisualState::<f32, Cpu>::default();
        for i in 0..3u8 {
            state.cache.insert(
                [i; 32],
                EncoderEntry {
                    embedding: Tensor::zeros([2, 4], &Cpu).unwrap(),
                    users: if i == 0 {
                        HashSet::from([42])
                    } else {
                        HashSet::new()
                    },
                    touched: i as u64,
                },
            );
            state.bytes += 32;
        }
        state.evict_for(32, 96).unwrap();
        assert!(state.cache.contains_key(&[0; 32]));
        assert!(!state.cache.contains_key(&[1; 32]));
        assert!(state.cache.contains_key(&[2; 32]));
        assert_eq!(state.bytes, 64);
        assert!(state.evict_for(97, 96).is_err());
        assert!(state.evict_for(96, 96).is_err());
        state.release(42);
        assert!(state.cache.contains_key(&[0; 32]));
        state.evict_for(96, 96).unwrap();
        assert_eq!(state.bytes, 0);
        assert!(state.cache.is_empty());
    }
}
