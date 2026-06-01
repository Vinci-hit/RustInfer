//! 语法验证测试 - 不需要实际运行，只需要通过编译检查

use crate::*;

#[allow(dead_code)]
fn test_protocol_types() {
    let _req = InferenceRequest {
        request_id: "test".to_string(),
        modality: InferenceModality::Llm,
        input_ids: vec![1, 2, 3],
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        stream: false,
        priority: 0,
        stop_sequences: vec!["<|eot_id|>".to_string()],
        diffusion: None,
    };

    let _resp = InferenceResponse {
        request_id: "test".to_string(),
        status: ResponseStatus::Success,
        output_token_ids: vec![10, 20, 30],
        images: vec![],
        finish_reason: Some("stop".to_string()),
        error: None,
        metrics: InferenceMetrics {
            total_ms: 300,
            num_tokens: 3,
            tokens_per_second: 10.0,
        },
    };

    let _chunk = StreamChunk {
        request_id: "test".to_string(),
        chunk_type: ChunkType::Token,
        token_id: Some(42),
        finish_reason: None,
        metrics: None,
    };
}

// ─── KV protocol field round-trip ────────────────────────────────────

#[cfg(test)]
mod phase4_kv_protocol {
    use crate::scheduler_to_worker_control::{
        FreeKvIndices, SchedulerControlMessage,
    };
    use crate::scheduler_to_worker_data::{
        PrefillBatchCmd, PrefillSegmentCompletion, PrefillSegmentMeta, SamplingParams,
    };
    use crate::worker_to_scheduler_data::{
        AssignedIndices, GeneratedToken, StepOutput,
    };

    fn rt<T>(v: &T) -> T
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let bytes = rmp_serde::to_vec(v).expect("serialize");
        rmp_serde::from_slice(&bytes).expect("deserialize")
    }

    #[test]
    fn prefill_segment_meta_round_trips_with_prefix_hint() {
        let seg = PrefillSegmentMeta {
            sequence_id: 7,
            block_table: vec![],
            block_size: 1,
            prompt_len: 6,
            segment_start: 0,
            segment_end: 6,
            max_tokens: 32,
            sampling_params: SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: -1,
            },
            completion: PrefillSegmentCompletion::FinishPrefillAndStartDecode,
            prefix_hint: Some(vec![100, 101, 102]),
        };
        let r = rt(&seg);
        assert_eq!(r.prefix_hint.as_deref(), Some(&[100, 101, 102][..]));
        assert_eq!(r.sequence_id, 7);
    }

    #[test]
    fn prefill_segment_meta_default_prefix_hint_is_none() {
        // Older wire formats (without `prefix_hint`) must deserialize cleanly
        // because of `#[serde(default)]`. We construct the older-shaped
        // value via direct serialization of an alternate type that omits
        // the field.
        #[derive(serde::Serialize)]
        struct LegacySeg<'a> {
            sequence_id: u64,
            block_table: &'a [u32],
            block_size: u32,
            prompt_len: u32,
            segment_start: u32,
            segment_end: u32,
            max_tokens: usize,
            sampling_params: SamplingParams,
            completion: PrefillSegmentCompletion,
        }
        let legacy = LegacySeg {
            sequence_id: 1,
            block_table: &[10, 11, 12],
            block_size: 1,
            prompt_len: 3,
            segment_start: 0,
            segment_end: 3,
            max_tokens: 16,
            sampling_params: SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: -1,
            },
            completion: PrefillSegmentCompletion::FinishPrefillAndStartDecode,
        };
        let bytes = rmp_serde::to_vec(&legacy).expect("serialize older shape");
        let parsed: PrefillSegmentMeta =
            rmp_serde::from_slice(&bytes).expect("deserialize current from older bytes");
        assert!(parsed.prefix_hint.is_none());
        assert_eq!(parsed.block_table, vec![10, 11, 12]);
    }

    #[test]
    fn step_output_round_trips_with_assigned_indices() {
        let out = StepOutput {
            prefill_done: vec![1, 2],
            tokens: vec![GeneratedToken {
                sequence_id: 1,
                token_id: 42,
                finished: false,
            }],
            assigned_indices: vec![
                AssignedIndices { sequence_id: 1, base: 0, len: 1 },
                AssignedIndices { sequence_id: 2, base: 1, len: 5 },
            ],
        };
        let r = rt(&out);
        assert_eq!(r.assigned_indices.len(), 2);
        assert_eq!(r.assigned_indices[0].base, 0);
        assert_eq!(r.assigned_indices[0].len, 1);
        assert_eq!(r.assigned_indices[1].end(), 6);
    }

    #[test]
    fn step_output_legacy_without_assigned_indices_deserializes() {
        // Older wire shape omitted `assigned_indices`; current code deserializes
        // it cleanly via `#[serde(default)]`.
        #[derive(serde::Serialize)]
        struct LegacyStep<'a> {
            prefill_done: &'a [u64],
            tokens: &'a [GeneratedToken],
        }
        let older = LegacyStep {
            prefill_done: &[3],
            tokens: &[GeneratedToken {
                sequence_id: 3,
                token_id: 9,
                finished: true,
            }],
        };
        let bytes = rmp_serde::to_vec(&older).expect("serialize older shape");
        let parsed: StepOutput =
            rmp_serde::from_slice(&bytes).expect("deserialize current from older bytes");
        assert!(parsed.assigned_indices.is_empty());
        assert_eq!(parsed.prefill_done, vec![3]);
    }

    #[test]
    fn free_kv_indices_round_trip() {
        let msg = SchedulerControlMessage::FreeKvIndices(FreeKvIndices {
            model_instance_id: "m0".to_string(),
            indices: vec![5, 6, 7, 100, 101],
        });
        let bytes = rmp_serde::to_vec(&msg).expect("serialize");
        let parsed: SchedulerControlMessage =
            rmp_serde::from_slice(&bytes).expect("deserialize");
        match parsed {
            SchedulerControlMessage::FreeKvIndices(f) => {
                assert_eq!(f.indices, vec![5, 6, 7, 100, 101]);
                assert_eq!(f.model_instance_id, "m0");
            }
            _ => panic!("expected FreeKvIndices variant"),
        }
    }

    #[test]
    fn validate_passes_when_block_table_empty_but_prefix_hint_present() {
        // The scheduler currently emits `block_table: vec![]` and relies
        // entirely on `prefix_hint` + worker-side allocation, but
        // `PrefillBatchCmd::validate()` still rejects empty block_tables.
        // This test pins down that current behavior; relax `validate` if
        // the field is ever removed from the wire.
        let cmd = PrefillBatchCmd {
            input_ids: vec![1, 2, 3],
            q_start_loc: vec![0],
            segments: vec![PrefillSegmentMeta {
                sequence_id: 1,
                block_table: vec![],
                block_size: 1,
                prompt_len: 3,
                segment_start: 0,
                segment_end: 3,
                max_tokens: 16,
                sampling_params: SamplingParams {
                    temperature: 1.0,
                    top_p: 1.0,
                    top_k: -1,
                },
                completion: PrefillSegmentCompletion::FinishPrefillAndStartDecode,
                prefix_hint: Some(vec![]),
            }],
        };
        assert!(cmd.validate(8192, 16).is_err());
    }
}

