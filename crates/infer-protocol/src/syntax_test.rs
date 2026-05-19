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
