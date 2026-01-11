//! 语法验证测试 - 不需要实际运行，只需要通过编译检查

use crate::*;

#[allow(dead_code)]
fn test_protocol_types() {
    let _req = InferenceRequest {
        request_id: "test".to_string(),
        prompt: "hello".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        stop_sequences: vec![],
        stream: false,
        priority: 0,
    };

    let _resp = InferenceResponse {
        request_id: "test".to_string(),
        status: ResponseStatus::Success,
        text: Some("world".to_string()),
        tokens: None,
        num_tokens: 5,
        error: None,
        metrics: InferenceMetrics {
            prefill_ms: 100,
            decode_ms: 200,
            queue_ms: 10,
            batch_size: 1,
            tokens_per_second: 25.0,
            decode_iterations: 5,
        },
    };
}
