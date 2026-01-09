use anyhow::Result;
use infer_core::model::llama3::Llama3;
use crate::config::ServerConfig;
use crate::api::openai::*;
use crate::chat::template::get_template;
use tokio::sync::mpsc;
use axum::response::sse::Event;

pub struct InferenceEngine {
    model: Llama3,
    config: ServerConfig,
}

impl InferenceEngine {
    pub async fn new(config: ServerConfig) -> Result<Self> {
        // Load model synchronously (it's a one-time startup operation)
        let model = Llama3::new(&config.model_path, config.device.clone(), false)?;

        Ok(Self { model, config })
    }

    pub async fn generate(&mut self, req: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        // Apply chat template
        let template = get_template(&req.model);
        let prompt = template.apply(&req.messages)?;

        let max_tokens = req.max_tokens.unwrap_or(self.config.max_tokens);

        // Run inference - CAPTURE timing metrics
        let (text, num_tokens, prefill_ms, decode_ms, decode_iterations) =
            self.model.generate(&prompt, max_tokens, false)?;

        // Calculate tokens/sec
        let total_time_sec = (prefill_ms + decode_ms) as f64 / 1000.0;
        let tokens_per_second = if total_time_sec > 0.0 {
            num_tokens as f64 / total_time_sec
        } else {
            0.0
        };

        // Build response
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();

        Ok(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model: req.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: text,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: num_tokens,
                total_tokens: num_tokens,
                performance: Some(Performance {
                    prefill_ms,
                    decode_ms,
                    decode_iterations,
                    tokens_per_second,
                    time_to_first_token_ms: prefill_ms,
                }),
            },
        })
    }

    pub async fn generate_stream(
        &mut self,
        req: ChatCompletionRequest,
    ) -> Result<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
        // Apply template
        let template = get_template(&req.model);
        let prompt = template.apply(&req.messages)?;

        let max_tokens = req.max_tokens.unwrap_or(self.config.max_tokens);
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = chrono::Utc::now().timestamp();
        let model_name = req.model.clone();

        // Create channel for streaming
        let (tx, rx) = mpsc::unbounded_channel();

        // Generate (blocking call, but behind Mutex so it's fine)
        // TODO: Implement token-by-token streaming in infer-core
        // For now, simulate with chunked output
        let result = self.model.generate(&prompt, max_tokens, false);

        match result {
            Ok((text, num_tokens, prefill_ms, decode_ms, decode_iterations)) => {
                // Calculate performance
                let total_time_sec = (prefill_ms + decode_ms) as f64 / 1000.0;
                let tokens_per_second = if total_time_sec > 0.0 {
                    num_tokens as f64 / total_time_sec
                } else {
                    0.0
                };

                // Send chunks
                for chunk in text.split_whitespace() {
                    let delta = StreamChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_name.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(format!("{} ", chunk)),
                            },
                            finish_reason: None,
                        }],
                        usage: None,  // No usage in intermediate chunks
                    };

                    if let Ok(json) = serde_json::to_string(&delta) {
                        let _ = tx.send(Ok(Event::default().data(json)));
                    }
                }

                // Send final chunk with metrics
                let final_chunk = StreamChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_name.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Some(Usage {
                        prompt_tokens: 0,
                        completion_tokens: num_tokens,
                        total_tokens: num_tokens,
                        performance: Some(Performance {
                            prefill_ms,
                            decode_ms,
                            decode_iterations,
                            tokens_per_second,
                            time_to_first_token_ms: prefill_ms,
                        }),
                    }),
                };

                if let Ok(json) = serde_json::to_string(&final_chunk) {
                    let _ = tx.send(Ok(Event::default().data(json)));
                }

                // Send done
                let _ = tx.send(Ok(Event::default().data("[DONE]")));
            }
            Err(e) => {
                tracing::error!("Generation failed: {:?}", e);
            }
        }

        Ok(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
    }
}
