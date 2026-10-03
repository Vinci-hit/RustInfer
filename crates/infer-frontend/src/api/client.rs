use super::types::*;
use crate::state::metrics::SystemMetrics;
use anyhow::{Context, Result};
use std::time::Duration;

#[derive(Debug)]
pub enum ChatStreamEvent {
    Chunk(StreamChunk),
    Done,
    Error(String),
}

/// Incremental parser for Server-Sent Events received from chat completions.
///
/// The byte buffer is intentional: a network chunk may split both an SSE line
/// and a multi-byte UTF-8 character. Lines are decoded only after a newline is
/// available, so no response text is lost or replaced at chunk boundaries.
#[derive(Debug, Default)]
pub struct ChatSseParser {
    pending: Vec<u8>,
    event_name: String,
    data_lines: Vec<String>,
}

impl ChatSseParser {
    pub fn push(&mut self, bytes: &[u8]) -> Vec<Result<ChatStreamEvent, String>> {
        self.pending.extend_from_slice(bytes);
        let mut events = Vec::new();

        while let Some(newline) = self.pending.iter().position(|byte| *byte == b'\n') {
            let mut line = self.pending.drain(..=newline).collect::<Vec<_>>();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            self.process_line(line, &mut events);
        }

        events
    }

    /// Flushes a final unterminated line and event when the HTTP body ends.
    pub fn finish(&mut self) -> Vec<Result<ChatStreamEvent, String>> {
        let mut events = Vec::new();
        if !self.pending.is_empty() {
            let mut line = std::mem::take(&mut self.pending);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            self.process_line(line, &mut events);
        }
        if let Some(event) = self.dispatch() {
            events.push(event);
        }
        events
    }

    fn process_line(&mut self, line: Vec<u8>, events: &mut Vec<Result<ChatStreamEvent, String>>) {
        let line = match String::from_utf8(line) {
            Ok(line) => line,
            Err(error) => {
                events.push(Err(format!("SSE stream contained invalid UTF-8: {error}")));
                self.reset_event();
                return;
            }
        };

        if line.is_empty() {
            if let Some(event) = self.dispatch() {
                events.push(event);
            }
            return;
        }

        if line.starts_with(':') {
            return;
        }

        let (field, value) = line
            .split_once(':')
            .map(|(field, value)| (field, value.strip_prefix(' ').unwrap_or(value)))
            .unwrap_or((&line, ""));

        match field {
            "event" => self.event_name = value.to_string(),
            "data" => self.data_lines.push(value.to_string()),
            _ => {}
        }
    }

    fn dispatch(&mut self) -> Option<Result<ChatStreamEvent, String>> {
        if self.data_lines.is_empty() {
            let event_name = std::mem::take(&mut self.event_name);
            return event_name.eq_ignore_ascii_case("error").then(|| {
                Ok(ChatStreamEvent::Error(
                    "server reported a streaming error".to_string(),
                ))
            });
        }

        let event_name = std::mem::take(&mut self.event_name);
        let data = std::mem::take(&mut self.data_lines).join("\n");

        if event_name.eq_ignore_ascii_case("error") {
            return Some(Ok(ChatStreamEvent::Error(error_message(&data))));
        }
        if data.trim() == "[DONE]" {
            return Some(Ok(ChatStreamEvent::Done));
        }

        // Some compatible servers send an error object without a named event.
        if serde_json::from_str::<serde_json::Value>(&data)
            .ok()
            .is_some_and(|value| value.get("error").is_some())
        {
            return Some(Ok(ChatStreamEvent::Error(error_message(&data))));
        }

        Some(
            serde_json::from_str::<StreamChunk>(&data)
                .map(|chunk| {
                    if chunk
                        .choices
                        .iter()
                        .any(|choice| choice.finish_reason.as_deref() == Some("error"))
                    {
                        ChatStreamEvent::Error(
                            "The inference engine could not complete the response.".into(),
                        )
                    } else {
                        ChatStreamEvent::Chunk(chunk)
                    }
                })
                .map_err(|error| format!("invalid chat SSE payload: {error}")),
        )
    }

    fn reset_event(&mut self) {
        self.event_name.clear();
        self.data_lines.clear();
    }
}

fn error_message(data: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
        return data.trim().to_string();
    };

    value
        .get("error")
        .and_then(|error| {
            error
                .as_str()
                .or_else(|| error.get("message").and_then(|message| message.as_str()))
        })
        .or_else(|| value.get("message").and_then(|message| message.as_str()))
        .unwrap_or(data)
        .to_string()
}

#[derive(Clone)]
pub struct ApiClient {
    base_url: String,
    client: reqwest::Client,
}

impl ApiClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: normalize_base_url(base_url),
            client: reqwest::Client::new(),
        }
    }

    pub fn default() -> Self {
        Self::new("http://localhost:8000")
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    pub async fn chat_completion(&self, mut request: ChatRequest) -> Result<ChatResponse> {
        request.stream = false;
        request.stream_options = None;
        let response = self
            .client
            .post(self.endpoint("/v1/chat/completions"))
            .json(&request)
            .send()
            .await
            .context("Unable to reach the inference server")?;
        checked_response(response)
            .await?
            .json()
            .await
            .context("Invalid chat response")
    }

    /// The caller owns and drops the response stream to stop generation.
    pub async fn chat_completion_stream(
        &self,
        mut request: ChatRequest,
    ) -> Result<reqwest::Response> {
        request.stream = true;
        request.stream_options = Some(StreamOptions {
            include_usage: true,
        });
        let response = self
            .client
            .post(self.endpoint("/v1/chat/completions"))
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .json(&request)
            .send()
            .await
            .context("Unable to reach the inference server")?;
        checked_response(response).await
    }

    pub async fn list_models(&self) -> Result<Vec<ModelObject>> {
        let response = self
            .client
            .get(self.endpoint("/v1/models"))
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .context("Unable to load models")?;
        let models: ModelListResponse = checked_response(response)
            .await?
            .json()
            .await
            .context("Invalid model list response")?;
        Ok(models.data)
    }

    /// Older servers remain usable for text; failures other than an absent
    /// discovery endpoint stay visible instead of masquerading as success.
    pub async fn get_capabilities(&self) -> Result<ServerCapabilities> {
        let response = self
            .client
            .get(self.endpoint("/v1/capabilities"))
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .context("Unable to discover server capabilities")?;
        if matches!(response.status().as_u16(), 404 | 405 | 501) {
            return Ok(ServerCapabilities::default());
        }
        checked_response(response)
            .await?
            .json()
            .await
            .context("Invalid server capability response")
    }

    pub async fn get_metrics(&self) -> Result<SystemMetrics> {
        let response = self
            .client
            .get(self.endpoint("/metrics/system"))
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .context("Unable to load system metrics")?;
        checked_response(response)
            .await?
            .json()
            .await
            .context("Invalid system metrics response")
    }

    pub async fn health_check(&self) -> bool {
        self.client
            .get(self.endpoint("/health"))
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
    }

    /// Liveness is not readiness: the server can be up while the model loads.
    pub async fn readiness_check(&self) -> Result<()> {
        let response = self
            .client
            .get(self.endpoint("/ready"))
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .context("Unable to check model readiness")?;
        checked_response(response).await?;
        Ok(())
    }

    /// Reserved adapter: call only when capabilities.transcription is true.
    pub async fn transcribe(
        &self,
        request: TranscriptionRequest,
        bytes: Vec<u8>,
        filename: String,
        mime_type: &str,
    ) -> Result<TranscriptionResponse> {
        let part = reqwest::multipart::Part::bytes(bytes)
            .file_name(filename)
            .mime_str(mime_type)?;
        let mut form = reqwest::multipart::Form::new()
            .text("model", request.model)
            .text("response_format", "json")
            .part("file", part);
        if let Some(language) = request.language {
            form = form.text("language", language);
        }
        if let Some(prompt) = request.prompt {
            form = form.text("prompt", prompt);
        }
        let response = self
            .client
            .post(self.endpoint("/v1/audio/transcriptions"))
            .multipart(form)
            .send()
            .await
            .context("Unable to transcribe audio")?;
        checked_response(response)
            .await?
            .json()
            .await
            .context("Invalid transcription response")
    }

    /// Reserved adapter: call only when capabilities.speech is true.
    pub async fn speech(&self, request: SpeechRequest) -> Result<SpeechResponse> {
        let response = self
            .client
            .post(self.endpoint("/v1/audio/speech"))
            .json(&request)
            .send()
            .await
            .context("Unable to generate speech")?;
        let response = checked_response(response).await?;
        let mime_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();
        let bytes = response
            .bytes()
            .await
            .context("Unable to read generated audio")?
            .to_vec();
        Ok(SpeechResponse { bytes, mime_type })
    }

    /// Reserved adapter: call only when capabilities.file_upload is true.
    pub async fn upload_file(
        &self,
        bytes: Vec<u8>,
        filename: String,
        mime_type: &str,
        purpose: String,
    ) -> Result<UploadedFile> {
        let part = reqwest::multipart::Part::bytes(bytes)
            .file_name(filename)
            .mime_str(mime_type)?;
        let form = reqwest::multipart::Form::new()
            .text("purpose", purpose)
            .part("file", part);
        let response = self
            .client
            .post(self.endpoint("/v1/files"))
            .multipart(form)
            .send()
            .await
            .context("Unable to upload file")?;
        checked_response(response)
            .await?
            .json()
            .await
            .context("Invalid file upload response")
    }

    /// Reserved RustInfer session negotiation. WebRTC/WebSocket transport is
    /// attached by the voice UI once a backend advertises realtime support.
    pub async fn create_realtime_session(
        &self,
        request: RealtimeSessionRequest,
    ) -> Result<RealtimeSession> {
        let response = self
            .client
            .post(self.endpoint("/v1/realtime/sessions"))
            .json(&request)
            .send()
            .await
            .context("Unable to create a realtime session")?;
        checked_response(response)
            .await?
            .json()
            .await
            .context("Invalid realtime session response")
    }
}

fn normalize_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    trimmed.strip_suffix("/v1").unwrap_or(trimmed).to_string()
}

async fn checked_response(response: reqwest::Response) -> Result<reqwest::Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let message = error_message(&body);
    let message: String = message.chars().take(1000).collect();
    if message.trim().is_empty() {
        anyhow::bail!("Server returned HTTP {status}");
    }
    anyhow::bail!("HTTP {status}: {message}")
}

#[cfg(test)]
mod tests {
    use super::{normalize_base_url, ChatSseParser, ChatStreamEvent};

    #[test]
    fn parses_events_split_across_arbitrary_chunks() {
        let mut parser = ChatSseParser::default();
        let payload = "data: {\"choices\":[{\"delta\":{\"content\":\"hello 世界\"},\"finish_reason\":null}],\"usage\":null}\r\n\r\ndata: [DONE]\n\n";
        let first_multibyte = payload.find('世').expect("test payload contains UTF-8");
        let chunks = [
            &payload.as_bytes()[..23],
            &payload.as_bytes()[23..first_multibyte + 1],
            &payload.as_bytes()[first_multibyte + 1..first_multibyte + 2],
            &payload.as_bytes()[first_multibyte + 2..],
        ];

        let events = chunks
            .into_iter()
            .flat_map(|chunk| parser.push(chunk))
            .collect::<Result<Vec<_>, _>>()
            .expect("valid stream");

        assert_eq!(events.len(), 2);
        let ChatStreamEvent::Chunk(chunk) = &events[0] else {
            panic!("expected content chunk");
        };
        assert_eq!(
            chunk.choices[0].delta.content.as_deref(),
            Some("hello 世界")
        );
        assert!(matches!(events[1], ChatStreamEvent::Done));
    }

    #[test]
    fn joins_multiline_data_and_flushes_an_unterminated_event() {
        let mut parser = ChatSseParser::default();
        let input = b"data: {\"choices\":[],\ndata: \"usage\":null}";

        assert!(parser.push(input).is_empty());
        let events = parser
            .finish()
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("valid multiline payload");

        let ChatStreamEvent::Chunk(chunk) = &events[0] else {
            panic!("expected content chunk");
        };
        assert!(chunk.choices.is_empty());
    }

    #[test]
    fn surfaces_named_error_events() {
        let mut parser = ChatSseParser::default();
        let events = parser
            .push(b"event: error\ndata: {\"error\":{\"message\":\"model unavailable\"}}\n\n")
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("valid error event");

        assert!(matches!(
            events.as_slice(),
            [ChatStreamEvent::Error(message)] if message == "model unavailable"
        ));
    }

    #[test]
    fn surfaces_empty_named_error_events() {
        let mut parser = ChatSseParser::default();
        let events = parser
            .push(b"event: error\n\n")
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("valid error event");

        assert!(matches!(
            events.as_slice(),
            [ChatStreamEvent::Error(message)] if message == "server reported a streaming error"
        ));
    }

    #[test]
    fn reports_malformed_json_instead_of_dropping_it() {
        let mut parser = ChatSseParser::default();
        let events = parser.push(b"data: not-json\n\n");

        assert_eq!(events.len(), 1);
        assert!(events[0]
            .as_ref()
            .expect_err("payload must fail")
            .contains("invalid chat SSE payload"));
    }
    #[test]
    fn normalizes_copied_api_roots_without_losing_proxy_paths() {
        assert_eq!(
            normalize_base_url(" http://localhost:8000/v1/ "),
            "http://localhost:8000"
        );
        assert_eq!(
            normalize_base_url("https://host/infer/v1"),
            "https://host/infer"
        );
        assert_eq!(
            normalize_base_url("https://host/infer/"),
            "https://host/infer"
        );
    }

    #[test]
    fn surfaces_unnamed_errors_and_terminal_engine_failures() {
        for input in [
            b"data: {\"error\":{\"message\":\"offline\"}}\n\n".as_slice(),
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"error\"}]}\n\n".as_slice(),
        ] {
            let events = ChatSseParser::default().push(input);
            assert!(matches!(events.as_slice(), [Ok(ChatStreamEvent::Error(_))]));
        }
    }
}
