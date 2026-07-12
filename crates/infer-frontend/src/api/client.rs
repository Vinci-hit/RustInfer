use super::types::*;
use crate::state::metrics::SystemMetrics;
use anyhow::Result;

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

        Some(
            serde_json::from_str(&data)
                .map(ChatStreamEvent::Chunk)
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
            base_url: base_url.to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn default() -> Self {
        Self::new("http://localhost:8000")
    }

    /// 非流式 chat completion
    pub async fn chat_completion(&self, request: ChatRequest) -> Result<ChatResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(response)
    }

    /// 流式 chat completion — 返回 Response 供逐行读取 SSE
    pub async fn chat_completion_stream(&self, request: ChatRequest) -> Result<reqwest::Response> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?;
        Ok(response)
    }

    /// 获取可用模型列表
    pub async fn list_models(&self) -> Result<Vec<ModelObject>> {
        let url = format!("{}/v1/models", self.base_url);
        let resp: ModelListResponse = self
            .client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(resp.data)
    }

    /// 获取系统 metrics
    pub async fn get_metrics(&self) -> Result<SystemMetrics> {
        let url = format!("{}/metrics", self.base_url);
        let response = self.client.get(&url).send().await?.json().await?;
        Ok(response)
    }

    /// 健康检查
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        self.client.get(&url).send().await.is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatSseParser, ChatStreamEvent};

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
}
