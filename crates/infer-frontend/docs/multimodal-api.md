# Multimodal client contract

The frontend uses the RustInfer HTTP server root (default `http://localhost:8000`). A pasted trailing `/v1` is normalized, so `https://host/infer/v1/` correctly targets `https://host/infer/v1/chat/completions`. Reverse proxy prefixes are preserved. The current server supports streamed text and, when a compatible Qwen3.5 vision processor is loaded, PNG/JPEG image input. Audio, document, video, and realtime types and client adapters are reserved for future backend implementations.

## Capability discovery

`GET /v1/capabilities` returns versioned capabilities. The same object appears as the additive `capabilities` field of each entry in `GET /v1/models`. Select a model's advertised capabilities when available; otherwise use the server response. Older servers without discovery (HTTP 404, 405, or 501) get a conservative text-only client default. Network failures, malformed responses, and other HTTP errors remain errors. Model names alone never enable a modality.

```json
{
  "version": 1,
  "inputs": { "text": true, "image": true, "audio": false, "file": false, "video": false },
  "outputs": { "text": true, "image": false, "audio": false, "file": false, "video": false },
  "transcription": false,
  "speech": false,
  "realtime": false,
  "file_upload": false,
  "greedy_sampling": false,
  "limits": {
    "max_images": 4,
    "max_image_bytes": 10485760,
    "image_mime_types": ["image/png", "image/jpeg"],
    "max_image_dimension": 8192,
    "max_visual_tokens": 1024,
    "max_file_bytes": 0
  }
}
```

The image flag is derived from the actual image processor and is disabled for speculative decoding, which rejects image requests. `greedy_sampling: true` tells clients that the server requires temperature zero (or equivalent greedy top-p/top-k). Every reserved capability is currently false. Capability discovery does not indicate readiness: `/health` checks process liveness, while `/ready` returns HTTP 503 until the model, scheduler, and worker are ready.

Limits apply to the complete request, including previous conversation turns. The image decoder also enforces a 512-token per-image limit, a 1024-token aggregate visual budget, and an aspect ratio at most 200:1. Visual token count depends on resizing and remains server-validated. Data URLs must use `data:image/png;base64,...` or `data:image/jpeg;base64,...`. Remote image URLs, WebP, GIF, and detail values other than `auto` are not supported by the current server. Decoder memory limits can reject an image even when its dimensions and compressed bytes pass client checks.

## Chat messages

Plain text remains wire-compatible with earlier servers:

```json
{ "role": "user", "content": "你好" }
```

Image messages use the server's existing content-part format:

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "Describe this image" },
    { "type": "image_url", "image_url": { "url": "data:image/png;base64,...", "detail": "auto" } }
  ]
}
```

Rust API: `ChatContent::Text(String)` / `ChatContent::Parts(Vec<ContentPart>)`, and `ChatMessage::text(role, text)`. A system prompt is a normal message with role `system`. `ChatRequest` supports `model`, `messages`, `max_tokens`, `temperature`, `top_p`, and stream options. `ApiClient::chat_completion_stream` always requests `stream_options.include_usage: true` so the final usage-only SSE event contains actual token counts.

The SSE parser handles split UTF-8 bytes, CRLF, multiline data, named and unnamed errors, engine `finish_reason: "error"`, and `[DONE]`. The caller retains ownership of the response stream and stops generation by dropping it. A transport EOF without a terminal event must be displayed as interrupted. The current server does not supply per-request performance in usage; UI timing is measured locally and must not be presented as worker telemetry.

## Reserved transports

These methods perform real HTTP requests, but the current backend has **no handlers** for them. UI controls must remain disabled until the corresponding capability is advertised and the required capture/playback transport has been implemented. Do not silently replace an attachment with its filename or drop it from a request.

| Capability | Client method | Route and request | Response |
| --- | --- | --- | --- |
| `transcription` | `transcribe(request, bytes, filename, mime_type)` | `POST /v1/audio/transcriptions`, multipart `file`, `model`, optional `language` / `prompt`, `response_format=json` | `{ "text": "..." }` |
| `speech` | `speech(SpeechRequest)` | `POST /v1/audio/speech`, JSON `model`, `input`, `voice`, optional `response_format` / `speed` | Audio bytes plus response MIME type |
| `file_upload` | `upload_file(bytes, filename, mime_type, purpose)` | `POST /v1/files`, multipart `file` and `purpose` | `{ "id", "filename", "bytes", "purpose" }` |
| `realtime` | `create_realtime_session(RealtimeSessionRequest)` | `POST /v1/realtime/sessions`, JSON `model`, `modalities`, optional `voice` / `instructions` | `{ "id", "transport", "url", "token"?, "expires_at"? }` |

The reserved realtime session response and event enums are a **RustInfer contract proposal**, not a claim of compatibility with a specific provider. A future adapter maps this contract to its backend. `RealtimeClientEvent` reserves audio append/commit, response create/cancel, and session close. `RealtimeServerEvent` reserves transcript/audio deltas, response completion, and errors. Opening a session is separate from attaching a WebSocket or WebRTC connection. Do not persist session tokens or microphone recordings in local storage; acquire microphone permission only in response to a user action, and stop tracks when a voice session ends.

Reserved content parts:

```json
{ "type": "input_audio", "input_audio": { "data": "BASE64_BYTES", "format": "wav" } }
{ "type": "file", "file": { "file_id": "uploaded-id" } }
{ "type": "file", "file": { "filename": "notes.pdf", "file_data": "data:application/pdf;base64,..." } }
{ "type": "video_url", "video_url": { "url": "..." } }
```

The current server rejects these parts. `input_audio.data` contains base64 bytes without a data-URL prefix. A file must use either a server file ID or inline data; future backend validation must enforce this. `video_url` is a reserved RustInfer extension and requires a decoder/transport implementation before enabling it. Advertised `inputs.audio`, `inputs.file`, or `inputs.video` must reflect actual chat support, independently of standalone upload or transcription endpoints.

## Implementation references

The client stays on reqwest 0.12 with `json`, `stream`, and `multipart`, whose browser multipart implementation supports byte-backed file parts. Verify browser support before upgrading HTTP major versions. See the [reqwest 0.12 multipart source](https://docs.rs/reqwest/0.12.28/src/reqwest/wasm/multipart.rs.html) and [request builder documentation](https://docs.rs/reqwest/0.12.28/reqwest/struct.RequestBuilder.html). CORS remains opt-in through explicit server origins; the new discovery route uses the same existing middleware as the chat and model routes.
