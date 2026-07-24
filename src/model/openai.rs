//! `OpenAiModel`: OpenAI-compatible `/chat/completions` client.
//!
//! Works with Ollama (`http://localhost:11434/v1`), OpenAI, and all compatible
//! providers. Request building (`build_payload`) and response parsing
//! (`parse_response`) are pure/testable; the actual HTTP call is inside
//! `complete`.

use super::{Completion, Model, Prompt, Role, TokenStream};
use crate::error::{LoreError, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default request timeout — generous for slow local models (Ollama),
/// but prevents a stuck connection from locking the agent indefinitely.
/// For one-shot requests, TOTAL duration; for streaming, max INTER-CHUNK
/// idle silence — a slow but progressing stream is never cut off because of
/// this.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Max connection setup time (does not cover body reads).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

fn make_client() -> reqwest::Client {
    reqwest::Client::builder()
        // No total request timeout: in streaming, this would cut off a long but
        // healthy response. The timeout layer is on the caller side:
        // one-shot = total, stream = per-chunk idle (tokio::time::timeout).
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .expect("reqwest client must be buildable")
}

/// An OpenAI-compatible chat model client.
pub struct OpenAiModel {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    temperature: f32,
    max_tokens: Option<u32>,
    /// For one-shot requests, total timeout; for streaming, per-chunk idle timeout.
    timeout: Duration,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: String,
    /// Reasoning area of reasoning models (GLM-4.6, o-series alike).
    /// Never written to the request body; falls back to this when content is
    /// empty in the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

impl ChatMessage {
    fn new(role: &str, content: String) -> Self {
        Self {
            role: role.into(),
            content,
            reasoning_content: None,
        }
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChatMessage,
}

/// Strips `<think>...</think>` blocks that reasoning models (qwen3,
/// deepseek-r1, etc.) leak into the response. Some templates skip the opening
/// `<think>` tag and start directly with the thought, closing with `</think>` —
/// therefore the text AFTER the last `</think>` is the real response. Complete
/// pairs are also cleaned. If there is no `</think>`, the text is returned
/// as-is (models without think, like glm, are not affected).
fn strip_think(text: &str) -> String {
    let no_pairs = strip_think_pairs(text);
    let tail = match no_pairs.rfind("</think>") {
        Some(i) => &no_pairs[i + "</think>".len()..],
        None => no_pairs.as_str(),
    };
    tail.trim().to_string()
}

/// Removes complete `<think>...</think>` pairs from the text. An unclosed
/// `<think>` is left as-is — real content is not accidentally deleted.
fn strip_think_pairs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        match rest[start..].find("</think>") {
            Some(end_rel) => {
                out.push_str(&rest[..start]);
                rest = &rest[start + end_rel + "</think>".len()..];
            }
            None => break,
        }
    }
    out.push_str(rest);
    out
}

impl OpenAiModel {
    /// Client with a specific base URL and model name (temperature = 0.7).
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: make_client(),
            base_url: base_url.into(),
            api_key: None,
            model: model.into(),
            temperature: 0.7,
            max_tokens: None,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Ollama shortcut (`http://localhost:11434/v1`).
    pub fn ollama(model: impl Into<String>) -> Self {
        Self::new("http://localhost:11434/v1", model)
    }

    /// Adds API key (builder).
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets temperature (builder).
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Sets timeout (builder; default 120 s).
    /// For one-shot requests, total duration; for streaming, per-chunk idle
    /// duration.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Max response tokens (builder). If not set, it is not written to the
    /// payload — the provider default applies. Note that low values on
    /// reasoning models may spend the entire budget on thinking
    /// (see `parse_response`).
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Converts the prompt into chat messages: system (identity + context) +
    /// conversation history (with actual user/assistant roles) + user (input).
    fn build_messages(&self, prompt: &Prompt) -> Vec<ChatMessage> {
        let mut system = prompt.system.clone();
        if !prompt.context.is_empty() {
            system.push_str("\n\nWhat you recall:\n");
            for c in &prompt.context {
                system.push_str("- ");
                system.push_str(c);
                system.push('\n');
            }
        }

        let mut msgs = Vec::new();
        if !system.trim().is_empty() {
            msgs.push(ChatMessage::new("system", system));
        }
        for t in &prompt.history {
            let role = match t.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            msgs.push(ChatMessage::new(role, t.text.clone()));
        }
        msgs.push(ChatMessage::new("user", prompt.user.clone()));
        msgs
    }

    /// Builds the request body (JSON).
    fn build_payload(&self, prompt: &Prompt, stream: bool) -> serde_json::Value {
        let mut v = serde_json::json!({
            "model": self.model,
            "messages": self.build_messages(prompt),
            "temperature": self.temperature,
            "stream": stream,
        });
        if let Some(mt) = self.max_tokens {
            v["max_tokens"] = mt.into();
        }
        v
    }

    /// Builds an authorized POST request.
    fn request(&self, payload: &serde_json::Value) -> reqwest::RequestBuilder {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let mut req = self.client.post(url).json(payload);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        req
    }

    /// Extracts text from the response body. Reasoning models (e.g. GLM-4.6)
    /// may spend the entire token budget on thinking and leave `content` empty;
    /// in this case, falls back to `reasoning_content` — better than an empty
    /// response. If `content` is present, the thinking text is never leaked.
    fn parse_response(body: &str) -> Result<Completion> {
        let resp: ChatResponse = serde_json::from_str(body)?;
        let msg = resp
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or_else(|| LoreError::Model("empty 'choices' in response".into()))?;
        // Strip <think> blocks that reasoning models leak into content.
        let content = strip_think(&msg.content);
        if content.trim().is_empty() {
            if let Some(r) = msg.reasoning_content {
                let r = strip_think(&r);
                if !r.trim().is_empty() {
                    // Chain-of-thought is serving as the response — flag it so
                    // the storing side can trim it (CoT should not pollute
                    // memory).
                    return Ok(Completion::reasoning_fallback(r));
                }
            }
            // Both empty: valid but content-less response — staying silent
            // makes diagnosis harder (e.g. max_tokens too low on a reasoning
            // model).
            tracing::warn!("model returned empty response (content and reasoning_content empty)");
        }
        Ok(Completion::new(content))
    }
}

/// An SSE stream event.
enum SseEvent {
    /// New text chunk.
    Token(String),
    /// Stream finished (`data: [DONE]`).
    Done,
}

/// Pulls the next complete line from the buffer (`\n` boundary). UTF-8 safe:
/// conversion happens after the line is complete (chunk boundaries do not split
/// characters).
fn next_line(buf: &mut Vec<u8>) -> Option<String> {
    let pos = buf.iter().position(|&b| b == b'\n')?;
    let line: Vec<u8> = buf.drain(..=pos).collect();
    Some(String::from_utf8_lossy(&line).trim().to_string())
}

/// Parses an SSE line: `data: {json}` → delta content; `data: [DONE]` →
/// done; other lines (comment, blank, role/finish chunk) → `None` (skipped).
/// Falls back to `reasoning_content` when `content` is absent/empty (parity
/// with `complete()` — reasoning models may stream only reasoning deltas).
/// Parse a single SSE line from the OpenAI-compatible streaming format.
///
/// Returns `None` for:
/// - keepalive lines (`: ping`, empty lines)
/// - `role` deltas (start-of-message metadata)
/// - `reasoning_content` with empty content (filtered by ThinkFilter)
fn parse_sse_line(line: &str) -> Option<SseEvent> {
    let payload = line.strip_prefix("data:")?.trim();
    if payload == "[DONE]" {
        return Some(SseEvent::Done);
    }
    let v: serde_json::Value = serde_json::from_str(payload).ok()?;
    let delta = &v["choices"][0]["delta"];
    // Prefer content; fall back to reasoning_content (reasoning model parity).
    let tok = delta["content"]
        .as_str()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            delta["reasoning_content"]
                .as_str()
                .filter(|s| !s.is_empty())
        })?;
    Some(SseEvent::Token(tok.to_string()))
}

/// Streaming, line-boundary-safe `<think>` filter. Strips the thinking that
/// reasoning models inject into delta content; also catches tags split at
/// token boundaries (`</thi` + `nk>`) via buffer carry.
///
/// - **Standard** (`<think>…</think>` in content): block is fully suppressed.
/// - **Orphan** `</think>` (no opening tag, "nothink" templates): at least
///   the marker is not leaked (preceding text may already have been sent due
///   to streaming nature — the non-streaming path cleans completely).
/// - **think-less** models (glm keeps reasoning in a separate
///   `reasoning_content` field; plain models produce no tags): content flows
///   without delay.
struct ThinkFilter {
    /// Whether `<think>` is open (content is discarded until closing).
    in_think: bool,
    /// Trim leading whitespace after thinking / at start.
    trim_leading: bool,
    /// Partial buffer carried over for split tags.
    pending: String,
}

impl ThinkFilter {
    fn new() -> Self {
        Self {
            in_think: false,
            trim_leading: true,
            pending: String::new(),
        }
    }

    /// Processes a raw delta token; returns text to emit (`None` if
    /// suppressed).
    fn push(&mut self, tok: &str) -> Option<String> {
        const OPEN: &str = "<think>";
        const CLOSE: &str = "</think>";
        self.pending.push_str(tok);
        let mut out = String::new();
        loop {
            if self.in_think {
                if let Some(i) = self.pending.find(CLOSE) {
                    self.pending.drain(..i + CLOSE.len());
                    self.in_think = false;
                    self.trim_leading = true;
                    continue;
                }
                // Still inside thinking: keep a possibly split </think> suffix,
                // discard the rest.
                let keep = partial_suffix_len(&self.pending, CLOSE);
                let cut = self.pending.len() - keep;
                self.pending.drain(..cut);
                break;
            }
            // Outside thinking: earliest <think> (opening) or orphan </think>.
            let open_i = self.pending.find(OPEN);
            let close_i = self.pending.find(CLOSE);
            let next = match (open_i, close_i) {
                (Some(o), Some(c)) => Some((o.min(c), o <= c)),
                (Some(o), None) => Some((o, true)),
                (None, Some(c)) => Some((c, false)),
                (None, None) => None,
            };
            match next {
                Some((idx, is_open)) => {
                    out.push_str(&self.pending[..idx]);
                    let taglen = if is_open { OPEN.len() } else { CLOSE.len() };
                    self.pending.drain(..idx + taglen);
                    if is_open {
                        self.in_think = true;
                    } else {
                        self.trim_leading = true; // orphan </think> — only drop the marker
                    }
                    continue;
                }
                None => {
                    // No full tag: keep the possible partial tag suffix, emit
                    // the rest.
                    let keep = partial_suffix_len(&self.pending, OPEN)
                        .max(partial_suffix_len(&self.pending, CLOSE));
                    let cut = self.pending.len() - keep;
                    out.push_str(&self.pending[..cut]);
                    self.pending.drain(..cut);
                    break;
                }
            }
        }
        if self.trim_leading {
            let t = out.trim_start();
            if !t.is_empty() {
                self.trim_leading = false;
            }
            out = t.to_string();
        }
        (!out.is_empty()).then_some(out)
    }

    /// Flushes remaining buffer when the stream ends (unclosed thinking is
    /// discarded).
    fn finish(&mut self) -> Option<String> {
        if self.in_think {
            self.pending.clear();
            return None;
        }
        let out = std::mem::take(&mut self.pending);
        let out = if self.trim_leading {
            out.trim_start().to_string()
        } else {
            out
        };
        (!out.is_empty()).then_some(out)
    }
}

/// Length of the longest suffix of `s` that is also a true prefix of `needle`.
/// Used to keep in buffer when a tag (`<think>`/`</think>`) is split at a
/// token boundary.
fn partial_suffix_len(s: &str, needle: &str) -> usize {
    let max = needle.len().min(s.len());
    for k in (1..=max).rev() {
        let start = s.len() - k;
        // needle is ASCII; we must cut at a valid UTF-8 boundary in s.
        if s.is_char_boundary(start) && s.as_bytes()[start..] == needle.as_bytes()[..k] {
            return k;
        }
    }
    0
}

#[async_trait]
impl Model for OpenAiModel {
    async fn complete(&self, prompt: &Prompt) -> Result<Completion> {
        // One-shot: timeout is the total duration (no progress signal).
        let work = async {
            let resp = self
                .request(&self.build_payload(prompt, false))
                .send()
                .await?;
            let status = resp.status();
            let body = resp.text().await?;
            if !status.is_success() {
                return Err(LoreError::Model(format!("{status}: {body}")));
            }
            let completion = Self::parse_response(&body)?;
            Ok(completion)
        };
        match tokio::time::timeout(self.timeout, work).await {
            Ok(r) => r,
            Err(_) => Err(LoreError::Model(format!(
                "request timeout ({}s)",
                self.timeout.as_secs()
            ))),
        }
    }

    /// Real token stream: SSE body is parsed line-by-line with `stream:true`,
    /// each delta chunk is emitted as it arrives. Timeout is per-chunk IDLE:
    /// slow but progressing streams (long-thinking reasoning models) are not
    /// cut off; stalled streams error out without waiting for total duration.
    async fn complete_stream(&self, prompt: &Prompt) -> Result<TokenStream> {
        // Initial bytes (headers) must also arrive within the idle window.
        let resp = match tokio::time::timeout(
            self.timeout,
            self.request(&self.build_payload(prompt, true)).send(),
        )
        .await
        {
            Ok(r) => r?,
            Err(_) => {
                return Err(LoreError::Model(format!(
                    "stream start timeout ({}s)",
                    self.timeout.as_secs()
                )))
            }
        };
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LoreError::Model(format!("{status}: {body}")));
        }

        let idle = self.timeout;
        let stream = futures::stream::unfold(
            (
                resp.bytes_stream(),
                Vec::<u8>::new(),
                ThinkFilter::new(),
                false,
            ),
            move |(mut body, mut buf, mut filter, ended)| async move {
                if ended {
                    return None;
                }
                loop {
                    // If there are complete lines in the buffer, consume them
                    // first.
                    while let Some(line) = next_line(&mut buf) {
                        match parse_sse_line(&line) {
                            // Stream done: flush remaining (post-think) content
                            // from the filter.
                            Some(SseEvent::Done) => {
                                return filter
                                    .finish()
                                    .map(|out| (Ok(out), (body, buf, filter, true)));
                            }
                            // Pass raw token through <think> filter — if
                            // suppressed, continue.
                            Some(SseEvent::Token(t)) => {
                                if let Some(out) = filter.push(&t) {
                                    return Some((Ok(out), (body, buf, filter, false)));
                                }
                            }
                            None => {}
                        }
                    }
                    // Wait for next byte chunk — with idle timeout: stalled
                    // streams error out without waiting for total duration.
                    match tokio::time::timeout(idle, body.next()).await {
                        Ok(Some(Ok(chunk))) => buf.extend_from_slice(&chunk),
                        Ok(Some(Err(e))) => {
                            return Some((Err(LoreError::Http(e)), (body, buf, filter, true)))
                        }
                        Err(_) => {
                            return Some((
                                Err(LoreError::Model(format!(
                                    "stream stalled: {}s without chunk",
                                    idle.as_secs()
                                ))),
                                (body, buf, filter, true),
                            ))
                        }
                        // Premature close: body ended without [DONE].
                        // Intentionally discards any buffered ThinkFilter content —
                        // partial output from a broken stream should not be treated
                        // as valid. Only the error is emitted; no `finish()` flush.
                        Ok(None) => {
                            return Some((
                                Err(LoreError::Model(
                                    "stream ended without terminal event".into(),
                                )),
                                (body, buf, filter, true),
                            ));
                        }
                    }
                }
            },
        );
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_has_model_temperature_and_two_messages() {
        let m = OpenAiModel::ollama("llama3.2").with_temperature(0.3);
        let prompt = Prompt {
            system: "You are Aria.".into(),
            context: vec!["[fact] Rust is fast".into()],
            user: "hello".into(),
            ..Default::default()
        };
        let v = m.build_payload(&prompt, false);

        assert_eq!(v["model"], "llama3.2");
        assert_eq!(v["stream"], false);
        assert!((v["temperature"].as_f64().unwrap() - 0.3).abs() < 1e-6);

        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        let sys = msgs[0]["content"].as_str().unwrap();
        assert!(sys.contains("You are Aria."));
        assert!(sys.contains("Rust is fast")); // context embedded in system
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "hello");
    }

    #[test]
    fn history_becomes_role_tagged_messages() {
        use super::super::Turn;
        let m = OpenAiModel::ollama("llama3.2");
        let prompt = Prompt {
            system: "You are Aria.".into(),
            history: vec![Turn::user("hello"), Turn::assistant("hello to you too")],
            user: "what did I say?".into(),
            ..Default::default()
        };
        let msgs = m.build_messages(&prompt);
        // system + 2 history turns + current user = 4 messages, correct order
        // + roles.
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].content, "hello");
        assert_eq!(msgs[2].role, "assistant");
        assert_eq!(msgs[2].content, "hello to you too");
        assert_eq!(msgs[3].role, "user");
        assert_eq!(msgs[3].content, "what did I say?");
    }

    #[test]
    fn parse_extracts_first_choice_content() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"hello I am Aria"}}]}"#;
        let c = OpenAiModel::parse_response(body).unwrap();
        assert_eq!(c.text, "hello I am Aria");
        assert!(!c.reasoning_fallback);
    }

    #[test]
    fn parse_falls_back_to_reasoning_content() {
        // Reasoning models like GLM-4.6 may spend the entire budget on
        // thinking and leave content empty — must fall back to
        // reasoning_content (§5.2).
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"","reasoning_content":"Step by step: 2+2=4"}}]}"#;
        let c = OpenAiModel::parse_response(body).unwrap();
        assert_eq!(c.text, "Step by step: 2+2=4");
        assert!(
            c.reasoning_fallback,
            "fallback flagged — storer should truncate"
        );
        // If content is present, reasoning is ignored (thinking is not leaked).
        let body2 = r#"{"choices":[{"message":{"role":"assistant","content":"4","reasoning_content":"thinking"}}]}"#;
        let c2 = OpenAiModel::parse_response(body2).unwrap();
        assert_eq!(c2.text, "4");
        assert!(!c2.reasoning_fallback);
    }

    #[test]
    fn payload_max_tokens_only_when_set() {
        let m = OpenAiModel::ollama("llama3.2");
        let p = Prompt {
            user: "hello".into(),
            ..Default::default()
        };
        // Default: max_tokens is not sent (provider default applies).
        assert!(m.build_payload(&p, false).get("max_tokens").is_none());
        // When set via builder, it appears in the payload.
        let m = m.with_max_tokens(256);
        assert_eq!(m.build_payload(&p, false)["max_tokens"], 256);
    }

    #[test]
    fn parse_errors_on_empty_choices() {
        assert!(OpenAiModel::parse_response(r#"{"choices":[]}"#).is_err());
    }

    #[test]
    fn sse_line_parsing_variants() {
        // Valid delta.
        let t = parse_sse_line(r#"data: {"choices":[{"delta":{"content":"hello"}}]}"#);
        assert!(matches!(t, Some(SseEvent::Token(s)) if s == "hello"));
        // Done signal.
        assert!(matches!(
            parse_sse_line("data: [DONE]"),
            Some(SseEvent::Done)
        ));
        // Role chunk (no content) → skip.
        assert!(parse_sse_line(r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#).is_none());
        // Comment / blank line → skip.
        assert!(parse_sse_line(": ping").is_none());
        assert!(parse_sse_line("").is_none());
    }

    #[test]
    fn reasoning_content_stream_parity() {
        // Fix 5: reasoning_content delta is used when content is empty/absent
        // (parity with complete() reasoning_content fallback).
        let rc = r#"data: {"choices":[{"delta":{"reasoning_content":"step1"}}]}"#;
        assert!(matches!(parse_sse_line(rc), Some(SseEvent::Token(s)) if s == "step1"));
        // Content takes priority over reasoning_content.
        let both =
            r#"data: {"choices":[{"delta":{"content":"answer","reasoning_content":"thought"}}]}"#;
        assert!(matches!(parse_sse_line(both), Some(SseEvent::Token(s)) if s == "answer"));
        // Empty content falls back to reasoning_content.
        let empty = r#"data: {"choices":[{"delta":{"content":"","reasoning_content":"reason"}}]}"#;
        assert!(matches!(parse_sse_line(empty), Some(SseEvent::Token(s)) if s == "reason"));
    }

    #[test]
    fn next_line_handles_partial_and_multibyte() {
        let mut buf = b"data: a\ndata: ".to_vec();
        assert_eq!(next_line(&mut buf).as_deref(), Some("data: a"));
        assert!(next_line(&mut buf).is_none(), "partial line should wait");
        // Even if a multi-byte character is split at a chunk boundary, the
        // result is correct once the line is complete.
        let g = "sun".as_bytes();
        buf.extend_from_slice(&g[..3]);
        assert!(next_line(&mut buf).is_none());
        buf.extend_from_slice(&g[3..]);
        buf.push(b'\n');
        assert_eq!(next_line(&mut buf).as_deref(), Some("data: sun"));
    }

    #[tokio::test]
    async fn streams_tokens_from_sse_endpoint() {
        use axum::{routing::post, Router};

        // Fake OpenAI-compatible SSE server.
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                (
                    [("content-type", "text/event-stream")],
                    "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n\
                     data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n\
                     data: {\"choices\":[{\"delta\":{\"content\":\"lo!\"}}]}\n\n\
                     data: [DONE]\n\n",
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let m = OpenAiModel::new(format!("http://{addr}/v1"), "test-model");
        let prompt = Prompt {
            user: "hello".into(),
            ..Default::default()
        };
        let mut s = m.complete_stream(&prompt).await.unwrap();
        let mut toks = Vec::new();
        while let Some(r) = s.next().await {
            toks.push(r.unwrap());
        }
        assert_eq!(toks, vec!["Hel".to_string(), "lo!".to_string()]);
    }

    #[tokio::test]
    async fn slow_but_progressing_stream_completes() {
        use axum::{routing::post, Router};

        // Server sending a chunk every 60ms: total duration ~300ms but
        // inter-chunk silence never exceeds 100ms. With correct idle-timeout
        // semantics the stream completes; a total-timeout would kill it at
        // 100ms.
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                axum::body::Body::from_stream(futures::stream::unfold(0, |i| async move {
                    if i >= 5 {
                        return Some((
                            Ok::<_, std::convert::Infallible>("data: [DONE]\n\n".to_string()),
                            i + 1,
                        ));
                    }
                    if i > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
                    }
                    Some((
                        Ok(format!(
                            "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"p{i}\"}}}}]}}\n\n"
                        )),
                        i + 1,
                    ))
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let m = OpenAiModel::new(format!("http://{addr}/v1"), "test-model")
            .with_timeout(std::time::Duration::from_millis(100));
        let prompt = Prompt {
            user: "hello".into(),
            ..Default::default()
        };
        let mut s = m.complete_stream(&prompt).await.unwrap();
        let mut toks = Vec::new();
        while let Some(r) = s.next().await {
            toks.push(r.unwrap());
        }
        assert_eq!(
            toks.len(),
            5,
            "slow but progressing stream should complete: {toks:?}"
        );
    }

    #[tokio::test]
    async fn stream_errors_on_stalled_chunks() {
        use axum::{routing::post, Router};

        // Server that sends one chunk then goes SILENT forever (stalled model).
        // Without per-chunk idle-timeout instead of total-timeout, the stream
        // would hang until the client's total timeout (default 120s).
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                axum::body::Body::from_stream(futures::stream::unfold(false, |sent| async move {
                    if sent {
                        // No more chunks will ever arrive.
                        futures::future::pending::<()>().await;
                        None
                    } else {
                        Some((
                            Ok::<_, std::convert::Infallible>(
                                "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
                            ),
                            true,
                        ))
                    }
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let m = OpenAiModel::new(format!("http://{addr}/v1"), "test-model")
            .with_timeout(std::time::Duration::from_millis(100));
        let prompt = Prompt {
            user: "hello".into(),
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let mut s = m.complete_stream(&prompt).await.unwrap();
        let mut toks = Vec::new();
        let mut errs = 0;
        while let Some(r) = s.next().await {
            match r {
                Ok(t) => toks.push(t),
                Err(_) => errs += 1,
            }
        }
        assert_eq!(toks, vec!["Hel".to_string()], "first chunk streamed");
        assert_eq!(errs, 1, "stall propagated as error");
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(5),
            "idle-timeout fast termination: {:?}",
            t0.elapsed()
        );
    }

    #[tokio::test]
    async fn stream_premature_close_errors() {
        // Fix 4: body ends without [DONE] → error, not silent partial.
        use axum::{routing::post, Router};

        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                (
                    [("content-type", "text/event-stream")],
                    "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let m = OpenAiModel::new(format!("http://{addr}/v1"), "test-model");
        let prompt = Prompt {
            user: "hello".into(),
            ..Default::default()
        };
        let mut s = m.complete_stream(&prompt).await.unwrap();
        let mut toks = Vec::new();
        let mut errs = Vec::new();
        while let Some(r) = s.next().await {
            match r {
                Ok(t) => toks.push(t),
                Err(e) => errs.push(e.to_string()),
            }
        }
        assert_eq!(toks, vec!["Hel".to_string()], "partial token streamed");
        assert_eq!(errs.len(), 1, "premature close produces an error");
        assert!(
            errs[0].contains("terminal event"),
            "error message: {}",
            errs[0]
        );
    }

    #[tokio::test]
    async fn stream_premature_close_before_any_token() {
        // Edge test: server sends zero deltas then closes — error, no tokens.
        use axum::{routing::post, Router};

        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async { ([("content-type", "text/event-stream")], "") }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let m = OpenAiModel::new(format!("http://{addr}/v1"), "test-model");
        let prompt = Prompt {
            user: "hello".into(),
            ..Default::default()
        };
        let mut s = m.complete_stream(&prompt).await.unwrap();
        let mut toks = Vec::new();
        let mut errs = Vec::new();
        while let Some(r) = s.next().await {
            match r {
                Ok(t) => toks.push(t),
                Err(e) => errs.push(e.to_string()),
            }
        }
        assert!(toks.is_empty(), "no tokens from empty body: {toks:?}");
        assert_eq!(errs.len(), 1, "empty body produces an error");
        assert!(errs[0].contains("terminal event"), "error: {}", errs[0]);
    }

    #[test]
    fn strip_think_removes_well_formed_block() {
        assert_eq!(
            strip_think("<think>reasoning</think>\n\nAnswer: 4"),
            "Answer: 4"
        );
    }

    #[test]
    fn strip_think_handles_missing_opening_tag() {
        // qwen3-"nothink": no opening tag, closes directly with </think>.
        assert_eq!(strip_think("\n</think>\n\n35"), "35");
        assert_eq!(strip_think("short thought </think>\n\n4"), "4");
    }

    #[test]
    fn strip_think_passthrough_without_think() {
        // Models without think (like glm) are not affected.
        assert_eq!(strip_think("4"), "4");
        assert_eq!(strip_think("Hello, how are you?"), "Hello, how are you?");
    }

    #[test]
    fn strip_think_removes_multiple_pairs() {
        assert_eq!(strip_think("<think>a</think>X<think>b</think>Y"), "XY");
    }

    #[test]
    fn strip_think_keeps_unclosed_tag() {
        // Unclosed <think> does not swallow real content (only complete pairs
        // are removed).
        assert_eq!(
            strip_think("normal <think> content"),
            "normal <think> content"
        );
    }

    #[test]
    fn parse_response_strips_leaked_think() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"<think>calculating</think>\n\n42"}}]}"#;
        assert_eq!(OpenAiModel::parse_response(body).unwrap().text, "42");
    }

    fn run_filter(tokens: &[&str]) -> String {
        let mut f = ThinkFilter::new();
        let mut out = String::new();
        for t in tokens {
            if let Some(s) = f.push(t) {
                out.push_str(&s);
            }
        }
        if let Some(s) = f.finish() {
            out.push_str(&s);
        }
        out
    }

    #[test]
    fn think_filter_strips_wellformed_across_split_tokens() {
        // Tags split at token boundary (`<thi`+`nk>`) — carry buffer catches
        // them.
        let out = run_filter(&["<thi", "nk>re", "ason", "</thi", "nk>", "\n\nAns", "wer"]);
        assert_eq!(out, "Answer");
    }

    #[test]
    fn think_filter_scrubs_bare_close_marker() {
        // No opening tag: raw </think> marker must not leak into output.
        let out = run_filter(&["short", " thought", "</think>", "\n\n42"]);
        assert!(!out.contains("think"), "marker leaked: {out:?}");
        assert!(out.ends_with("42"), "response streamed: {out:?}");
    }

    #[test]
    fn think_filter_passthrough_plain_text() {
        // Plain/glm content (no tags) flows without delay or corruption.
        assert_eq!(run_filter(&["Hel", "lo, ", "world"]), "Hello, world");
        // A stray '<' in the middle is not held up.
        assert_eq!(run_filter(&["a < b", " = c"]), "a < b = c");
    }

    #[tokio::test]
    async fn stream_filters_think_tokens() {
        use axum::{routing::post, Router};

        // <think> block that a reasoning model mixes into deltas — token by
        // token.
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                (
                    [("content-type", "text/event-stream")],
                    "data: {\"choices\":[{\"delta\":{\"content\":\"<think>\"}}]}\n\n\
                     data: {\"choices\":[{\"delta\":{\"content\":\"thinking\"}}]}\n\n\
                     data: {\"choices\":[{\"delta\":{\"content\":\"</think>\"}}]}\n\n\
                     data: {\"choices\":[{\"delta\":{\"content\":\"\\n\\n4\"}}]}\n\n\
                     data: {\"choices\":[{\"delta\":{\"content\":\"2\"}}]}\n\n\
                     data: [DONE]\n\n",
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let m = OpenAiModel::new(format!("http://{addr}/v1"), "test-model");
        let prompt = Prompt {
            user: "hello".into(),
            ..Default::default()
        };
        let mut s = m.complete_stream(&prompt).await.unwrap();
        let mut full = String::new();
        while let Some(r) = s.next().await {
            full.push_str(&r.unwrap());
        }
        assert_eq!(full, "42", "thinking suppressed, only response streamed");
        assert!(
            !full.contains("thinking") && !full.to_lowercase().contains("think"),
            "think block did not leak: {full:?}"
        );
    }

    /// Smoke test against a real OpenAI-compatible server (default: local
    /// Ollama).
    /// Run with: `LORE_LLM_BASE=... LORE_LLM_MODEL=... cargo test -- --ignored`
    #[tokio::test]
    #[ignore = "requires real LLM (LORE_LLM_BASE, default http://localhost:11434/v1)"]
    async fn live_llm_complete_and_stream() {
        let base =
            std::env::var("LORE_LLM_BASE").unwrap_or_else(|_| "http://localhost:11434/v1".into());
        let model = std::env::var("LORE_LLM_MODEL").unwrap_or_else(|_| "llama3.2".into());
        let mut m = OpenAiModel::new(base, model);
        if let Ok(k) = std::env::var("LORE_LLM_KEY") {
            m = m.with_api_key(k);
        }
        let prompt = Prompt {
            system: "Reply briefly.".into(),
            user: "2+2? Answer with a single number.".into(),
            ..Default::default()
        };

        // Completion path.
        let c = m.complete(&prompt).await.unwrap();
        assert!(!c.text.trim().is_empty(), "complete returned empty");

        // Real streaming path.
        let mut s = m.complete_stream(&prompt).await.unwrap();
        let mut full = String::new();
        while let Some(r) = s.next().await {
            full.push_str(&r.unwrap());
        }
        assert!(!full.trim().is_empty(), "stream returned empty");
    }

    mod props {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// SSE buffer reassembler produces the same lines under arbitrary
            /// chunking — even when multi-byte UTF-8 characters are split at
            /// boundaries.
            #[test]
            fn next_line_reassembles_under_arbitrary_chunking(
                lines in prop::collection::vec("[a-zA-Z0-9 áéíóúñãõ]{0,40}", 1..8),
                cuts in prop::collection::vec(1usize..20, 0..30),
            ) {
                let joined: String = lines.iter().map(|l| format!("{l}\n")).collect();
                let bytes = joined.as_bytes();

                // Split at cut points, feed sequentially.
                let mut buf: Vec<u8> = Vec::new();
                let mut got: Vec<String> = Vec::new();
                let mut pos = 0usize;
                let mut cut_iter = cuts.iter();
                while pos < bytes.len() {
                    let step = cut_iter.next().copied().unwrap_or(7).min(bytes.len() - pos);
                    buf.extend_from_slice(&bytes[pos..pos + step]);
                    pos += step;
                    while let Some(l) = next_line(&mut buf) {
                        got.push(l);
                    }
                }
                let expected: Vec<String> =
                    lines.iter().map(|l| l.trim().to_string()).collect();
                prop_assert_eq!(got, expected);
            }

            /// Line parser must never panic with arbitrary input.
            #[test]
            fn parse_sse_line_never_panics(s in "\\PC*") {
                let _ = parse_sse_line(&s);
            }
        }
    }

    #[test]
    fn ollama_shortcut_sets_base_url() {
        let m = OpenAiModel::ollama("qwen2.5");
        assert_eq!(m.base_url, "http://localhost:11434/v1");
        assert_eq!(m.model, "qwen2.5");
    }
}
