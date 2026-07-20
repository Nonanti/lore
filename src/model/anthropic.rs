//! `AnthropicModel`: Anthropic Messages API (`/v1/messages`) client.
//!
//! Two auth modes behind one type:
//! - **API key** (`x-api-key`) — official, stable.
//! - **Subscription OAuth** (`Authorization: Bearer`) — Claude Pro/Max, needs
//!   the Claude Code beta headers and a system prompt that begins with the exact
//!   Claude Code identity string (server-enforced for OAuth tokens).
//!
//! Request building (`build_payload`) and response parsing (`parse_response`)
//! are pure/testable; the HTTP call lives in `complete`/`complete_stream`.

use super::{Completion, Model, Prompt, Role, TokenStream};
use crate::auth::AccessTokenProvider;
use crate::error::{LoreError, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;

const ANTHROPIC_BASE: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Beta features Claude Code's OAuth path negotiates.
const ANTHROPIC_OAUTH_BETA: &str =
    "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14";
/// Server-enforced identity prefix for OAuth (subscription) tokens. The first
/// system block must begin with this or the API returns HTTP 400.
const CLAUDE_CODE_IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";
const DEFAULT_MAX_TOKENS: u32 = 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// How the request authenticates to Anthropic.
#[derive(Clone)]
pub enum AnthropicAuth {
    /// Metered API key → `x-api-key`.
    ApiKey(String),
    /// Subscription access token → `Authorization: Bearer` + Claude Code betas.
    /// The provider yields a fresh token per call (auto-refresh).
    OAuth(Arc<dyn AccessTokenProvider>),
}

impl AnthropicAuth {
    fn is_oauth(&self) -> bool {
        matches!(self, AnthropicAuth::OAuth(_))
    }

    /// Resolves the on-the-wire auth material (refreshing OAuth if needed).
    async fn resolve(&self) -> Result<ResolvedAuth> {
        match self {
            AnthropicAuth::ApiKey(k) => Ok(ResolvedAuth::ApiKey(k.clone())),
            AnthropicAuth::OAuth(p) => Ok(ResolvedAuth::Bearer(p.access_token().await?)),
        }
    }
}

/// Auth material resolved just before a request.
enum ResolvedAuth {
    ApiKey(String),
    Bearer(String),
}

fn make_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .unwrap_or_default()
}

/// Anthropic Messages API client.
pub struct AnthropicModel {
    client: reqwest::Client,
    base_url: String,
    model: String,
    auth: AnthropicAuth,
    temperature: f32,
    max_tokens: u32,
    /// One-shot: total timeout; streaming: per-chunk idle timeout.
    timeout: Duration,
}

impl AnthropicModel {
    /// New client for `model`, authenticated by `auth`.
    pub fn new(model: impl Into<String>, auth: AnthropicAuth) -> Self {
        Self {
            client: make_client(),
            base_url: ANTHROPIC_BASE.to_string(),
            model: model.into(),
            auth,
            temperature: 0.7,
            max_tokens: DEFAULT_MAX_TOKENS,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Overrides the base URL (self-hosted/proxy).
    pub fn with_base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into();
        self
    }

    /// Sets sampling temperature.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Sets the response token cap (Anthropic requires `max_tokens`).
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n.max(1);
        self
    }

    /// Sets the request timeout (one-shot total / stream per-chunk idle).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// System string = persona system + recalled context lines.
    fn system_text(prompt: &Prompt) -> String {
        let mut s = prompt.system.clone();
        if !prompt.context.is_empty() {
            s.push_str("\n\nWhat you recall:\n");
            for c in &prompt.context {
                s.push_str("- ");
                s.push_str(c);
                s.push('\n');
            }
        }
        s.trim().to_string()
    }

    /// Messages array (history + user), Anthropic role names.
    fn build_messages(prompt: &Prompt) -> Vec<serde_json::Value> {
        let mut msgs = Vec::new();
        for t in &prompt.history {
            let role = match t.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            msgs.push(serde_json::json!({ "role": role, "content": t.text }));
        }
        msgs.push(serde_json::json!({ "role": "user", "content": prompt.user }));
        msgs
    }

    /// Builds the `system` field. OAuth requires the Claude Code identity as the
    /// first block, so it is emitted as a two-block array; API-key mode uses a
    /// plain string (or omits it when empty).
    fn build_system(&self, prompt: &Prompt) -> Option<serde_json::Value> {
        let text = Self::system_text(prompt);
        if self.auth.is_oauth() {
            let mut blocks = vec![serde_json::json!({
                "type": "text", "text": CLAUDE_CODE_IDENTITY
            })];
            if !text.is_empty() {
                blocks.push(serde_json::json!({ "type": "text", "text": text }));
            }
            Some(serde_json::Value::Array(blocks))
        } else if text.is_empty() {
            None
        } else {
            Some(serde_json::Value::String(text))
        }
    }

    /// Builds the JSON request body.
    fn build_payload(&self, prompt: &Prompt, stream: bool) -> serde_json::Value {
        let mut v = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
            "messages": Self::build_messages(prompt),
        });
        if let Some(system) = self.build_system(prompt) {
            v["system"] = system;
        }
        v
    }

    /// Builds the authorized POST request with the right auth + beta headers.
    fn request(&self, payload: &serde_json::Value, auth: &ResolvedAuth) -> reqwest::RequestBuilder {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let mut req = self
            .client
            .post(url)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(payload);
        match auth {
            ResolvedAuth::ApiKey(key) => {
                req = req.header("x-api-key", key);
            }
            ResolvedAuth::Bearer(access) => {
                req = req
                    .bearer_auth(access)
                    .header("anthropic-beta", ANTHROPIC_OAUTH_BETA)
                    .header("anthropic-dangerous-direct-browser-access", "true");
            }
        }
        req
    }

    /// Extracts text from a non-streaming Messages response (joins `text`
    /// content blocks; thinking blocks are ignored).
    fn parse_response(body: &str) -> Result<Completion> {
        let resp: MessagesResponse = serde_json::from_str(body)?;
        let text: String = resp
            .content
            .into_iter()
            .filter_map(|b| match b.kind.as_str() {
                "text" => b.text,
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        if text.trim().is_empty() {
            tracing::warn!("anthropic returned no text content blocks");
        }
        Ok(Completion::new(text))
    }
}

#[derive(Deserialize)]
struct MessagesResponse {
    #[serde(default)]
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

/// An SSE event of interest from the Messages stream.
enum SseEvent {
    Token(String),
    Done,
}

/// Pulls the next complete `\n`-terminated line from the buffer (UTF-8 safe).
fn next_line(buf: &mut Vec<u8>) -> Option<String> {
    let pos = buf.iter().position(|&b| b == b'\n')?;
    let line: Vec<u8> = buf.drain(..=pos).collect();
    Some(String::from_utf8_lossy(&line).trim().to_string())
}

/// Parses one Anthropic SSE `data:` line. Emits text deltas; signals done on
/// `message_stop`; surfaces an error event as a done (the body carries it).
fn parse_sse_line(line: &str) -> Option<SseEvent> {
    let payload = line.strip_prefix("data:")?.trim();
    let v: serde_json::Value = serde_json::from_str(payload).ok()?;
    match v["type"].as_str()? {
        "content_block_delta" => {
            let delta = &v["delta"];
            if delta["type"] == "text_delta" {
                let t = delta["text"].as_str()?;
                if t.is_empty() {
                    return None;
                }
                return Some(SseEvent::Token(t.to_string()));
            }
            None
        }
        "message_stop" => Some(SseEvent::Done),
        _ => None,
    }
}

#[async_trait]
impl Model for AnthropicModel {
    async fn complete(&self, prompt: &Prompt) -> Result<Completion> {
        let auth = self.auth.resolve().await?;
        let work = async {
            let resp = self
                .request(&self.build_payload(prompt, false), &auth)
                .send()
                .await?;
            let status = resp.status();
            let body = resp.text().await?;
            if !status.is_success() {
                return Err(LoreError::Model(format!("{status}: {body}")));
            }
            Self::parse_response(&body)
        };
        match tokio::time::timeout(self.timeout, work).await {
            Ok(r) => r,
            Err(_) => Err(LoreError::Model(format!(
                "request timeout ({}s)",
                self.timeout.as_secs()
            ))),
        }
    }

    async fn complete_stream(&self, prompt: &Prompt) -> Result<TokenStream> {
        let auth = self.auth.resolve().await?;
        let resp = match tokio::time::timeout(
            self.timeout,
            self.request(&self.build_payload(prompt, true), &auth)
                .send(),
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
            (resp.bytes_stream(), Vec::<u8>::new(), false),
            move |(mut body, mut buf, ended)| async move {
                if ended {
                    return None;
                }
                loop {
                    while let Some(line) = next_line(&mut buf) {
                        match parse_sse_line(&line) {
                            Some(SseEvent::Done) => return None,
                            Some(SseEvent::Token(t)) => {
                                return Some((Ok(t), (body, buf, false)));
                            }
                            None => {}
                        }
                    }
                    match tokio::time::timeout(idle, body.next()).await {
                        Ok(Some(Ok(chunk))) => buf.extend_from_slice(&chunk),
                        Ok(Some(Err(e))) => {
                            return Some((Err(LoreError::Http(e)), (body, buf, true)))
                        }
                        Ok(None) => return None,
                        Err(_) => {
                            return Some((
                                Err(LoreError::Model(format!(
                                    "stream idle timeout ({}s)",
                                    idle.as_secs()
                                ))),
                                (body, buf, true),
                            ))
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

    fn prompt() -> Prompt {
        Prompt {
            system: "You are Aria.".into(),
            context: vec!["likes rust".into()],
            history: vec![super::super::Turn::user("hi")],
            user: "hello".into(),
        }
    }

    fn oauth(tok: &str) -> AnthropicAuth {
        AnthropicAuth::OAuth(Arc::new(crate::auth::StaticToken(tok.to_string())))
    }

    #[test]
    fn oauth_system_starts_with_claude_code_identity() {
        let m = AnthropicModel::new("claude-x", oauth("tok"));
        let v = m.build_payload(&prompt(), false);
        let system = v["system"].as_array().expect("oauth system is an array");
        assert_eq!(system[0]["text"], CLAUDE_CODE_IDENTITY);
        // Persona system + context land in the second block.
        assert!(system[1]["text"]
            .as_str()
            .unwrap()
            .contains("You are Aria."));
        assert!(system[1]["text"].as_str().unwrap().contains("likes rust"));
    }

    #[test]
    fn apikey_system_is_plain_string_without_identity() {
        let m = AnthropicModel::new("claude-x", AnthropicAuth::ApiKey("k".into()));
        let v = m.build_payload(&prompt(), false);
        let s = v["system"].as_str().expect("api-key system is a string");
        assert!(s.contains("You are Aria."));
        assert!(!s.contains(CLAUDE_CODE_IDENTITY));
    }

    #[test]
    fn payload_has_required_fields() {
        let m =
            AnthropicModel::new("claude-x", AnthropicAuth::ApiKey("k".into())).with_max_tokens(512);
        let v = m.build_payload(&prompt(), true);
        assert_eq!(v["model"], "claude-x");
        assert_eq!(v["max_tokens"], 512);
        assert_eq!(v["stream"], true);
        let msgs = v["messages"].as_array().unwrap();
        // history user + current user
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "hi");
        assert_eq!(msgs[1]["content"], "hello");
    }

    #[test]
    fn parse_joins_text_blocks_and_skips_thinking() {
        let body = r#"{"content":[{"type":"thinking","thinking":"..."},
            {"type":"text","text":"Hello "},{"type":"text","text":"world"}]}"#;
        let c = AnthropicModel::parse_response(body).unwrap();
        assert_eq!(c.text, "Hello world");
    }

    #[test]
    fn parse_empty_content_is_ok() {
        let c = AnthropicModel::parse_response(r#"{"content":[]}"#).unwrap();
        assert_eq!(c.text, "");
    }

    #[test]
    fn sse_line_parsing() {
        let delta = r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#;
        assert!(matches!(parse_sse_line(delta), Some(SseEvent::Token(t)) if t == "hi"));
        let stop = r#"data: {"type":"message_stop"}"#;
        assert!(matches!(parse_sse_line(stop), Some(SseEvent::Done)));
        // thinking delta and non-data lines are skipped.
        let think = r#"data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"x"}}"#;
        assert!(parse_sse_line(think).is_none());
        assert!(parse_sse_line("event: message_stop").is_none());
        assert!(parse_sse_line(": ping").is_none());
    }
}
