//! `CodexModel`: OpenAI ChatGPT (Codex) Responses API over subscription OAuth.
//!
//! Talks to `https://chatgpt.com/backend-api/codex/responses` with a
//! `Bearer` access token and the `ChatGPT-Account-Id` header. The request/
//! response follow the OpenAI **Responses** API (not Chat Completions):
//! `instructions` + `input[]`, and an SSE stream of `response.output_text.delta`
//! events.
//!
//! **Unverified against the live backend** in this build (no live subscription
//! token was available); shapes follow the Codex CLI / community references.
//! The metered OpenAI API-key path (`OpenAiModel`, Chat Completions) is stable.

use super::{Completion, Model, Prompt, Role, TokenStream};
use crate::auth::AccessTokenProvider;
use crate::error::{LoreError, Result};
use async_trait::async_trait;
use futures::StreamExt;
use std::sync::Arc;
use std::time::Duration;

const CODEX_BASE: &str = "https://chatgpt.com/backend-api/codex";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

fn make_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .unwrap_or_default()
}

/// OpenAI Codex (ChatGPT subscription) Responses API client.
pub struct CodexModel {
    client: reqwest::Client,
    base_url: String,
    model: String,
    token: Arc<dyn AccessTokenProvider>,
    account_id: Option<String>,
    timeout: Duration,
}

impl CodexModel {
    /// New client. `account_id` populates the `ChatGPT-Account-Id` header
    /// (from the login id-token); the `token` provider yields a fresh access
    /// token per call (auto-refresh).
    pub fn new(
        model: impl Into<String>,
        token: Arc<dyn AccessTokenProvider>,
        account_id: Option<String>,
    ) -> Self {
        Self {
            client: make_client(),
            base_url: CODEX_BASE.to_string(),
            model: model.into(),
            token,
            account_id,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    /// Overrides the base URL (proxy/testing).
    pub fn with_base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into();
        self
    }

    /// Sets the request timeout (one-shot total / stream per-chunk idle).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// System string = persona system + recalled context lines.
    fn instructions(prompt: &Prompt) -> String {
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

    /// Responses `input[]`: history + current user. User content uses
    /// `input_text`, assistant content uses `output_text`.
    fn build_input(prompt: &Prompt) -> Vec<serde_json::Value> {
        let mut items = Vec::new();
        let mut push = |role: &str, kind: &str, text: &str| {
            items.push(serde_json::json!({
                "type": "message",
                "role": role,
                "content": [{ "type": kind, "text": text }],
            }));
        };
        for t in &prompt.history {
            match t.role {
                Role::User => push("user", "input_text", &t.text),
                Role::Assistant => push("assistant", "output_text", &t.text),
            }
        }
        push("user", "input_text", &prompt.user);
        items
    }

    /// Builds the JSON request body (Responses API).
    fn build_payload(&self, prompt: &Prompt) -> serde_json::Value {
        let mut v = serde_json::json!({
            "model": self.model,
            "input": Self::build_input(prompt),
            "stream": true,
            "store": false,
        });
        let instructions = Self::instructions(prompt);
        if !instructions.is_empty() {
            v["instructions"] = instructions.into();
        }
        v
    }

    /// Builds the authorized POST request.
    fn request(&self, payload: &serde_json::Value, access: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/responses", self.base_url.trim_end_matches('/'));
        let mut req = self
            .client
            .post(url)
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", "codex_cli_rs")
            .header("Accept", "text/event-stream")
            .bearer_auth(access)
            .json(payload);
        if let Some(acc) = &self.account_id {
            req = req.header("ChatGPT-Account-Id", acc);
        }
        req
    }

    /// Opens the SSE stream and returns the HTTP response (status-checked).
    async fn open_stream(&self, prompt: &Prompt) -> Result<reqwest::Response> {
        let access = self.token.access_token().await?;
        let payload = self.build_payload(prompt);
        let resp = match tokio::time::timeout(self.timeout, self.request(&payload, &access).send())
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
        Ok(resp)
    }
}

/// An SSE event of interest from the Responses stream.
enum SseEvent {
    Token(String),
    Done,
    /// The backend reported a failed/incomplete response; carries its message.
    Failed(String),
}

/// Pulls the next complete `\n`-terminated line from the buffer (UTF-8 safe).
fn next_line(buf: &mut Vec<u8>) -> Option<String> {
    let pos = buf.iter().position(|&b| b == b'\n')?;
    let line: Vec<u8> = buf.drain(..=pos).collect();
    Some(String::from_utf8_lossy(&line).trim().to_string())
}

/// Parses one Responses SSE `data:` line. Emits `output_text` deltas; signals
/// done on `response.completed`; surfaces `response.failed`/`incomplete` as an
/// error (so a failed run is not silently truncated).
fn parse_sse_line(line: &str) -> Option<SseEvent> {
    let payload = line.strip_prefix("data:")?.trim();
    if payload == "[DONE]" {
        return Some(SseEvent::Done);
    }
    let v: serde_json::Value = serde_json::from_str(payload).ok()?;
    match v["type"].as_str()? {
        "response.output_text.delta" => {
            let t = v["delta"].as_str()?;
            if t.is_empty() {
                return None;
            }
            Some(SseEvent::Token(t.to_string()))
        }
        "response.completed" => Some(SseEvent::Done),
        "response.failed" | "response.incomplete" => {
            let msg = v["response"]["error"]["message"]
                .as_str()
                .or_else(|| v["error"]["message"].as_str())
                .unwrap_or("codex response failed");
            Some(SseEvent::Failed(msg.to_string()))
        }
        _ => None,
    }
}

/// Drives the byte stream to completion, invoking `on_token` per text delta.
async fn drive_stream<F: FnMut(String)>(
    resp: reqwest::Response,
    idle: Duration,
    mut on_token: F,
) -> Result<()> {
    let mut body = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    loop {
        while let Some(line) = next_line(&mut buf) {
            match parse_sse_line(&line) {
                Some(SseEvent::Done) => return Ok(()),
                Some(SseEvent::Token(t)) => on_token(t),
                Some(SseEvent::Failed(m)) => return Err(LoreError::Model(m)),
                None => {}
            }
        }
        match tokio::time::timeout(idle, body.next()).await {
            Ok(Some(Ok(chunk))) => buf.extend_from_slice(&chunk),
            Ok(Some(Err(e))) => return Err(LoreError::Http(e)),
            Ok(None) => return Ok(()),
            Err(_) => {
                return Err(LoreError::Model(format!(
                    "stream idle timeout ({}s)",
                    idle.as_secs()
                )))
            }
        }
    }
}

#[async_trait]
impl Model for CodexModel {
    async fn complete(&self, prompt: &Prompt) -> Result<Completion> {
        let resp = self.open_stream(prompt).await?;
        let mut text = String::new();
        drive_stream(resp, self.timeout, |t| text.push_str(&t)).await?;
        if text.trim().is_empty() {
            tracing::warn!("codex returned no output_text deltas");
        }
        Ok(Completion::new(text))
    }

    async fn complete_stream(&self, prompt: &Prompt) -> Result<TokenStream> {
        let resp = self.open_stream(prompt).await?;
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
                            Some(SseEvent::Token(t)) => return Some((Ok(t), (body, buf, false))),
                            Some(SseEvent::Failed(m)) => {
                                return Some((Err(LoreError::Model(m)), (body, buf, true)))
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
    use crate::auth::StaticToken;

    fn model() -> CodexModel {
        CodexModel::new(
            "gpt-5",
            Arc::new(StaticToken("tok".into())),
            Some("acc-123".into()),
        )
    }

    fn prompt() -> Prompt {
        Prompt {
            system: "You are Aria.".into(),
            context: vec!["likes rust".into()],
            history: vec![super::super::Turn::assistant("earlier")],
            user: "hello".into(),
        }
    }

    #[test]
    fn payload_uses_responses_shape() {
        let v = model().build_payload(&prompt());
        assert_eq!(v["model"], "gpt-5");
        assert_eq!(v["stream"], true);
        assert_eq!(v["store"], false);
        assert!(v["instructions"]
            .as_str()
            .unwrap()
            .contains("You are Aria."));
        assert!(v["instructions"].as_str().unwrap().contains("likes rust"));
        let input = v["input"].as_array().unwrap();
        // history assistant (output_text) + current user (input_text)
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["role"], "assistant");
        assert_eq!(input[0]["content"][0]["type"], "output_text");
        assert_eq!(input[1]["role"], "user");
        assert_eq!(input[1]["content"][0]["type"], "input_text");
        assert_eq!(input[1]["content"][0]["text"], "hello");
    }

    #[test]
    fn sse_parsing() {
        let d = r#"data: {"type":"response.output_text.delta","delta":"hi","sequence_number":1}"#;
        assert!(matches!(parse_sse_line(d), Some(SseEvent::Token(t)) if t == "hi"));
        assert!(matches!(
            parse_sse_line(r#"data: {"type":"response.completed"}"#),
            Some(SseEvent::Done)
        ));
        assert!(matches!(
            parse_sse_line("data: [DONE]"),
            Some(SseEvent::Done)
        ));
        // failed surfaces the error message instead of a silent stop.
        let failed = r#"data: {"type":"response.failed","response":{"error":{"message":"boom"}}}"#;
        assert!(matches!(parse_sse_line(failed), Some(SseEvent::Failed(m)) if m == "boom"));
        // reasoning/other events skipped; non-data lines skipped.
        assert!(parse_sse_line(r#"data: {"type":"response.created"}"#).is_none());
        assert!(parse_sse_line("event: response.completed").is_none());
    }
}
