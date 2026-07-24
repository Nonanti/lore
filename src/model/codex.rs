//! `CodexModel`: OpenAI ChatGPT (Codex) Responses API over subscription OAuth.
//!
//! Talks to `https://chatgpt.com/backend-api/codex/responses` with a
//! `Bearer` access token and the `ChatGPT-Account-Id` header. The request/
//! response follow the OpenAI **Responses** API (not Chat Completions):
//! `instructions` + `input[]`, and an SSE stream of `response.output_text.delta`
//! events. Native tool calling uses Responses `function_call` /
//! `function_call_output` items (flat `tools` entries, `name` at top level);
//! completed calls are read from `response.output_item.done` events.
//!
//! **Unverified against the live backend** in this build (no live subscription
//! token was available); shapes follow the Codex CLI / community references.
//! The metered OpenAI API-key path (`OpenAiModel`, Chat Completions) is stable.

use super::{
    ChatRole, Completion, ContentBlock, Model, Prompt, Role, StopReason, Thread, ThreadReply,
    TokenStream, ToolSpec,
};
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

    /// Thread → Responses `input[]` items. Text keeps the message/
    /// `input_text`⁄`output_text` shape; ToolUse becomes a `function_call`
    /// item (arguments as a JSON string, the Responses convention) and
    /// ToolResult a `function_call_output` item correlated by `call_id`
    /// (no error flag on this wire — the "ERROR: .." output text carries it).
    fn build_thread_input(thread: &Thread) -> Vec<serde_json::Value> {
        let mut items = Vec::new();
        for m in &thread.messages {
            for b in &m.blocks {
                match (m.role, b) {
                    (ChatRole::User, ContentBlock::Text { text }) => {
                        items.push(serde_json::json!({
                            "type": "message", "role": "user",
                            "content": [{ "type": "input_text", "text": text }],
                        }));
                    }
                    (ChatRole::Assistant, ContentBlock::Text { text }) => {
                        items.push(serde_json::json!({
                            "type": "message", "role": "assistant",
                            "content": [{ "type": "output_text", "text": text }],
                        }));
                    }
                    (ChatRole::Assistant, ContentBlock::ToolUse { id, name, input }) => {
                        items.push(serde_json::json!({
                            "type": "function_call",
                            "call_id": id,
                            "name": name,
                            "arguments": serde_json::to_string(input).unwrap_or_default(),
                        }));
                    }
                    (
                        ChatRole::User,
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error: _,
                        },
                    ) => {
                        items.push(serde_json::json!({
                            "type": "function_call_output",
                            "call_id": tool_use_id,
                            "output": content,
                        }));
                    }
                    (role, block) => {
                        tracing::warn!(?role, ?block, "block/role mismatch in thread; skipped");
                    }
                }
            }
        }
        items
    }

    /// Builds the request body for a native tool-calling thread. Responses
    /// tools are FLAT (`name` at top level), unlike chat-completions'
    /// nested `function` object.
    fn build_thread_payload(&self, thread: &Thread, tools: &[ToolSpec]) -> serde_json::Value {
        let mut v = serde_json::json!({
            "model": self.model,
            "input": Self::build_thread_input(thread),
            "stream": true,
            "store": false,
        });
        if !tools.is_empty() {
            let tools_json: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    })
                })
                .collect();
            v["tools"] = serde_json::Value::Array(tools_json);
        }
        let instructions = thread.system.trim();
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
        self.open_stream_payload(self.build_payload(prompt), false)
            .await
    }

    /// Opens the SSE stream for an arbitrary payload. `classify_tools` maps
    /// tools-unsupported error bodies to the typed downgrade error — set
    /// only when the payload actually carries tools (plain chat 400s must
    /// never masquerade as downgrades).
    async fn open_stream_payload(
        &self,
        payload: serde_json::Value,
        classify_tools: bool,
    ) -> Result<reqwest::Response> {
        let access = self.token.access_token().await?;
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
            if classify_tools && super::openai::OpenAiModel::tools_unsupported(&body) {
                return Err(LoreError::NativeToolsUnsupported(format!(
                    "{status}: {body}"
                )));
            }
            return Err(LoreError::Model(format!("{status}: {body}")));
        }
        Ok(resp)
    }
}

/// An SSE event of interest from the Responses stream.
enum SseEvent {
    Token(String),
    /// A completed `function_call` output item (call_id, name, arguments).
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
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

/// Parses one Responses SSE `data:` line. Emits `output_text` deltas and
/// completed `function_call` items; signals done on `response.completed`;
/// surfaces `response.failed`/`incomplete` as an error (so a failed run is
/// not silently truncated).
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
        // Complete function-call item: arguments arrive fully assembled here
        // (`response.function_call_arguments.delta` events are skipped — the
        // done item is the single source of truth).
        "response.output_item.done" => {
            let item = &v["item"];
            if item["type"].as_str()? != "function_call" {
                return None;
            }
            Some(SseEvent::FunctionCall {
                call_id: item["call_id"].as_str().unwrap_or_default().to_string(),
                name: item["name"].as_str()?.to_string(),
                arguments: item["arguments"].as_str().unwrap_or("{}").to_string(),
            })
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
    drive_events(resp, idle, |ev| {
        if let SseEvent::Token(t) = ev {
            on_token(t);
        }
    })
    .await
}

/// Drives the byte stream to completion, invoking `on_event` per parsed
/// event (Done/Failed are handled here, not passed through).
async fn drive_events<F: FnMut(SseEvent)>(
    resp: reqwest::Response,
    idle: Duration,
    mut on_event: F,
) -> Result<()> {
    let mut body = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    loop {
        while let Some(line) = next_line(&mut buf) {
            match parse_sse_line(&line) {
                Some(SseEvent::Done) => return Ok(()),
                Some(SseEvent::Failed(m)) => return Err(LoreError::Model(m)),
                Some(ev) => on_event(ev),
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
    async fn complete_thread(&self, thread: &Thread, tools: &[ToolSpec]) -> Result<ThreadReply> {
        let resp = self
            .open_stream_payload(self.build_thread_payload(thread, tools), !tools.is_empty())
            .await?;
        let mut text = String::new();
        let mut calls: Vec<(String, String, String)> = Vec::new();
        drive_events(resp, self.timeout, |ev| match ev {
            SseEvent::Token(t) => text.push_str(&t),
            SseEvent::FunctionCall {
                call_id,
                name,
                arguments,
            } => calls.push((call_id, name, arguments)),
            _ => {}
        })
        .await?;

        let mut blocks = Vec::new();
        if !text.trim().is_empty() {
            blocks.push(ContentBlock::Text { text });
        }
        for (i, (call_id, name, arguments)) in calls.into_iter().enumerate() {
            // Same tolerance as the chat-completions wire: unparseable
            // arguments become a raw string; missing ids are synthesized.
            let input = match serde_json::from_str(&arguments) {
                Ok(v) => v,
                Err(_) => serde_json::Value::String(arguments),
            };
            let id = if call_id.is_empty() {
                format!("call_{i}")
            } else {
                call_id
            };
            blocks.push(ContentBlock::ToolUse { id, name, input });
        }
        let stop = if blocks
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
        {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        };
        Ok(ThreadReply {
            blocks,
            stop,
            reasoning_fallback: false,
        })
    }

    fn supports_native_tools(&self) -> bool {
        true
    }

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
                            // Plain chat streams never request tools; a stray
                            // function_call item has no consumer here.
                            Some(SseEvent::FunctionCall { .. }) | None => {}
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
    fn thread_payload_maps_items_and_flat_tools() {
        use super::super::{ChatMessage as ThreadMsg, ContentBlock};
        let mut t = Thread::new("You are Aria.");
        t.push(ThreadMsg::user_text("what is 3+4?"));
        t.push(ThreadMsg::assistant_blocks(vec![
            ContentBlock::Text {
                text: "Computing.".into(),
            },
            ContentBlock::ToolUse {
                id: "c1".into(),
                name: "calc".into(),
                input: serde_json::json!({"args": "3 + 4"}),
            },
        ]));
        t.push(ThreadMsg::tool_results(vec![ContentBlock::ToolResult {
            tool_use_id: "c1".into(),
            content: "7".into(),
            is_error: false,
        }]));
        let specs = vec![ToolSpec {
            name: "calc".into(),
            description: "evaluates arithmetic".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];

        let v = model().build_thread_payload(&t, &specs);
        assert_eq!(v["instructions"], "You are Aria.");
        // Responses tools are FLAT — name at top level.
        let tools = v["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "calc");
        assert_eq!(tools[0]["parameters"]["type"], "object");
        assert!(tools[0].get("function").is_none());

        let input = v["input"].as_array().unwrap();
        assert_eq!(input.len(), 4);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[1]["type"], "message");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "c1");
        let args: serde_json::Value =
            serde_json::from_str(input[2]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["args"], "3 + 4");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "c1");
        assert_eq!(input[3]["output"], "7");
    }

    #[test]
    fn sse_function_call_item_parsed() {
        let done_item = r#"data: {"type":"response.output_item.done","item":{
            "type":"function_call","call_id":"c9","name":"calc",
            "arguments":"{\"args\":\"2+2\"}"}}"#
            .replace('\n', "");
        match parse_sse_line(&done_item) {
            Some(SseEvent::FunctionCall {
                call_id,
                name,
                arguments,
            }) => {
                assert_eq!(call_id, "c9");
                assert_eq!(name, "calc");
                assert!(arguments.contains("2+2"));
            }
            other => panic!("expected FunctionCall, got {:?}", other.is_some()),
        }
        // Non-function items are skipped.
        let msg_item = r#"data: {"type":"response.output_item.done","item":{"type":"message"}}"#;
        assert!(parse_sse_line(msg_item).is_none());
    }

    #[tokio::test]
    async fn complete_thread_round_trips_over_sse() {
        use axum::{routing::post, Router};
        let sse_body = concat!(
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"calc\",\"arguments\":\"{\\\"args\\\":\\\"3 + 4\\\"}\"}}\n\n",
            "data: {\"type\":\"response.completed\"}\n\n",
        );
        let app = Router::new().route(
            "/responses",
            post(move |body: axum::Json<serde_json::Value>| async move {
                assert!(body.0["tools"].is_array());
                ([("content-type", "text/event-stream")], sse_body)
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let m = model().with_base_url(format!("http://{addr}"));
        assert!(m.supports_native_tools());
        let mut t = Thread::new("sys");
        t.push(super::super::ChatMessage::user_text("3+4?"));
        let specs = vec![ToolSpec {
            name: "calc".into(),
            description: "d".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let r = m.complete_thread(&t, &specs).await.unwrap();
        assert_eq!(r.stop, StopReason::ToolUse);
        let uses = r.tool_uses();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].id, "c1");
        assert_eq!(uses[0].input["args"], "3 + 4");
    }

    #[tokio::test]
    async fn complete_thread_maps_unsupported_error() {
        use axum::{http::StatusCode, routing::post, Router};
        let app = Router::new().route(
            "/responses",
            post(|| async {
                (
                    StatusCode::BAD_REQUEST,
                    r#"{"error":{"message":"this backend does not support tools"}}"#,
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let m = model().with_base_url(format!("http://{addr}"));
        let mut t = Thread::new("sys");
        t.push(super::super::ChatMessage::user_text("q"));
        let specs = vec![ToolSpec {
            name: "calc".into(),
            description: "d".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let err = m.complete_thread(&t, &specs).await.unwrap_err();
        assert!(
            matches!(err, LoreError::NativeToolsUnsupported(_)),
            "got: {err}"
        );

        // Without tools, the same body is a PLAIN model error (no fake downgrade).
        let err2 = m.complete_thread(&t, &[]).await.unwrap_err();
        assert!(matches!(err2, LoreError::Model(_)), "got: {err2}");
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
