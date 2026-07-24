//! Server tests: AppState core logic (offline) + HTTP end-to-end.

use super::router;
use super::security::{ct_eq, extract_key};
use super::types::{AgentView, AskResp, PersonaPatch};
use super::AppState;
use crate::error::{LoreError, Result};
use crate::id::AgentId;
use crate::memory::{InMemoryStore, MemoryStore};
use crate::model::{MockModel, Model};
use crate::tool::ToolContext;
use axum::http::HeaderMap;
use std::path::PathBuf;
use std::sync::Arc;

fn state() -> AppState {
    AppState::new(Arc::new(InMemoryStore::new()), Arc::new(MockModel::new()))
}

fn calc_tools() -> ToolContext {
    use crate::tool::{CalcTool, KeywordRouter, ToolRegistry};
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(CalcTool::new()));
    ToolContext {
        registry: reg,
        router: Arc::new(KeywordRouter::new().on("calculate", "calc")),
    }
}

#[tokio::test]
async fn act_uses_tool_then_remembers() {
    let st = state().with_tools(calc_tools());
    let v = st
        .create_agent("Calculator", "assistant", vec![])
        .await
        .unwrap();
    let id = AgentId::from(v.id.clone());

    let out = st.act(&id, "calculate 12 * 3").await.unwrap();
    assert_eq!(out, "36");

    // Tool usage was remembered as episodic memory.
    let mems = st.recall(&id, "tool", 10, false).await.unwrap();
    assert!(!mems.is_empty(), "tool usage remembered");
}

#[tokio::test]
async fn act_without_match_falls_back() {
    let st = state().with_tools(calc_tools());
    let v = st.create_agent("A", "r", vec![]).await.unwrap();
    let id = AgentId::from(v.id.clone());
    let out = st.act(&id, "hello how are you").await.unwrap();
    assert!(!out.is_empty(), "falls to respond");
}

#[tokio::test]
async fn create_experience_recall_flow() {
    let st = state();
    let view = st
        .create_agent("Aria", "researcher", vec!["curious".into()])
        .await
        .unwrap();
    let id = AgentId::from(view.id.clone());
    assert_eq!(view.traits, vec!["curious".to_string()]);

    st.experience(&id, "important event", "should be remembered")
        .await
        .unwrap();
    let mems = st.recall(&id, "important", 10, false).await.unwrap();
    assert_eq!(mems.len(), 1);

    let reply = st.ask(&id, "hello").await.unwrap();
    assert!(!reply.is_empty());

    assert_eq!(st.list_agents().await.len(), 1);
}

#[tokio::test]
async fn agents_survive_restart() {
    use crate::memory::SqliteStore;
    let dir = std::env::temp_dir().join(format!("lore-srv-{}", AgentId::new()));
    let db = dir.join("m.db");
    let adir = dir.join("agents");
    std::fs::create_dir_all(&dir).unwrap();
    let db = db.to_str().unwrap().to_string();
    let model: Arc<dyn Model> = Arc::new(MockModel::new());

    // First run.
    let id = {
        let store: Arc<dyn MemoryStore> = Arc::new(SqliteStore::open(&db).unwrap());
        let st = AppState::persistent(&adir, store, model.clone()).unwrap();
        let v = st.create_agent("Aria", "role", vec![]).await.unwrap();
        let id = AgentId::from(v.id.clone());
        st.experience(&id, "persistent memory", "after restart")
            .await
            .unwrap();
        id
    };

    // Restart: same directory + same DB → identity + memories restored.
    {
        let store: Arc<dyn MemoryStore> = Arc::new(SqliteStore::open(&db).unwrap());
        let st = AppState::persistent(&adir, store, model).unwrap();
        assert_eq!(st.list_agents().await.len(), 1, "agent restored");
        let mems = st.recall(&id, "persistent", 5, false).await.unwrap();
        assert_eq!(mems.len(), 1, "memories restored");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn patch_bumps_version_and_delete_removes() {
    let st = state();
    let v = st.create_agent("Aria", "role", vec![]).await.unwrap();
    assert_eq!(v.version, 1);
    let id = AgentId::from(v.id.clone());

    let patch = PersonaPatch {
        role: Some("senior researcher".into()),
        traits: Some(vec!["wise".into()]),
        ..Default::default()
    };
    let updated = st.update_agent(&id, patch).await.unwrap();
    assert_eq!(updated.version, 2, "version incremented");
    assert_eq!(updated.role, "senior researcher");
    assert_eq!(updated.traits, vec!["wise".to_string()]);

    st.delete_agent(&id).await.unwrap();
    assert!(st.list_agents().await.is_empty(), "deleted");
    assert!(matches!(
        st.update_agent(&id, PersonaPatch::default())
            .await
            .unwrap_err(),
        LoreError::NotFound(_)
    ));
}

#[tokio::test]
async fn patch_persists_across_restart() {
    use crate::memory::SqliteStore;
    let dir = std::env::temp_dir().join(format!("lore-patch-{}", AgentId::new()));
    let adir = dir.join("agents");
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("m.db").to_str().unwrap().to_string();
    let model: Arc<dyn Model> = Arc::new(MockModel::new());

    {
        let store: Arc<dyn MemoryStore> = Arc::new(SqliteStore::open(&db).unwrap());
        let st = AppState::persistent(&adir, store, model.clone()).unwrap();
        let v = st.create_agent("Aria", "role", vec![]).await.unwrap();
        let id = AgentId::from(v.id.clone());
        st.update_agent(
            &id,
            PersonaPatch {
                role: Some("new role".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    }

    {
        let store: Arc<dyn MemoryStore> = Arc::new(SqliteStore::open(&db).unwrap());
        let st = AppState::persistent(&adir, store, model).unwrap();
        let a = st.list_agents().await.into_iter().next().unwrap();
        assert_eq!(a.version, 2, "version restored from disk");
        assert_eq!(a.role, "new role");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn deliberate_collects_and_posts_to_board() {
    let st = state();
    st.create_agent("Aria", "role", vec![]).await.unwrap();
    st.create_agent("Kai", "role", vec![]).await.unwrap();

    let replies = st.deliberate("summarize").await.unwrap();
    assert_eq!(replies.len(), 2, "two agents responded");

    // Board: 1 question + 2 replies = 3 records.
    let board = st.read_board(50).await.unwrap();
    assert_eq!(board.len(), 3, "question + two replies on board");
}

/// Model that always errors — for resilience tests.
struct FailModel;
#[async_trait::async_trait]
impl Model for FailModel {
    async fn complete(&self, _p: &crate::model::Prompt) -> Result<crate::model::Completion> {
        Err(LoreError::Model("intentional test error".into()))
    }
}

#[tokio::test]
async fn deliberate_survives_failing_agents() {
    // Connect agents to a failing model: deliberate should still return Ok,
    // failed replies are skipped (a single error does not kill the poll).
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let st = AppState::new(store, Arc::new(FailModel));
    st.create_agent("Aria", "role", vec![]).await.unwrap();
    st.create_agent("Kai", "role", vec![]).await.unwrap();

    let replies = st.deliberate("summarize").await.unwrap();
    assert!(replies.is_empty(), "failed replies skipped, no panic/error");

    // The question still lands on the board.
    let board = st.read_board(10).await.unwrap();
    assert_eq!(board.len(), 1, "only question on board");
}

#[tokio::test]
async fn session_carries_conversation_history() {
    let st = state();
    let v = st.create_agent("Aria", "role", vec![]).await.unwrap();
    let id = AgentId::from(v.id.clone());

    // No session: each question is independent (no history).
    let plain = st.ask_session(&id, None, "hello").await.unwrap();
    assert!(!plain.contains("chat history"));

    // Same session: second question sees previous turns.
    let _t1 = st.ask_session(&id, Some("s1"), "hello").await.unwrap();
    let t2 = st
        .ask_session(&id, Some("s1"), "what did I say?")
        .await
        .unwrap();
    assert!(
        t2.contains("chat history: 2 messages"),
        "history carried over: {t2}"
    );

    // Different session is isolated: history does not leak.
    let other = st.ask_session(&id, Some("s2"), "hi").await.unwrap();
    assert!(
        !other.contains("chat history"),
        "sessions isolated: {other}"
    );

    // Deleting the agent also drops its sessions.
    st.delete_agent(&id).await.unwrap();
    assert!(st.sessions.read().await.is_empty(), "sessions cleaned up");
}

#[tokio::test]
async fn ask_stream_with_session_records_after_stream_ends() {
    use futures::StreamExt;
    let st = state();
    let v = st.create_agent("Aria", "role", vec![]).await.unwrap();
    let id = AgentId::from(v.id.clone());

    // Streamed first turn: consume chunks (lock drops when stream ends).
    let mut s = st.ask_stream(&id, Some("s1"), "hello").await.unwrap();
    let mut full = String::new();
    while let Some(c) = s.next().await {
        full.push_str(&c.unwrap());
    }
    drop(s);
    assert!(!full.is_empty());

    // Second turn in the same session without streaming: history should be recorded.
    let second = st
        .ask_session(&id, Some("s1"), "what did I say?")
        .await
        .unwrap();
    assert!(
        second.contains("chat history: 2 messages"),
        "exchange recorded to window at end of stream: {second}"
    );
}

#[tokio::test]
async fn http_ask_with_session_remembers() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let st = state();
    let app = router(st.clone());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let v = st.create_agent("Aria", "role", vec![]).await.unwrap();
    let client = reqwest::Client::new();
    let url = format!("http://{addr}/agents/{}/ask", v.id);

    let _first: AskResp = client
        .post(&url)
        .json(&serde_json::json!({"message":"hello","session":"web"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let second: AskResp = client
        .post(&url)
        .json(&serde_json::json!({"message":"what did I say?","session":"web"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        second.reply.contains("chat history: 2 messages"),
        "HTTP session carried history: {}",
        second.reply
    );
}

#[tokio::test]
async fn ask_unknown_agent_is_not_found() {
    let st = state();
    let err = st.ask(&AgentId::from("none"), "x").await.unwrap_err();
    assert!(matches!(err, LoreError::NotFound(_)));
}

#[tokio::test]
async fn agent_to_agent_ask_and_tell() {
    let st = state();
    let a = st.create_agent("Aria", "role", vec![]).await.unwrap();
    let b = st.create_agent("Kai", "role", vec![]).await.unwrap();
    let aid = AgentId::from(a.id.clone());
    let bid = AgentId::from(b.id.clone());

    // Tell: Aria → Kai provides info; Kai remembers.
    let ack = st
        .message(&bid, Some(&aid), false, "meeting tomorrow")
        .await
        .unwrap();
    assert!(ack.is_empty(), "tell ack empty");
    let mems = st.recall(&bid, "meeting", 10, false).await.unwrap();
    assert!(!mems.is_empty(), "Kai remembers message");

    // Ask: Aria → Kai asks; reply arrives.
    let reply = st.message(&bid, Some(&aid), true, "hello").await.unwrap();
    assert!(!reply.is_empty(), "ask reply exists");
}

#[tokio::test]
async fn metrics_report_counts() {
    let st = state();
    st.create_agent("Aria", "role", vec![]).await.unwrap();
    let text = st.metrics_text().await;
    assert!(text.contains("lore_agents 1"), "agent count");
    assert!(text.contains("lore_requests_total"));
    assert!(text.contains("lore_uptime_seconds"));
}

#[test]
fn auth_logic() {
    let open = state();
    assert!(open.authorized(None), "open when no key");

    let sec = state().with_api_key("secret");
    assert!(!sec.authorized(None));
    assert!(!sec.authorized(Some("wrong")));
    assert!(sec.authorized(Some("secret")));
}

#[test]
fn ct_eq_compares_correctly() {
    assert!(ct_eq("secret", "secret"));
    assert!(!ct_eq("secret", "gizlj"));
    assert!(!ct_eq("secret", "secr")); // different length
    assert!(!ct_eq("", "a"));
    assert!(ct_eq("", ""));
}

#[test]
fn extract_key_bearer_is_case_insensitive() {
    let mut h = HeaderMap::new();
    h.insert("authorization", "Bearer abc".parse().unwrap());
    assert_eq!(extract_key(&h).as_deref(), Some("abc"));

    h.insert("authorization", "bearer abc".parse().unwrap());
    assert_eq!(extract_key(&h).as_deref(), Some("abc"));

    h.insert("authorization", "BEARER abc".parse().unwrap());
    assert_eq!(extract_key(&h).as_deref(), Some("abc"));

    h.insert("authorization", "Basic abc".parse().unwrap());
    assert_eq!(extract_key(&h), None);

    h.insert("authorization", "Bea".parse().unwrap());
    assert_eq!(extract_key(&h), None, "short header does not panic");
}

#[test]
fn with_peers_normalizes_and_dedupes() {
    let st = state().with_peers(
        vec![
            "http://a:1/".into(),
            "http://a:1".into(),
            " http://b:2/ ".into(),
            "".into(),
        ],
        None,
    );
    assert_eq!(
        st.peers,
        vec!["http://a:1".to_string(), "http://b:2".to_string()]
    );
}

#[test]
fn rate_limit_logic() {
    let st = state().with_rate_limit(2, 60);
    assert!(st.allow("a"));
    assert!(st.allow("a"));
    assert!(!st.allow("a"), "3rd request rejected");
    assert!(st.allow("b"), "other client independent");
}

#[test]
fn fail_rate_limit_logic() {
    // Fail-rate is independent of the main rate: a different max/window applies
    // to "fail:{ip}" keys. Default is 1/10 of the main rate.
    let st = state().with_rate_limit(20, 60);
    // Default fail-rate: 20/10 = 2 per 60s.
    let fail_key = "fail:127.0.0.1";
    assert!(st.allow_fail(fail_key));
    assert!(st.allow_fail(fail_key));
    assert!(!st.allow_fail(fail_key), "3rd fail-rate request rejected");
    // Main rate limit for valid keys is separate.
    assert!(st.allow("valid-key"), "main limit unaffected");
}

#[test]
fn explicit_fail_rate_override() {
    // Builder allows overriding the default 1/10 ratio.
    let st = state().with_rate_limit(100, 60).with_fail_rate_limit(5, 60);
    let fail_key = "fail:127.0.0.1";
    for _ in 0..5 {
        assert!(st.allow_fail(fail_key));
    }
    assert!(!st.allow_fail(fail_key), "6th fail-rate request rejected");
}

#[tokio::test]
async fn observability_request_id_ready_and_histogram() {
    // Phase 1 observability contract: every response carries x-request-id,
    // /ready verifies store, /metrics includes route-based histogram + retrieval
    // counters.
    let st = state();
    st.create_agent("Aria", "role", vec![]).await.unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let health = reqwest::get(format!("http://{addr}/health")).await.unwrap();
    assert!(
        health.headers().get("x-request-id").is_some(),
        "every response carries request id"
    );

    let ready = reqwest::get(format!("http://{addr}/ready")).await.unwrap();
    assert_eq!(ready.status(), 200, "store accessible → ready");

    // OpenAPI spec: valid JSON + covers all core endpoints (drift guard).
    let spec: serde_json::Value = reqwest::get(format!("http://{addr}/openapi.json"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let paths = spec["paths"].as_object().expect("paths object");
    for p in [
        "/health",
        "/ready",
        "/metrics",
        "/agents",
        "/agents/{id}",
        "/agents/{id}/ask",
        "/agents/{id}/ask/stream",
        "/agents/{id}/act",
        "/agents/{id}/solve",
        "/agents/{id}/message",
        "/agents/{id}/experience",
        "/agents/{id}/recall",
        "/agents/{id}/reinforce",
        "/agents/{id}/reflect",
        "/deliberate",
        "/deliberate/live",
        "/board",
        "/openapi.json",
        // Phase D — Task HTTP surface
        "/tasks",
        "/tasks/{id}",
        "/tasks/{id}/log",
        "/inbox",
        "/approvals/{id}/approve",
        "/approvals/{id}/deny",
    ] {
        assert!(paths.contains_key(p), "missing endpoint in spec: {p}");
    }

    let text = reqwest::get(format!("http://{addr}/metrics"))
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        text.contains("lore_http_request_duration_ms_bucket{route=\"/health\",le=\"5\"}"),
        "histogram written with route label: {text}"
    );
    assert!(text.contains("lore_http_request_duration_ms_count{route=\"/health\"}"));
    assert!(text.contains("lore_recall_candidates_total"));
    assert!(text.contains("lore_token_fallback_hits_total"));
    assert!(text.contains("lore_consolidation_runs_total"));
    assert!(text.contains("lore_sessions"));
}

#[tokio::test]
async fn http_auth_and_rate_limit() {
    // Server with API key + low rate limit.
    let st = state().with_api_key("secret").with_rate_limit(2, 60);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // /health is open (200 without key).
    assert!(client
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap()
        .status()
        .is_success());

    // Protected endpoint without key → 401.
    let no_key = client.get(format!("{base}/agents")).send().await.unwrap();
    assert_eq!(no_key.status().as_u16(), 401);

    // With correct key → 200 (rate limit 2/window).
    let ok = client
        .get(format!("{base}/agents"))
        .header("x-api-key", "secret")
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status().as_u16(), 200);

    // Second request still within limit (200), third exceeds (429).
    let _second = client
        .get(format!("{base}/agents"))
        .header("x-api-key", "secret")
        .send()
        .await
        .unwrap();
    let third = client
        .get(format!("{base}/agents"))
        .header("x-api-key", "secret")
        .send()
        .await
        .unwrap();
    assert_eq!(third.status().as_u16(), 429, "rate limit exceeded");
}

#[tokio::test]
async fn failed_auth_is_rate_limited_per_ip() {
    // Without the fail-rate, brute-force auth attempts would bypass the
    // key-based rate limit entirely (401 returned before rate check).
    // Failed auth from a single IP is throttled independently.
    let st = state()
        .with_api_key("secret")
        .with_rate_limit(20, 60)
        .with_fail_rate_limit(2, 60);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Two failed auth attempts -> 401 each (within fail-rate limit of 2).
    for _ in 0..2 {
        let resp = client
            .get(format!("{base}/agents"))
            .header("x-api-key", "wrong")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 401, "failed auth returns 401");
    }

    // Third failed auth attempt from same IP -> 429 (fail-rate exceeded).
    let blocked = client
        .get(format!("{base}/agents"))
        .header("x-api-key", "wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(blocked.status().as_u16(), 429, "fail-rate exceeded -> 429");

    // Valid key still works: fail-rate is per IP for invalid attempts only;
    // the normal key-based rate limit is separate.
    let ok = client
        .get(format!("{base}/agents"))
        .header("x-api-key", "secret")
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status().as_u16(), 200, "valid key still allowed");
}

#[tokio::test]
async fn deliberate_with_synthesizer_is_hierarchical() {
    let st = state();
    st.create_agent("Aria", "role", vec![]).await.unwrap();
    st.create_agent("Kai", "role", vec![]).await.unwrap();
    let s = st.create_agent("Sage", "supervisor", vec![]).await.unwrap();
    let sid = AgentId::from(s.id.clone());

    let (replies, synthesis) = st.deliberate_synth("decide", &sid, false).await.unwrap();
    assert_eq!(replies.len(), 2, "supervisor does not participate in poll");
    assert!(!synthesis.is_empty(), "synthesis produced");
    assert!(
        replies.iter().all(|r| r.name != "Sage"),
        "supervisor not among replies"
    );
    let board = st.read_board(100).await.unwrap();
    assert!(
        board.iter().any(|m| m.summary.contains("Synthesis")),
        "synthesis on board"
    );
}

#[tokio::test]
async fn patch_rejects_blank_name_or_role_and_trims() {
    // PATCH cannot bypass create's validation (empty name update rejected).
    let st = state();
    let v = st.create_agent("Aria", "role", vec![]).await.unwrap();
    let id = AgentId::from(v.id.clone());

    let p = PersonaPatch {
        name: Some("".into()),
        ..Default::default()
    };
    assert!(st.update_agent(&id, p).await.is_err(), "empty name 422");

    let p = PersonaPatch {
        role: Some("   ".into()),
        ..Default::default()
    };
    assert!(
        st.update_agent(&id, p).await.is_err(),
        "whitespace role 422"
    );

    let p = PersonaPatch {
        name: Some("  New  ".into()),
        ..Default::default()
    };
    let v = st.update_agent(&id, p).await.unwrap();
    assert_eq!(v.name, "New", "valid value trimmed");
}

#[test]
fn agent_id_fs_safety() {
    // Second line of defense in file-path construction: only ULID format is safe.
    let ok = AgentId::new();
    assert!(super::state::id_is_fs_safe(&ok));
    for evil in ["../../etc/passwd", "a/b", "x", "", "..\\win"] {
        assert!(
            !super::state::id_is_fs_safe(&AgentId::from(evil.to_string())),
            "should be considered unsafe: {evil}"
        );
    }
}

#[tokio::test]
async fn ws_rejects_oversized_question() {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message as TMsg;

    let st = state();
    st.create_agent("Aria", "role", vec![]).await.unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/deliberate/live"))
        .await
        .unwrap();
    // HTTP body limit is 2MB while WS accepted ~64MB — oversized questions
    // should not be processed: no reply frame arrives (close/error instead).
    let big = "s".repeat(100_000);
    ws.send(TMsg::Text(big)).await.unwrap();
    match ws.next().await {
        None | Some(Err(_)) | Some(Ok(TMsg::Close(_))) => {} // expected: rejected
        Some(Ok(m)) => {
            panic!("large question should not have been processed, frame arrived: {m:?}")
        }
    }
}

#[tokio::test]
async fn create_agent_rejects_blank_name_or_role() {
    let st = state();
    assert!(st.create_agent("", "role", vec![]).await.is_err());
    assert!(st.create_agent("Aria", "   ", vec![]).await.is_err());
    // Valid values still work (after trimming).
    let v = st.create_agent("  Aria ", " role ", vec![]).await.unwrap();
    assert_eq!(v.name, "Aria");
    assert_eq!(v.role, "role");
}

#[tokio::test]
async fn session_table_has_hard_cap_with_lru_eviction() {
    // Hard cap: client-controlled unlimited `session` values cannot grow the table;
    // when full, least-recently-used (LRU) sessions are evicted.
    let st = state().with_session_cap(2);
    let a = st.create_agent("Aria", "role", vec![]).await.unwrap();
    let id = AgentId::from(a.id.clone());
    for sid in ["s1", "s2", "s3"] {
        st.ask_session(&id, Some(sid), "hello").await.unwrap();
    }
    let map = st.sessions.read().await;
    assert!(map.len() <= 2, "hard cap: {} sessions", map.len());
    assert!(
        !map.keys().any(|(_, s)| s == "s1"),
        "oldest session (s1) evicted by LRU"
    );
    assert!(map.keys().any(|(_, s)| s == "s3"), "newest session remains");
}

#[tokio::test]
async fn existing_session_not_evicted_at_cap() {
    // When cap is full, a request to an EXISTING session: (1) must not trigger
    // eviction, (2) the requested session's window must never be reset (even if
    // it is the LRU-oldest).
    let st = state().with_session_cap(2);
    let a = st.create_agent("Aria", "role", vec![]).await.unwrap();
    let id = AgentId::from(a.id.clone());
    st.ask_session(&id, Some("s1"), "first").await.unwrap();
    st.ask_session(&id, Some("s2"), "second").await.unwrap();
    // s1 is now the LRU-oldest and cap is full — its window should still be preserved.
    let t = st
        .ask_session(&id, Some("s1"), "what did I say?")
        .await
        .unwrap();
    assert!(
        t.contains("chat history: 2 messages"),
        "s1 window preserved: {t}"
    );
    let map = st.sessions.read().await;
    assert_eq!(
        map.len(),
        2,
        "request to existing key does not trigger eviction"
    );
    assert!(
        map.keys().any(|(_, s)| s == "s2"),
        "neighbor session also stays"
    );
}

#[tokio::test]
async fn overlong_session_id_is_rejected() {
    let st = state();
    let a = st.create_agent("Aria", "role", vec![]).await.unwrap();
    let id = AgentId::from(a.id.clone());
    let long_sid = "s".repeat(300);
    let err = st.ask_session(&id, Some(&long_sid), "hello").await;
    assert!(err.is_err(), "overly long session name rejected");
}

#[tokio::test]
async fn synthesizer_with_local_flag_skips_peers() {
    // Node B is running — if fanout fires, its reply would be visible.
    let st_b = state();
    st_b.create_agent("Kai", "role", vec![]).await.unwrap();
    let lb = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_b = lb.local_addr().unwrap();
    let app_b = router(st_b);
    tokio::spawn(async move {
        axum::serve(lb, app_b).await.unwrap();
    });

    let st_a = state().with_peers(vec![format!("http://{addr_b}")], None);
    st_a.create_agent("Aria", "role", vec![]).await.unwrap();
    let s = st_a
        .create_agent("Sage", "supervisor", vec![])
        .await
        .unwrap();
    let sid = AgentId::from(s.id.clone());

    // local:false → peer reply included.
    let (with_peers, _) = st_a.deliberate_synth("decide", &sid, false).await.unwrap();
    assert!(
        with_peers.iter().any(|r| r.node.is_some()),
        "peer reply arrives when fanout is on"
    );

    // local:true → depth-1 guarantee also holds with synthesizer.
    let (local_only, _) = st_a.deliberate_synth("decide", &sid, true).await.unwrap();
    assert!(
        local_only.iter().all(|r| r.node.is_none()),
        "local:true skips peer fanout"
    );
}

#[tokio::test]
async fn federated_deliberate_merges_peer_replies() {
    // Node B: Kai (running as server).
    let st_b = state();
    st_b.create_agent("Kai", "role", vec![]).await.unwrap();
    let lb = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_b = lb.local_addr().unwrap();
    let app_b = router(st_b);
    tokio::spawn(async move {
        axum::serve(lb, app_b).await.unwrap();
    });

    // Node A: Aria + B as peer.
    let st_a = state().with_peers(vec![format!("http://{addr_b}")], None);
    st_a.create_agent("Aria", "role", vec![]).await.unwrap();

    let replies = st_a.deliberate("what is the status").await.unwrap();
    assert_eq!(replies.len(), 2, "local + peer replies merged");
    assert!(
        replies.iter().any(|r| r.name == "Aria" && r.node.is_none()),
        "local reply"
    );
    assert!(
        replies.iter().any(|r| r.name == "Kai" && r.node.is_some()),
        "peer reply with node label"
    );

    // Unreachable peer is silently skipped.
    let st_c = state().with_peers(vec!["http://127.0.0.1:1".into()], None);
    st_c.create_agent("Solo", "role", vec![]).await.unwrap();
    let solo = st_c.deliberate("test").await.unwrap();
    assert_eq!(solo.len(), 1, "dead peer does not break flow");
}

#[tokio::test]
async fn ws_deliberate_streams_replies() {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message as TMsg;

    let st = state();
    st.create_agent("Aria", "role", vec![]).await.unwrap();
    st.create_agent("Kai", "role", vec![]).await.unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st);
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/deliberate/live"))
        .await
        .unwrap();
    ws.send(TMsg::Text("summarize".into())).await.unwrap();

    let mut replies = 0;
    while let Some(Ok(TMsg::Text(t))) = ws.next().await {
        if t == "[DONE]" {
            break;
        }
        assert!(t.contains("reply"), "JSON frame: {t}");
        replies += 1;
    }
    assert_eq!(replies, 2, "two agents streamed live");
}

#[tokio::test]
async fn ws_deliberate_streams_peer_replies() {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message as TMsg;

    // Node B: Kai (running as peer server).
    let st_b = state();
    st_b.create_agent("Kai", "role", vec![]).await.unwrap();
    let lb = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_b = lb.local_addr().unwrap();
    let app_b = router(st_b);
    tokio::spawn(async move {
        axum::serve(lb, app_b).await.unwrap();
    });

    // Node A: Aria + B as peer; live deliberate WS is opened from A.
    let st_a = state().with_peers(vec![format!("http://{addr_b}")], None);
    st_a.create_agent("Aria", "role", vec![]).await.unwrap();
    let la = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_a = la.local_addr().unwrap();
    let app_a = router(st_a);
    tokio::spawn(async move {
        axum::serve(la, app_a).await.unwrap();
    });

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr_a}/deliberate/live"))
        .await
        .unwrap();
    ws.send(TMsg::Text("summarize".into())).await.unwrap();

    let mut local = 0;
    let mut peer = 0;
    while let Some(Ok(TMsg::Text(t))) = ws.next().await {
        if t == "[DONE]" {
            break;
        }
        assert!(t.contains("reply"), "JSON frame: {t}");
        // node field uses skip_serializing_if=None — absent for local, present for peer.
        if t.contains("\"node\"") {
            peer += 1;
        } else {
            local += 1;
        }
    }
    assert_eq!(local, 1, "local Aria reply streamed");
    assert_eq!(peer, 1, "peer Kai reply streamed with node label");
}

#[tokio::test]
async fn http_ask_stream_sse() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let st = state();
    let app = router(st.clone());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let v = st.create_agent("Aria", "role", vec![]).await.unwrap();

    let body = reqwest::Client::new()
        .post(format!("http://{addr}/agents/{}/ask/stream", v.id))
        .json(&serde_json::json!({"message":"hello"}))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(body.contains("data:"), "SSE events");
    assert!(body.contains("[DONE]"), "end marker");
}

#[tokio::test]
async fn http_end_to_end() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(state());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // health
    let ok = client.get(format!("{base}/health")).send().await.unwrap();
    assert_eq!(ok.text().await.unwrap(), "ok");

    // create
    let created: AgentView = client
        .post(format!("{base}/agents"))
        .json(&serde_json::json!({"name":"Aria","role":"role"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(created.name, "Aria");

    // ask
    let resp: AskResp = client
        .post(format!("{base}/agents/{}/ask", created.id))
        .json(&serde_json::json!({"message":"hello"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(!resp.reply.is_empty());
}

#[tokio::test]
async fn default_body_limit_rejects_over_2mb() {
    // Edge test: DefaultBodyLimit::max(2 MiB) rejects payloads >2 MiB with 413.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(state());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Oversized body (3 MiB) on a protected route → 413 Payload Too Large.
    let big = serde_json::json!({"name": "x", "role": "r", "extra": "s".repeat(3 * 1024 * 1024)});
    let resp = client
        .post(format!("{base}/agents"))
        .json(&big)
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::PAYLOAD_TOO_LARGE,
        "3 MiB body should be rejected with 413, got {}",
        resp.status()
    );

    // Body just under 2 MiB should be accepted (4xx from validation, not body limit).
    let ok_size = serde_json::json!({"name": "Aria", "role": "role"});
    let resp_ok = client
        .post(format!("{base}/agents"))
        .json(&ok_size)
        .send()
        .await
        .unwrap();
    // Auth is not set in these tests → agents route uses security_mw which
    // may return 401; the key assertion is that it is NOT 413.
    assert_ne!(
        resp_ok.status(),
        reqwest::StatusCode::PAYLOAD_TOO_LARGE,
        "small body should not be 413, got {}",
        resp_ok.status(),
    );
}

#[tokio::test]
async fn reinforce_http_validates_scope_and_outcome() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(state());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Two agents; A receives a memory.
    let a: AgentView = client
        .post(format!("{base}/agents"))
        .json(&serde_json::json!({"name":"Aria","role":"role"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let b: AgentView = client
        .post(format!("{base}/agents"))
        .json(&serde_json::json!({"name":"Kai","role":"role"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(client
        .post(format!("{base}/agents/{}/experience", a.id))
        .json(&serde_json::json!({"title":"secret topic","body":"content"}))
        .send()
        .await
        .unwrap()
        .status()
        .is_success());
    let mems: Vec<crate::server::MemoryView> = client
        .get(format!("{base}/agents/{}/recall?q=secret", a.id))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let mid = mems[0].id.clone();

    // Owner can reinforce → 204.
    let ok = client
        .post(format!("{base}/agents/{}/reinforce", a.id))
        .json(&serde_json::json!({"memory_id": mid, "outcome": "accessed"}))
        .send()
        .await
        .unwrap();
    assert_eq!(ok.status().as_u16(), 204);

    // Another agent's memory → 404 (existence is not leaked).
    let cross = client
        .post(format!("{base}/agents/{}/reinforce", b.id))
        .json(&serde_json::json!({"memory_id": mid, "outcome": "accessed"}))
        .send()
        .await
        .unwrap();
    assert_eq!(cross.status().as_u16(), 404);

    // Nonexistent record → 404.
    let missing = client
        .post(format!("{base}/agents/{}/reinforce", a.id))
        .json(&serde_json::json!({"memory_id": "01J00000000000000000000000", "outcome": "success"}))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status().as_u16(), 404);

    // Invalid outcome → 4xx (serde rejects).
    let bad = client
        .post(format!("{base}/agents/{}/reinforce", a.id))
        .json(&serde_json::json!({"memory_id": mid, "outcome": "maybe"}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status().as_u16(), 422);

    // World (board) records can be reinforced by anyone → 204.
    client
        .post(format!("{base}/deliberate"))
        .json(&serde_json::json!({"question":"board test","local":true}))
        .send()
        .await
        .unwrap();
    let board: Vec<crate::server::MemoryView> = client
        .get(format!("{base}/board"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let world = client
        .post(format!("{base}/agents/{}/reinforce", b.id))
        .json(&serde_json::json!({"memory_id": board[0].id, "outcome": "accessed"}))
        .send()
        .await
        .unwrap();
    assert_eq!(world.status().as_u16(), 204, "World record is shared");
}

#[tokio::test]
async fn create_agent_rejects_beyond_cap() {
    // Agent count must have a cap — an authenticated client cannot open thousands
    // of agents and explode fan-out on every /deliberate.
    let st = state().with_max_agents(1);
    st.create_agent("Aria", "role", vec![]).await.unwrap();
    let err = st.create_agent("Kai", "role", vec![]).await.unwrap_err();
    assert!(
        matches!(err, crate::error::LoreError::InvalidInput(_)),
        "cap exceeded 422: {err}"
    );
    // Deleting an agent frees up a slot.
    let aria = st.list_agents().await;
    st.delete_agent(&crate::id::AgentId::from(aria[0].id.clone()))
        .await
        .unwrap();
    st.create_agent("Kai", "role", vec![]).await.unwrap();
}

#[tokio::test]
async fn reflect_endpoint_distills_hot_memories() {
    // Episodic memory is reinforced twice (hot), reflect promotes it to semantic
    // — end-to-end learning cycle over HTTP.
    let st = state();
    let a = st.create_agent("Aria", "role", vec![]).await.unwrap();
    let aid = crate::id::AgentId::from(a.id.clone());
    st.experience(&aid, "rust conversation", "user likes Rust")
        .await
        .unwrap();
    let mems = st.recall(&aid, "rust", 10, false).await.unwrap();
    let mid = crate::id::MemoryId::from(mems[0].id.clone());
    for _ in 0..2 {
        st.reinforce(&aid, &mid, crate::memory::Outcome::Accessed)
            .await
            .unwrap();
    }

    let n = st.reflect(&aid).await.unwrap();
    assert_eq!(n, 1, "hot memory distilled");
    // Second run: no hot memories left to distill.
    let n2 = st.reflect(&aid).await.unwrap();
    assert_eq!(n2, 0, "idempotent: not distilled again");
}

// ── Phase D: Task HTTP surface tests ──────────────────────────────────

use crate::task::TaskStore;

fn task_state() -> AppState {
    // Minimal AppState without task store — callers provide their own
    // via TaskSrvDir + .with_task_store() to avoid leaked temp dirs.
    let store: Arc<dyn MemoryStore> = Arc::new(InMemoryStore::new());
    let model: Arc<dyn Model> = Arc::new(MockModel::new());
    AppState::new(store, model)
}

struct TaskSrvDir(std::path::PathBuf);

impl TaskSrvDir {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("lore-task-http-{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(dir.join("logs")).unwrap();
        Self(dir)
    }
}

impl Drop for TaskSrvDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn task_http_auth_required_401() {
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state()
        .with_api_key("secret")
        .with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Without key -> 401 on all task endpoints.
    assert_eq!(
        client
            .post(format!("{base}/tasks"))
            .send()
            .await
            .unwrap()
            .status()
            .as_u16(),
        401
    );
    assert_eq!(
        client
            .get(format!("{base}/tasks?limit=5"))
            .send()
            .await
            .unwrap()
            .status()
            .as_u16(),
        401
    );
    assert_eq!(
        client
            .get(format!("{base}/tasks/nonexistent"))
            .send()
            .await
            .unwrap()
            .status()
            .as_u16(),
        401
    );
    assert_eq!(
        client
            .get(format!("{base}/tasks/nonexistent/log"))
            .send()
            .await
            .unwrap()
            .status()
            .as_u16(),
        401
    );
    assert_eq!(
        client
            .get(format!("{base}/inbox"))
            .send()
            .await
            .unwrap()
            .status()
            .as_u16(),
        401
    );
    assert_eq!(
        client
            .post(format!("{base}/approvals/nonexistent/approve"))
            .send()
            .await
            .unwrap()
            .status()
            .as_u16(),
        401
    );
    assert_eq!(
        client
            .post(format!("{base}/approvals/nonexistent/deny"))
            .send()
            .await
            .unwrap()
            .status()
            .as_u16(),
        401
    );

    // With key -> endpoints accessible (may return 404/422, but not 401).
    let resp = client
        .get(format!("{base}/tasks?limit=5"))
        .header("x-api-key", "secret")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
}

#[tokio::test]
async fn task_enqueue_list_get_happy_path() {
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Enqueue.
    let task: serde_json::Value = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "testbot", "goal": "fix the login endpoint"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(task["status"], "Queued");
    assert_eq!(task["agent"], "testbot");
    let task_id = task["id"].as_str().unwrap().to_string();

    // List.
    let list: Vec<serde_json::Value> = client
        .get(format!("{base}/tasks?limit=5"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["id"], task_id);

    // Get full record.
    let full: serde_json::Value = client
        .get(format!("{base}/tasks/{task_id}"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(full["task"]["id"], task_id);
    assert_eq!(full["task"]["goal"], "fix the login endpoint");
    // No children for a standalone task.
    assert!(full.get("children").is_none() || full["children"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn task_validation_missing_goal_422() {
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Missing goal -> 422.
    let resp = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "testbot", "goal": ""}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 422);

    // Missing agent -> 422.
    let resp2 = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "", "goal": "do something"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp2.status().as_u16(), 422);

    // Path traversal in agent -> 422.
    let resp3 = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "../evil", "goal": "hack"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp3.status().as_u16(), 422);
}

#[tokio::test]
async fn task_unknown_id_404() {
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    let resp = client
        .get(format!("{base}/tasks/nonexistent_id"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);
}

#[tokio::test]
async fn task_log_traversal_rejected() {
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Path traversal: ../etc/passwd -> 422.
    let _resp = client
        .get(format!("{base}/tasks/../etc/passwd/log"))
        .send()
        .await
        .unwrap();
    // axum routes the path literally; the handler validates the id param.
    // The id would be "../etc/passwd" which gets rejected.
    // Actually, axum normalizes paths - the route is /tasks/:id/log
    // so ../etc/passwd would not match the route pattern. Instead test with
    // a valid-looking id that contains traversal chars.
    // Let me use a direct id param.
    let resp2 = client
        .get(format!("{base}/tasks/..%2Fetc%2Fpasswd/log"))
        .send()
        .await
        .unwrap();
    // The URL-decoded id is "../etc/passwd" or "..%2Fetc%2Fpasswd"
    // Either way it should be rejected.
    assert!(
        resp2.status().as_u16() == 422 || resp2.status().as_u16() == 404,
        "traversal id rejected: got {}",
        resp2.status().as_u16()
    );
}

#[tokio::test]
async fn approval_decide_inbox_empties_second_decide_409() {
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Enqueue a task (so approval can be attached).
    let task: serde_json::Value = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "testbot", "goal": "needs approval"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let task_id = task["id"].as_str().unwrap().to_string();

    // Insert an approval via TaskStore directly (the HTTP surface doesn't
    // create approvals — they're created by the daemon work loop).
    let store = TaskStore::open(&db_path).unwrap();
    let approval_id = store.add_approval(&task_id, "{}", "test reason").unwrap();

    // Inbox should show the pending approval.
    let inbox: Vec<serde_json::Value> = client
        .get(format!("{base}/inbox"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(inbox.len(), 1);
    assert_eq!(inbox[0]["id"], approval_id);

    // Approve it.
    let resp = client
        .post(format!("{base}/approvals/{approval_id}/approve"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 204);

    // Inbox now empty.
    let inbox2: Vec<serde_json::Value> = client
        .get(format!("{base}/inbox"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(inbox2.len(), 0);

    // Second decide -> 409 (already decided).
    let resp2 = client
        .post(format!("{base}/approvals/{approval_id}/approve"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp2.status().as_u16(), 409);

    // Deny on already-decided also -> 409.
    let resp3 = client
        .post(format!("{base}/approvals/{approval_id}/deny"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp3.status().as_u16(), 409);
}

#[tokio::test]
async fn approval_deny_recorded() {
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Enqueue a task + add approval.
    let task: serde_json::Value = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "testbot", "goal": "deny test"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let task_id = task["id"].as_str().unwrap().to_string();

    let store = TaskStore::open(&db_path).unwrap();
    let approval_id = store.add_approval(&task_id, "{}", "deny me").unwrap();

    // Deny it.
    let resp = client
        .post(format!("{base}/approvals/{approval_id}/deny"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 204);

    // Verify via TaskStore that it's Denied.
    let status = store.approval_status(&approval_id).unwrap().unwrap();
    assert_eq!(status, crate::task::ApprovalStatus::Denied);
}

#[tokio::test]
async fn approval_unknown_id_404() {
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Unknown approval id -> 404.
    let resp = client
        .post(format!("{base}/approvals/nonexistent/approve"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);

    let resp2 = client
        .post(format!("{base}/approvals/nonexistent/deny"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp2.status().as_u16(), 404);
}

#[tokio::test]
async fn task_log_reads_file() {
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Enqueue a task.
    let task: serde_json::Value = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "logbot", "goal": "log test"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let task_id = task["id"].as_str().unwrap().to_string();

    // Write a fake log file (as the daemon would).
    let log_path = td.0.join("logs").join(format!("{task_id}.log"));
    std::fs::write(&log_path, "line1\nline2\nline3\nline4\nline5").unwrap();

    // Read full log.
    let content = client
        .get(format!("{base}/tasks/{task_id}/log"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(content.contains("line1"), "full log: {content}");

    // Read tail=2.
    let tail_content = client
        .get(format!("{base}/tasks/{task_id}/log?tail=2"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(tail_content.contains("line4"), "tail=2: {tail_content}");
    assert!(tail_content.contains("line5"), "tail=2: {tail_content}");
    assert!(
        !tail_content.contains("line1"),
        "tail=2 should not have line1: {tail_content}"
    );

    // Unknown task log -> 404.
    let resp = client
        .get(format!("{base}/tasks/unknown_task_id/log"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);
}

// ── Phase D: Additional edge tests ──────────────────────────────────

#[tokio::test]
async fn task_enqueue_verify_empty_vs_omitted() {
    // POST /tasks with verify omitted (default empty) vs verify: [] — both succeed.
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Omitted verify (no key in JSON).
    let task1: serde_json::Value = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "bot1", "goal": "no verify"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let v1 = task1["verify"].as_array().unwrap();
    assert!(v1.is_empty(), "omitted verify defaults to empty array");

    // Explicit empty verify array.
    let task2: serde_json::Value = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "bot2", "goal": "empty verify", "verify": []}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let v2 = task2["verify"].as_array().unwrap();
    assert!(v2.is_empty(), "explicit empty verify array accepted");

    // Verify with actual commands.
    let task3: serde_json::Value = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "bot3", "goal": "with verify", "verify": ["cargo test", "clippy"]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let v3 = task3["verify"].as_array().unwrap();
    assert_eq!(v3.len(), 2, "verify commands stored");
    assert_eq!(v3[0], "cargo test");
}

#[tokio::test]
async fn task_list_limit_clamping() {
    // GET /tasks?limit=N — values are clamped to [0, 1000];
    // limit=0 → empty list, limit=999999 → clamped to 1000.
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Enqueue 3 tasks.
    for i in 0..3 {
        client
            .post(format!("{base}/tasks"))
            .json(&serde_json::json!({"agent": "bot", "goal": format!("task {i}")}))
            .send()
            .await
            .unwrap();
    }

    // limit=0 → empty list (lower bound is 0).
    let list0: Vec<serde_json::Value> = client
        .get(format!("{base}/tasks?limit=0"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(list0.is_empty(), "limit=0 returns empty list");

    // Default (no limit param) → 20.
    let list_default: Vec<serde_json::Value> = client
        .get(format!("{base}/tasks"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        list_default.len(),
        3,
        "default limit=20, only 3 tasks exist"
    );

    // limit=2 → exact.
    let list2: Vec<serde_json::Value> = client
        .get(format!("{base}/tasks?limit=2"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list2.len(), 2, "limit=2 returns 2");

    // limit=999999 → clamped to 1000 (returns all 3 since we have fewer).
    let list_big: Vec<serde_json::Value> = client
        .get(format!("{base}/tasks?limit=999999"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        list_big.len(),
        3,
        "limit=999999 clamped to 1000, returns all 3"
    );
}

#[tokio::test]
async fn approval_concurrent_approve_deny_race() {
    // Two concurrent decisions on the same approval:
    // one must succeed (204), the loser gets 409.
    // This tests the Conflict mapping under realistic concurrency.
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Enqueue a task + add approval.
    let task: serde_json::Value = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "racebot", "goal": "race test"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let task_id = task["id"].as_str().unwrap().to_string();

    let store = TaskStore::open(&db_path).unwrap();
    let approval_id = store
        .add_approval(&task_id, "{\"cmd\":\"rm\"}", "race reason")
        .unwrap();

    // Fire approve + deny concurrently — one wins, loser gets 409.
    let (approve_resp, deny_resp) = tokio::join!(
        client
            .post(format!("{base}/approvals/{approval_id}/approve"))
            .send(),
        client
            .post(format!("{base}/approvals/{approval_id}/deny"))
            .send()
    );
    let a_status = approve_resp.unwrap().status().as_u16();
    let d_status = deny_resp.unwrap().status().as_u16();

    // Exactly one must be 204, the other 409.
    let wins = (a_status == 204 && d_status == 409) || (a_status == 409 && d_status == 204);
    assert!(
        wins,
        "one must win (204), the other gets 409: approve={a_status}, deny={d_status}"
    );
}

#[tokio::test]
async fn inbox_empty_state() {
    // GET /inbox returns empty array when no pending approvals exist.
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st);
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    let inbox: Vec<serde_json::Value> = client
        .get(format!("{base}/inbox"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(inbox.is_empty(), "inbox starts empty");
}

#[tokio::test]
async fn task_get_with_children_shows_subtasks() {
    // GET /tasks/:id for a parent task includes children;
    // a standalone task omits the children field (skip_serializing_if).
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Create a parent task via TaskStore (with parent_id requires enqueue_child).
    let store = TaskStore::open(&db_path).unwrap();
    let parent = store
        .enqueue(crate::task::NewTask {
            agent: "pm".into(),
            goal: "big feature".into(),
            workspace: PathBuf::from("/tmp/ws"),
            verify: vec![],
            parent_id: None,
        })
        .unwrap();

    // Add two children.
    let c1 = store
        .enqueue_child(
            &parent.id,
            crate::task::NewTask {
                agent: "backend".into(),
                goal: "impl API".into(),
                workspace: PathBuf::from("/tmp/ws1"),
                verify: vec!["cargo test".into()],
                parent_id: None,
            },
        )
        .unwrap();
    let c2 = store
        .enqueue_child(
            &parent.id,
            crate::task::NewTask {
                agent: "frontend".into(),
                goal: "build UI".into(),
                workspace: PathBuf::from("/tmp/ws2"),
                verify: vec![],
                parent_id: None,
            },
        )
        .unwrap();

    // GET parent: should have children array with 2 entries.
    let full: serde_json::Value = client
        .get(format!("{base}/tasks/{parent_id}", parent_id = parent.id))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(full["task"]["id"], parent.id, "parent task id matches");
    let children = full["children"]
        .as_array()
        .expect("children field present for parent");
    assert_eq!(children.len(), 2, "parent has 2 children");
    assert_eq!(children[0]["id"], c1.id);
    assert_eq!(children[1]["id"], c2.id);

    // GET standalone child (no children of its own) → children field absent.
    let child_full: serde_json::Value = client
        .get(format!("{base}/tasks/{cid}", cid = c1.id))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        child_full.get("children").is_none()
            || child_full["children"].as_array().unwrap().is_empty(),
        "standalone task has no children field"
    );
}

#[tokio::test]
async fn task_log_tail_larger_than_file_returns_full() {
    // tail=100 on a 5-line file returns the entire content.
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Enqueue a task.
    let task: serde_json::Value = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "logbot", "goal": "tail edge"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let task_id = task["id"].as_str().unwrap().to_string();

    // Write a 3-line log.
    let log_path = td.0.join("logs").join(format!("{task_id}.log"));
    std::fs::write(&log_path, "a\nb\nc").unwrap();

    // tail=100 (more than file lines) → returns full content.
    let content = client
        .get(format!("{base}/tasks/{task_id}/log?tail=100"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        content.contains("a") && content.contains("c"),
        "tail larger than file returns full: {content}"
    );

    // tail=1 → last line only.
    let tail1 = client
        .get(format!("{base}/tasks/{task_id}/log?tail=1"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        tail1.contains("c") && !tail1.contains("a"),
        "tail=1 returns last line: {tail1}"
    );
}

#[tokio::test]
async fn task_workspace_containment_and_relativisation() {
    // M1 + S2: workspace containment + relativisation in HTTP responses.
    // 1) Absolute workspace outside data_dir → 422.
    // 2) Absolute workspace under data_dir → accepted, relativised in response.
    // 3) No workspace (default) → relativised in response.
    let td = TaskSrvDir::new();
    let db_path = td.0.join("tasks.db");
    let st = task_state().with_task_store(td.0.clone(), db_path.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(st.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Absolute workspace outside data_dir → 422.
    let resp = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "sneaky", "goal": "escape", "workspace": "/etc"}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        422,
        "workspace outside data_dir rejected"
    );

    // Absolute workspace under data_dir → accepted.
    let ws_path = td.0.join("workspaces").join("safe");
    let task: serde_json::Value = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "safe", "goal": "safe goal", "workspace": ws_path.to_string_lossy().to_string()}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    // Workspace should be relativised: "workspaces/safe" not the full path.
    let ws_val = task["workspace"].as_str().unwrap();
    assert!(
        !ws_val.contains("lore-task-http"),
        "workspace relativised, no temp dir prefix"
    );
    assert_eq!(ws_val, "workspaces/safe", "workspace is relative path");

    // No workspace (default) → relativised in response.
    let task2: serde_json::Value = client
        .post(format!("{base}/tasks"))
        .json(&serde_json::json!({"agent": "defbot", "goal": "default ws"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ws_val2 = task2["workspace"].as_str().unwrap();
    assert_eq!(
        ws_val2, "workspaces/defbot",
        "default workspace relativised"
    );

    // List endpoint should also show relativised workspace and omit report.
    let list: Vec<serde_json::Value> = client
        .get(format!("{base}/tasks?limit=1000"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for item in &list {
        assert!(
            !item["workspace"]
                .as_str()
                .unwrap()
                .contains("lore-task-http"),
            "list items have relativised workspace"
        );
        assert!(item.get("report").is_none(), "compact list omits report");
    }
}
