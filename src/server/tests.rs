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
        "/deliberate",
        "/deliberate/live",
        "/board",
        "/openapi.json",
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
