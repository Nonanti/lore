//! HTTP layer: router, `serve()`, and thin handler wrappers.
//!
//! Handlers contain no business logic — they call `AppState` methods and map
//! results to HTTP. Error mapping is centralized in `ApiError`.

use super::security::security_mw;
use super::state::AppState;
use super::types::{
    ActReq, ActResp, AgentView, AskReq, AskResp, BoardParams, CreateReq, DeliberateReply,
    DeliberateReq, DeliberateResp, EnqueueTaskReq, ExperienceReq, MemoryView, MessageReq, MsgKind,
    PersonaPatch, RecallParams, ReflectResp, ReinforceReq, SolveReq, TaskListParams, TaskLogParams,
};
use crate::agent::DEFAULT_SOLVE_STEPS;
use crate::error::{LoreError, Result};
use crate::id::AgentId;

use axum::{
    extract::ws::{Message as WsMsg, WebSocket, WebSocketUpgrade},
    extract::{DefaultBodyLimit, MatchedPath, Path, Query as AxQuery, Request, State},
    http::{HeaderValue, StatusCode},
    middleware::{self, Next},
    response::sse::{Event, Sse},
    response::{IntoResponse, Response},
    routing::{get, patch, post},
    Json, Router,
};
use futures::stream::{self, Stream, StreamExt};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// Default interval for periodic memory consolidation
/// (configurable via `LORE_CONSOLIDATE_SECS` env).
const CONSOLIDATE_PERIOD_SECS: u64 = 3600;

/// Periodic reflect (episodic → semantic distillation) interval; 0 = disabled.
const REFLECT_PERIOD_SECS: u64 = 3600;

/// WS deliberate: timeout for the initial question frame — prevents idle
/// connections from holding resources indefinitely.
const WS_QUESTION_TIMEOUT: Duration = Duration::from_secs(30);

/// Upper bound for query `limit` (client-controlled field — excessive values are clamped).
pub(super) const MAX_QUERY_LIMIT: usize = 1000;

/// WS message/frame size limit — HTTP bodies are capped at 2MB, but WS's
/// default ~64MB acceptance was an inconsistent DoS surface; 64KB is ample for a question.
const WS_MAX_MESSAGE_BYTES: usize = 64 * 1024;

/// Sets up the Lore HTTP router. `/health` is open; other endpoints go through auth + rate-limit middleware.
pub fn router(state: AppState) -> Router {
    let protected = Router::new()
        .route("/agents", post(create_h).get(list_h))
        .route("/agents/:id", patch(patch_h).delete(delete_h))
        .route("/agents/:id/ask", post(ask_h))
        .route("/agents/:id/ask/stream", post(ask_stream_h))
        .route("/agents/:id/act", post(act_h))
        .route("/agents/:id/solve", post(solve_h))
        .route("/agents/:id/message", post(message_h))
        .route("/agents/:id/experience", post(exp_h))
        .route("/agents/:id/reinforce", post(reinforce_h))
        .route("/agents/:id/reflect", post(reflect_h))
        .route("/agents/:id/recall", get(recall_h))
        .route("/deliberate", post(deliberate_h))
        .route("/deliberate/live", get(deliberate_ws_h))
        .route("/board", get(board_h))
        // Metrics are observability data — protected when API key is configured.
        .route("/metrics", get(metrics_h))
        // Task queue HTTP surface (Phase D).
        .route("/tasks", post(enqueue_task_h).get(list_tasks_h))
        .route("/tasks/:id", get(get_task_h))
        .route("/tasks/:id/log", get(task_log_h))
        .route("/inbox", get(inbox_h))
        .route("/approvals/:id/approve", post(approve_h))
        .route("/approvals/:id/deny", post(deny_h))
        .route_layer(middleware::from_fn_with_state(state.clone(), security_mw))
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024)); // 2 MiB

    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready_h))
        .route("/openapi.json", get(openapi_h))
        .merge(protected)
        .layer(middleware::from_fn_with_state(state.clone(), request_mw))
        .with_state(state)
}

/// Runs the HTTP service at the given address (blocking; shuts down gracefully on ctrl-c).
pub async fn serve(addr: &str, state: AppState) -> Result<()> {
    // Memory maintenance: periodic consolidation (merge + decay) in the background —
    // without it, memory grows unboundedly.
    // Zero/invalid values are rejected: `interval(Duration::ZERO)` would panic and
    // the janitor task would silently die.
    let period = match std::env::var("LORE_CONSOLIDATE_SECS") {
        Ok(s) => match s.parse::<u64>() {
            Ok(n) if n > 0 => n,
            _ => {
                tracing::warn!(value = %s, "LORE_CONSOLIDATE_SECS invalid, using default");
                CONSOLIDATE_PERIOD_SECS
            }
        },
        Err(_) => CONSOLIDATE_PERIOD_SECS,
    };
    let janitor = crate::memory::evolution::spawn_periodic(
        state.inner.store.clone(),
        Duration::from_secs(period),
    );
    // Autonomous reflection (learning loop): periodic reflect — frequently recalled
    // episodic memories are distilled by the model and promoted to the semantic tier.
    // The agent's memory matures even without calls. `LORE_REFLECT_SECS=0` disables it.
    let reflect_secs = match std::env::var("LORE_REFLECT_SECS") {
        Ok(s) => s.parse::<u64>().unwrap_or_else(|_| {
            tracing::warn!(value = %s, "LORE_REFLECT_SECS invalid, using default");
            REFLECT_PERIOD_SECS
        }),
        Err(_) => REFLECT_PERIOD_SECS,
    };
    let reflect_task = if reflect_secs > 0 {
        let st = state.clone();
        Some(tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(reflect_secs));
            loop {
                tick.tick().await;
                for (id, agent) in st.team().await {
                    if let Err(e) = agent.reflect().await {
                        tracing::warn!(agent = %id, error = %e, "periodic reflect error");
                    }
                }
            }
        }))
    } else {
        tracing::info!("periodic reflect disabled (LORE_REFLECT_SECS=0)");
        None
    };
    // Security posture: listening without a key on a non-loopback address must be
    // a conscious choice — silence implies an accidentally exposed service.
    if state.api_key.is_none()
        && !(addr.starts_with("127.") || addr.starts_with("localhost") || addr.starts_with("[::1]"))
    {
        tracing::warn!(
            %addr,
            "LORE_API_KEY not set and address is not loopback — API LISTENING WITHOUT AUTHENTICATION"
        );
    }
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| LoreError::Server(format!("could not bind to {addr}: {e}")))?;
    let served = axum::serve(
        listener,
        router(state).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
        println!("\n👋 shutdown signal received, stopping server");
    })
    .await
    .map_err(|e| LoreError::Server(e.to_string()));
    janitor.abort();
    if let Some(t) = reflect_task {
        t.abort();
    }
    served?;
    Ok(())
}

/// Request metadata middleware (all endpoints): increments the counter, generates a
/// unique request ID (`x-request-id` response header + span field in all logs), feeds
/// route-based latency histograms, and writes a structured completion log.
/// Route label is a template (`/agents/:id/ask`) — cardinality does not explode;
/// unmatched (404) requests are grouped under a single "(unmatched)" label.
async fn request_mw(State(st): State<AppState>, req: Request, next: Next) -> Response {
    use tracing::Instrument;

    st.inner.requests.fetch_add(1, Ordering::Relaxed);
    let rid = ulid::Ulid::new().to_string();
    let method = req.method().clone();
    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| "(unmatched)".into());
    let start = std::time::Instant::now();

    let span = tracing::info_span!("http", %rid, %method, %route);
    let mut resp = next.run(req).instrument(span).await;

    let ms = start.elapsed().as_millis() as u64;
    st.record_latency(&route, ms);
    tracing::info!(
        %rid,
        %method,
        %route,
        status = resp.status().as_u16(),
        ms,
        "request completed"
    );
    if let Ok(v) = HeaderValue::from_str(&rid) {
        resp.headers_mut().insert("x-request-id", v);
    }
    resp
}

// --- Handlers (thin wrappers) ---

async fn health() -> &'static str {
    "ok"
}

/// OpenAPI 3.1 specification — embedded at compile time, hand-maintained
/// (`openapi.json`). Open endpoint: tools (Swagger UI, code generators) can access
/// without a key; the spec contains no secrets.
async fn openapi_h() -> Response {
    (
        [("content-type", "application/json")],
        include_str!("../../openapi.json"),
    )
        .into_response()
}

/// Readiness check: unlike liveness, verifies store accessibility.
async fn ready_h(State(st): State<AppState>) -> Response {
    match st.ready().await {
        Ok(()) => (StatusCode::OK, "ready").into_response(),
        Err(e) => {
            tracing::error!(error = %e, "readiness: store unreachable");
            (StatusCode::SERVICE_UNAVAILABLE, "store unreachable").into_response()
        }
    }
}

async fn metrics_h(State(st): State<AppState>) -> String {
    st.metrics_text().await
}

async fn create_h(
    State(st): State<AppState>,
    Json(req): Json<CreateReq>,
) -> std::result::Result<Json<AgentView>, ApiError> {
    Ok(Json(
        st.create_agent(&req.name, &req.role, req.traits).await?,
    ))
}

async fn list_h(State(st): State<AppState>) -> Json<Vec<AgentView>> {
    Json(st.list_agents().await)
}

async fn patch_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
    Json(patch): Json<PersonaPatch>,
) -> std::result::Result<Json<AgentView>, ApiError> {
    Ok(Json(st.update_agent(&AgentId::from(id), patch).await?))
}

async fn delete_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> std::result::Result<StatusCode, ApiError> {
    st.delete_agent(&AgentId::from(id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn ask_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<AskReq>,
) -> std::result::Result<Json<AskResp>, ApiError> {
    let reply = st
        .ask_session(&AgentId::from(id), req.session.as_deref(), &req.message)
        .await?;
    Ok(Json(AskResp { reply }))
}

/// Streams the response as real-time SSE: if the model supports streaming (OpenAI
/// `stream:true`), tokens arrive as they are generated; otherwise, a single chunk.
/// `session` preserves conversation history. Ends with `[DONE]`; errors are emitted
/// as `event: error`.
async fn ask_stream_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<AskReq>,
) -> std::result::Result<Sse<impl Stream<Item = std::result::Result<Event, Infallible>>>, ApiError>
{
    let chunks = st
        .ask_stream(&AgentId::from(id), req.session.as_deref(), &req.message)
        .await?;
    let events = chunks
        .map(|r| match r {
            // \r cannot be carried in SSE (axum panics) — stripped out.
            Ok(t) => {
                // \r cannot be carried in SSE (axum panics) — stripped only when
                // present to avoid allocating a new String on every chunk (hot path).
                let data = if t.contains('\r') {
                    t.replace('\r', "")
                } else {
                    t
                };
                Event::default().data(data)
            }
            Err(e) => {
                tracing::error!(error = %e, "stream error");
                Event::default().event("error").data("stream interrupted")
            }
        })
        .chain(stream::once(async { Event::default().data("[DONE]") }))
        .map(Ok::<_, Infallible>);
    Ok(Sse::new(events))
}

async fn act_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ActReq>,
) -> std::result::Result<Json<ActResp>, ApiError> {
    Ok(Json(ActResp {
        result: st.act(&AgentId::from(id), &req.input).await?,
    }))
}

/// Multi-step tool loop (ReAct): tools are chained, final response returned.
async fn solve_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<SolveReq>,
) -> std::result::Result<Json<ActResp>, ApiError> {
    let steps = req.max_steps.unwrap_or(DEFAULT_SOLVE_STEPS);
    Ok(Json(ActResp {
        result: st.solve(&AgentId::from(id), &req.input, steps).await?,
    }))
}

/// Inter-agent message: `ask` returns a reply (JSON), `tell` returns `204 No Content`.
async fn message_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<MessageReq>,
) -> std::result::Result<Response, ApiError> {
    let ask = matches!(req.kind, Some(MsgKind::Ask));
    let from = req.from.map(AgentId::from);
    let reply = st
        .message(&AgentId::from(id), from.as_ref(), ask, &req.content)
        .await?;
    if ask {
        Ok(Json(AskResp { reply }).into_response())
    } else {
        Ok(StatusCode::NO_CONTENT.into_response())
    }
}

async fn exp_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ExperienceReq>,
) -> std::result::Result<StatusCode, ApiError> {
    st.experience(&AgentId::from(id), &req.title, &req.body)
        .await?;
    Ok(StatusCode::CREATED)
}

/// Memory reinforcement (decay/Wilson feed): scope validation is in state.
async fn reinforce_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ReinforceReq>,
) -> std::result::Result<StatusCode, ApiError> {
    st.reinforce(
        &AgentId::from(id),
        &crate::id::MemoryId::from(req.memory_id),
        req.outcome.into(),
    )
    .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Reflection: distills frequently recalled episodic memories into the semantic tier.
async fn reflect_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> std::result::Result<Json<ReflectResp>, ApiError> {
    let distilled = st.reflect(&AgentId::from(id)).await?;
    Ok(Json(ReflectResp { distilled }))
}

async fn recall_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
    AxQuery(p): AxQuery<RecallParams>,
) -> std::result::Result<Json<Vec<MemoryView>>, ApiError> {
    let q = p.q.unwrap_or_default();
    let limit = p.limit.unwrap_or(10).min(MAX_QUERY_LIMIT);
    let semantic = p.semantic.unwrap_or(false);
    Ok(Json(
        st.recall(&AgentId::from(id), &q, limit, semantic).await?,
    ))
}

async fn deliberate_h(
    State(st): State<AppState>,
    Json(req): Json<DeliberateReq>,
) -> std::result::Result<Json<DeliberateResp>, ApiError> {
    if let Some(s) = &req.synthesizer {
        // The `local` contract also applies with a synthesizer: peer fanout is skipped.
        let (replies, synthesis) = st
            .deliberate_synth(&req.question, &AgentId::from(s.clone()), req.local)
            .await?;
        return Ok(Json(DeliberateResp {
            replies,
            synthesis: Some(synthesis),
        }));
    }
    let replies = if req.local {
        st.deliberate_local(&req.question).await?
    } else {
        st.deliberate(&req.question).await?
    };
    Ok(Json(DeliberateResp {
        replies,
        synthesis: None,
    }))
}

/// WebSocket live deliberate: the client sends the question as a single text frame;
/// each agent reply streams as a JSON frame as soon as ready, ending with `[DONE]`.
async fn deliberate_ws_h(State(st): State<AppState>, ws: WebSocketUpgrade) -> Response {
    ws.max_message_size(WS_MAX_MESSAGE_BYTES)
        .max_frame_size(WS_MAX_MESSAGE_BYTES)
        .on_upgrade(move |socket| deliberate_ws(st, socket))
}

async fn deliberate_ws(st: AppState, mut socket: WebSocket) {
    use futures::stream::FuturesUnordered;
    use futures::StreamExt;

    // First text frame = question (with timeout — idle connections are not held indefinitely).
    let question = match tokio::time::timeout(WS_QUESTION_TIMEOUT, socket.recv()).await {
        Ok(Some(Ok(WsMsg::Text(q)))) => q,
        _ => return,
    };
    if let Err(e) = st.board_note("Question", question.clone()).await {
        // Inform the client why the connection is closing (silent drops are
        // hard to debug from the client side).
        tracing::warn!(error = %e, "ws deliberate: board note could not be written");
        let err_frame = serde_json::json!({"error": "board note failed"}).to_string();
        let _ = socket.send(WsMsg::Text(err_frame)).await;
        return;
    }
    // Replies are collected IN PARALLEL and streamed as soon as ready — serial waiting
    // would stack per-agent model latency (equivalent to poll_team).
    let mut futs: FuturesUnordered<_> = st
        .team()
        .await
        .into_iter()
        .map(|(id, agent)| {
            let q = question.clone();
            async move {
                let res = agent.respond(&q).await;
                (id, agent.persona.name.clone(), res)
            }
        })
        .collect();
    while let Some((id, name, res)) = futs.next().await {
        let reply = match res {
            Ok(r) => r,
            Err(e) => {
                // A failing agent does not halt the stream, but is not silently swallowed either.
                tracing::warn!(
                    agent = %super::log_safe(&name),
                    error = %e,
                    "ws deliberate: no response"
                );
                continue;
            }
        };
        let _ = st
            .board_note(format!("{name} response"), reply.clone())
            .await;
        let frame = serde_json::to_string(&DeliberateReply {
            id: id.to_string(),
            name,
            reply,
            node: None,
        })
        .unwrap_or_default();
        if socket.send(WsMsg::Text(frame)).await.is_err() {
            return; // client disconnected
        }
    }
    // Federation: stream peer node replies live too (with node label) — same scope
    // as HTTP /deliberate. peer_fanout also writes to the local board.
    for reply in st.peer_fanout(&question).await {
        let frame = serde_json::to_string(&reply).unwrap_or_default();
        if socket.send(WsMsg::Text(frame)).await.is_err() {
            return; // client disconnected
        }
    }
    let _ = socket.send(WsMsg::Text("[DONE]".into())).await;
}

async fn board_h(
    State(st): State<AppState>,
    AxQuery(p): AxQuery<BoardParams>,
) -> std::result::Result<Json<Vec<MemoryView>>, ApiError> {
    Ok(Json(
        st.read_board(p.limit.unwrap_or(20).min(MAX_QUERY_LIMIT))
            .await?,
    ))
}

// ── Task queue handlers ──────────────────────────────────────────────

async fn enqueue_task_h(
    State(st): State<AppState>,
    Json(req): Json<EnqueueTaskReq>,
) -> std::result::Result<(StatusCode, Json<crate::task::Task>), ApiError> {
    let workspace = req.workspace.map(std::path::PathBuf::from);
    let task = st.enqueue_task(&req.agent, &req.goal, workspace, req.verify)?;
    Ok((StatusCode::CREATED, Json(task)))
}

async fn list_tasks_h(
    State(st): State<AppState>,
    AxQuery(p): AxQuery<TaskListParams>,
) -> std::result::Result<Json<Vec<crate::task::Task>>, ApiError> {
    let limit = p.limit.unwrap_or(20);
    Ok(Json(st.list_tasks(limit)?))
}

async fn get_task_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> std::result::Result<Json<super::types::TaskFullView>, ApiError> {
    Ok(Json(st.get_task_full(&id)?))
}

async fn task_log_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
    AxQuery(p): AxQuery<TaskLogParams>,
) -> std::result::Result<String, ApiError> {
    Ok(st.read_task_log(&id, p.tail)?)
}

async fn inbox_h(
    State(st): State<AppState>,
) -> std::result::Result<Json<Vec<crate::task::ApprovalEntry>>, ApiError> {
    Ok(Json(st.pending_approvals()?))
}

async fn approve_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> std::result::Result<StatusCode, ApiError> {
    st.decide_approval(&id, true)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn deny_h(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> std::result::Result<StatusCode, ApiError> {
    st.decide_approval(&id, false)?;
    Ok(StatusCode::NO_CONTENT)
}

/// HTTP error wrapper: `LoreError` → appropriate status code.
struct ApiError(LoreError);

impl From<LoreError> for ApiError {
    fn from(e: LoreError) -> Self {
        Self(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (code, msg) = match &self.0 {
            LoreError::NotFound(m) => (StatusCode::NOT_FOUND, m.clone()),
            LoreError::InvalidInput(m) => (StatusCode::UNPROCESSABLE_ENTITY, m.clone()),
            LoreError::Conflict(m) => (StatusCode::CONFLICT, m.clone()),
            other => {
                // Internal details are not leaked to the client; logged server-side.
                tracing::error!(error = %other, "internal error (500)");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal error".to_string(),
                )
            }
        };
        (code, Json(serde_json::json!({ "error": msg }))).into_response()
    }
}
