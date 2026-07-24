//! `AppState`: shared application state and core business logic.
//!
//! All logic lives here in axum-independent async methods — offline-testable.
//! Collective deliberation (deliberate/federation) is in `deliberate.rs`,
//! the HTTP layer is in `api.rs`.

use super::types::{AgentView, CompactTaskView, MemoryView, PersonaPatch, TaskFullView, TaskView};
use crate::agent::{Agent, Conversation, Persona};
use crate::error::{LoreError, Result};
use crate::id::AgentId;
use crate::memory::{Memory, MemoryStore, Query, Scope};
use crate::model::{Model, TokenStream};
use crate::tool::ToolContext;
use futures::StreamExt;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex as AsyncMutex, RwLock};

/// When the rate-limit table exceeds this size, expired windows are evicted
/// (records are kept per key, so this prevents unbounded growth — DoS mitigation).
const HITS_EVICT_THRESHOLD: usize = 10_000;

/// Timeout for requests to peer nodes (seconds).
const PEER_TIMEOUT_SECS: u64 = 10;

/// Maximum concurrent sessions — HARD cap (DoS mitigation): when exceeded, idle
/// sessions are evicted first, then least-recently-used (LRU).
const MAX_SESSIONS: usize = 1000;

/// Hard cap on agent count (default; configurable via `with_max_agents`/`LORE_MAX_AGENTS`).
/// Without a cap, an authenticated client could open unlimited agents, exploding
/// fan-out + board writes on every `/deliberate` (DoS).
const MAX_AGENTS: usize = 1024;

/// Sessions idle for this duration become eviction candidates when the table is full.
const SESSION_IDLE_TTL: Duration = Duration::from_secs(3600);

/// Session name length limit (client-controlled field — prevents key bloat).
const MAX_SESSION_ID_LEN: usize = 128;

/// Is the identity safe as a file name? (ULID: 26 alphanumeric characters.)
/// All IDs are server-generated ULIDs and map membership already guards them —
/// this is a SECOND line of defense in file-path construction (path traversal).
pub(super) fn id_is_fs_safe(id: &AgentId) -> bool {
    let s = id.to_string();
    s.len() == 26 && s.bytes().all(|b| b.is_ascii_alphanumeric())
}

/// Makes the ID in error messages log-safe (newline sanitization).
fn log_safe_id(id: &AgentId) -> String {
    super::log_safe(&id.to_string())
}

/// Is the peer URL plain-http AND non-loopback? (key is sent in cleartext → warning)
pub(crate) fn is_insecure_peer(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("http://") else {
        return false; // https or no scheme — not the concern of this check
    };
    // Strip userinfo, extract host: IPv6 `[::1]`, IPv4/name `host[:port][/path]`.
    let after_auth = rest.rsplit('@').next().unwrap_or(rest);
    let host = if let Some(end) = after_auth.strip_prefix('[').and_then(|s| s.find(']')) {
        &after_auth[1..end + 1] // Inside brackets (e.g. ::1)
    } else {
        after_auth.split(['/', ':']).next().unwrap_or("")
    };
    let host = host.to_ascii_lowercase();
    !matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1")
}

#[cfg(test)]
mod peer_tests {
    #[test]
    fn insecure_peer_only_plain_http_and_remote() {
        assert!(super::is_insecure_peer("http://remote.server:3777"));
        assert!(super::is_insecure_peer("http://192.168.1.10:3777"));
        assert!(!super::is_insecure_peer("http://127.0.0.1:3777"));
        assert!(!super::is_insecure_peer("http://localhost:3777"));
        assert!(!super::is_insecure_peer("http://LOCALHOST:3777"));
        assert!(!super::is_insecure_peer("http://[::1]:3777"));
        assert!(super::is_insecure_peer("http://[2001:db8::1]:3777"));
        assert!(!super::is_insecure_peer("https://remote.server:3777"));
    }
}

#[cfg(test)]
mod rate_limit_tests {
    use super::*;
    use crate::memory::InMemoryStore;
    use crate::model::MockModel;

    fn test_state() -> AppState {
        AppState::new(Arc::new(InMemoryStore::new()), Arc::new(MockModel::new()))
    }

    #[test]
    fn cross_eviction_does_not_kill_live_fail_window() {
        // Fix 3: fail_rate.per > rate.per — eviction must not drop a live
        // fail-rate window just because rate.per is shorter.
        let st = test_state()
            .with_rate_limit(100, 10) // 100/10s
            .with_fail_rate_limit(5, 60); // 5/60s

        // Fill the hits map above HITS_EVICT_THRESHOLD with expired main-rate
        // entries so eviction triggers.
        {
            let mut hits = st.hits.lock().unwrap();
            for i in 0..HITS_EVICT_THRESHOLD + 1 {
                hits.insert(
                    format!("filler-{i}"),
                    Window {
                        start: Instant::now() - Duration::from_secs(120),
                        count: 1,
                    },
                );
            }
            // A live fail-rate window (created 30s ago — within fail_rate.per=60s
            // but outside rate.per=10s).
            hits.insert(
                "fail:attacker".into(),
                Window {
                    start: Instant::now() - Duration::from_secs(30),
                    count: 4,
                },
            );
        }

        // Trigger eviction via allow_with (main rate).
        let rl = st.rate.unwrap();
        st.allow_with("new-client", &rl);

        // The fail-rate window must survive (not evicted by rate.per=10s).
        let hits = st.hits.lock().unwrap();
        assert!(
            hits.contains_key("fail:attacker"),
            "fail-rate window must survive cross-eviction"
        );
    }
}

/// Latency histogram buckets (ms) — Prometheus `le` boundaries.
const LATENCY_BUCKETS_MS: [u64; 10] = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000];

/// Cumulative latency histogram (Prometheus semantics: each bucket counts
/// toward all `le` boundaries greater than or equal to the value).
#[derive(Clone, Default)]
pub(super) struct Histogram {
    buckets: [u64; LATENCY_BUCKETS_MS.len()],
    sum_ms: u64,
    count: u64,
}

impl Histogram {
    fn record(&mut self, ms: u64) {
        for (i, b) in LATENCY_BUCKETS_MS.iter().enumerate() {
            if ms <= *b {
                self.buckets[i] += 1;
            }
        }
        self.sum_ms += ms;
        self.count += 1;
    }

    fn render_into(&self, out: &mut String, route: &str) {
        use std::fmt::Write;
        for (i, b) in LATENCY_BUCKETS_MS.iter().enumerate() {
            let _ = writeln!(
                out,
                "lore_http_request_duration_ms_bucket{{route=\"{route}\",le=\"{b}\"}} {}",
                self.buckets[i]
            );
        }
        let _ = writeln!(
            out,
            "lore_http_request_duration_ms_bucket{{route=\"{route}\",le=\"+Inf\"}} {}",
            self.count
        );
        let _ = writeln!(
            out,
            "lore_http_request_duration_ms_sum{{route=\"{route}\"}} {}",
            self.sum_ms
        );
        let _ = writeln!(
            out,
            "lore_http_request_duration_ms_count{{route=\"{route}\"}} {}",
            self.count
        );
    }
}

/// Shared application state (cheaply cloneable — inner is `Arc`).
#[derive(Clone)]
pub struct AppState {
    pub(super) inner: Arc<Inner>,
    /// Platform-level shared tools (used by all agents' `act`).
    pub(super) tools: Option<Arc<ToolContext>>,
    /// If set, requests must carry this API key (otherwise open).
    pub(super) api_key: Option<String>,
    /// If set, fixed-window rate limit per key/client.
    pub(super) rate: Option<RateLimit>,
    /// Stricter rate limit for failed auth attempts (per client IP). Applied
    /// BEFORE the 401 response so brute-force attacks cannot bypass the normal
    /// rate limit by sending invalid keys. Defaults to 1/10 of the main rate.
    pub(super) fail_rate: Option<RateLimit>,
    /// Federation peers (base URLs of other Lore nodes).
    pub(super) peers: Vec<String>,
    /// Session table hard cap (default [`MAX_SESSIONS`]; can be reduced in tests).
    pub(super) session_cap: usize,
    /// Agent count hard cap (default [`MAX_AGENTS`]).
    pub(super) max_agents: usize,
    /// Route-based HTTP latency histograms (metrics).
    pub(super) latency: Arc<Mutex<HashMap<String, Histogram>>>,
    /// API key to use when calling peers (optional).
    pub(super) peer_key: Option<String>,
    /// Rate-limit counter (key → window).
    pub(super) hits: Arc<Mutex<HashMap<String, Window>>>,
    /// Data directory root (LORE_DATA). Used for task log paths.
    /// None when the server runs without persistent task support.
    pub(super) data_dir: Option<PathBuf>,
    /// Task database path (<data>/tasks.db). None when task support is off.
    /// Opening per-request is acceptable: `TaskStore` owns a non-Sync
    /// `rusqlite::Connection`; same pattern as `QueueApprover` and CLI.
    pub(super) task_db_path: Option<PathBuf>,
    /// Shared HTTP client for peer node calls (connection pool).
    pub(super) http: reqwest::Client,
    /// Chat sessions: (agent, session name) → working memory.
    /// The outer `RwLock` is held briefly; the session's own lock is held only
    /// for that session during a model call (different sessions flow in parallel).
    pub(super) sessions: SessionMap,
}

/// Session table type (dedicated locked session cells).
pub(super) type SessionMap = Arc<RwLock<HashMap<(AgentId, String), Arc<AsyncMutex<Session>>>>>;

/// A chat session: working memory + last-used timestamp (for eviction).
pub(super) struct Session {
    convo: Conversation,
    last_used: Instant,
}

/// Fixed-window rate-limit configuration.
///
/// Note: fixed windows allow a short burst at the window boundary
/// (e.g. max at end of window + max at start). Simplicity is a conscious choice;
/// switch to sliding window / token bucket for smoother limiting if needed.
#[derive(Clone, Copy)]
pub(super) struct RateLimit {
    pub(super) max: u32,
    pub(super) per: Duration,
}

/// A client's current window.
pub(super) struct Window {
    start: Instant,
    count: u32,
}

pub(super) struct Inner {
    pub(super) store: Arc<dyn MemoryStore>,
    pub(super) model: Arc<dyn Model>,
    pub(super) agents: RwLock<HashMap<AgentId, Agent>>,
    /// If set, personas are saved/loaded from this directory as `<id>.json`.
    pub(super) agents_dir: Option<PathBuf>,
    /// Service start time (for uptime).
    pub(super) started: Instant,
    /// Total processed request count.
    pub(super) requests: AtomicU64,
}

/// Creates an HTTP client with timeout for peer node calls.
fn make_peer_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(PEER_TIMEOUT_SECS))
        .build()
        .expect("reqwest client must be buildable")
}

impl AppState {
    /// New state with shared store + model.
    pub fn new(store: Arc<dyn MemoryStore>, model: Arc<dyn Model>) -> Self {
        Self::assemble(store, model, None, HashMap::new())
    }

    /// Persistent state: loads persona files under `dir` (with the given store + model);
    /// new agents are also saved to this directory. Combined with SQLite store,
    /// the service survives restarts (identity + memories are restored).
    pub fn persistent(
        dir: impl AsRef<std::path::Path>,
        store: Arc<dyn MemoryStore>,
        model: Arc<dyn Model>,
    ) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| LoreError::Storage(e.to_string()))?;
        let mut agents = HashMap::new();
        for entry in std::fs::read_dir(&dir).map_err(|e| LoreError::Storage(e.to_string()))? {
            let path = entry.map_err(|e| LoreError::Storage(e.to_string()))?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                // A single corrupt persona file must not prevent the entire
                // service from starting: warn and skip.
                match Agent::load_from(&path, store.clone(), model.clone()) {
                    Ok(agent) => {
                        agents.insert(agent.id.clone(), agent);
                    }
                    Err(e) => {
                        tracing::warn!(
                            path = %path.display(),
                            error = %e,
                            "persona could not be loaded, skipping"
                        )
                    }
                }
            }
        }
        Ok(Self::assemble(store, model, Some(dir), agents))
    }

    /// Single-point assembly: `new`/`persistent` differ only in the agents source
    /// (memory/disk) — fields cannot drift across two places.
    fn assemble(
        store: Arc<dyn MemoryStore>,
        model: Arc<dyn Model>,
        agents_dir: Option<std::path::PathBuf>,
        agents: HashMap<AgentId, Agent>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                store,
                model,
                agents: RwLock::new(agents),
                agents_dir,
                started: Instant::now(),
                requests: AtomicU64::new(0),
            }),
            tools: None,
            api_key: None,
            rate: None,
            fail_rate: None,
            peers: Vec::new(),
            session_cap: MAX_SESSIONS,
            max_agents: MAX_AGENTS,
            peer_key: None,
            hits: Arc::new(Mutex::new(HashMap::new())),
            http: make_peer_client(),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            latency: Arc::new(Mutex::new(HashMap::new())),
            data_dir: None,
            task_db_path: None,
        }
    }

    /// Sets the session cap (builder — test/config; 0 is clamped to at least 1).
    pub fn with_session_cap(mut self, cap: usize) -> Self {
        self.session_cap = cap.max(1);
        self
    }

    /// Sets the agent cap (builder — test/config; 0 is clamped to at least 1).
    pub fn with_max_agents(mut self, cap: usize) -> Self {
        self.max_agents = cap.max(1);
        self
    }

    /// Connects federation peers (builder): `deliberate` includes peers in questions.
    /// URLs are normalized (trailing `/` removed) and duplicates are culled.
    pub fn with_peers(mut self, peers: Vec<String>, key: Option<String>) -> Self {
        let mut seen = std::collections::HashSet::new();
        self.peers = peers
            .into_iter()
            .map(|p| p.trim().trim_end_matches('/').to_string())
            .filter(|p| !p.is_empty() && seen.insert(p.clone()))
            .collect();
        // Security posture: plain-http remote peer carries the key (if any) in
        // CLEARTEXT — the operator must choose this consciously, not silently.
        for p in &self.peers {
            if is_insecure_peer(p) {
                tracing::warn!(peer = %p, "federation: plain-http remote peer — key sent in cleartext (https recommended)");
            }
        }
        self.peer_key = key;
        self
    }

    /// Requires an API key (builder). All endpoints except `/health` are protected.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// `max` requests / `per_secs` seconds fixed-window rate limit (builder).
    /// Also sets a default fail-rate (1/10 of the main rate) to throttle
    /// brute-force auth attempts — can be overridden with `with_fail_rate_limit`.
    pub fn with_rate_limit(mut self, max: u32, per_secs: u64) -> Self {
        self.rate = Some(RateLimit {
            max,
            per: Duration::from_secs(per_secs),
        });
        // Default fail-rate: 1/10 of main max, same window (clamped to >=1).
        self.fail_rate = Some(RateLimit {
            max: (max / 10).max(1),
            per: Duration::from_secs(per_secs),
        });
        self
    }

    /// Explicit fail-rate override (builder). When set, failed auth attempts
    /// from a single IP are throttled independently of the main rate limit.
    pub fn with_fail_rate_limit(mut self, max: u32, per_secs: u64) -> Self {
        self.fail_rate = Some(RateLimit {
            max,
            per: Duration::from_secs(per_secs),
        });
        self
    }

    /// Connects the shared tool set (builder). `act` uses these.
    pub fn with_tools(mut self, tools: ToolContext) -> Self {
        self.tools = Some(Arc::new(tools));
        self
    }

    /// Connects task database path (builder). Required for task/approval endpoints.
    /// Data dir is also set (used for log file reading).
    /// When not set, task/approval endpoints return 503.
    pub fn with_task_store(mut self, data_dir: PathBuf, db_path: PathBuf) -> Self {
        self.data_dir = Some(data_dir);
        self.task_db_path = Some(db_path);
        self
    }

    /// Is the request authorized? (Always yes when no key is configured.)
    pub(super) fn authorized(&self, provided: Option<&str>) -> bool {
        match &self.api_key {
            None => true,
            Some(k) => provided.is_some_and(|p| super::security::ct_eq(p, k)),
        }
    }

    /// Does the rate limit allow the request? (Always yes when not configured.)
    /// `rl` overrides the stored `self.rate` — allows checking the fail-rate
    /// table with the same logic while using a different limit config.
    pub(super) fn allow_with(&self, key: &str, rl: &RateLimit) -> bool {
        let now = Instant::now();
        // Poison recovery: a panicked thread must not kill the entire service.
        let mut hits = self.hits.lock().unwrap_or_else(|e| e.into_inner());
        // If the table has grown, evict expired windows (memory DoS mitigation).
        // Use max(rate.per, fail_rate.per) so we never evict a live window
        // belonging to the other limiter (both share the same map).
        if hits.len() > HITS_EVICT_THRESHOLD {
            let evict_per = match (self.rate, self.fail_rate) {
                (Some(r), Some(f)) => r.per.max(f.per),
                (Some(r), None) => r.per,
                (None, Some(f)) => f.per,
                (None, None) => rl.per,
            };
            hits.retain(|_, w| now.duration_since(w.start) <= evict_per);
        }
        let w = hits.entry(key.to_string()).or_insert(Window {
            start: now,
            count: 0,
        });
        if now.duration_since(w.start) > rl.per {
            w.start = now;
            w.count = 0;
        }
        if w.count >= rl.max {
            return false;
        }
        w.count += 1;
        true
    }

    /// Convenience: check against the default rate limit.
    pub(super) fn allow(&self, key: &str) -> bool {
        let Some(rl) = self.rate else {
            return true;
        };
        self.allow_with(key, &rl)
    }

    /// Convenience: check against the fail-rate limit.
    pub(super) fn allow_fail(&self, key: &str) -> bool {
        let Some(rl) = self.fail_rate else {
            return true;
        };
        self.allow_with(key, &rl)
    }

    pub(super) fn view(a: &Agent) -> AgentView {
        AgentView {
            id: a.id.to_string(),
            name: a.persona.name.clone(),
            role: a.persona.role.clone(),
            traits: a.persona.traits.clone(),
            version: a.persona.version,
        }
    }

    fn persist(&self, agent: &Agent) -> Result<()> {
        if let Some(dir) = &self.inner.agents_dir {
            // Second line of defense: id is always a server-generated ULID, but
            // file-path construction should not rely on a single invariant (path traversal).
            if !id_is_fs_safe(&agent.id) {
                return Err(LoreError::Storage(format!(
                    "unsafe agent id not written to file: {}",
                    log_safe_id(&agent.id)
                )));
            }
            agent.save_to(dir.join(format!("{}.json", agent.id)))?;
        }
        Ok(())
    }

    /// Creates a new agent (with shared store + model). If persistence is enabled,
    /// the persona is written to disk. Name and role cannot be empty (422).
    pub async fn create_agent(
        &self,
        name: &str,
        role: &str,
        traits: Vec<String>,
    ) -> Result<AgentView> {
        let (name, role) = (name.trim(), role.trim());
        if name.is_empty() || role.is_empty() {
            return Err(LoreError::InvalidInput(
                "name and role cannot be empty".into(),
            ));
        }
        let persona = Persona::new(name, role).with_traits(traits);
        let agent = Agent::new(persona, self.inner.store.clone(), self.inner.model.clone());
        // Cap check + insert under the same write lock: no race can exceed the cap.
        // Check BEFORE persist — a rejected agent must not leave a file behind.
        let mut agents = self.inner.agents.write().await;
        if !agents.contains_key(&agent.id) && agents.len() >= self.max_agents {
            return Err(LoreError::InvalidInput(format!(
                "agent cap reached ({})",
                self.max_agents
            )));
        }
        self.persist(&agent)?;
        let view = Self::view(&agent);
        agents.insert(agent.id.clone(), agent);
        Ok(view)
    }

    /// Lists agents sorted by identity.
    pub async fn list_agents(&self) -> Vec<AgentView> {
        let mut v: Vec<AgentView> = self
            .inner
            .agents
            .read()
            .await
            .values()
            .map(Self::view)
            .collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    /// Snapshot of the team (clone under lock, release; deterministic order).
    pub(super) async fn team(&self) -> Vec<(AgentId, Agent)> {
        let mut team: Vec<(AgentId, Agent)> = {
            let g = self.inner.agents.read().await;
            g.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        };
        team.sort_by_key(|(id, _)| id.to_string());
        team
    }

    /// Drops a note on the shared board (World). Auto-record: deliberation
    /// drops many notes per run — decay must be able to undo them.
    pub(super) async fn board_note(
        &self,
        title: impl Into<String>,
        body: impl Into<String>,
    ) -> Result<()> {
        self.inner
            .store
            .remember(
                Memory::episodic(Scope::World, title, body)
                    .with_importance(Memory::AUTO_IMPORTANCE),
            )
            .await?;
        Ok(())
    }

    /// Records a request's route-based latency into the histogram.
    pub(super) fn record_latency(&self, route: &str, ms: u64) {
        let mut map = self.latency.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(route.to_string()).or_default().record(ms);
    }

    /// Readiness check: is the store actually reachable? (readiness probe)
    pub async fn ready(&self) -> Result<()> {
        self.inner.store.count(&Scope::World).await.map(|_| ())
    }

    /// Produces Prometheus-style metrics text: basic gauges/counters +
    /// route-based latency histogram + retrieval/consolidation counters.
    pub async fn metrics_text(&self) -> String {
        use crate::memory::evolution::stats as cstats;
        use crate::memory::retrieval::stats as rstats;
        use std::fmt::Write;
        use std::sync::atomic::Ordering as AOrd;

        let agents = self.inner.agents.read().await.len();
        let requests = self.inner.requests.load(Ordering::Relaxed);
        let uptime = self.inner.started.elapsed().as_secs();
        let board = self.inner.store.count(&Scope::World).await.unwrap_or(0);
        let sessions = self.sessions.read().await.len();

        let mut out = format!(
            "# HELP lore_agents Number of registered agents\n\
             # TYPE lore_agents gauge\n\
             lore_agents {agents}\n\
             # HELP lore_requests_total Total processed requests\n\
             # TYPE lore_requests_total counter\n\
             lore_requests_total {requests}\n\
             # HELP lore_uptime_seconds Service uptime\n\
             # TYPE lore_uptime_seconds gauge\n\
             lore_uptime_seconds {uptime}\n\
             # HELP lore_board_memories Number of board (World) memories\n\
             # TYPE lore_board_memories gauge\n\
             lore_board_memories {board}\n\
             # HELP lore_sessions Open conversation session count\n\
             # TYPE lore_sessions gauge\n\
             lore_sessions {sessions}\n"
        );

        let _ = writeln!(
            out,
            "# HELP lore_recall_candidates_total Number of candidate records exceeding score threshold\n\
             # TYPE lore_recall_candidates_total counter\n\
             lore_recall_candidates_total {}\n\
             # HELP lore_token_fallback_hits_total Token-level fallback matches\n\
             # TYPE lore_token_fallback_hits_total counter\n\
             lore_token_fallback_hits_total {}",
            rstats::RECALL_CANDIDATES.load(AOrd::Relaxed),
            rstats::TOKEN_FALLBACK_HITS.load(AOrd::Relaxed),
        );
        let _ = writeln!(
            out,
            "# HELP lore_consolidation_runs_total Number of consolidation runs\n\
             # TYPE lore_consolidation_runs_total counter\n\
             lore_consolidation_runs_total {}\n\
             # HELP lore_consolidation_forgotten_total Records forgotten by decay\n\
             # TYPE lore_consolidation_forgotten_total counter\n\
             lore_consolidation_forgotten_total {}\n\
             # HELP lore_consolidation_merged_total Near-duplicate merges\n\
             # TYPE lore_consolidation_merged_total counter\n\
             lore_consolidation_merged_total {}\n\
             # HELP lore_consolidation_last_duration_ms Last consolidation duration\n\
             # TYPE lore_consolidation_last_duration_ms gauge\n\
             lore_consolidation_last_duration_ms {}",
            cstats::RUNS.load(AOrd::Relaxed),
            cstats::FORGOTTEN.load(AOrd::Relaxed),
            cstats::MERGED.load(AOrd::Relaxed),
            cstats::LAST_MS.load(AOrd::Relaxed),
        );

        out.push_str("# HELP lore_http_request_duration_ms HTTP request duration (per route)\n");
        out.push_str("# TYPE lore_http_request_duration_ms histogram\n");
        let map = self.latency.lock().unwrap_or_else(|e| e.into_inner());
        let mut routes: Vec<&String> = map.keys().collect();
        routes.sort(); // deterministic output (testable + diffable)
        for r in routes {
            map[r].render_into(&mut out, r);
        }
        out
    }

    /// Reads the shared board (World scope).
    pub async fn read_board(&self, limit: usize) -> Result<Vec<MemoryView>> {
        let res = self
            .inner
            .store
            .recall(&Scope::World, &Query::new("").limit(limit))
            .await?;
        Ok(res
            .into_iter()
            .map(|s| MemoryView {
                id: s.item.id.to_string(),
                score: s.score,
                summary: s.item.summary(),
            })
            .collect())
    }

    /// Partially updates the persona; if any field is set, `version` increments and the
    /// change is written to disk (empty patch is a no-op). Disk write is done OUTSIDE the
    /// lock — slow I/O must not block the entire agent table.
    pub async fn update_agent(&self, id: &AgentId, patch: PersonaPatch) -> Result<AgentView> {
        let (snapshot, changed) = {
            let mut guard = self.inner.agents.write().await;
            let agent = guard
                .get_mut(id)
                .ok_or_else(|| LoreError::NotFound(format!("agent {id}")))?;
            let mut changed = false;
            // Same contract as create: name/role cannot be emptied; values are
            // trimmed before writing (PATCH validation cannot be bypassed).
            if let Some(n) = patch.name {
                let n = n.trim();
                if n.is_empty() {
                    return Err(LoreError::InvalidInput("name cannot be empty".into()));
                }
                agent.persona.name = n.to_string();
                changed = true;
            }
            if let Some(r) = patch.role {
                let r = r.trim();
                if r.is_empty() {
                    return Err(LoreError::InvalidInput("role cannot be empty".into()));
                }
                agent.persona.role = r.to_string();
                changed = true;
            }
            if let Some(d) = patch.description {
                agent.persona.description = d;
                changed = true;
            }
            if let Some(t) = patch.traits {
                agent.persona.traits = t;
                changed = true;
            }
            if let Some(s) = patch.system_prompt {
                agent.persona.system_prompt = s;
                changed = true;
            }
            if changed {
                agent.persona.version += 1;
            }
            (agent.clone(), changed)
        }; // write lock released here
        if changed {
            // Race protection: if a concurrent delete overtook us, do not
            // "resurrect" the deleted agent to disk; for concurrent PATCHes, always
            // write the MOST CURRENT version. A read lock is held during persist —
            // a delete waiting for the write lock will be serialized (persona files
            // are small, I/O is short).
            let g = self.inner.agents.read().await;
            if let Some(current) = g.get(id) {
                self.persist(current)?;
            }
        }
        Ok(Self::view(&snapshot))
    }

    /// Removes an agent (persona file is also deleted; memories remain in the store).
    pub async fn delete_agent(&self, id: &AgentId) -> Result<()> {
        let removed = self.inner.agents.write().await.remove(id);
        if removed.is_none() {
            return Err(LoreError::NotFound(format!("agent {id}")));
        }
        // The agent's open chat sessions are also dropped.
        self.sessions.write().await.retain(|(aid, _), _| aid != id);
        if let Some(dir) = &self.inner.agents_dir {
            // Second line of defense (see `persist`): validate id before path construction.
            if !id_is_fs_safe(id) {
                return Ok(()); // removed from map; unsafe id must not enter file path
            }
            // If the file cannot be deleted, return an error (except NotFound) —
            // otherwise the persona would be restored from disk on restart, "resurrecting"
            // the agent.
            if let Err(e) = std::fs::remove_file(dir.join(format!("{id}.json"))) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    return Err(LoreError::Storage(format!(
                        "persona file could not be deleted: {e}"
                    )));
                }
            }
        }
        Ok(())
    }

    pub(super) async fn get_agent(&self, id: &AgentId) -> Result<Agent> {
        self.inner
            .agents
            .read()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| LoreError::NotFound(format!("agent {id}")))
    }

    /// Asks an agent, returns the reply.
    pub async fn ask(&self, id: &AgentId, msg: &str) -> Result<String> {
        self.get_agent(id).await?.respond(msg).await
    }

    /// Sessioned chat: if `session` is provided, that session's working memory (last N turns)
    /// is included in the prompt and the exchange is recorded to the window; `None` falls
    /// back to plain `ask`. Concurrent requests to the same session are serialized (turn
    /// integrity is preserved).
    pub async fn ask_session(
        &self,
        id: &AgentId,
        session: Option<&str>,
        msg: &str,
    ) -> Result<String> {
        let Some(sid) = session else {
            return self.ask(id, msg).await;
        };
        Self::validate_session_id(sid)?;
        let agent = self.get_agent(id).await?;
        let cell = self.session_cell(id, sid).await;
        let mut s = cell.lock().await;
        s.last_used = Instant::now();
        agent.converse(&mut s.convo, msg).await
    }

    /// Streams the response in real time. If `session` is provided, working memory is
    /// included in the prompt and the session lock is held UNTIL THE STREAM ENDS (turn
    /// integrity — only that session waits, others flow in parallel); the exchange is
    /// recorded to the window at the end of the stream.
    pub async fn ask_stream(
        &self,
        id: &AgentId,
        session: Option<&str>,
        msg: &str,
    ) -> Result<TokenStream> {
        let agent = self.get_agent(id).await?;
        let Some(sid) = session else {
            return agent.respond_stream(msg).await;
        };
        Self::validate_session_id(sid)?;
        let cell = self.session_cell(id, sid).await;
        let mut guard = cell.lock_owned().await;
        guard.last_used = Instant::now();
        let inner = agent.think_stream(msg, guard.convo.history()).await?;
        let msg = msg.to_string();
        let wrapped = futures::stream::unfold(
            Some((inner, String::new(), guard, msg)),
            |state| async move {
                let (mut inner, mut acc, mut guard, msg) = state?;
                match inner.next().await {
                    Some(Ok(chunk)) => {
                        acc.push_str(&chunk);
                        Some((Ok(chunk), Some((inner, acc, guard, msg))))
                    }
                    Some(Err(e)) => Some((Err(e), None)),
                    None => {
                        // Stream ended: exchange is recorded to working memory, lock drops.
                        guard.convo.record(&msg, &acc);
                        None
                    }
                }
            },
        );
        Ok(Box::pin(wrapped))
    }

    /// Validates the session name (client-controlled — length limit).
    fn validate_session_id(sid: &str) -> Result<()> {
        if sid.len() > MAX_SESSION_ID_LEN {
            return Err(LoreError::InvalidInput(format!(
                "session name too long (max {MAX_SESSION_ID_LEN} bytes)"
            )));
        }
        Ok(())
    }

    /// Retrieves or creates a session cell. Cap is HARD: when full, idle sessions (TTL)
    /// are evicted first, then least-recently-used (LRU) — the table cannot grow
    /// unboundedly with client-controlled `session` values. Only currently locked
    /// (active model call) sessions are safe from eviction.
    ///
    /// Eviction runs ONLY when a new key is added: a request to an existing session
    /// neither evicts its neighbor nor resets its own window (even if it is the
    /// LRU-oldest) — otherwise every request at cap would kill a live session.
    async fn session_cell(&self, id: &AgentId, sid: &str) -> Arc<AsyncMutex<Session>> {
        let key = (id.clone(), sid.to_string());
        let mut map = self.sessions.write().await;
        if !map.contains_key(&key) && map.len() >= self.session_cap {
            // 1) Evict idle sessions.
            map.retain(|_, cell| match cell.try_lock() {
                Ok(s) => s.last_used.elapsed() <= SESSION_IDLE_TTL,
                Err(_) => true, // currently in use — keep
            });
            // 2) Still full: evict LRU (excluding locked — those are active work).
            while map.len() >= self.session_cap {
                let oldest = map
                    .iter()
                    .filter_map(|(k, cell)| cell.try_lock().ok().map(|s| (k.clone(), s.last_used)))
                    .min_by_key(|&(_, t)| t)
                    .map(|(k, _)| k);
                match oldest {
                    Some(k) => {
                        map.remove(&k);
                    }
                    // All locked: cap is consumed by active work (bounded by concurrency).
                    None => break,
                }
            }
        }
        map.entry(key)
            .or_insert_with(|| {
                Arc::new(AsyncMutex::new(Session {
                    convo: Conversation::new(),
                    last_used: Instant::now(),
                }))
            })
            .clone()
        // outer lock released here — model call does not block other sessions
    }

    /// Inter-agent message: if `ask`, the recipient replies (sender remembers the exchange);
    /// if `tell`, the recipient records the message as episodic memory. Orchestrator Ask/Tell semantics.
    pub async fn message(
        &self,
        to: &AgentId,
        from: Option<&AgentId>,
        ask: bool,
        content: &str,
    ) -> Result<String> {
        let target = self.get_agent(to).await?;
        if ask {
            let reply = target.respond(content).await?;
            if let Some(f) = from {
                let tname = target.persona.name.clone();
                self.get_agent(f)
                    .await?
                    .note(format!("asked {tname}"), format!("{content} → {reply}"))
                    .await?;
            }
            Ok(reply)
        } else {
            let sname = match from {
                Some(f) => self.get_agent(f).await?.persona.name.clone(),
                None => "System".to_string(),
            };
            // DELIBERATE CHOICE: `tell` is an explicit "remember this" action — the
            // recipient's record is preserved at standard importance (0.5); it is NOT
            // an automatic trace. (The sender's "asked X" trace is automatic → `note`.)
            target
                .experience(format!("message from {sname}"), content.to_string())
                .await?;
            Ok(String::new())
        }
    }

    /// Multi-step tool loop (ReAct): the model chains tools to solve the task,
    /// observations are fed back. Falls back to plain `respond` if no tools are available.
    pub async fn solve(&self, id: &AgentId, input: &str, max_steps: usize) -> Result<String> {
        let agent = self.get_agent(id).await?;
        match &self.tools {
            Some(ctx) => agent.solve(ctx, input, max_steps).await,
            None => agent.respond(input).await,
        }
    }

    /// The agent performs an action: if one of the shared tools matches, it is executed
    /// and the usage is remembered as episodic; otherwise falls back to `respond`.
    pub async fn act(&self, id: &AgentId, input: &str) -> Result<String> {
        let agent = self.get_agent(id).await?;
        if let Some(ctx) = &self.tools {
            if let Some(call) = ctx.router.route(input, &ctx.registry).await {
                if let Some(tool) = ctx.registry.get(&call.tool) {
                    let result = tool.run(&call.args).await?;
                    agent
                        .note(
                            format!("used {} tool", call.tool),
                            format!("input: {input} → result: {result}"),
                        )
                        .await?;
                    return Ok(result);
                }
            }
        }
        agent.respond(input).await
    }

    /// Adds an episodic memory to the agent.
    pub async fn experience(&self, id: &AgentId, title: &str, body: &str) -> Result<()> {
        self.get_agent(id).await?.experience(title, body).await
    }

    /// Runs reflection on the agent: frequently recalled episodic memories are
    /// distilled by the model and promoted to the semantic tier. Returns the number
    /// of distilled memories.
    pub async fn reflect(&self, id: &AgentId) -> Result<usize> {
        self.get_agent(id).await?.reflect().await
    }

    /// Reinforces a memory record (access/success/failure — external feed for decay and
    /// Wilson signals; e.g. an orchestrator saying "this procedure worked").
    ///
    /// Scope validation: the record must belong to the agent or be World — another
    /// agent's record returns 404 (even its existence is not leaked). Otherwise, any
    /// client could inflate another's Wilson score.
    pub async fn reinforce(
        &self,
        agent_id: &AgentId,
        memory_id: &crate::id::MemoryId,
        outcome: crate::memory::Outcome,
    ) -> Result<()> {
        let agent = self.get_agent(agent_id).await?;
        let mem = agent
            .memory
            .get(memory_id)
            .await?
            .ok_or_else(|| LoreError::NotFound(format!("record {memory_id}")))?;
        let visible = matches!(mem.scope, Scope::World) || mem.scope == agent.scope();
        if !visible {
            return Err(LoreError::NotFound(format!("record {memory_id}")));
        }
        agent.memory.reinforce(memory_id, outcome).await
    }

    /// Recalls from the agent's memory (`semantic` enables semantic recall).
    pub async fn recall(
        &self,
        id: &AgentId,
        q: &str,
        limit: usize,
        semantic: bool,
    ) -> Result<Vec<MemoryView>> {
        let mut query = Query::new(q).limit(limit);
        if semantic {
            query = query.semantic();
        }
        let res = self.get_agent(id).await?.recall(&query).await?;
        Ok(res
            .into_iter()
            .map(|s| MemoryView {
                id: s.item.id.to_string(),
                score: s.score,
                summary: s.item.summary(),
            })
            .collect())
    }

    // ── Task queue HTTP surface ────────────────────────────────────────────

    /// Opens a per-request TaskStore connection.
    /// Design choice: `rusqlite::Connection` is not `Sync`; the CLI and
    /// `QueueApprover` each open per-call connections too. WAL mode permits
    /// concurrent reads/writes. No new dependency needed.
    fn open_task_store(&self) -> Result<crate::task::TaskStore> {
        let path = self
            .task_db_path
            .as_ref()
            .ok_or_else(|| LoreError::Server("task store not configured".to_string()))?
            .as_path();
        crate::task::TaskStore::open(path)
    }

    /// Relativises a workspace path against data_dir to avoid leaking
    /// absolute server filesystem paths in HTTP responses.
    /// Returns the relative portion ("workspaces/myagent") when under data_dir,
    /// or just the basename ("myagent") as a fallback.
    fn relativise_workspace(&self, workspace: &std::path::Path) -> String {
        if let Some(data_dir) = &self.data_dir {
            if let Ok(rel) = workspace.strip_prefix(data_dir) {
                // Remove leading separator so it looks like "workspaces/myagent"
                // instead of "/workspaces/myagent".
                let s = rel.to_string_lossy();
                return s.strip_prefix('/').unwrap_or(&s).to_string();
            }
        }
        // Fallback: basename only (prevents full path disclosure).
        workspace
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| workspace.to_string_lossy().to_string())
    }

    /// Converts a `Task` into an HTTP view (workspace relativised, report included).
    fn task_to_view(&self, task: &crate::task::Task) -> TaskView {
        TaskView {
            id: task.id.clone(),
            agent: task.agent.clone(),
            goal: task.goal.clone(),
            workspace: self.relativise_workspace(&task.workspace),
            verify: task.verify.clone(),
            status: task.status.clone(),
            created_at: task.created_at,
            updated_at: task.updated_at,
            report: task.report.clone(),
            parent_id: task.parent_id.clone(),
        }
    }

    /// Converts a `Task` into a compact HTTP view (workspace relativised, report omitted).
    fn task_to_compact(&self, task: &crate::task::Task) -> CompactTaskView {
        CompactTaskView {
            id: task.id.clone(),
            agent: task.agent.clone(),
            goal: task.goal.clone(),
            workspace: self.relativise_workspace(&task.workspace),
            verify: task.verify.clone(),
            status: task.status.clone(),
            created_at: task.created_at,
            updated_at: task.updated_at,
            parent_id: task.parent_id.clone(),
        }
    }

    /// Enqueues a new task (HTTP: POST /tasks).
    /// Workspace defaults to `<data_dir>/workspaces/<agent>` when not provided.
    /// If an explicit absolute workspace is provided, it must be under data_dir
    /// (server-side containment check; daemon-side Gate provides a second layer).
    pub fn enqueue_task(
        &self,
        agent: &str,
        goal: &str,
        workspace: Option<PathBuf>,
        verify: Vec<String>,
    ) -> Result<TaskView> {
        let store = self.open_task_store()?;
        let agent = agent.trim();
        let goal = goal.trim();
        if agent.is_empty() {
            return Err(LoreError::InvalidInput("agent cannot be empty".into()));
        }
        if goal.is_empty() {
            return Err(LoreError::InvalidInput("goal cannot be empty".into()));
        }
        // Path traversal validation: agent name is used in persona file lookup.
        if agent.contains('/') || agent.contains('\\') || agent.contains("..") {
            return Err(LoreError::InvalidInput(
                "agent name must not contain path separators".into(),
            ));
        }
        let workspace = workspace.unwrap_or_else(|| {
            self.data_dir
                .as_ref()
                .map(|d| d.join("workspaces").join(agent))
                .unwrap_or_else(|| PathBuf::from(format!("/tmp/lore-{agent}")))
        });
        // Server-side containment: absolute workspace must be under data_dir.
        if workspace.is_absolute() {
            if let Some(data_dir) = &self.data_dir {
                if !workspace.starts_with(data_dir) {
                    return Err(LoreError::InvalidInput(
                        "workspace must be within data directory".into(),
                    ));
                }
            }
        }
        let new_task = crate::task::NewTask {
            agent: agent.to_string(),
            goal: goal.to_string(),
            workspace,
            verify,
            parent_id: None,
        };
        let task = store.enqueue(new_task)?;
        Ok(self.task_to_view(&task))
    }

    /// Lists tasks (compact, newest-first). Omits report (available on GET /tasks/:id).
    pub fn list_tasks(&self, limit: usize) -> Result<Vec<CompactTaskView>> {
        let store = self.open_task_store()?;
        let limit = limit.clamp(0, 1000); // 0 → empty list; MAX_QUERY_LIMIT same as api.rs
        let tasks = store.list(limit)?;
        Ok(tasks.iter().map(|t| self.task_to_compact(t)).collect())
    }

    /// Gets a full task record (HTTP: GET /tasks/:id).
    /// Includes report + children when present.
    pub fn get_task_full(&self, id: &str) -> Result<TaskFullView> {
        let store = self.open_task_store()?;
        let task = store
            .get(id)?
            .ok_or_else(|| LoreError::NotFound(format!("task {id}")))?;
        let children = store.children_of(id)?;
        Ok(TaskFullView {
            task: self.task_to_view(&task),
            children: children.iter().map(|c| self.task_to_view(c)).collect(),
        })
    }

    /// Reads a task log file (HTTP: GET /tasks/:id/log?tail=N).
    /// Uses the same path traversal validation as the CLI.
    pub fn read_task_log(&self, id: &str, tail: Option<usize>) -> Result<String> {
        // Path traversal validation (same as CLI: reject / \ ..).
        if id.contains('/') || id.contains('\\') || id.contains("..") {
            return Err(LoreError::InvalidInput("invalid task id".into()));
        }
        let data_dir = self
            .data_dir
            .as_ref()
            .ok_or_else(|| LoreError::Server("data directory not configured".to_string()))?
            .clone();
        let log_path = data_dir.join("logs").join(format!("{id}.log"));
        if !log_path.exists() {
            return Err(LoreError::NotFound(format!(
                "log file not found for task {id}"
            )));
        }
        let content =
            std::fs::read_to_string(&log_path).map_err(|e| LoreError::Storage(e.to_string()))?;
        match tail {
            Some(n) => {
                let lines: Vec<&str> = content.lines().collect();
                let start = lines.len().saturating_sub(n);
                Ok(lines[start..].join("\n"))
            }
            None => Ok(content),
        }
    }

    /// Lists pending approvals (HTTP: GET /inbox).
    pub fn pending_approvals(&self) -> Result<Vec<crate::task::ApprovalEntry>> {
        let store = self.open_task_store()?;
        store.pending_approvals()
    }

    /// Decides an approval: approve or deny (HTTP: POST /approvals/:id/approve|deny).
    /// Idempotent: deciding a non-Pending approval -> Conflict (409) with clear message;
    /// unknown id -> NotFound (404).
    pub fn decide_approval(&self, id: &str, approve: bool) -> Result<()> {
        let store = self.open_task_store()?;
        // TaskStore::decide_approval returns NotFound for unknown id and
        // InvalidInput for already-decided. We map InvalidInput -> Conflict
        // (409 in the handler layer).
        store.decide_approval(id, approve).map_err(|e| match e {
            LoreError::InvalidInput(msg) => LoreError::Conflict(msg),
            other => other,
        })
    }
}
