# Lore — Design and Roadmap

> **Lore**: an identity + orchestration + memory core for AI agents. It gives each agent
> a persistent identity (persona) and a personal memory. **Fully self-contained** —
> not tied to any external service or API; the memory engine is written from scratch in
> native Rust, inside Lore (“Alaz from zero”).

> **Status (540 tests passing + 1 ignored [live-LLM], clippy clean, with CI):** M0–M31 ✅ + AI coworkers ✅ + code review fixes ✅ + 4-way review hardening ✅. Next roadmap: `docs/superpowers/specs/2026-07-24-next-roadmap.md`.

This document captures the design decisions and phased roadmap made after research
(the 2026 “Memory in the Age of AI Agents” survey + Alaz's current architecture).

---

## 1. Vision and Scope

Two things make an AI agent "that agent":

1. **Identity** — who it is: its name, role, character, capabilities, permissions.
2. **Memory** — what it has lived through, what it knows, what it has learned.

Lore brings these two together and manages agents jointly (orchestration).
The goal is not a heavy framework but a **lightweight, embeddable core** (as described
in Cargo.toml, "a lightweight orchestration core").

**Out of scope (deliberately):** LLM training, fine-tuning, latent/parametric memory
(requires access to the model's internal state; not possible with hosted APIs). Lore
works entirely on **token-level memory** — the only design space that anyone working
with hosted models can actually use.

---

## 2. Positioning: Fully Self-Contained, Its Own Memory Engine

**Lore is not tied to any external service.** It does not connect to Alaz (or any other
memory server); we write the memory capabilities **from scratch, in native Rust, inside
Lore**. The Alaz architecture was studied only as **reference/inspiration** (Rust +
Postgres/pgvector, 6-signal hybrid search, HyDE, cross-encoder rerank, Wilson score, 5W
recall); we build the equivalent of those capabilities ourselves, from scratch.

So Lore is a whole on its own — both identity+orchestration and its own memory engine:

| Subsystem         | Responsibility                | Question                   |
|-------------------|-------------------------------|----------------------------|
| **Identity**      | Persona, identity, capabilities | "Who am I?"              |
| **Orchestration** | Managing agents, messaging    | "Who am I talking to?"     |
| **Memory**        | Its own native memory engine  | "What do I know, what have I lived?" |

Critical decision: **memory is still abstracted behind a `trait`** — but not to plug in
external backends, rather to make our own implementations (in-memory ↔ persistent)
swappable and testable:

- `InMemoryStore` → fast prototype + tests (zero dependencies)
- `SqliteStore` → persistent native engine (embedding + BM25 + graph all embedded)

No external HTTP dependency, no external server, no external API. Everything lives inside
the binary.

---

## 3. Research Summary (rationale for decisions)

The three-axis framework of the 2026 survey and its practical takeaways:

- **Forms (where memory lives):** token-level / latent / parametric. → We are **token-level**.
  Topology spectrum: **flat (1D vector) → planar (2D graph/tree) → hierarchical (3D tiers)**.
  Guideline: **start with flat**; move to a graph when multi-hop retrieval can't be solved with flat.
- **Functions (what memory is for):**
  - **Factual** = episodic (events) + semantic (general knowledge) → "what do I know"
  - **Experiential** = procedural (case → strategy → skill) → "how do I do it better"
    (the biggest gap in the literature; agents don't learn from their own success/failure)
  - **Working** = context-window management (runtime, not persistent storage)
- **Dynamics (how it operates over time):**
  - **Formation** (what is stored): knowledge distillation (discrete facts) + semantic summarization
  - **Evolution** (maintenance): consolidation (merge), updating (soft-delete + timestamp),
    forgetting (time decay/Ebbinghaus + access frequency + importance)
  - **Retrieval** (access): timing (don't always search) → query construction (**HyDE**)
    → strategy (**hybrid: BM25 + semantic**) → post-process (**MMR + rerank + aggressive filtering**)

Additional practical golden rules:
- **Retrieval quality is bounded by formation+evolution quality.** What you store first matters.
- **Write-time conflict detection**: raise a conflict warning if a new record falls in the 0.6–0.9 cosine similarity band.
- **Soft-delete + timestamp** (Zep-style), not hard-delete — auditability + "unlearning".
- **Multi-tenancy from the start**: isolation must be enforced at the retrieval level too, not just in storage.
- Add experiential memory — let the agent remember "how it solved it", not just "what it knew".

Rust ecosystem (local, lightweight stack):
- `rig` — LLM framework (20+ providers, type-safe tools, structured output)
- `fastembed` — local ONNX embeddings (nomic-embed-text-v1.5, 768 dim, Matryoshka), no server
- `tantivy` — full-text/BM25 search
- `usearch` — vector ANN (if needed)
- `rusqlite` — lightweight persistent storage

---

## 4. Design Decisions (locked)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Fully self-contained; memory written from scratch in native code. `MemoryStore` trait + `InMemoryStore`/`SqliteStore` (both native) | No dependency on external services; the engine is inside the binary |
| D2 | 3 memory tiers: **Episodic**, **Semantic**, **Procedural** (+ runtime Working memory) | The survey's Functions axis; maps one-to-one with Alaz |
| D3 | Phased storage: `InMemory` + JSON snapshot → `SQLite` (rusqlite). Postgres = Alaz's job | Start with zero dependencies, then persistence |
| D4 | Phased retrieval: keyword+recency → +semantic (fastembed) hybrid (RRF) → +graph/HyDE/rerank | "Start with flat"; add complexity when the evidence arrives |
| D5 | `Agent` = `AgentId` (ULID) + `Persona` + capabilities + memory handle + model handle | Identity = persona (lore) + personal memory |
| D6 | `Model` trait; first `MockModel`, then an OpenAI-compatible client (Ollama `:11434/v1`) / `rig` | Compatible with Alaz's current LLM setup, testable |
| D7 | Orchestration: **supervisor + message-passing** (tokio mpsc mailbox); blackboard later | The most common, most understandable pattern; single-writer safety |
| D8 | Evolution: soft-delete+timestamp, time/frequency/importance decay, write-time conflict | Survey best practice; "forgetting matters as much as remembering" |
| D9 | Scoping: memory isolated by `AgentId` (+ optional shared `world`); retrieval-level enforcement | Multi-tenant security from the start |
| D10 | Language: Rust 2021, `async` core (tokio), errors via `thiserror`+`anyhow` | Compatible with the current Cargo.toml |
| D11 | Hands: write/exec tools (`ShellTool`, `FileWriteTool`, `FileEditTool`) gated by a `Policy` engine (allow/deny/approve) | Personal-use-first pragmatic security; policy-based autonomy — agents run freely inside the policy, anything outside falls to an approval gate. Shell metacharacter chaining denied by default |
| D12 | Work loop: `Agent::work(WorkSpec)` — plan → apply → verify → iterate until verification passes | Victory is declared by the verify command's exit code, never by the model. Failed verification feeds back as data for the next iteration |
| D13 | Daemon + task queue: SQLite-backed `TaskStore` + `lore daemon` foreground worker + CLI (`task add/list/status/log`, `inbox`, `approve|deny`) | Single operator, single machine; everything coordinates through SQLite (WAL). Crash recovery: orphaned Running/WaitingApproval tasks are re-queued on restart. `--concurrency N` enables atomic-claim parallel workers (1–8) |
| D14 | Team + PM: role presets (`backend/frontend/reviewer/pm`) with per-agent `ModelConfig` + factory; `task add --team` decomposes via PM → child tasks → reviewer → synthesis | Each role gets tailored tools (reviewer is read-only). The PM flow is crash-safe: no duplicate children, no wedged parents, compare-and-swap finalization |
| D15 | Distillation: `Agent::distill_work` extracts conventions/constraints/facts from successful tasks into semantic memory; failed tasks produce constraint-only lessons | Procedural memory (Wilson-reinforced) is recorded after every run. Recalled conventions seed the next task's goal. `--no-distill` opts out per agent |
| D16 | Sandbox: `Policy.sandbox_exec` (bubblewrap: `--ro-bind / /`, workspace rw, `--die-with-parent`) — `Off`/`IfAvailable`/`Required` | `Required` without bwrap fails closed; argv-built, never string-joined. Opt-in: the default build runs plain |
| D17 | Parallel daemon: `lore daemon --concurrency N` — atomic `claim_next_queued` (`UPDATE…RETURNING`), per-worker connections, graceful shutdown with in-flight re-queue | Two workers can never take the same task. Team finalization is compare-and-swap; reviewer child enqueue via atomic `INSERT…WHERE NOT EXISTS` |
| D18 | Orchestrator is a lib-API for embedded/demo use, not the production routing layer. Production team flow (daemon+pm) lives in `daemon.rs` + `task/store.rs`; `pm.rs` stays in `orchestrator/` as a decomposition helper (moving it to `team/` would churn without benefit — the Orchestrator's mailbox pattern is used by demo and tests, but HTTP+TaskStore drives production). Demo/tests exercise the Orchestrator's mailbox+blackboard directly; the daemon bypasses it for stateless SQL-backed operations. | Keeps the Orchestrator lightweight and purpose-scoped; avoids conflating the in-memory mailbox model with the persistent task queue that production actually uses |

---

## 5. Module Structure

```
src/
  lib.rs            # public API surface, re-exports
  main.rs           # demo / CLI entry point
  error.rs          # LoreError (thiserror)
  daemon.rs         # task queue daemon (foreground worker, crash recovery, parallel)

  id.rs             # AgentId, MemoryId (ULID-based)

  agent/
    mod.rs          # Agent struct: identity + handles
    persona.rs      # Persona: name, role, description, traits, system_prompt
    conversation.rs # Conversation: bounded verbatim window + Prompt.history
    work/            # WorkSpec / WorkReport / Agent::work — plan→apply→verify→iterate
      mod.rs          # loop, helpers (tail_bytes, extract_exit_code)
      tests.rs        # work-loop + seeding + strategy tests
    distill.rs      # Agent::distill_work — extract conventions/facts into semantic memory
    roles.rs        # Role presets (backend, frontend, reviewer, pm) + identity extras

  memory/
    mod.rs          # MemoryStore trait + shared types
    types.rs        # Memory, Episodic, Semantic, Procedural, Scope, Query
    in_memory.rs    # InMemoryStore (+ JSON snapshot) — Phase 1
    sqlite.rs       # SqliteStore (native persistent engine) — Phase 2
    embed.rs        # local embeddings (fastembed) — Phase 2
    retrieval.rs    # scoring: keyword+recency → hybrid → HyDE/rerank
    rerank.rs       # native reranker (coverage + phrase + bigram)
    evolution.rs    # consolidation, decay, conflict detection (background task)
    graph.rs        # entity/relationship graph (native)

  model/
    mod.rs          # Model trait (complete/chat/embed/stream)
    mock.rs         # MockModel (deterministic, for tests)
    openai.rs       # OpenAI-compatible client (including Ollama)
    anthropic.rs    # Anthropic Messages API (key + subscription auth)
    codex.rs        # CodexModel — OpenAI Responses API (ChatGPT subscription)
    factory.rs      # ModelConfig + build_model — per-agent model construction

  auth/
    mod.rs          # PKCE helpers, TokenStore (0600 atomic), AccessTokenProvider
    oauth.rs        # Provider-specific OAuth (Anthropic, OpenAI loopback/device)

  orchestrator/
    mod.rs          # Orchestrator: registry + mailbox + routing + blackboard
    message.rs      # Message, Envelope (from AgentId → to AgentId)
    registry.rs     # AgentId -> Agent registration table
    pm.rs           # PM decomposition: team task → child tasks → reviewer → synthesis

  policy/
    mod.rs          # Policy engine: allowed roots, auto-allow, deny, default_exec, sandbox
    approval.rs     # Gate + Approver (CliApprover, AllowAll, DenyAll, QueueApprover)

  task/
    mod.rs          # Types (TaskStatus, Task, NewTask, ApprovalEntry, ApprovalStatus) + re-exports
    store.rs        # TaskStore (SQLite): task + approval CRUD, atomic claim, idempotent decisions
    approver.rs     # QueueApprover: approval requests stored in DB until answered

  tool/
    mod.rs          # Tool trait + ToolRegistry + ToolRouter (KeywordRouter + LlmRouter)
    builtin.rs      # CalcTool, TimeTool, WebFetchTool, FileReadTool
    shell.rs        # ShellTool (timeout, output truncation, policy-gated)
    fs_write.rs     # FileWriteTool (atomic tmp+rename), FileEditTool (exact replace)

  server/
    mod.rs          # Module entry
    api.rs          # Router + thin handler wrappers + serve()
    state.rs        # AppState core (shared store+model, agent map, team)
    security.rs     # API key validation + rate-limit middleware
    deliberate.rs   # Collective reasoning + federation + WebSocket
    types.rs        # DTOs (CreateReq, AskReq, etc.)
```

---

## 6. Data Model (draft)

```rust
// Identity
pub struct AgentId(String);              // ULID: time-ordered, collision-free

pub struct Persona {
    pub name: String,                     // "Aria"
    pub role: String,                     // "researcher"
    pub description: String,              // free-form character text
    pub traits: Vec<String>,             // ["curious", "cautious"]
    pub system_prompt: String,           // identity injection sent to the model
    pub version: u32,                     // the persona is versioned
}

// Memory tiers — shared envelope
pub struct Memory {
    pub id: MemoryId,
    pub scope: Scope,                     // Agent(AgentId) | World
    pub kind: MemoryKind,
    pub created_at: DateTime<Utc>,
    pub last_access: DateTime<Utc>,
    pub access_count: u32,
    pub importance: f32,                  // 0..1, decay/forgetting signal
    pub deleted_at: Option<DateTime<Utc>>,// soft-delete
    pub embedding: Option<Vec<f32>>,     // Phase 2+
}

pub enum MemoryKind {
    // Factual → episodic
    Episodic { title: String, body: String, cues: FiveW },   // 5W: who/what/where/when/why
    // Factual → semantic
    Semantic { key: Option<String>, statement: String, category: SemanticCat },
    // Experiential → procedural
    Procedural { title: String, steps: Vec<String>,
                 successes: u32, failures: u32 },             // Wilson score comes from here
}

pub enum SemanticCat { Fact, Preference, Convention, Constraint }  // same as Alaz core_memory

pub struct FiveW { who: Vec<String>, what: Vec<String>, where_: Vec<String>,
                   when: Vec<String>, why: Vec<String> }
```

**Wilson score** (procedural confidence): as in Alaz, a lower-bound confidence interval
is computed from `successes`/`failures`; it becomes a signal in retrieval ranking.

---

## 7. The `MemoryStore` Trait (the heart)

```rust
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn remember(&self, mem: Memory) -> Result<MemoryId>;
    async fn recall(&self, scope: &Scope, query: &Query) -> Result<Vec<Scored<Memory>>>;
    async fn reinforce(&self, id: &MemoryId, outcome: Outcome) -> Result<()>; // access/success
    async fn forget(&self, id: &MemoryId) -> Result<()>;                      // soft-delete
    async fn consolidate(&self) -> Result<ConsolidationReport>;               // background
}
```

- `Query`: text + tier filter + limit + optional `hyde: bool`, `rerank: bool`
  (deliberately parallel to Alaz's search interface).
- `Scored<Memory>`: score + which signals contributed (explainability).
- Retrieval-level scope enforcement: a `recall` call never leaks outside its `scope`.

---

## 8. Retrieval Strategy (phased)

1. **Phase 1 — Flat + cheap:** substring/token keyword score + **recency decay**
   (Ebbinghaus-style time weighting) + importance + Wilson (for procedural). Dependencies: none.
2. **Phase 2 — Hybrid:** local embeddings via `fastembed`; **BM25 (tantivy) + cosine** fusion
   (Reciprocal Rank Fusion). Write-time conflict detection (0.6–0.9 band).
3. **Phase 3 — Smart:** **HyDE** (question→answer-shaped hypothesis), cross-encoder/LLM **rerank**,
   **MMR** for diversity, aggressive post-filtering.
4. **Phase 4 — Graph:** only if flat proves insufficient for multi-hop; entity extraction +
   native graph traversal (our own graph engine — `memory/graph.rs`).

---

## 9. Orchestration Model

- The `Orchestrator` holds a `Registry` (`AgentId -> Agent`).
- Each agent has a **mailbox** (tokio `mpsc`). Messages are `Envelope { from, to, payload }`.
- Starting pattern: **supervisor** — the orchestrator routes a message to the right agent
  and manages its lifecycle (spawn/stop). Single-writer safety is preserved.
- Later: **blackboard** (shared context across agents via a shared `World`-scope memory),
  pub/sub, hierarchical teams.
- Each agent-turn template: **recall (memory) → persona system_prompt → model → act →
  remember (new episode/fact/procedure)** — the memory loop closes.

---

## 10. Phased Roadmap (milestones)

> **Status (540 tests passing + 1 ignored [live-LLM], clippy clean, with CI):** M0–M31 ✅ + AI coworkers ✅ + code review fixes ✅ + 4-way review hardening ✅. **Next roadmap:** see [`docs/superpowers/specs/2026-07-24-next-roadmap.md`](../docs/superpowers/specs/2026-07-24-next-roadmap.md) (Phase A: correctness sweep, Phase B: doc sync, Phase C: e2e harness, Phase D: task HTTP surface, Phase E: maintenance).

- ✅ **M0 — Skeleton:** crate layout, `error.rs`, `id.rs`, empty modules, `lib.rs` compiles.
- ✅ **M1 — Memory core:** `Memory` types + `MemoryStore` trait + `InMemoryStore`
  + Phase 1 retrieval (keyword+recency) + **tests**. JSON snapshot persistence.
- ✅ **M2 — Identity:** `AgentId` + `Persona` + `Agent`; a "live → remember → recall" vertical slice
  end-to-end with `MockModel` (demo `main.rs`).
- ✅ **M3 — Orchestration:** `Orchestrator` + registry + mailbox; 2 agents message each other,
  each keeping its own memory.
- ✅ **M4 — Real model:** OpenAI-compatible `OpenAiModel` client (Ollama `:11434/v1`); persona
  system_prompt + recall injection.
- ✅ **M5 — Persistence + hybrid:** `SqliteStore` (rusqlite bundled) + **native `HashingEmbedder`**
  (char n-gram feature hashing — offline, no model download, captures Turkish morphology)
  + hybrid retrieval (keyword + cosine, `Query::semantic()` opt-in) + write-time conflict
  detection (`is_conflict`, cosine 0.6..0.9). Note: `fastembed` can later be plugged in behind the
  same `Embedder` trait; the native embedder is length-sensitive — short-query × long-document
  dilution is compensated in short (≤2 token) queries by a **token-level cosine fallback**
  (`Embedder::token_fallback`, signal: `cosine_tok` — disabled for neural).
- ✅ **M6 — Evolution:** `spawn_periodic` consolidation/decay task; soft-delete; forgetting
  signals (idle+importance+access+wilson); near-duplicate merge (cosine≥0.92).
- ✅ **M7 — Full engine:** **MMR** diversification (`Query::diverse()`) + **native graph**
  (`MemoryGraph`: entity index, neighbors, BFS `related`, multi-hop `path`) + **HyDE**
  (`Query::embed_text`, `Agent::recall_hyde`) + **native rerank** (`Reranker`/`NativeReranker`,
  `Query::rerank()`). The signals of "Alaz from zero" are native; no external dependency.
  Note: a neural cross-encoder / fastembed can later be plugged into the relevant traits.
- ✅ **M8 — Blackboard & collective reasoning:** `Orchestrator::with_blackboard` (shared
  World-scope board), `post_to_board`/`read_board`, `poll` (ask the whole team, collect
  replies), `deliberate` (blackboard pattern: question→board, team replies→board). Multi-agent collaboration.
- ✅ **M9 — Identity persistence:** `Agent::save_to`/`load_from` (persona+id JSON). Combined with
  SQLite memory, when an agent is reborn with the same `AgentId` → both character and memories
  return (proven by the `identity_survives_restart` test).
- ✅ **M10 — Tool use:** `Tool` trait + `ToolRegistry` + `ToolRouter` (native
  `KeywordRouter`) + `Agent::act` loop + embedded `CalcTool`. Agents touch the world
  through tools; usage is remembered episodically. LLM tool-calling can later plug into
  the same `ToolRouter` trait.
- ✅ **M11 — HTTP API/daemon:** `server` module (axum). `AppState` (shared store+model,
  agent map) + pure async methods (testable) + thin handlers. Endpoints:
  `GET /health`, `POST/GET /agents`, `POST /agents/:id/ask`, `POST /agents/:id/experience`,
  `GET /agents/:id/recall`. `LoreError`→HTTP status. Service mode via `LORE_SERVE=host:port`.
  End-to-end reqwest test (health/create/ask).
- ✅ **M12 — Persistence for the service:** `AppState::persistent(dir, store, model)` — personas at
  `<dir>/<id>.json`, memories in SQLite. All personas load at startup; `create` writes to disk.
  `LORE_DATA` directory. The service survives restarts (identity + memories return) — proven both
  by the `agents_survive_restart` test and a live binary smoke test.
- ✅ **M13 — CLI (clap):** `lore serve|new-agent|list|ask|remember|recall|demo`. CLI and service
  share the same persistent data directory (create from terminal → access over the network).
  `recall --semantic` captures Turkish morphology. `build_model`/`build_state` shared helpers.
- ✅ **M14 — Orchestration exposed as a service:** `POST /deliberate` (collective reasoning — ask the
  whole team, write replies to the shared World-scope board) + `GET /board`. CLI: `deliberate`,
  `board`. Because agents see the board, they reflect on each other's replies (emergent).
- ✅ **M15 — Agent lifecycle:** `PATCH /agents/:id` (partial persona update → `version`
  increments, written to disk — identity evolution) + `DELETE /agents/:id` (204). CLI: `update`, `delete`.
  `PersonaPatch` (all fields Option). `version` survives restarts (`patch_persists_across_restart`).
- ✅ **M16 — Tool use exposed as a service:** `AppState.with_tools(ToolContext)` a platform-level
  shared tool set; `AppState.act` runs the tool if it matches and remembers usage episodically,
  otherwise `respond`. `POST /agents/:id/act` + CLI `act`. Tools survive restarts (build_state
  rebuilds the toolset), and usage memory persists in SQLite.
- ✅ **M17 — Auth + rate limit:** `with_api_key` (optional; `X-API-Key` or `Bearer`) +
  `with_rate_limit` (fixed-window, per-key) native middleware. `/health` open,
  other endpoints protected (401/429). Env: `LORE_API_KEY`, `LORE_RATE_LIMIT` (per min). No extra dependencies.
- ✅ **M18 — Observability:** `GET /metrics` (Prometheus-style: lore_agents / requests_total /
  uptime_seconds / board_memories) + `count_mw` request-counter middleware. `/metrics` open.
- ✅ **M19 — LLM tool-calling:** `LlmRouter` (`ToolRouter` impl) presents the tool catalog to the
  model, the model selects a tool via JSON; `parse_tool_call` extracts JSON from prose/code-fences,
  a fabricated tool is rejected. Same trait as `KeywordRouter` — plugs seamlessly into `act`.
- ✅ **M20 — Streaming (SSE):** `POST /agents/:id/ask/stream` streams the reply word by word as SSE
  events (`[DONE]` terminator). Even if the model doesn't support streaming, the frontend does (`futures::stream`).
- ✅ **M21 — Inter-agent messaging:** `POST /agents/:id/message` (`ask` → the recipient replies +
  the sender remembers the exchange; `tell` → the recipient remembers). CLI `message`. Orchestrator Ask/Tell
  semantics over HTTP/CLI.
- ✅ **M22 — WebSocket live deliberate:** `GET /deliberate/live` — the question as a single text frame;
  each agent's reply streams as a JSON frame the moment it's ready, `[DONE]` at the end.
- ✅ **M23 — Federation (multi-node):** `LORE_PEERS` — deliberate merges the local team and peer Lore
  nodes (`DeliberateReply.node` label). Requests to peers carry `local:true` (loop broken); a dead
  peer is silently skipped. Lore talks to Lore, no external service.
- ✅ **M24 — Supervisor synthesis:** `deliberate --synthesizer <id>` / `{synthesizer}` —
  the supervisor doesn't participate in the poll, it synthesizes all replies (`Agent::respond_with` extra
  context); the result lands on the board as "Synthesis". Hierarchical team pattern.
- ✅ **M25 — Neural embedder (optional):** `--features neural` + `LORE_EMBEDDER=neural` →
  `NeuralEmbedder` (fastembed/ONNX, multilingual-e5-small, 384d, rustls). The default build
  stays fully offline; the native `HashingEmbedder` is always a fallback. Evidence: the query
  "pets" ranks the "love of cats" memory to the top despite zero shared tokens.
- ✅ **M26 — Conversation history (working memory):** the two-layer memory is complete —
  `Conversation` (bounded verbatim window, VecDeque) + `Prompt.history` (role-tagged
  `Turn`s; real user/assistant messages in OpenAI). `Agent::converse` adds working memory
  to the request; turns that fall out of the window are already in episodic memory (they
  return via retrieval). Service: a `session` field on `ask` — a dedicated lock per session
  (the same session is serialized, different sessions run in parallel), MAX_SESSIONS + idle TTL
  eviction, sessions dropped when the agent is deleted. CLI: interactive `lore chat <id>` + `ask --session`.
- ✅ **M27 — Real token streaming:** `Model::complete_stream` (`TokenStream`;
  the default impl streams `complete` as a single chunk — Mock-compatible). The OpenAI client
  parses the `stream:true` SSE body incrementally (UTF-8-safe line buffer,
  delta extraction; tested with a fake SSE server). `Agent::respond_stream`
  records the full reply episodically once the stream ends (a partial reply is not recorded);
  `AppState::ask_stream` holds the session lock until the stream ends and commits the exchange
  to the window at the end. The SSE endpoint is now real-time; the CLI `chat` prints chunks
  as they arrive.
- ✅ **M28 — ReAct loop:** `Agent::solve(ctx, input, max_steps)` — think → call
  tool → feed the observation back → final reply. A tool error/fabricated name also returns
  as an observation (the model self-corrects); on the last step the tool budget is exhausted
  (guaranteed termination); the tool trace is remembered as procedural experience.
  `POST /agents/:id/solve` + `lore solve`.
- ✅ **M29 — CI + rustfmt:** GitHub Actions — `fmt --check` + `clippy -D warnings` +
  `cargo test` on every push/PR; the entire codebase was run through fmt.
- ✅ **M30 — Embedder migration:** `Embedder::signature()` (hash-512-n3 / e5-small-384);
  the sqlite `meta` table stores the signature and warns at startup on a mismatch (old vectors
  won't match in the new space — silent death is over); `SqliteStore::reembed()` + `lore reembed`
  migrate all live records to the new space.
- ✅ **M31 — Backup + maintenance tuning:** `MemoryStore::export()`; `lore export/import`
  (deterministic JSON, ids preserved), `lore consolidate` manual maintenance,
  `LORE_CONSOLIDATE_SECS` period setting. Also: Dockerfile (multi-stage),
  README security notes (TLS/federation trust), a real-LLM ignored smoke test.

**Principle:** every milestone compiles + ships with tests. Complexity is added only when a
concrete need is proven (flat → hybrid → graph).

### Code review fixes (after M25, 4 phases)

The entire codebase was reviewed; 2 critical, 10 major, 18 minor findings were addressed:

- **Security:** the rate-limit key is derived from a validated API key / client IP instead of an
  attacker-controlled header; eviction on the `hits` table (memory DoS); constant-time key
  comparison; 500 responses don't leak internal details (logged on the server); `/metrics`
  protected; 401/429 JSON body + standard headers.
- **Resilience:** sqlite WAL + busy_timeout (CLI ↔ service sharing); OpenAiModel 120s
  timeout; deliberate doesn't die on a single agent error (warn + skip); a corrupt persona file
  doesn't stop startup; a deleted agent can't be resurrected on restart; graceful shutdown.
- **Performance/async hygiene:** SqliteStore does all work in the `spawn_blocking` pool;
  deliberate + federation in parallel (`join_all`, order preserved); persist happens outside the
  heavy lock; metrics use `COUNT` instead of a full scan (`MemoryStore::count`); a shared reqwest client.
- **Correctness:** evolution wired into the service (hourly consolidation — memory no longer grows
  unbounded); the semantic gate / conflict band are embedder-specific (`Embedder::semantic_gate`
  0.40 hashing / 0.80 e5); CalcTool requires an operator + i64 overflow protection;
  `LoreError::InvalidInput` → HTTP 422; `KeywordRouter` deterministic (BTreeMap);
  SSE preserves whitespace; an empty PATCH doesn't increment the version.
- **Modularization:** `server/mod.rs` (~1600-line monolith) split into focused files:
  `state.rs` (AppState core), `deliberate.rs` (collective reasoning + federation),
  `security.rs` (auth + rate limit), `api.rs` (router + handlers + serve),
  `types.rs` (DTOs), `tests.rs`. The public API was preserved exactly (`pub use` re-exports).

---

## 11. Open Questions

1. **Identity persistence:** should persona + memory live in a single file (one `.lore`
   folder per agent) or in a single central DB?
2. **Depth of the engine:** how far do we take "Alaz from zero" — are all 6 signals
   (FTS+vector+ColBERT+graph+RAPTOR+decay) the target, or is hybrid (BM25+vector)
   + decay + Wilson enough? (ColBERT/RAPTOR are serious additional engineering.)
3. **Model provider:** target local Ollama first, or OpenAI/Anthropic/Gemini/ZAI?
4. **`rig` dependency:** should we lean on `rig` on the LLM side, or a thin client of our own?
5. **Theme/lore:** how important is story/character depth in the persona system (e.g.
   relationships between agents, "memory sharing") — is it the product's differentiator,
   or should it stay minimal?
