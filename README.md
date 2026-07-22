# Lore

> An **identity + orchestration + memory** core for AI agents — fully self-contained,
> native Rust. Not tied to any external service, HTTP, or API; the memory engine is
> written from scratch.

Lore gives AI agents a persistent **identity** (persona) and a personal **memory**, and
**manages them together** in a community. The two things that make an agent *that* agent —
who it is and what it has experienced — come together here.

![Lore demo — identity, orchestration, memory](demo.gif)

*Create an agent → teach it a fact → it answers using its memory → semantic recall
(`lore recall --semantic`). Recorded against a local Ollama model; `lore demo` also
runs fully offline on the built-in MockModel — no API key needed.*

## Why?

A stateless LLM starts every conversation from scratch. By giving the agent a memory that
accumulates over time and a consistent identity, Lore turns it into an **evolving entity**.

## Architecture

Three subsystems, all within a single binary:

| Subsystem         | Responsibility                     | Main types |
|-------------------|------------------------------------|-----------|
| **Identity**      | Persona, identity, capabilities    | `Agent`, `Persona`, `AgentId` |
| **Orchestration** | Managing agents, messaging, collaboration | `Orchestrator`, `Envelope`, `Registry`, blackboard |
| **Memory**        | Native memory engine               | `MemoryStore`, `InMemoryStore`, `SqliteStore` |

### Memory — three tiers

- **Episodic** — experienced events (5W cues, timestamped)
- **Semantic** — facts/preferences (`Fact`/`Preference`/`Convention`/`Constraint`)
- **Procedural** — learned skills (confidence tracking via Wilson score)

### Retrieval (hybrid)

A fusion of `keyword (coverage+tf)` + `cosine (native char n-gram embedding)`, topped
with a `recency (Ebbinghaus decay)` + `importance` + `Wilson` boost. For short (≤2 token)
queries a token-level cosine fallback kicks in — the query "learning" finds the "Learned
Rust ..." memory without getting lost in document dilution. Options:

- `Query::semantic()` — retrieve semantically (morphology/synonym) even if keywords don't match
- `Query::diverse()` — diversify results with MMR
- `Query::rerank()` — native second-pass rerank (coverage + phrase + bigram)
- `Agent::recall_hyde()` — HyDE: generate a hypothetical answer → embed → search
- `MemoryGraph` — entity graph: `neighbors`, `related` (BFS), `path` (multi-hop)
- retrieval-level **scope isolation** (agents cannot see each other's private memory)

**Scale** (schema v2): in SQLite, keyword queries run over an **FTS5 index** (no
full-table scan); semantic queries are pre-selected with a lightweight scan
(`id, search_text, emb BLOB`) without paying the JSON-parse cost; embeddings are kept
in a separate BLOB column. Old files are migrated automatically on startup. `cargo bench`
baselines (10k records, single core): keyword recall **~4 ms** (206 ms before FTS), short
semantic recall ~66 ms (token-level fallback cost — only for ≤2 token queries), browse
under ~1 ms.

### Evolution (maintenance)

Background consolidation: **forgetting via decay** (old + unimportant + unused),
**near-duplicate merge** (cosine ≥ 0.92), **soft-delete** (auditability). Also write-time
**conflict detection** (cosine 0.6–0.9). System-generated records (exchange/tool/message/board
traces) are born with `Memory::AUTO_IMPORTANCE` (0.2): below the forgetting threshold —
decay can reclaim unused ones, while accessed ones are preserved via `access_count`.
Explicit `remember`/`experience` records are preserved at 0.5. On file-based stores,
consolidation runs **on a separate connection** (WAL) — the O(n²) dedup scan does not block
the hot recall/remember path.

## Quick start

```rust
use lore::{Agent, InMemoryStore, MemoryStore, Persona, Query, MockModel, HashingEmbedder};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store: Arc<dyn MemoryStore> =
        Arc::new(InMemoryStore::new().with_embedder(Arc::new(HashingEmbedder::new())));

    let aria = Agent::new(
        Persona::new("Aria", "researcher").with_traits(["curious", "meticulous"]),
        store.clone(),
        Arc::new(MockModel::new()),
    );

    aria.experience("Learned Rust", "ownership and the borrow checker").await?;
    let reply = aria.respond("what do you know about rust").await?;
    println!("{reply}");
    Ok(())
}
```

For a real model (if not offline):

```bash
LORE_LLM_BASE=http://localhost:11434/v1 LORE_LLM_MODEL=llama3.2 cargo run
```

### Providers & auth (OpenAI / Anthropic — API key *or* subscription)

Lore ships **its own** provider auth — fully self-contained, no external
credential store is read. You can drive an agent with a metered **API key** or a
consumer **subscription** — **Anthropic** (Claude Pro/Max) and **OpenAI**
(ChatGPT Plus/Pro, Codex).

```bash
# Anthropic with a subscription (Claude Pro/Max) — native PKCE OAuth login.
lore login anthropic            # browser loopback flow (or: --device to paste a code)
lore auth                       # show configured credentials + expiry
LORE_PROVIDER=anthropic LORE_LLM_MODEL=claude-sonnet-4-5-20250929 \
  lore ask <agent> "..."        # tokens auto-refresh on use

# OpenAI with a subscription (ChatGPT Plus/Pro) — Codex Responses API.
lore login openai               # browser loopback (redirect http://localhost:1455/auth/callback)
LORE_PROVIDER=openai LORE_LLM_MODEL=gpt-5 lore ask <agent> "..."

# Metered API keys.
LORE_PROVIDER=anthropic LORE_AUTH=key ANTHROPIC_API_KEY=sk-ant-... \
  LORE_LLM_MODEL=claude-sonnet-4-5-20250929 lore ask <agent> "..."
LORE_PROVIDER=openai LORE_AUTH=key OPENAI_API_KEY=sk-... \
  LORE_LLM_MODEL=gpt-4o lore ask <agent> "..."

lore logout <anthropic|openai>  # remove the stored credential
```

> **Verification status:** the Anthropic subscription path is validated
> end-to-end against the live API. The OpenAI/Codex subscription path is wired
> and reaches the live ChatGPT backend (auth/endpoint confirmed) but a full
> completion was **not** live-verified in this build (no live ChatGPT token was
> available); its request shape follows the Codex CLI references.

OAuth tokens are stored under `LORE_DATA/auth/<provider>.json` (`0600` files in a
`0700` dir, atomic write) and refreshed by Lore itself (with `state` verified on
the callback). `LORE_AUTH=key|subs` forces a mode; with it unset, Lore
auto-detects (subscription if a stored OAuth credential exists, else an API key
from `ANTHROPIC_API_KEY`/`LORE_LLM_KEY`). Note: refresh uses an in-process lock
only — running two Lore processes against one `LORE_DATA` narrows but does not
fully close the token-refresh race (single-operator use is assumed).

> **Fragility note:** subscription OAuth relies on provider constants (client id,
> endpoints, the Claude Code identity/beta headers) that are **not officially
> published for third-party use** and can change or be revoked. The metered
> API-key path is the stable one. Anthropic's OAuth is scoped to Claude
> Code/claude.ai — using it elsewhere may violate provider terms.

## HTTP service

```bash
LORE_SERVE=127.0.0.1:3777 LORE_DATA=./lore-data cargo run
```

Identities live in `LORE_DATA/agents/<id>.json`, memories in `LORE_DATA/lore.db` (SQLite) —
the service survives restarts: agents come back with the same identity + memories.

Security (optional): `LORE_API_KEY=secret` protects every endpoint (except `/health`) via
`X-API-Key` / `Authorization: Bearer`; `LORE_RATE_LIMIT=60` sets a per-minute request limit
(401/429). The session table has a hard cap (1000; when full, idle entries are evicted first,
then LRU) and a `session` name may be at most 128 bytes. Note: when auth is off, the
rate-limit key is the client IP — behind a reverse proxy all clients share the proxy IP
(a single bucket); use `LORE_API_KEY` behind a proxy (key-based limiting).

Model tuning (optional): `LORE_LLM_MAX_TOKENS=1024` sends the response token limit
(default: the provider's default). On reasoning models (e.g. GLM-4.6) low values may spend
the budget on thinking — if content stays empty, Lore falls back to `reasoning_content`.
`LORE_LLM_TIMEOUT=300` sets the request timeout in seconds (default 120) — can be raised
for slow local models (14B+ CPU). `<think>…</think>` blocks that reasoning models leak into
the response are stripped in both single-shot and streaming responses (models that keep
reasoning in a separate field, like glm, are unaffected).

Federation (multi-node): `LORE_PEERS=http://nodeB:3777,http://nodeC:3777` — `deliberate`
makes peer Lore nodes' teams respond alongside the local team (responses arrive tagged with
`node`; the WebSocket `deliberate/live` also streams peer responses live). Lore only talks
to Lore — still no external service.

### Learning loop and tools

- **Reflect (episodic → semantic distillation):** frequently recalled memories (access ≥ 2)
  are distilled by the model into one-sentence persistent knowledge and promoted to the
  semantic tier; the original memory is archived (soft-delete). Runs automatically on a
  schedule (`LORE_REFLECT_SECS`, default 3600s; `0` = off) or is triggered manually:
  `POST /agents/:id/reflect` / `lore reflect <id>`.
- **Reinforce (external feedback):** decay/Wilson signals can be fed externally via
  `POST /agents/:id/reinforce` (`accessed|success|failure`); it is scope-validated
  (another agent's record → 404). CLI: `lore reinforce <agent> <memory> <outcome>`.
- **Native tools:** `time` (UTC), `web` (http GET, capped at 64KB, **SSRF-protected** —
  private/loopback addresses blocked by default; use `LORE_WEB_ALLOW_PRIVATE=1` for your
  own services), `file` (reading within the data-directory sandbox), `calc` (calculator).
- **Agent cap:** `LORE_MAX_AGENTS` (default 1024) — a fan-out DoS brake.

Neural embedder (optional): `cargo build --features neural` + `LORE_EMBEDDER=neural` →
multilingual real semantic recall (fastembed/ONNX, multilingual-e5-small). The default build
is fully offline; the native embedder is always the fallback. After switching embedders,
`lore reembed` migrates old records into the new space (a signature mismatch is warned about
on startup).

### Security notes (threat model)

**Assumptions**: Lore is a single-operator, trusted-network-first core. Exposure to the
internet = the trio of reverse proxy + `LORE_API_KEY` + rate limit.

- **No TLS (by design)** — Lore speaks plain HTTP; TLS termination is the reverse proxy's
  (nginx/caddy) job. The API key is only secure over TLS. When listening keyless +
  off-loopback, the service logs a warning on startup.
- **Single API key** — no per-user key/permission separation (deliberately simple); key
  verification is constant-time (no timing leak).
- **Input limits** — HTTP body 2MB (axum), WS message/frame 64KB, `session` name 128B,
  query `limit` ≤1000, peer response body 2MB. The session table is hard-capped (TTL+LRU) —
  memory cannot be grown via client-controlled fields.
- **Federation trust** — peers are authenticated with a shared secret (`LORE_PEER_KEY`);
  agent names in peer responses are THE PEER'S CLAIM (unsigned) — only pair with nodes you
  trust. Questions go to peers as `local:true`: they stay at depth 1, no loops.
- **Log hygiene** — user text enters logs after newline sanitization (no log forging);
  internal errors are not leaked to the client (500 + generic message).
- **Supply chain** — `cargo audit` on every CI run; 16 runtime dependencies
  (`sha2` added for the PKCE challenge in native OAuth login).

| Endpoint | Description |
|----------|----------|
| `GET  /health` | health |
| `POST /agents` | create an agent (`{name, role, traits}`) |
| `GET  /agents` | list agents |
| `PATCH  /agents/:id` | update persona (`{name?, role?, traits?, ...}`) — `version` increments |
| `DELETE /agents/:id` | delete the agent |
| `POST /agents/:id/ask` | ask the agent (`{message, session?}` — `session` preserves conversation history) |
| `POST /agents/:id/act` | make it do something (`{input}`) — runs if a tool matches |
| `POST /agents/:id/ask/stream` | stream the response in REAL time via SSE (token by token; `session` supported) |
| `POST /agents/:id/message` | inter-agent message (`{from?, kind, content}`) |
| `GET  /metrics` | Prometheus-style metrics (open) |
| `POST /agents/:id/experience` | add a memory (`{title, body}`) |
| `GET  /agents/:id/recall?q=&limit=&semantic=` | recall |
| `POST /deliberate` | collective reasoning (`{question, synthesizer?, local?}`) — team + peer nodes |
| `GET  /deliberate/live` | WebSocket: responses stream live as they become ready |
| `GET  /board?limit=` | read the shared board |

## CLI

The CLI and the service share the same persistent data directory — an agent you create
from the terminal is also reachable over the network via `serve`.

```bash
cargo run -- new-agent --name Aria --role researcher --traits curious,meticulous
cargo run -- list
cargo run -- remember <id> --title "Rust" --body "ownership is strong"
cargo run -- recall <id> philosophy --semantic   # captures morphology
cargo run -- update <id> --role senior --traits wise,patient   # version increments
cargo run -- delete <id>
cargo run -- ask <id> "what do you know"
cargo run -- chat <id>                           # interactive multi-turn chat
cargo run -- act <id> "calculate 12 * 3"        # runs if a tool matches
cargo run -- solve <id> "what is (3+4)*6?"      # multi-step tool chain (ReAct)
cargo run -- message <kai> "sprint tomorrow" --from <aria> --kind tell   # inter-agent
cargo run -- deliberate "what should we do tomorrow"     # the whole team responds
cargo run -- deliberate "make a decision" --synthesizer <id>   # the supervisor synthesizes
cargo run -- board                              # shared board
cargo run -- login anthropic                     # subscription OAuth login (Claude Pro/Max)
cargo run -- login openai                        # subscription OAuth login (ChatGPT/Codex)
cargo run -- auth                                # show provider credentials + status
cargo run -- logout anthropic                    # remove a stored credential
cargo run -- serve --addr 127.0.0.1:3777
cargo run -- export --out backup.json           # export memory
cargo run -- import backup.json                 # restore (ids preserved)
cargo run -- consolidate                        # trigger memory maintenance manually
cargo run -- reembed                            # after an embedder migration
cargo run -- demo
```

Docker: `docker build -t lore . && docker run -p 3777:3777 -v lore-data:/data lore`

## Commands

```bash
cargo run      # demo (when no subcommand)
cargo test     # 193 tests passing (1 ignored: live-LLM smoke test)
cargo clippy --all-targets
```

## Status

M0–M31 complete (identity, memory, orchestration, real model, persistence, evolution, graph+HyDE+rerank,
blackboard/collective reasoning, identity persistence, tool use, persistent HTTP service + CLI, collective-reasoning API, agent lifecycle + persona versioning, tool-use API, auth + rate limit,
observability /metrics, LLM tool-calling, SSE streaming, inter-agent messaging, live WebSocket deliberate,
multi-node federation, supervisor synthesis, optional neural embedder,
conversation history — `lore chat` interactive chat + HTTP `session` support,
real token streaming — OpenAI `stream:true` flows end-to-end into SSE,
ReAct multi-step tool chain — `lore solve`, CI, embedder migration — `lore reembed`,
backup — `lore export/import`, Dockerfile).
A full code review was done after M25: fixes landed for security (rate-limit key, constant-time comparison,
500 detail leak), resilience (WAL, timeouts, fault-tolerant deliberate) and async
hygiene (spawn_blocking, parallel fanout); the `server` module was split into focused
files (state / deliberate / security / api / types).

Maturation phases (see [`CHANGELOG.md`](CHANGELOG.md)): structured logging + request-id +
latency histograms + `/ready` · FTS5 + emb BLOB + separate-connection consolidation
(keyword recall @10k: ~4ms) · property tests + real-binary e2e (SIGKILL/restart
persistence) · `/openapi.json` + threat model · soak/chaos harness (`scripts/soak.sh`:
continuous load + periodic SIGKILL; sample run: 2948 requests, 2 restarts, 0 errors,
ask p95=9ms, full data integrity). Details: [`DESIGN.md`](DESIGN.md).

## Design philosophy

- **Full self-containment** — no external services, everything inside the binary
- **Start flat** — complexity is added only when evidence arrives (flat → hybrid → graph)
- **Abstraction behind traits** — `MemoryStore` / `Model` / `Embedder` are swappable
- **Every milestone compiles + ships with tests**

License: MIT
