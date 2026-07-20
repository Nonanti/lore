# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/tr/1.1.0/) · Versioning: [SemVer](https://semver.org/lang/tr/).
During the 0.x series, minor bumps may contain breaking changes; all are marked here.

## [Unreleased]

### Fixed
- **Recalled memories now reach the model**: prompt context injection used
  `Memory::summary()` — which for an **episodic** record is only its *title*, so
  the body (the actual remembered content) never reached the model. `ask`/
  `respond` now inject a new `Memory::recall_context()` that includes the
  episodic body / semantic statement / procedural steps (capped per line to keep
  the prompt lean). `remember`/`experience` knowledge is now actually used in
  answers; `summary()` stays the compact one-liner for CLI/board listings.

### Added
- **Native provider auth (Anthropic + OpenAI)**: Lore now has its
  **own** self-contained credential subsystem (`src/auth/`) — it reads no other
  tool's credentials. An agent can be driven by a metered **API key** or a
  consumer **subscription** (Claude Pro/Max; ChatGPT Plus/Pro via Codex).
  - `CodexModel` (`src/model/codex.rs`): OpenAI **Responses** API over the
    ChatGPT subscription backend (`/backend-api/codex/responses`), `Bearer` +
    `ChatGPT-Account-Id` (account id decoded from the login id-token JWT),
    SSE `response.output_text.delta` streaming. OpenAI OAuth uses the
    `auth.openai.com` endpoints (form-encoded) with the Codex client id.
    `lore login openai` (browser loopback on `localhost:1455`).
    *Wired and reaches the live backend (auth/endpoint confirmed); a full
    completion was not live-verified in this build.* The metered OpenAI API-key
    path uses `OpenAiModel` (Chat Completions) unchanged.
  - `AnthropicModel` (`src/model/anthropic.rs`): Anthropic Messages API
    (`/v1/messages`) with real SSE streaming and the shared idle-timeout
    discipline. Two auth modes: `x-api-key` (official) and subscription
    `Authorization: Bearer` (adds the Claude Code beta headers and forces the
    server-required Claude Code identity as the first system block).
  - `src/auth/`: PKCE (S256) helpers, a `0600` atomic `TokenStore` under
    `LORE_DATA/auth/<provider>.json`, and an `AccessTokenProvider` that
    auto-refreshes OAuth tokens on use (persisting the rotation) — so a
    long-running server survives token expiry mid-session.
  - CLI: `lore login <provider> [--device]` (browser loopback or paste-the-code
    flow), `lore logout <provider>`, `lore auth` (status + expiry).
  - Selection: `LORE_PROVIDER=anthropic` + `LORE_AUTH=key|subs` (auto-detected
    when unset); `LORE_LLM_BASE` (OpenAI-compatible) path is unchanged.
  - New dependency: `sha2` (PKCE challenge). Fragility of the unofficial
    subscription OAuth constants is documented in the README.
  - **Review hardening**: OAuth `state` is verified on the callback (CSRF
    guard); loopback login binds IPv4 **and** IPv6 and uses a matching redirect
    (no hang on IPv6-first hosts) with a 5-minute timeout instead of a blocking
    accept; the token store dir is `0700` and files `0600` (with a warning on
    non-Unix); `RefreshingToken` re-reads disk before minting to narrow the
    multi-process refresh race; `CodexModel` surfaces `response.failed` as an
    error instead of a silent truncation; Anthropic default `max_tokens` raised
    to 4096.

- **Learning loop (reflect)**: frequently recalled episodic memories (access ≥ 2)
  are distilled by the model into one-sentence persistent knowledge and promoted
  to the semantic tier (`Agent::reflect`, `POST /agents/:id/reflect`, `lore reflect`);
  the category (Fact/Preference) is parsed from the model output; the original memory
  is archived via soft-delete. Periodic autonomous run in the service janitor
  (`LORE_REFLECT_SECS`, default 3600, 0 = off).
- **Native tools**: `time` (UTC timestamp), `web` (http GET, capped at 64KB,
  **SSRF-protected** — private/loopback/link-local/CGNAT addresses blocked by
  default, enabled with `LORE_WEB_ALLOW_PRIVATE=1`), `file` (root-sandboxed
  reading, traversal denied, root created lazily) + English router keys.
- **Quality harness**: `tests/eval.rs` golden set (hit@5 100%) + keyword
  baseline — a regression alarm for retrieval calibration.
- **Review findings, round 2**:
  - `reembed` and consolidation writes are a **single transaction** (no per-row
    autocommit; reembed is atomic — if interrupted midway, no mixed embedding
    space forms); the v1→v2 backfill consumes rows lazily (the whole table isn't
    pulled into RAM); only the 6 features actually used instead of tokio `full` (L9).
  - `respond` memory access is now semantic — morphology capture is enabled in
    the agent's own thinking loop too (H2).
  - Consolidation dedup uses **multi-probe sort-LSH** instead of an O(n²) full
    scan (64-bit random-hyperplane signature, 4 deterministic bit-permutations,
    sorted sliding window; candidates verified with exact cosine) — the candidate
    count is linear, independent of data clustering. On clustered 10k records
    ~75s (full scan) → **~0.32s**; `cargo bench` baseline `dedup_lsh` (M2).
  - Persona writes are **atomic** (tmp + rename): the "corrupt JSON mid-crash →
    agent silently lost on restart" problem is closed (M3).
  - Streaming timeout is **per-chunk idle**: a slow but progressing stream
    (a long-thinking reasoning model) isn't cut off; a stalled stream raises an
    error quickly. For single-shot requests the total duration is preserved (M4).
  - `LORE_API_KEY=""` is rejected (an open-door trap), a security warning for a
    plain-http remote federation peer, a cap on the agent count
    (`LORE_MAX_AGENTS`, default 1024) (L1/L5/L6).
  - Reasoning-fallback responses are written to memory truncated (no CoT
    pollution; `Completion.reasoning_fallback` flag) (L7).
- **Memory feedback (H1 fix)**: `reinforce` is now used in production paths —
  decay's "preserve what's used" rule and procedural Wilson tracking are no
  longer dead code.
  - `Agent::recall` automatically reinforces the records it returns on text
    queries (`last_access` refreshed, `access_count` incremented; batch
    `reinforce_many` — a single blocking call in sqlite). Browse (text-less)
    bulk scans do not reinforce: graph building/board reading does not kill decay.
  - `Agent::solve` learns procedures: a successful tool chain becomes a Procedural
    record; a similar task following the same tool sequence strengthens the
    existing procedure with `Success` INSTEAD OF duplicating it. Proven procedures
    (Wilson ≥ threshold) enter the next solve prompt as guidance; a run that ends
    at the step limit marks the injected procedures with `Failure`.
  - New endpoint: `POST /agents/:id/reinforce` (`accessed|success|failure`,
    scope-validated — another agent's record → 404) + `lore reinforce` CLI
    command; `lore recall` output now shows the record id.
  - `get(id)` and `reinforce_many(ids, outcome)` added to the `MemoryStore` trait
    (with defaults; stores may override with a batch impl).
- **Observability**: `tracing`-based structured logging (`LORE_LOG` env-filter),
  `x-request-id` on every response, `/metrics` extended with a route-based latency
  histogram and retrieval/consolidation counters, a `/ready` readiness endpoint.
- **Performance (schema v2)**: SQLite FTS5 index — keyword recall does no
  full-table scan (206ms → ~4ms on 10k records); embedding in a separate BLOB
  column (semantic pre-selection doesn't parse JSON); on file-based stores
  consolidation runs on a separate connection (the hot path isn't blocked). Old
  files are migrated automatically on startup. `cargo bench` (criterion) baseline
  measurements.
- **Test depth**: proptest property tests (parser/tokenizer/SSE/auth never panic +
  correctness properties), real-binary e2e (`tests/e2e.rs`: CLI lifecycle,
  SIGKILL + restart persistence), weekly live-LLM smoke + `cargo audit` in CI.
- **API**: `/openapi.json` (OpenAPI 3.1, hand-maintained, guarded by a spec-drift
  test); this CHANGELOG.
- Tool contract: `Tool::args_hint()` + shared `catalog()`; `CalcTool` flexible
  argument parsing (comma/JSON-like/adjacent; sign, scientific notation and hex
  prefix preserved).
- Reasoning model support: fallback to `reasoning_content` when content is empty;
  `with_max_tokens` / `LORE_LLM_MAX_TOKENS`; `<think>…</think>` block stripping
  (single-shot + line-safe streaming filter, with split-tag carry);
  request timeout via `LORE_LLM_TIMEOUT` (for slow local models).
- WebSocket `deliberate/live` federation: peer node responses also stream live
  (tagged with `node`) — same scope as HTTP `/deliberate`.
- Token-level cosine fallback for short (≤2 token) queries (`cosine_tok`
  signal) — morphological recall isn't lost to dilution.

### Changed
- **[BREAKING]** `AppState::deliberate_synth` takes `local: bool` as a third
  parameter (the depth-1 guarantee is preserved with the synthesizer too).
- **[BREAKING]** `orchestrator::Delivery` carries a new `error: Option<String>`
  field; on a single-agent error `Orchestrator::run()` marks the delivery with
  the error and continues instead of losing the transcript.
- Automatic records (exchange/tool/message/board traces) are born with
  `Memory::AUTO_IMPORTANCE` (0.2) — decay can now reclaim them; explicit
  `remember`/`experience` and `tell` stay at 0.5.
- `score_with_gate` is **deprecated** — use `Scorer`/`score_with_embedder`.

### Security
- Session table is hard-capped (TTL + LRU; no eviction applied to the active
  session); `session` name ≤128 bytes. WS message/frame limit 64KB. Peer response
  body ≤2MB + status-code check. `create`/`PATCH` reject an empty name/role.
  ULID format check on the persona file path (a second line of defense).
  Log-forging protection (`log_safe`). `LORE_RATE_LIMIT`/`LORE_CONSOLIDATE_SECS`
  warn on an invalid value (no silent disabling).

## [0.1.0] — 2026-07-17

First whole: identity (Persona/Agent), three-tier memory (episodic/semantic/
procedural; hybrid retrieval + MMR + HyDE + rerank + graph + evolution),
orchestration (Envelope/blackboard/poll), HTTP API (15+ endpoints, SSE, WebSocket,
federation), CLI, SQLite persistence, optional neural embedder. M0–M31.
