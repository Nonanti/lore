# Lore — Comprehensive Test Report

| | |
|---|---|
| **Project** | `lore` — an identity + orchestration + memory core for AI agents (Rust) |
| **Version** | 0.1.0 (M0–M31 done) |
| **Test date** | 17 July 2026 |
| **Tested by** | Berkant (pi agent) |
| **Test scope** | Quality gates, CLI, HTTP API, security, real LLM (2 providers), federation, WebSocket, memory behaviors |
| **Result** | ✅ **108/108 tests passed · clippy clean · runtime works end-to-end** · 1 medium + 1 minor finding |

---

## 1. Summary

The Lore project was tested end to end: build/lint/test gates, CLI flow, HTTP service
(15 endpoints), auth + rate-limit, two different real LLMs (Ollama qwen3:14b + Z.ai GLM-4.5-air),
multi-node federation, WebSocket live deliberate, runtime scope isolation, memory maintenance
commands, and export determinism.

The project is **very solid**: 8071 lines of Rust, 108 unit tests, clean clippy (`-D warnings`),
a well-documented design (DESIGN.md), and disciplined phased growth. No serious bugs were found.
The only medium finding is that LlmRouter doesn't specify the tool argument format to the model
(ReAct solve gets stuck with qwen3).

---

## 2. Test Environment

| Component | Detail |
|-----------|--------|
| Machine | local (x86_64 Linux) |
| Binary | `./target/debug/lore` (default features, offline) |
| Real model 1 | **Ollama** `qwen3:14b` — over SSH `nonaserver@192.168.1.26`, OpenAI-compatible `http://192.168.1.26:11434/v1` |
| Real model 2 | **Z.ai GLM-4.5-air** — `https://api.z.ai/api/paas/v4` (Bearer auth) |
| Models (Z.ai) | glm-4.5, glm-4.5-air, glm-4.6, glm-4.7, glm-5, glm-5-turbo, glm-5.1, glm-5.2 |
| Test data | isolated `LORE_DATA` directories via `mktemp` (cleaned up after each test) |

---

## 3. Methodology

1. **Quality gates** — `cargo fmt --check`, `cargo build`, `cargo clippy --all-targets -- -D warnings`, `cargo test`.
2. **Static reading** — DESIGN.md, lib.rs, main.rs, server/{api,security}.rs, memory/{sqlite,retrieval}.rs, model/openai.rs, tool/{mod,builtin}.rs.
3. **CLI flow** — an end-to-end command chain with a temporary data directory (new-agent → ... → delete + restart simulation).
4. **HTTP API** — the service brought up and every endpoint hit with `curl`; error codes (404/422) verified.
5. **Security** — a separate protected service (`LORE_API_KEY` + `LORE_RATE_LIMIT`); auth and rate-limit behaviors.
6. **Real LLM** — ask / ask-stream (SSE) / solve (ReAct) with Ollama and Z.ai.
7. **Distributed** — two-node federation (`LORE_PEERS`); WebSocket `/deliberate/live` (Python stdlib raw WS client).
8. **Memory** — scope isolation (runtime), reembed, export determinism, semantic recall, consolidate.
9. **Performance** — cold-start time measured with `strace` + a Python subprocess.

---

## 4. Test Results

### 4.1 Quality gates

| Gate | Command | Result |
|------|---------|--------|
| Format | `cargo fmt --check` | ✅ clean (exit 0) |
| Build | `cargo build` | ✅ successful |
| Lint | `cargo clippy --all-targets -- -D warnings` | ✅ clean (exit 0) |
| Unit test | `cargo test` | ✅ **108 passed · 1 ignored · 0 failed** (0.03 s) |

`1 ignored` = the real-LLM smoke test (`live_llm_complete_and_stream`, `#[ignore]`).

### 4.2 CLI flow (persistence)

A sequential command chain with a temporary `LORE_DATA`. All commands ran successfully:

| Step | Command | Result |
|------|---------|--------|
| new-agent | `lore new-agent --name Aria --role researcher --traits curious,meticulous` | ✅ ULID generated |
| list | `lore list` | ✅ 2 agents listed |
| remember | `lore remember <id> --title "Learned Rust" --body "ownership..."` | ✅ |
| recall (keyword) | `lore recall <id> rust` | ✅ score 0.824 |
| recall (semantic) | `lore recall <id> "learning" --semantic` | ⚠️ empty (see §5.3) |
| ask | `lore ask <id> "what do you know about rust"` | ✅ MockModel reply |
| act (calc) | `lore act <id> "calculate 5 * 7"` | ✅ → **35** |
| message | `lore message <kai> "sprint tomorrow" --from <aria> --kind tell` | ✅ delivered |
| update | `lore update <id> --role senior --traits wise,patient` | ✅ version **v2** |
| deliberate | `lore deliberate "give a summary"` | ✅ 2 agent replies |
| board | `lore board` | ✅ 3 records (question + replies) |
| export | `lore export --out yedek.json` | ✅ 9 records |
| **restart sim** | `lore list` (new process, same data) | ✅ **persistent** — Aria v2 is retained |
| consolidate | `lore consolidate` | ✅ scanned: 9 · merged: 1 · forgotten: 0 |
| delete | `lore delete <kai>` | ✅ deleted |
| import | `lore import yedek.json` | ✅ 9 records round-trip |

### 4.3 HTTP API (15 endpoints)

MockModel service (`127.0.0.1:13788`, health ready in **15 ms**):

| Endpoint | Method | Result |
|----------|--------|--------|
| `/health` | GET | ✅ `ok` (200) |
| `/agents` | POST | ✅ agent created (version 1) |
| `/agents` | GET | ✅ list |
| `/agents/:id` | PATCH | ✅ persona updated (version 2) |
| `/agents/:id` | DELETE | ✅ 204 |
| `/agents/:id/ask` | POST | ✅ `{reply:...}` |
| `/agents/:id/ask/stream` | POST | ✅ **SSE**: `data: ...` + `data: [DONE]` |
| `/agents/:id/act` | POST | ✅ `{result:"72"}` (9*8) |
| `/agents/:id/solve` | POST | ⚠️ MockModel didn't call a tool (see §4.6) |
| `/agents/:id/experience` | POST | ✅ 201 |
| `/agents/:id/recall?q=` | GET | ✅ score 0.904 |
| `/agents/:id/message` | POST | ✅ |
| `/deliberate` | POST | ✅ collective reasoning |
| `/board` | GET | ✅ |
| `/metrics` | GET | ✅ Prometheus format |

**Error mapping:**
- Unknown agent → `POST /agents/WRONG/ask` → **404** ✅
- Empty create body → `POST /agents {}` → **422** ✅

### 4.4 Security — Auth + Rate Limit

Protected service: `LORE_API_KEY=gizli123 LORE_RATE_LIMIT=3`

| Scenario | Expected | Result |
|----------|----------|--------|
| `/health` no key | 200 (open) | ✅ 200 |
| `/agents` no key | 401 | ✅ 401 + `{"error":"unauthorized..."}` |
| `/agents` wrong key | 401 | ✅ 401 |
| `/agents` `X-API-Key: gizli123` | 200 | ✅ 200 |
| `/agents` `Authorization: Bearer gizli123` | 200 | ✅ 200 (case-insensitive scheme) |
| `/metrics` no key | 401 (protected) | ✅ 401 |
| `/metrics` with key | 200 | ✅ 200 |
| 6 rapid requests (limit=3) | first 3 → 200, then 429 | ✅ **429** + `Retry-After: 60` |

Constant-time comparison (`ct_eq`) and the fact that the rate-limit key is derived from the
validated key/IP (the attacker's header is not trusted) were verified in the code (`src/server/security.rs`).

### 4.5 Real LLM — Ollama qwen3:14b

`LORE_LLM_BASE=http://192.168.1.26:11434/v1 LORE_LLM_MODEL=qwen3:14b`

| Test | Result |
|------|--------|
| ask | ✅ "Waking up early lets you gain more time from the start of the day..." |
| ask/stream (SSE) | ✅ **real token stream**: `T` · `he` · ` sky` · ` is` · ` wri` · `ting`... |
| solve (ReAct) | ⚠️ tool-call format mismatch (see §5.1) |

### 4.6 Real LLM — Z.ai GLM-4.5-air

`LORE_LLM_BASE=https://api.z.ai/api/paas/v4 LORE_LLM_MODEL=glm-4.5-air`

| Test | Result |
|------|--------|
| ask | ✅ "Waking up early makes for a more productive and well-planned start to the day." |
| ask/stream (SSE) | ✅ token stream: `A` · `l` · `o` · `ne` · ` I` · ` stay`... |
| **solve (ReAct)** "calculate 23 + 17" | ✅ **`"23 + 17 = 40"`** (correct) |

### 4.7 Federation (multi-node)

Two nodes: A (`:13830`, agent Alpha) + B (`:13831`, agent Beta, `LORE_PEERS=http://127.0.0.1:13830`).

`POST /deliberate` from B:

```
Beta  node=None                       -> (local reply)
Alpha node=http://127.0.0.1:13830     -> (peer reply, node-labeled)
```
✅ **2 replies** — the local team and the peer Lore node merged. The peer `node` label is correct.

### 4.8 WebSocket — `/deliberate/live`

Python stdlib raw WebSocket client (handshake + masked text frame). A two-agent service:

```
handshake: HTTP/1.1 101 Switching Protocols
frame: {"id":"...","name":"Aria","reply":"..."}   ← streams the moment it's ready
frame: {"id":"...","name":"Kai","reply":"..."}
frame: [DONE]
```
✅ Live deliberate, replies stream as frames as they become ready.

### 4.9 Memory behaviors

| Test | Result |
|------|--------|
| **Scope isolation** — "secret password" to A, recall from B | ✅ A: found at score 1.037 · B: **empty** (no leak, keyword + semantic) |
| **reembed** | ✅ 2 records re-embedded with the active embedder |
| **export determinism** (2× diff) | ✅ **identical** (diffable backup) |
| **consolidate** | ✅ scanned: 2 · merged: 0 · forgotten: 0 (fresh records) |
| **semantic recall** "pets" → "love of cats" | ✅ score **0.935** (cosine, synonym) |

### 4.10 Performance — cold start

> ⚠️ **Important correction:** In the first measurements I thought "cold start was 31–42 s".
> That was an **artifact** of the bash `&` + `&&` chain + curl polling interaction. Verification:

| Method | Result |
|--------|--------|
| `strace -f -c` (MockModel serve) | total syscalls **117 ms**, bind 6 µs |
| Python subprocess + `urllib` poll | **health READY: 0.015 s**, create 0.015 s |

**Conclusion:** the service is ready in 15 ms. No performance issue.

---

## 5. Findings

### 5.1 🟡 MEDIUM — solve/ReAct tool-call format mismatch with qwen3

**Location:** `src/tool/mod.rs` — `LlmRouter::route` and `parse_tool_call`.

**Problem:** `LlmRouter` presents the tool catalog to the model (`- hesap: does arithmetic
with two numbers and an operator`) and asks for the `{"tool":"<name>","args":"<string>"}`
template. But it **doesn't specify what format args should be in**. `CalcTool`
(`src/tool/builtin.rs`) expects whitespace-separated `23 17 +`, while qwen3 returned
`{"tool":"hesap","args":"23,17,\"+\""}` (comma-separated).

- `parse_tool_call` accepts this as valid (tool="hesap" exists).
- `CalcTool::run("23,17,\"+\"")` → `split_whitespace` yields one token → parse error ("two numbers required").
- Despite the ReAct observe feedback, qwen3 didn't correct itself; the raw JSON became the final reply.
- GLM-4.5-air bypassed the tool and computed `"23 + 17 = 40"` itself (correct result, but the tool wasn't called).

**Impact:** solve can't use a tool with MockModel (expected — it requires tool-calling). With qwen3
the tool chain breaks. With GLM the result works but doesn't use the tool path.

**Recommendation:** an **args format example/signature per tool** can be added to the `LlmRouter`
prompt: like `hesap: "<number> <operator> <number>"`. Or the CalcTool args parser can be made more
flexible to also accept a comma as a separator.

### 5.2 🔵 MINOR — GLM-4.6 reasoning model incompatibility

**Location:** `src/model/openai.rs` — `parse_response`.

**Problem:** The GLM-4.6 reasoning model produces a `reasoning_content` field in its response.
lore only reads `message.content`. If `max_tokens` is low, the entire budget goes to reasoning
and content stays empty.

**Impact:** Limited — lore doesn't send `max_tokens` in `build_payload` (the provider default),
so content fills in normal usage. But on reasoning models `reasoning_content` is deliberately
ignored (token waste / compatibility note).

**Recommendation (optional):** a `max_tokens` parameter can be added to `Prompt`/`OpenAiModel`;
`reasoning_content` can be logged optionally.

### 5.3 🔵 MINOR — HashingEmbedder weak on short-text morphology

**Location:** `src/memory/embed.rs` + `retrieval.rs`.

**Problem:** `recall "learning" --semantic` (memory: "Learned Rust") returned empty — char n-gram
feature hashing can fall below the 0.40 semantic gate on short single-token queries.
It works well on longer text ("pets" → "love of cats" 0.935).

**Impact:** Low — the gate calibration is embedder-specific (`Embedder::semantic_gate`), and the
neural embedder (`--features neural`) closes this scenario. The design note already says "best on
short phrases"; there's an inconsistency.

### 5.4 ❌ FALSE POSITIVE — cold start

The first tests assumed 31–42 s; a bash test-harness artifact. **The truth is 15 ms** (§4.10).
No problem, removed from the report.

---

## 6. Overall Assessment

| Criterion | Assessment |
|-----------|------------|
| **Code quality** | ⭐⭐⭐⭐⭐ Clean, modular, well-commented. `clippy -D warnings` clean. |
| **Test coverage** | ⭐⭐⭐⭐⭐ 108 unit tests, end-to-end HTTP/streaming/federation/ws tests. |
| **Security** | ⭐⭐⭐⭐⭐ Constant-time comparison, rate-limit key from validated key/IP, scope isolation (runtime), WAL, no 500 detail leakage. |
| **Resilience** | ⭐⭐⭐⭐⭐ spawn_blocking, timeouts, fault-tolerant deliberate, graceful shutdown, embedder signature tracking. |
| **Independence** | ⭐⭐⭐⭐⭐ No external service, everything inside the binary (the default build is fully offline). |
| **Real-world compatibility** | ⭐⭐⭐⭐½ Works with Ollama + Z.ai; a single tool-call format finding. |
| **Documentation** | ⭐⭐⭐⭐⭐ DESIGN.md + README.md are comprehensive, the phased roadmap is clear. |

**Conclusion:** Lore is a near-production-quality core, written with good engineering discipline.
No serious bugs. The only medium finding (LlmRouter args format) can be resolved with a small
improvement.

---

## 7. Recommendations

1. **(Medium)** Add an args format example per tool to the `LlmRouter` prompt — so ReAct solve
   works with models like qwen3.
2. **(Minor)** An optional `max_tokens` on `OpenAiModel` + a reasoning model (`reasoning_content`)
   note.
3. **(Minor)** Let the `CalcTool` args parser also accept comma separators (model flexibility).

---

## Appendix 0 — Resolution Status (17 Jul 2026, post-report)

All findings + additional findings from an independent double review were closed (108 → **126 tests**):

| Finding | Status | Resolution |
|---------|--------|------------|
| §5.1 tool-call format | ✅ | `Tool::args_hint()` + a shared `catalog()` (LlmRouter+solve) + tolerant `normalize_args` (comma/JSON-like/concatenated; sign & scientific notation & hex preserved) |
| §5.2 reasoning_content | ✅ | fall back to `reasoning_content` when content is empty + `with_max_tokens` / `LORE_LLM_MAX_TOKENS` |
| §5.3 short-query semantics | ✅ | Token-level cosine fallback (`Embedder::token_fallback`, `cosine_tok` signal, Scorer token cache) — `recall "learning" --semantic` now finds it |
| Review: unbounded memory growth | ✅ | Automatic records get `AUTO_IMPORTANCE=0.2` (< forgetting threshold 0.25) — decay now works; explicit records + `tell` keep 0.5 |
| Review: session table DoS | ✅ | Hard cap + TTL + LRU (eviction not applied to an existing session) + session name ≤128B |
| Review: SQL deleted scan | ✅ | `AND deleted = 0` (with_deleted still reaches it) |
| Review: update/delete race | ✅ | Persist writes the latest state under a read lock |
| Review: WS deliberate | ✅ | Parallel collection (FuturesUnordered) + 30 s question timeout + errors logged |
| Review: peer body limit | ✅ | 2MB limit + status code check |
| Review: `local`+synthesizer | ✅ | The depth-1 guarantee is preserved with the synthesizer too |
| Review: CLI/env hygiene | ✅ | `--kind` validated; warns if `LORE_RATE_LIMIT`/`LORE_CONSOLIDATE_SECS` are invalid (0 would panic the janitor); empty name/role 422; recall/board limit ≤1000; near-dup scan is scope-partitioned |

---

## Appendix A — Core commands run

```bash
# Quality
cargo fmt --check
cargo build
cargo clippy --all-targets -- -D warnings
cargo test

# CLI demo
cargo run -- demo

# Service + test
./target/debug/lore serve --addr 127.0.0.1:13777
curl http://127.0.0.1:13777/health

# Real model (Ollama)
LORE_LLM_BASE=http://192.168.1.26:11434/v1 LORE_LLM_MODEL=qwen3:14b \
  ./target/debug/lore serve --addr 127.0.0.1:13801

# Real model (Z.ai)
LORE_LLM_BASE=https://api.z.ai/api/paas/v4 LORE_LLM_MODEL=glm-4.5-air \
  LORE_LLM_KEY=<key> ./target/debug/lore serve --addr 127.0.0.1:13803

# Cold-start verification
strace -f -c -o /tmp/strace.txt ./target/debug/lore serve --addr 127.0.0.1:13811
```

## Appendix B — Files tested

- `src/lib.rs`, `src/main.rs`, `src/error.rs`, `src/id.rs`
- `src/agent/{mod,persona,conversation}.rs`
- `src/memory/{mod,types,in_memory,sqlite,embed,retrieval,rerank,evolution,graph}.rs`
- `src/model/{mod,mock,openai}.rs`
- `src/orchestrator/{mod,message,registry}.rs`
- `src/server/{mod,api,state,security,deliberate,types,tests}.rs`
- `src/tool/{mod,builtin}.rs`
- `Cargo.toml`, `Dockerfile`, `DESIGN.md`, `README.md`
