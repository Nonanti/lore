# Next Roadmap — post "AI coworkers" (approved 2026-07-24)

Source: 4-way general review (memory engine, server+auth, orchestration+
providers, architecture/health). All Class-1 findings verified against code;
OAuth-state finding rejected (caller verifies — main.rs:1249/1279/1303).

## Phase A — Correctness & performance sweep (first; memory quality is the product)

1. `retrieval.rs:326` — recency uses `last_access` not `created_at`
   (align with `should_forget`); unit test: old-created/recently-accessed
   outranks fresh-created/never-accessed.
2. `sqlite.rs load_by_ids` — replace N+1 loop with parameterized
   `IN (...)` batch (chunked at 500 ids); keep ordering by input ids.
3. `server/state.rs` — split rate-limit maps (main vs fail) or evict with
   `max(rate.per, fail_rate.per)`; test the cross-eviction case.
4. `anthropic.rs` + `openai.rs` streams — premature close without terminal
   event (`message_stop`/`[DONE]`) → `Err(LoreError::Model("stream ended
   without terminal event"))` instead of silent partial.
5. `openai.rs complete_stream` — handle `reasoning_content` deltas (or
   fallback at stream end if content empty), parity with `complete()`.
6. `api.rs` — explicit `DefaultBodyLimit::max(2 MiB)` layer.
7. `sqlite.rs reinforce_many` — single transaction around the loop.
8. `sqlite.rs blob_to_emb` — `tracing::warn!` on non-multiple-of-4 BLOB.
9. `embed.rs NeuralEmbedder` — poison recovery `unwrap_or_else(into_inner)`.

## Phase B — Documentation sync (parallel with A; highest leverage/cost)

1. README: new section "AI coworkers" — hands+policy, work loop, daemon,
   task CLI (`task add/list/status/log`, `inbox`, `approve|deny`), team
   (`agent create/list`, `--team`, roles), distillation, sandbox
   (`sandbox_exec`), `--concurrency`; quick-start flow example.
2. README fixes: endpoint table (`/ready`, `/openapi.json`, metrics auth
   note), test count (507+), remove stale M0–M31 status block.
3. DESIGN.md: new chapter for the coworkers platform (hands/work/daemon/
   team/distill as D11+ decisions), §5 module map update, §10 pointer to
   the new roadmap docs, status header refresh.
4. TEST_REPORT.md: add "historical snapshot (2026-07-17)" header note.

## Phase C — E2E harness (regression insurance for the race-prone area)

1. `tests/e2e.rs`: real-binary daemon + `task add` + work + `inbox`/`approve`
   flow (MockModel via env), SIGKILL-during-task → restart → recovery sweep
   observed via `task status`.
2. Team flow e2e: `--team` → children → reviewer → synthesis (scripted).
3. bwrap smoke test, skipped when bwrap absent.
4. Distill golden-set in `tests/eval.rs`-style: fixed scripted model →
   expected distilled kinds/categories; alarm on prompt regressions.

## Phase D — Task HTTP surface (completes the self-hosted positioning)

1. Routes (protected, same API-key auth): `POST /tasks`, `GET /tasks`,
   `GET /tasks/:id`, `GET /tasks/:id/log`, `GET /inbox`,
   `POST /approvals/:id/approve`, `POST /approvals/:id/deny`.
2. openapi.json regeneration + server tests (auth required, validation,
   404s, approval idempotency).
3. CLI stays functional as-is; thin-client migration is a later decision.

## Phase E — Maintenance & consolidation (after features settle)

1. axum 0.7→0.8 (`{id}` path syntax migration incl. tests), rusqlite bump.
2. Module splits at the M25 threshold: `task/mod.rs` (1943) → store+types,
   `agent/work.rs` (1449) → loop+tests split, `memory/sqlite.rs` (1411).
3. Orchestrator role decision: document mailbox Orchestrator as lib-API
   (embedded use) or consolidate; move `orchestrator/pm.rs` → `team/`.
4. Persona field sanitization at creation (reject control chars/newlines
   in name/role/traits) + doc the residual prompt-injection boundary.

## Gates per phase

fmt + clippy `-D warnings` zero + full suite green + 5× stress where
concurrency-adjacent. Conventional commits. Spec/review docs under
`docs/superpowers/specs/`.
