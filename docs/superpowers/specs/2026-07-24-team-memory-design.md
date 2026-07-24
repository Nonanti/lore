# Team Memory (Agent-to-Agent Sharing) — Design (approved 2026-07-24)

Sub-project 2 of the B → D → C roadmap. Answers DESIGN.md §11 open
question 5's engineering half: agents on a team should **learn together** —
a convention one agent distills ("tests run with cargo nextest") must reach
its teammates' next task without any human relay.

## 1. The gap

Distillation writes conventions/constraints/facts into the agent's OWN
scope, and daemon agents live in **separate SQLite files**
(`data/memory/<agent>.db`) — so team members can never see each other's
lessons. `Scope::World` ("shared memory accessible to all agents") already
expresses the right visibility, but in the daemon it is trapped inside each
private file.

## 2. Locked decisions

| # | Decision | Rationale |
|---|----------|-----------|
| T1 | **No new Scope variant.** `Scope::World` is the team scope | The enum is serialized in every record; a new variant is a migration for zero expressive gain — World already means "visible to all" (`Scope::sees`) |
| T2 | **`CompositeStore`**: personal + shared handles behind one `MemoryStore` — writes route by scope (World → shared, Agent → personal), recalls merge | Storage-level solution; agents/distill/recall code stays store-agnostic |
| T3 | Only **Convention + Constraint** distillates are shared; Facts and procedures stay personal | Conventions/constraints are durable and universal; facts are often task-local; procedures carry per-agent Wilson stats that must not mix |
| T4 | Sharing is on by default when distillation is on; `--no-share` (CLI) / `with_share(false)` opts out per agent | Team flows exist to collaborate; the flag mirrors `--no-distill` exactly |
| T5 | Shared file: `data/memory/team.db`, WAL, opened by every worker | Same concurrency posture as `tasks.db`; B's `BEGIN IMMEDIATE` lesson already applied in `remember` |
| T6 | Cross-agent near-duplicates are consolidation's job (shared store consolidates like any store); no write-time cross-agent dedup | Consolidation already merges near-duplicates; duplicating that logic at write time is complexity without evidence |
| T7 | No attribution field on shared memories | `Memory` has no author field today; inventing one for this feature is YAGNI — revisit with a real need |

## 3. Design

### 3.1 `CompositeStore` (`src/memory/composite.rs`)

```rust
pub struct CompositeStore {
    personal: Arc<dyn MemoryStore>,
    shared: Arc<dyn MemoryStore>,
}
```

- `remember`: `Scope::World` → shared; `Scope::Agent(_)` → personal.
- `recall`: query BOTH with the same scope/query (stores enforce
  `Scope::sees` themselves — an `Agent(a)` query returns World records
  from the shared store, personal records from the own store), concat,
  sort by score desc, truncate to `query.limit`. Per-store finalize
  (rerank/MMR/graph) runs before the merge; MMR diversity is not
  re-applied across stores (documented tradeoff).
- `get`/`reinforce`/`forget`: personal first, `NotFound` → shared.
- `reinforce_many`: split ids by which store resolves them (via `get`),
  batch each side.
- `count`: sum; `export`: concat; `consolidate`: run on both, sum
  reports (shared file consolidation under concurrent workers is safe:
  WAL + IMMEDIATE, and merges are idempotent).

### 3.2 Distill routing (`agent/distill.rs`)

`distill_work`'s semantic write chooses scope per item:
`share_enabled && matches!(cat, Convention | Constraint)` → `Scope::World`,
else `self.scope()`. Failed-task constraint-only guard is unchanged (and
still shares — a "don't do X" lesson is exactly what teammates need).
Without a composite store, World records land in the agent's own store
where recall already sees them — graceful degradation, no daemon required.

### 3.3 Agent flag + CLI

`Agent.share: Option<bool>` (None = true), `with_share` builder,
`share_enabled()` accessor; `AgentRecord.share` with the same serde shape
as `distill` (absent = default, old records load unchanged);
`lore agent create --no-share`.

### 3.4 Daemon wiring (`daemon.rs`)

Per-task store becomes
`CompositeStore::new(personal_sqlite, team_sqlite)` where team =
`data/memory/team.db` (same embedder). Every worker opens both; the
existing store-open concurrency tests cover the shape.

## 4. Testing

- Composite unit: write routing by scope, recall merge order + limit,
  get/reinforce/forget fallback, count/export/consolidate aggregation,
  reinforce_many split.
- Sharing flow: two `Agent`s over two composites sharing ONE shared store
  — agent A's distilled convention appears in agent B's context recall;
  facts/procedures do NOT cross; `with_share(false)` keeps everything
  personal.
- Record compat: old persona JSON (no `share` field) loads with sharing on.
- Daemon: team.db created and populated through a worked task (scripted
  model), teammate's recall sees the convention.
- Eval untouched (recall semantics unchanged for single stores).

## 5. Gates & process

fmt + clippy `-D warnings` (default + `neural`) + full suite green + 5×
stress on daemon tests (new shared-file concurrency surface). Conventional
commits, independent review at the end, push.

## 6. Out of scope

- Dashboard visibility of team memory (sub-project C).
- Attribution/provenance fields (T7).
- Cross-team hierarchies, per-role sharing policies — no evidence yet.
- Server mode changes: single shared `lore.db` already gives World
  visibility; composite is a daemon concern.
