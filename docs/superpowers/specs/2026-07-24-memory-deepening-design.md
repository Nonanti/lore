# Memory Engine Deepening — Design (approved 2026-07-24)

Sub-project 1 of the post-native-tool-calling roadmap (B → D → C: engine
deepening → team memory sharing → dashboard). Answers DESIGN.md §11 open
question 2: *how far do we take "Alaz from zero"?*

## 1. Spine: measure first, build only what the numbers justify

The retrieval eval harness (`tests/eval.rs`) is **saturated**: hit@5 = 15/15
(100%) on a 15-record corpus of mostly single-token exact-match queries, with
a threshold literally marked `PLACEHOLDER`. Against a saturated benchmark no
signal addition is measurable — so every phase below is gated on the hardened
eval built in Phase 1, and any signal that does not move a metric is not
merged. This turns §11-Q2 from an opinion into a measurement.

## 2. Locked decisions

| # | Decision | Rationale |
|---|----------|-----------|
| M1 | **Offline-first**: the zero-dependency default build is the first-class target; neural stays an opt-in layer behind the existing `neural` feature (the `NeuralEmbedder` pattern) | "Fully self-contained" is the product identity (DESIGN §2); default-build users must feel the gains |
| M2 | **Eval before signals**: no retrieval change without a hardened-eval delta | Current benchmark is saturated; unmeasured tuning is superstition |
| M3 | **ColBERT rejected** | Token-level late interaction needs per-token vectors: storage blow-up, meaningless under `HashingEmbedder`, and the precision need is covered by cross-encoder rerank at a fraction of the cost |
| M4 | **RAPTOR-lite conditional** | Built only if the hardened eval shows abstraction/cluster misses that the graph signal does not fix; otherwise deferred with a decision record |
| M5 | Existing `Query` flag pattern (`semantic`/`diverse`/`rerank`) extends to the graph leg; defaults decided by eval numbers, not taste | Consistency + measurability |

## 3. Phases

### Phase 1 — Eval hardening (the yardstick)

`tests/eval.rs` grows from 15/15 to a corpus of **~60 records** and
**~35 queries**:

- **Distractors**: near-miss records sharing tokens with queries (keyword
  can now be fooled — rerank/cosine must earn their keep).
- **Clusters**: entity-bridged record groups ("Aylin owns cat Paspas" /
  "Paspas was vaccinated at the vet" / …) enabling **multi-hop queries**
  ("aylin pet health") that no single record answers lexically.
- **Paraphrase/synonym queries** with zero token overlap — the honest
  weak spot of `HashingEmbedder`; expected to miss at baseline (headroom
  for the neural layer to prove itself).
- **Morphology + mixed Turkish/English subset** (the harness comment
  already claims "Turkish-weighted"; make it true).
- **Metrics**: hit@1, hit@5, MRR@5 — all reported; thresholds set just
  below the measured baseline as regression alarms (`PLACEHOLDER` dies).
  Keyword-only baseline test stays.
- Query categories are tagged so per-category rates print per run
  (exact / morphology / paraphrase / multihop / distractor-resistance).

Deliverable: an honest baseline table in the eval output and in this spec's
addendum. **No production code changes in this phase.**

### Phase 2 — Graph signal in the hot path

`memory/graph.rs` exists (entity inverted index, BFS, path) but is wired to
nothing and rebuilds from the full scope per call. Integration:

- **Incremental entity index in both stores** — entities extracted at
  `remember` time (same `extract_entities` rules: 5W cues + tokens ≥4 chars
  minus stopwords, factored to be shared):
  - `InMemoryStore`: `HashMap<String, HashSet<MemoryId>>` maintained on
    remember/forget.
  - `SqliteStore`: new `entities(memory_id, entity)` table + index on
    `entity`, **schema v3** migration backfilling from existing records
    (same BEGIN IMMEDIATE transaction pattern as v1→v2).
- **Recall expansion leg** (`Query.graph: bool`): after first-pass scoring,
  take the top-K candidates' entities, find 1-hop neighbors via the index
  (capped), score them with a **damped boost**
  (`neighbor_score = parent_score × G_DAMP`, `G_DAMP ≈ 0.5`, tuned by
  eval), merge and re-sort. Neighbors already in the result set are not
  double-counted (max, not sum).
- **Default** decided by eval: `graph=true` becomes the default only if
  multi-hop rates improve without regressing exact/distractor rates;
  otherwise it stays opt-in like `rerank`.
- `MemoryGraph` (build/related/path) stays as the offline analysis API.

### Phase 3 — Fusion calibration + native rerank strengthening

Only-what-measures-up changes, all judged by the Phase-1 eval:

- `NativeReranker`: **IDF-weighted coverage** (rare terms count more than
  common ones) + a term-proximity feature; both cheap, offline.
- Fusion weights (`W_KEYWORD`/`W_COSINE`), semantic gate, and an **RRF
  (reciprocal rank fusion) experiment** vs the current weighted sum —
  adopt whichever wins on the eval, document the numbers.
- Token-fallback calibration for the morphology category.

### Phase 4 — Neural opt-in layer: cross-encoder rerank

- `NeuralReranker` implementing the existing `Reranker` trait via
  fastembed's text-rerank models (bge-reranker family), behind the
  **existing `neural` feature** (no new flags).
- Selection mirrors the embedder: `LORE_RERANKER=neural|native` env.
- Tests follow the `NeuralEmbedder` pattern: one `#[ignore]`d download
  test + the neural eval run reported in the addendum (expected to lift
  the paraphrase category specifically).

### Phase 5 (conditional) — RAPTOR-lite

Only if Phase 1–3 evidence shows cluster/abstraction queries failing:
consolidation-time clustering of related episodics (embedding similarity +
graph communities) into LLM-written summary records, graph-linked to their
members; recall hits the summary and expands via the Phase-2 leg. If the
evidence does not appear, defer with a decision record. **Not started
without the evidence.**

## 4. Gates & process

Per phase: `cargo fmt` + clippy `-D warnings` + full suite green + eval
metrics printed (and thresholds raised when quality rises — never lowered).
Conventional commits, one phase per commit minimum, independent review at
the end of the sub-project, live smoke where meaningful (graph leg exercised
through a real agent recall).

## 5. Out of scope

- Cross-agent/shared memory (sub-project D — next).
- Dashboard/visualization (sub-project C).
- ColBERT (M3), RAPTOR-lite without evidence (M4).
- Embedding-space migration tooling changes (`lore reembed` untouched).

## 6. Addendum (2026-07-24): measured results

Golden set: 56 records / 32 categorized queries (`tests/eval.rs`).

| Stack | hit@5 | MRR@5 | Exact | Morph | Paraphrase | MultiHop | Distractor |
|-------|-------|-------|-------|-------|------------|----------|------------|
| Hashing baseline (pre-graph) | 72% | .682 | 7/7 | 8/8 | 0/7 | 2/4 | 6/6 |
| + graph leg (Phase 2) | **78%** | .701 | 7/7 | 8/8 | 0/7 | **4/4** | 6/6 |
| e5 embedder + graph (opt-in) | **91%** | — | ✓ | ✓ | **6/7** | 2/4¹ | ✓ |
| + cross-encoder rerank (Phase 4) | **100%** | — | 7/7 | 8/8 | **7/7** | **4/4** | 6/6 |

¹ e5's wider candidate pool reshuffles graph seeds — exactly the precision
gap the cross-encoder closes.

Phase notes:

- **Phase 2 details**: acronym entities (2–4 char ALL-CAPS raw-text tokens:
  NAS/TLS/8TB) restored bridges the ≥4-char entity floor dropped — this
  single rule took MultiHop from 3/4 to 4/4. Agent context injection opts
  into `.graph()`; `Query` default stays off for API stability.
- **Phase 3 verdict (M2 applied)**: native-rerank feature work SKIPPED with
  evidence — rerank-on vs rerank-off measured **identical** on the golden
  set (all lexical categories already at hit@1); IDF/proximity features had
  nothing to move. Revisit only if a future eval exposes lexical-precision
  misses.
- **Phase 4 details**: `NeuralReranker` (fastembed `TextRerank`, default
  BGE-reranker-base, `with_model` escape hatch) behind the existing
  `neural` feature; store-attached via `with_reranker`, selected by
  `LORE_RERANKER=neural`; fail-open (rerank error keeps first-pass order).
  Agent context queries turn `.rerank()` on — native pass is
  measured-neutral and cheap; the cross-encoder upgrades it in place.
- **Phase 5 verdict (M4 applied)**: RAPTOR-lite DEFERRED with evidence —
  after Phases 2+4 the hardened set is fully solved (100%); no
  abstraction/cluster miss pattern remains to justify it. Revisit when an
  eval with long-horizon summarization queries shows the gap.
