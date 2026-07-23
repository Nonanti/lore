# Phase 5 — Memory Distillation: coworkers that learn on the job

Roadmap: Phase 1 (hands) ✔, Phase 2 (work loop), Phase 3 (daemon),
Phase 4 (team). This is Lore's differentiator: after each task, the agent
**distills what it learned** into its own memory — so the next task starts
smarter. The memory engine already supports everything needed; this phase
wires the work loop into it.

## Design

### Task-level strategy memory — `src/agent/work.rs` extension

After every `work()` completion (success OR failure), the agent writes a
procedural memory itself (no model call — deterministic):

```rust
MemoryKind::Procedural {
    title: format!("task: {}", goal_summary),          // first 80 chars of goal
    steps: vec![
        format!("workspace: {}", spec.workspace.display()),
        format!("verify: {}", spec.verify.join(" && ")),
        format!("iterations used: {}", report.iterations),
    ],
    successes: if success { 1 } else { 0 },
    failures: if success { 0 } { 1 },
}
```

Wilson scoring + dedup/merge already exist in the memory engine — repeated
similar tasks strengthen or weaken the strategy automatically. This closes
the experiential loop the DESIGN.md survey flagged as "the biggest gap in
the literature".

### Convention distillation — `Agent::distill_work`

```rust
pub async fn distill_work(&self, spec: &WorkSpec, report: &WorkReport) -> Result<usize>
```

One model call (cheap prompt, the final answer + verify log tail, not the
whole transcript):

> "From this completed task, extract durable facts worth remembering for
> future work in this project: conventions, gotchas, commands that worked.
> Return JSON: `[{"kind":"convention"|"constraint"|"fact","title":"...","body":"..."}]`.
> At most 3 items; empty list if nothing durable."

- Items → `MemoryKind::Semantic` variants (Convention/Constraint/Fact —
  existing kinds), stored via `self.remember`. Return count.
- Called by the daemon after each task (config flag `distill: bool` on the
  agent JSON, default true). Direct `work()` callers opt in explicitly.
- Distillation failures (bad JSON, model error) are logged and never fail
  the task — learning is best-effort.

### Priors at task start — `work()` seeding

Before iteration 1, `work()` recalls and injects into the goal context:

- procedural strategies (already done by `solve` internally), plus
- **semantic conventions**: `recall(Query::new(goal).tier(Tier::Semantic).semantic().limit(3))`
  formatted as `[project convention] title — body` lines prepended to the
  goal text. So an agent that learned "this repo uses conventional commits"
  applies it without being told again.

### Scope isolation

Everything stays within the acting agent's own scope (existing retrieval
isolation): Kaya's conventions are not Ece's. Team tasks (Phase 4): each
child distills into its own memory; the PM's synthesis is distilled by the
PM only.

## Error handling

- Distillation writes go through the normal memory path (conflict detection,
  AUTO vs explicit importance — distilled items use the explicit 0.5 level
  since they are deliberate).
- `distill_work` on an empty/trivial report (success in 1 iteration, no
  failures seen) still runs — small wins teach conventions too — but the
  prompt allows "empty list".

## Testing

- work() success writes one procedural memory with successes=1; failure
  writes failures=1 (query the store after run with ScriptedModel).
- Repeated same-goal runs merge/strengthen via existing dedup (count does
  not grow unboundedly; Wilson moves).
- distill_work: ScriptedModel returns 2-item JSON → 2 semantic memories of
  the right kinds; garbage JSON → Ok(0), no error; empty list → Ok(0).
- Seeding: pre-store a convention, run work(), assert the scripted model's
  captured first prompt contains the convention line.
- Scope: two agents on the same store kind (two scopes) — agent B's seeded
  context does not contain agent A's convention.

## Non-goals

Cross-agent shared memory (a "company wiki" scope), automatic forgetting
tuning, LLM-based consolidation, cost-aware distillation model choice
(distill uses the agent's own model; per-role cheap models come from Phase 4
config).
